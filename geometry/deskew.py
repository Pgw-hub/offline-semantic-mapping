#!/usr/bin/env python3
"""Livox 스캔 deskew — 점별 시각(CustomMsg ot: 스캔 시작 기준 ns 오프셋)에 odometry 를 보간해 각 점을 스캔 기준 시각의 프레임으로 되돌린다.

원리: 점 p 가 시각 t_i 에 측정됨 → world 좌표 P_w = T_odom(t_i) · T_bl · p. 스캔 기준 시각 t_0 의 base 프레임으로 표현하면 p' = (T_odom(t_0)·T_bl)⁻¹ · P_w.
(IMU 적분 없이 LIO odometry 10Hz 보간을 쓰므로, 100ms 스캔 안의 등속 운동 가정과 같다. 회전은 slerp.)
사용: python deskew.py <log.mcap> <calib_dir> <out_dir>   →  out_dir/%06d.npz {xyz_deskew (base 프레임), xyz_raw, stamp_ns, T_wb (스캔 시각 odometry)}
      python deskew.py <log.mcap> <calib_dir> --measure   →  deskew 전/후 지면 두께(z 표준편차) 비교만 출력
"""
import argparse, os, struct, sys, time
import numpy as np, yaml
from mcap.reader import make_reader

LIDAR_DT = np.dtype([("ot", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("refl", "u1"), ("tag", "u1"), ("line", "u1")])


def q2R(q):  # w,x,y,z
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp_batch(q0, q1, u):
    """q0,q1 (N,4) wxyz, u (N,) → (N,4)"""
    d = np.sum(q0 * q1, 1); flip = d < 0; q1 = np.where(flip[:, None], -q1, q1); d = np.abs(d)
    out = np.empty_like(q0); lin = d > 0.9995
    out[lin] = q0[lin] + u[lin, None] * (q1[lin] - q0[lin])
    th = np.arccos(np.clip(d[~lin], -1, 1)) * u[~lin]; q2 = q1[~lin] - q0[~lin] * d[~lin, None]; q2 /= np.linalg.norm(q2, axis=1, keepdims=True) + 1e-12
    out[~lin] = q0[~lin] * np.cos(th)[:, None] + q2 * np.sin(th)[:, None]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


class Odom:
    def __init__(self, ns, pos, quat):
        o = np.argsort(ns); self.ns = np.array(ns)[o]; self.pos = np.array(pos)[o]; self.q = np.array(quat)[o]; self.q /= np.linalg.norm(self.q, axis=1, keepdims=True)
    def at(self, t):
        """t (N,) ns → R (N,3,3), p (N,3); 범위 밖은 None 마스크"""
        t = np.asarray(t, np.int64); k = np.searchsorted(self.ns, t); ok = (k > 0) & (k < len(self.ns))
        k = np.clip(k, 1, len(self.ns) - 1); u = (t - self.ns[k - 1]) / np.maximum(self.ns[k] - self.ns[k - 1], 1)
        u = np.clip(u, 0, 1).astype(float); p = self.pos[k - 1] * (1 - u)[:, None] + self.pos[k] * u[:, None]; q = slerp_batch(self.q[k - 1], self.q[k], u)
        R = np.stack([q2R(qq) for qq in q]) if len(q) < 64 else R_batch(q)
        return R, p, ok & (t - self.ns[k - 1] < 250_000_000)


def R_batch(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3)); R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y); return R


def read_mcap(path):
    gt_ns, gt_pos, gt_q, scans = [], [], [], []
    with open(path, "rb") as f:
        for sch, ch, msg in make_reader(f).iter_messages(topics=["/Odometry/base", "/livox/lidar"]):
            if ch.topic == "/livox/lidar":
                sec, nsec = struct.unpack_from("<iI", msg.data, 0); N, = struct.unpack_from("<I", msg.data, 24)
                p = np.frombuffer(msg.data, dtype=LIDAR_DT, count=N, offset=28)
                scans.append((sec * 10**9 + nsec, np.stack([-p["x"], -p["y"], p["z"]], -1).astype(np.float64), p["ot"].astype(np.int64)))
            else:
                v = struct.unpack_from("<7d", msg.data, 8); sec, nsec = struct.unpack_from("<iI", msg.data, 0)
                gt_ns.append(sec * 10**9 + nsec); gt_pos.append(v[0:3]); gt_q.append([v[6], v[3], v[4], v[5]])
    return Odom(gt_ns, gt_pos, gt_q), scans


def deskew_scan(odom, t0, xyz, ot, t_bl):
    """스캔 점 xyz(lidar 프레임, 스캔 시작 t0 기준 오프셋 ot) → 스캔 시작 시각 base 프레임 점"""
    # Livox ot: 스캔 내 오프셋(ns). 값이 이상하게 크면(절대시각) 보정
    ot = ot if ot.max() < 2_000_000_000 else (ot - ot.min())
    ti = t0 + ot
    # 성능: 점을 시간 구간 64개로 묶어 구간 대표 시각으로 보간
    nb = 64; bins = np.minimum((ot * nb // max(int(ot.max()) + 1, 1)).astype(int), nb - 1)
    tb = t0 + (np.arange(nb) + 0.5) * (ot.max() + 1) / nb
    Rb, pb, okb = odom.at(tb.astype(np.int64)); R0, p0, ok0 = odom.at(np.array([t0]))
    if not ok0[0] or not okb.all(): return None
    Pl = xyz + t_bl                                  # base 프레임(회전 항등) — lidar 오프셋만
    Pw = np.einsum("nij,nj->ni", Rb[bins], Pl) + pb[bins]                 # 각 점의 측정 시각 world 좌표
    Pb0 = (Pw - p0[0]) @ R0[0]                                            # 스캔 시각 base 프레임으로
    return Pb0, R0[0], p0[0]


def ground_thickness(P_base):
    """base 프레임 점 중 지면(z ∈ [-1.2,-0.4], 반경 3~15m) 의 z 표준편차 — 작을수록 스캔이 덜 뭉개짐"""
    r = np.hypot(P_base[:, 0], P_base[:, 1]); m = (r > 3) & (r < 15) & (P_base[:, 2] > -1.2) & (P_base[:, 2] < -0.4)
    if m.sum() < 200: return np.nan
    z = P_base[m, 2]; A = np.c_[P_base[m, 0], P_base[m, 1], np.ones(m.sum())]; c, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(np.std(z - A @ c))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mcap"); ap.add_argument("calib"); ap.add_argument("out", nargs="?"); ap.add_argument("--measure", action="store_true"); ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    tr = yaml.safe_load(open(os.path.join(a.calib, "livox_lidar_0_lidar.yaml")))["transform"]; t_bl = np.array([tr["translation"][k] for k in "xyz"])
    odom, scans = read_mcap(a.mcap); t = time.time()
    if a.out: os.makedirs(a.out, exist_ok=True)
    th_raw, th_dsk, n_ok = [], [], 0
    for i, (t0, xyz, ot) in enumerate(scans[:: a.stride]):
        res = deskew_scan(odom, t0, xyz, ot, t_bl)
        if res is None: continue
        Pb, R0, p0 = res; n_ok += 1
        if a.measure or i % 50 == 0:
            th_raw.append(ground_thickness(xyz + t_bl)); th_dsk.append(ground_thickness(Pb))
        if a.out:
            T = np.eye(4); T[:3, :3] = R0; T[:3, 3] = p0
            np.savez(os.path.join(a.out, "%06d.npz" % i), xyz_deskew=Pb.astype(np.float32), xyz_raw=(xyz + t_bl).astype(np.float32), stamp_ns=np.int64(t0), T_wb=T.astype(np.float64), ot_max_ms=float(ot.max()) / 1e6)
    th_raw = np.array(th_raw); th_dsk = np.array(th_dsk); m = np.isfinite(th_raw) & np.isfinite(th_dsk)
    print(f"스캔 {n_ok}/{len(scans[::a.stride])} 처리 [{time.time()-t:.1f}s] · 지면 두께(z std) raw {np.median(th_raw[m])*100:.1f} cm → deskew {np.median(th_dsk[m])*100:.1f} cm (중앙값, n={m.sum()})")
    if a.out: print("저장:", a.out)
