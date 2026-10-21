#!/usr/bin/env python3
"""mcap 의 LiDAR 스캔을 world 좌표 점 + 센서 원점으로 내보내기 (occupancy 레이캐스팅용).
출력: <out_dir>/%06d.npz {xyz (N,3) float32 world, origin (3,) float32 world, stamp_ns}
사용: python3 export_lidar_scans.py <log.mcap> <calib_dir> <out_dir> [--max-range 40]
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # scripts/ (패키지 import)
import argparse, os, struct, sys
import numpy as np, yaml
from preprocess.geometry import raisin_msgs as RM   # 레코더 v1/v2 레이아웃 공용 디코더
from mcap.reader import make_reader

LIDAR_DT = np.dtype([("ot", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("refl", "u1"), ("tag", "u1"), ("line", "u1")])


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def slerp(q0, q1, u):
    d = float(np.dot(q0, q1))
    if d < 0: q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u*(q1-q0); return q/np.linalg.norm(q)
    th = np.arccos(d)*u; q2 = q1 - q0*d; q2 /= np.linalg.norm(q2)
    return q0*np.cos(th) + q2*np.sin(th)


ap = argparse.ArgumentParser()
ap.add_argument("mcap"); ap.add_argument("calib"); ap.add_argument("out"); ap.add_argument("--max-range", type=float, default=40.0)
ap.add_argument("--deskew", action="store_true", help="그래프 없이 원 오도메트리로 점별 시각 deskew (held-out 질의 스캔용)")
ap.add_argument("--pose-graph", help="graph/ 디렉터리 — PGO 보정 자세 + 점별 시각 deskew 로 world 점 생성")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from preprocess.geometry.extrinsics import lidar_t_bl
t_bl = lidar_t_bl(a.calib)                                   # 회전은 항등 (점의 180° 보정은 디코더에서). LIDAR_EXTRINSIC=slam|calib

gt_ns, gt_pos, gt_quat, scans = [], [], [], []
with open(a.mcap, "rb") as f:
    for sch, ch, msg in make_reader(f).iter_messages(topics=["/Odometry/base", "/livox/lidar"]):
        if ch.topic == "/livox/lidar":
            from preprocess.geometry.pc2_reader import decode_livox                          # CustomMsg / PointCloud2 자동, LIVOX_FLIP 환경변수
            ns_, xyz_, ot_, _ = RM.lidar_scan(msg.data, sch); scans.append((ns_, xyz_, ot_))
        else:
            ns_v, p_v, q_v = RM.odom(msg.data, sch); v = (*p_v, *q_v); sec, nsec = divmod(ns_v, 10**9)
            gt_ns.append(sec*10**9+nsec); gt_pos.append(v[0:3]); gt_quat.append([v[6], v[3], v[4], v[5]])
o = np.argsort(gt_ns); gt_ns = np.array(gt_ns)[o]; gt_pos = np.array(gt_pos)[o]; gt_quat = np.array(gt_quat)[o]
gt_quat /= np.linalg.norm(gt_quat, axis=1, keepdims=True)

pp = None
if a.pose_graph or a.deskew:
    if a.pose_graph:
        from preprocess.geometry.pose_provider import PoseProvider
        pp = PoseProvider(a.pose_graph); print("pose graph:", a.pose_graph, pp.meta, "· deskew on")
    else:                                                   # 원 오도메트리 보간기(Odom)를 같은 인터페이스로 감쌈
        from preprocess.geometry.deskew import Odom
        class _Raw:
            def __init__(s_, od): s_.od = od
            def at_many(s_, t):
                R, p, ok = s_.od.at(np.asarray(t, np.int64)); T = np.tile(np.eye(4), (len(t), 1, 1)); T[:, :3, :3] = R; T[:, :3, 3] = p; return T, ok
            def at(s_, ns):
                T, ok = s_.at_many([int(ns)]); return T[0] if ok[0] else None
        pp = _Raw(Odom(gt_ns, gt_pos, gt_quat)); print("deskew on (raw odometry)")
n_ok = 0
for i, (ns, xyz, ot) in enumerate(scans):
    rng = np.linalg.norm(xyz, axis=1); keep = (rng > 0.6) & (rng < a.max_range)
    if pp is not None:
        T0 = pp.at(ns)
        if T0 is None: continue
        ot = ot if ot.max() < 2_000_000_000 else ot - ot.min(); nb = 64
        bins = np.minimum((ot[keep] * nb // max(int(ot.max()) + 1, 1)).astype(int), nb - 1)
        Tb, okb = pp.at_many(ns + ((np.arange(nb) + 0.5) * (ot.max() + 1) / nb).astype(np.int64))
        if not okb.all(): continue
        Pl = xyz[keep].astype(np.float64) + t_bl
        pw = np.einsum("nij,nj->ni", Tb[bins, :3, :3], Pl) + Tb[bins, :3, 3]        # 점별 측정 시각의 보정 자세로 world
        origin = (T0[:3, :3] @ t_bl + T0[:3, 3]).astype(np.float32)
    else:
        k = np.searchsorted(gt_ns, ns)
        if k == 0 or k == len(gt_ns) or gt_ns[k]-gt_ns[k-1] > 250e6:
            continue
        u = (ns-gt_ns[k-1])/(gt_ns[k]-gt_ns[k-1])
        R = q2R(slerp(gt_quat[k-1], gt_quat[k], u)); t = gt_pos[k-1]*(1-u)+gt_pos[k]*u
        pw = xyz[keep] @ R.T + (R @ t_bl + t)
        origin = (R @ t_bl + t).astype(np.float32)
    np.savez(os.path.join(a.out, "%06d.npz" % i), xyz=pw.astype(np.float32), origin=origin, stamp_ns=np.int64(ns))
    n_ok += 1
print("스캔 %d/%d 저장 → %s" % (n_ok, len(scans), a.out))
