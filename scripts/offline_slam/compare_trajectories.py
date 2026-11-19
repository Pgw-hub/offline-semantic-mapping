#!/usr/bin/env python3
"""오프라인 SLAM 검증 — 같은 녹화의 실시간 SLAM(/Odometry/base, /path/pgo)과 오프라인 재생 결과를 헤더 시각별로 비교.

출력: 매칭 개수, 원점 그대로의 위치 차, 강체 정렬(Umeyama, 회전+이동) 후 위치 차, yaw 차, 출력 누락, 키프레임 수.
사용: python compare_trajectories.py <실시간 mcap_dir> <오프라인 mcap_dir> [--logs ...] [--out report.json]
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from preprocess.geometry import raisin_msgs as RM
from mcap.reader import make_reader


def load(d, logs):
    fs = sorted(glob.glob(os.path.join(d, "log_[0-9][0-9][0-9].mcap")))
    if logs is not None: fs = [f for f in fs if int(os.path.basename(f)[4:7]) in logs]
    od, kf, frames, lidar = {}, {}, {}, 0
    for f in fs:
        for sch, ch, m in make_reader(open(f, "rb")).iter_messages(topics=["/Odometry/base", "/path/pgo", "/livox/lidar"]):
            if ch.topic == "/livox/lidar": lidar += 1; continue
            if ch.topic == "/Odometry/base":
                c = RM._Cur(m.data, 8); _, fr = c.header(); frames[fr] = frames.get(fr, 0) + 1
                ns, p, q = RM.odom(m.data, sch); od[ns] = np.r_[p, q]
            else:
                P, pns = RM.path_poses(m.data, sch)
                if pns is not None:
                    for t, x in zip(pns, P): kf[int(t)] = x
    main = max(frames, key=frames.get) if frames else None
    return od, kf, frames, main, lidar


def yaw(q):  # xyzw
    x, y, z, w = q; return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def umeyama(A, B):  # B ≈ R A + t
    ma, mb = A.mean(0), B.mean(0); H = (A - ma).T @ (B - mb); U, S, Vt = np.linalg.svd(H); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T; return R, mb - R @ ma


ap = argparse.ArgumentParser(); ap.add_argument("live"); ap.add_argument("offline"); ap.add_argument("--logs", type=int, nargs="*"); ap.add_argument("--out")
a = ap.parse_args()
lo, lk, lf, lmain, nl = load(a.live, a.logs); oo, ok, of, omain, _ = load(a.offline, a.logs)
common = sorted(set(lo) & set(oo))
L = np.array([lo[t] for t in common]); O = np.array([oo[t] for t in common]); t = (np.array(common) - common[0]) / 1e9
raw = np.linalg.norm(L[:, :3] - O[:, :3], axis=1)
R, tr = umeyama(O[:, :3], L[:, :3]); Oa = O[:, :3] @ R.T + tr; al = np.linalg.norm(L[:, :3] - Oa, axis=1)
dyaw = np.degrees(np.angle(np.exp(1j * (np.array([yaw(q) for q in L[:, 3:]]) - np.array([yaw(q) for q in O[:, 3:]]) - np.arctan2(R[1, 0], R[0, 0])))))
path = lambda P: float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
rep = {"lidar_scans": nl, "live_odom": len(lo), "offline_odom": len(oo), "matched": len(common), "live_frames": lf, "offline_frames": of,
       "live_keyframes": len(lk), "offline_keyframes": len(ok), "path_live_m": round(path(L[:, :3]), 2), "path_offline_m": round(path(O[:, :3]), 2),
       "raw_pos_diff_cm": {"median": round(100 * float(np.median(raw)), 2), "p90": round(100 * float(np.percentile(raw, 90)), 2), "max": round(100 * float(raw.max()), 2)},
       "aligned_pos_diff_cm": {"median": round(100 * float(np.median(al)), 2), "p90": round(100 * float(np.percentile(al, 90)), 2), "max": round(100 * float(al.max()), 2)},
       "align_rot_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 3), "align_t_cm": np.round(100 * tr, 2).tolist(),
       "align_yaw_deg": round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 3), "align_tilt_deg": round(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), 3),   # 중력 방향 차 (높이 드리프트로 이어짐)
       "yaw_diff_deg": {"median": round(float(np.median(np.abs(dyaw))), 3), "max": round(float(np.abs(dyaw).max()), 3)}}
q = np.linspace(0, len(t) - 1, 9).astype(int)
rep["aligned_diff_along_time"] = [(round(float(t[i]), 1), round(100 * float(al[i]), 2)) for i in q]
print(json.dumps(rep, indent=1, ensure_ascii=False))
if a.out: json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)
