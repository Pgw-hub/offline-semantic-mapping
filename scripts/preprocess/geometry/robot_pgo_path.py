#!/usr/bin/env python3
"""로봇 lidar_slam 의 /path/pgo (마지막 메시지 = 최종 보정 키프레임 자세) 를 읽어 키프레임 시각을 복원하고 TUM 궤적으로 저장.
레코더가 pose 별 stamp 를 버리므로, 연속 키프레임 간 상대 이동(위치+yaw)이 오도메트리(/Odometry/base)의 상대 이동과 맞는 시각을 순차 탐색해 복원한다.
  python geometry/robot_pgo_path.py <mcap_dir> <out_dir> [--logs 0,13] [--rrd]
산출: <out>/robot_pgo_tum.txt (# timestamp tx ty tz qx qy qz qw), robot_pgo.npz (t, T_pgo[N,4,4], T_odom[N,4,4], match_err), robot_pgo_view.rrd
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))  # scripts/ (패키지 import)
import argparse, glob, os, struct, sys
import numpy as np
from preprocess.geometry import raisin_msgs as RM   # 레코더 v1/v2 레이아웃 공용 디코더
from mcap.reader import make_reader

def q2R(q):  # x,y,z,w
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def R2q(R):
    from scipy.spatial.transform import Rotation as Ro; return Ro.from_matrix(R).as_quat()
def T_of(p, q): T = np.eye(4); T[:3, :3] = q2R(q); T[:3, 3] = p; return T
def yaw_of(T): return np.arctan2(T[1, 0], T[0, 0])

def read(mcap_dir, logs):
    files = sorted(glob.glob(os.path.join(mcap_dir, "log_*.mcap")))
    if logs: files = [f for f in files if int(os.path.basename(f)[4:7]) in logs]
    od_t, od_T, last_path, last_path_t = [], [], None, None
    for f in files:
        for sch, ch, m in make_reader(open(f, "rb")).iter_messages(topics=["/Odometry/base", "/path/pgo"]):
            d = m.data; t = RM.stamp_ns(d, sch) / 1e9
            if ch.topic == "/Odometry/base":
                _, v = RM.odom13(d, sch); od_t.append(t); od_T.append(T_of(v[:3], v[3:7]))
            else:
                P, _ = RM.path_poses(d, sch)
                if last_path is None or len(P) >= len(last_path): last_path, last_path_t = P.copy(), t
    od_t = np.array(od_t); o = np.argsort(od_t)
    return od_t[o], np.array(od_T)[o], last_path, last_path_t

def match(od_t, od_T, P, max_gap_s=60.0, w_yaw=0.5):
    """키프레임 k 의 시각 인덱스 j_k: 이전 키프레임 시각 j_{k-1} 이후 max_gap 초 안에서 상대 이동이 PGO 상대 이동과 가장 맞는 오도메트리 샘플."""
    T_pgo = np.array([T_of(p[:3], p[3:7]) for p in P]); n = len(P)
    j = np.zeros(n, int); err = np.zeros(n)
    # 첫 키프레임: 시작 후 첫 오도메트리 (PGO 0 번은 원점 근처)
    j[0] = int(np.argmin(np.linalg.norm(od_T[:min(200, len(od_t)), :3, 3] - P[0, :3], axis=1)))
    for k in range(1, n):
        rel = np.linalg.inv(T_pgo[k-1]) @ T_pgo[k]; dp, dy = rel[:3, 3], yaw_of(rel)
        j0 = j[k-1]; j1 = int(np.searchsorted(od_t, od_t[j0] + max_gap_s)); cand = np.arange(j0 + 1, max(j0 + 2, j1))
        Ti = np.linalg.inv(od_T[j0]); R = np.einsum("ij,njk->nik", Ti[:3, :3], od_T[cand, :3, :3])
        p = od_T[cand, :3, 3] @ Ti[:3, :3].T + Ti[:3, 3]; y = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        dyaw = np.angle(np.exp(1j * (y - dy)))
        e = np.linalg.norm(p - dp, axis=1) + w_yaw * np.abs(dyaw); a = int(np.argmin(e)); j[k] = cand[a]; err[k] = e[a]
    return T_pgo, j, err

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mcap_dir"); ap.add_argument("out"); ap.add_argument("--logs"); ap.add_argument("--rrd", action="store_true")
    a = ap.parse_args(); logs = None
    if a.logs:
        lo, hi = (a.logs.split(",") + [None])[:2]; logs = set(range(int(lo), int(hi or lo) + 1))
    od_t, od_T, P, pt = read(a.mcap_dir, logs)
    print(f"odometry {len(od_t)} 샘플 {od_t[-1]-od_t[0]:.0f}s · /path/pgo 마지막 메시지 poses {len(P)} (stamp {pt:.2f})")
    T_pgo, j, err = match(od_t, od_T, P); t = od_t[j]
    corr = np.linalg.norm(T_pgo[:, :3, 3] - od_T[j, :3, 3], axis=1)
    dy = np.abs(np.degrees(np.angle(np.exp(1j * (np.array([yaw_of(T) for T in T_pgo]) - np.array([yaw_of(T) for T in od_T[j]]))))))
    gaps = np.diff(t)
    print(f"키프레임 시각 복원: 매칭 오차 중앙 {np.median(err[1:])*100:.1f}cm p90 {np.percentile(err[1:],90)*100:.1f}cm 최대 {err.max()*100:.1f}cm · 간격 중앙 {np.median(gaps):.1f}s 최대 {gaps.max():.1f}s")
    print(f"PGO−오도메트리 보정량: 위치 중앙 {np.median(corr)*100:.1f}cm p90 {np.percentile(corr,90)*100:.1f}cm 최대 {corr.max()*100:.1f}cm · yaw 중앙 {np.median(dy):.2f}° 최대 {dy.max():.2f}°")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "robot_pgo_tum.txt"), "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw  (from /path/pgo last msg; timestamps recovered by odometry matching)\n")
        for k in range(len(P)): f.write(f"{t[k]:.9f} " + " ".join(f"{v:.6f}" for v in P[k]) + "\n")
    np.savez(os.path.join(a.out, "robot_pgo.npz"), t=t, T_pgo=T_pgo, T_odom=od_T[j], match_err=err, odom_t=od_t, odom_T=od_T)
    if a.rrd:
        import rerun as rr, rerun.blueprint as rrb
        rr.init("robot_pgo"); rr.save(os.path.join(a.out, "robot_pgo_view.rrd"))
        rr.log("odometry", rr.LineStrips3D([od_T[:, :3, 3].astype(np.float32)], colors=[[150, 150, 150]], radii=0.05))
        rr.log("robot_pgo", rr.LineStrips3D([T_pgo[:, :3, 3].astype(np.float32)], colors=[[60, 120, 255]], radii=0.08))
        rr.log("robot_pgo/keyframes", rr.Points3D(T_pgo[:, :3, 3].astype(np.float32), colors=[[60, 120, 255]], radii=0.15))
        rr.log("correction", rr.LineStrips3D(np.stack([od_T[j, :3, 3], T_pgo[:, :3, 3]], 1).astype(np.float32), colors=[[230, 60, 60]], radii=0.04))
        rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="/", name="odometry(grey) vs robot PGO(blue), correction(red)"), collapse_panels=True))
    print("저장:", a.out)
