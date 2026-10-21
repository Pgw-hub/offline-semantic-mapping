"""연속 스캔 정합 잔차로 deskew 효과 검증 — 움직이는 구간의 인접 두 스캔을 각자 odom 포즈로 월드에 놓고 NN 거리 중앙값 비교.
결과(172935/log_002, 81쌍): raw 7.93cm → deskew(헤더=스캔 시작) 6.93cm(−12.6%) · 헤더=스캔 끝 가설 8.85cm(악화) ⇒ /Odometry/base 와 스캔 헤더 시각은 스캔 시작 기준.
사용: docker exec geom_rebuild python geometry/deskew_eval.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))  # 저장소 루트 (패키지 import)
import sys, numpy as np, open3d as o3d, yaml, os
from preprocess.geometry.deskew import read_mcap, deskew_scan
S = "/data2/20260827/20260827/KR-RRT-2604-008_20260827_172935"; C = S + "/raisin_config/calibration"
from preprocess.geometry.extrinsics import lidar_t_bl; t_bl = lidar_t_bl(a.calib if "a" in dir() else C)
odom, scans = read_mcap(S + "/log_002.mcap")
def pose(t):
    R, p, ok = odom.at(np.array([t])); return (R[0], p[0]) if ok[0] else (None, None)
def to_world(P, R, p): return P @ R.T + p
def nn_med(A, B):
    t = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(B))); d = [t.search_knn_vector_3d(q, 1)[2][0] for q in A[::7]]; return float(np.sqrt(np.median(d)))
res = {"raw@start": [], "raw@end": [], "deskew(start)": [], "deskew(end)": []}
for i in range(20, len(scans) - 1, 6):
    (t0, x0, o0), (t1, x1, o1) = scans[i], scans[i + 1]
    Ra, pa = pose(t0); Rb, pb = pose(t1)
    if Ra is None or Rb is None: continue
    v = np.linalg.norm(pb - pa) / ((t1 - t0) / 1e9)
    if v < 0.3: continue
    r0 = np.linalg.norm(x0, axis=1); r1 = np.linalg.norm(x1, axis=1); m0 = (r0 > 1) & (r0 < 25); m1 = (r1 > 1) & (r1 < 25)
    A = to_world(x0[m0] + t_bl, Ra, pa); B = to_world(x1[m1] + t_bl, Rb, pb); res["raw@start"].append(nn_med(B, A))
    Re, pe = pose(t0 + int(o0.max())); Rf, pf = pose(t1 + int(o1.max()))
    if Re is not None and Rf is not None:
        A = to_world(x0[m0] + t_bl, Re, pe); B = to_world(x1[m1] + t_bl, Rf, pf); res["raw@end"].append(nn_med(B, A))
    d0 = deskew_scan(odom, t0, x0, o0, t_bl); d1 = deskew_scan(odom, t1, x1, o1, t_bl)
    if d0 and d1:
        A = to_world(d0[0][m0], d0[1], d0[2]); B = to_world(d1[0][m1], d1[1], d1[2]); res["deskew(start)"].append(nn_med(B, A))
    d0b = deskew_scan(odom, t0 - int(o0.max()), x0, o0, t_bl); d1b = deskew_scan(odom, t1 - int(o1.max()), x1, o1, t_bl)   # 헤더 시각 = 스캔 끝 가설
    if d0b and d1b:
        A = to_world(d0b[0][m0], d0b[1], d0b[2]); B = to_world(d1b[0][m1], d1b[1], d1b[2]); res["deskew(end)"].append(nn_med(B, A))
print("moving scan pairs:", len(res["raw@start"]), "| consecutive-scan NN distance median (lower = better aligned)")
for k, v in res.items(): print(f"  {k:14s} {np.median(v)*100:6.2f} cm  (p75 {np.percentile(v,75)*100:.2f} cm)")
