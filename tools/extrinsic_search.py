#!/usr/bin/env python3
"""LiDAR 외부파라미터(base→lidar 병진) 를 지도 일관성으로 역추정 — 어느 값이 /Odometry/base(또는 PGO 자세) 와 일관되는지 데이터로 판정.
후보 오프셋 δ 를 calibration lidar yaml 값에 더해 스캔을 world 에 놓고, 스캔 간 최근접 거리(재방문 정합) 가 최소인 δ 를 찾는다.
  python tools/extrinsic_search.py <mcap_dir> <log> [--graph graph_dir] [--every 3] [--sub 8]
출력: 좌표별 1D 스윕 → 최적 부근 3D 격자, 그리고 calib(0,0,0) / SLAM(-0.011,-0.0233,+0.0441) / 최적 δ 의 지표 비교
"""
import argparse, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geometry"))
from deskew import read_mcap, deskew_scan
from extrinsics import lidar_extrinsic, SLAM_IMU_TO_LIDAR
ap = argparse.ArgumentParser(); ap.add_argument("mcap_dir"); ap.add_argument("log", type=int); ap.add_argument("--graph"); ap.add_argument("--every", type=int, default=3); ap.add_argument("--sub", type=int, default=8)
ap.add_argument("--rmin", type=float, default=1.0); ap.add_argument("--rmax", type=float, default=30.0)
a = ap.parse_args(); t0 = time.time()
calib = os.path.join(a.mcap_dir, "raisin_config/calibration"); t_cal = lidar_extrinsic(calib, "calib")[0]
odom, scans = read_mcap(os.path.join(a.mcap_dir, "log_%03d.mcap" % a.log)); scans = scans[::a.every]
pp = None
if a.graph:
    from pose_provider import PoseProvider; pp = PoseProvider(a.graph)
print(f"log_{a.log:03d}: 스캔 {len(scans)} (every {a.every}) · 자세 {'PGO ' + a.graph if pp else '/Odometry/base'} · calib t_bl {t_cal.round(4).tolist()} [{time.time()-t0:.0f}s]")

# deskew 는 t_bl 에 약하게만 의존 → 한 번만 계산해 두고(base 프레임 점 Pb, 스캔 자세 R0,p0), δ 는 base 프레임 병진으로 더한다: world = (Pb + δ) R0ᵀ + p0
base = []
for ts, xyz, ot in scans:
    d = deskew_scan(odom, ts, xyz, ot, t_cal)
    if not d: continue
    Pb, R0, p0 = d
    if pp is not None:
        T, ok = pp.at_many(np.array([ts]))
        if not ok[0]: continue
        R0, p0 = T[0, :3, :3], T[0, :3, 3]
    r = np.linalg.norm(Pb - t_cal, axis=1); keep = (r > a.rmin) & (r < a.rmax); base.append((Pb[keep][::a.sub], R0, p0))
print(f"유효 스캔 {len(base)} · 점 {sum(len(b[0]) for b in base):,}")

def score(delta, n_eval=None):
    W = [((Pb + delta) @ R0.T + p0) for Pb, R0, p0 in base]; allp = np.concatenate(W); idx = np.repeat(np.arange(len(W)), [len(w) for w in W]); tree = cKDTree(allp); med = []
    for k in range(0, len(W), max(1, len(W) // (n_eval or 40))):
        if len(W[k]) < 30: continue
        d, j = tree.query(W[k][::3], k=6); d = np.where(idx[j] == k, np.inf, d).min(1); d = d[np.isfinite(d)]
        if len(d): med.append(np.median(d))
    return float(np.median(med))

best = np.zeros(3); names = "xyz"
for rnd in range(2):
    for ax in range(3):
        grid = np.arange(-0.06, 0.0801, 0.01 if rnd == 0 else 0.005) if rnd == 0 else best[ax] + np.arange(-0.015, 0.0151, 0.005)
        sc = []
        for g in grid:
            d = best.copy(); d[ax] = g; sc.append(score(d))
        best[ax] = grid[int(np.argmin(sc))]
        print(f"  round {rnd+1} {names[ax]}: " + " ".join(f"{g*100:+.1f}:{s*100:.2f}" for g, s in zip(grid, sc)) + f" → {best[ax]*100:+.1f} cm", flush=True)
print(f"최적 δ (base 프레임, calib 대비) = {np.round(best*100,1).tolist()} cm [{time.time()-t0:.0f}s]")
for nm, d in (("calib (0,0,0)", np.zeros(3)), ("SLAM (-1.1,-2.3,+4.4)", SLAM_IMU_TO_LIDAR), ("최적", best)):
    print(f"  {nm:24s} 스캔 일관성 중앙 {score(d, 80)*100:.2f} cm")
