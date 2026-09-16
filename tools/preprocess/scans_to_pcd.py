#!/usr/bin/env python3
"""S 단계 산출물(lidar_scans/*.npz, world 점) 을 골라 0.1 m 지도 pcd 로 — 트랙 2 에서 로봇 저장 지도 대신 기하로 쓸 때.
  python scans_to_pcd.py <out.pcd> --scans <ds/log_000/lidar_scans> [...] [--keyframes traj.txt | --every N] [--vox 0.1] [--tol 0.06]
  --keyframes: 로봇 키프레임 궤적(TUM) 의 stamp 에 가장 가까운 스캔만 (tol 초 안) → 자세가 최종 키프레임 값과 정확히 일치하는 스캔 집합
  --every N : N 스캔마다 하나. 둘 다 없으면 전체.
"""
import argparse, glob, os
import numpy as np, open3d as o3d
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--scans", nargs="+", required=True); ap.add_argument("--keyframes"); ap.add_argument("--every", type=int, default=1)
ap.add_argument("--vox", type=float, default=0.1); ap.add_argument("--tol", type=float, default=0.06)
a = ap.parse_args()
files = sorted(f for d in a.scans for f in glob.glob(os.path.join(d, "*.npz")))
ts = np.array([int(np.load(f)["stamp_ns"]) for f in files]) / 1e9
if a.keyframes:
    T = np.loadtxt(a.keyframes, comments="#"); pick = []
    for t in T[:, 0]:
        j = int(np.argmin(np.abs(ts - t)))
        if abs(ts[j] - t) <= a.tol: pick.append(j)
    pick = sorted(set(pick)); print(f"키프레임 {len(T)} → 스캔 {len(pick)} (tol {a.tol}s)")
else: pick = list(range(0, len(files), a.every))
P = np.concatenate([np.load(files[i])["xyz"] for i in pick]).astype(np.float64)
pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P); pc = pc.voxel_down_sample(a.vox)
o3d.io.write_point_cloud(a.out, pc); print(f"스캔 {len(pick)}/{len(files)} · 점 {len(P):,} → {a.vox} m 복셀 {len(pc.points):,} → {a.out}")
