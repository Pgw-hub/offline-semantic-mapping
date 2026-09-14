#!/usr/bin/env python3
"""실로봇 기하 번짐의 지배 요인 분해 — 스캔 간 일관성(재방문 최근접 거리)을 로봇 운동 상태별로 나눠 본다.
정지(센서 노이즈+자세 지터) / 직진(자세 오차) / 회전(deskew 잔차·시각 정렬) 로 나뉘면 어느 요인이 큰지 바로 보인다. deskew 켬/끔도 비교.
  python tools/blur_by_motion.py <mcap_dir> <log> [--graph graph_dir] [--every 2]
"""
import argparse, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geometry"))
from deskew import read_mcap, deskew_scan
from extrinsics import lidar_t_bl
ap = argparse.ArgumentParser(); ap.add_argument("mcap_dir"); ap.add_argument("log", type=int); ap.add_argument("--graph"); ap.add_argument("--every", type=int, default=2); ap.add_argument("--sub", type=int, default=6)
a = ap.parse_args(); t_bl = lidar_t_bl(os.path.join(a.mcap_dir, "raisin_config/calibration"), "calib")
odom, scans = read_mcap(os.path.join(a.mcap_dir, "log_%03d.mcap" % a.log)); scans = scans[::a.every]
pp = None
if a.graph:
    from pose_provider import PoseProvider; pp = PoseProvider(a.graph)
# 운동 상태: 오도메트리에서 속도, yaw 속도
def yaw(qw, qx, qy, qz): return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
ot = odom.ns / 1e9; v = np.linalg.norm(np.gradient(odom.pos, ot, axis=0), axis=1); yw = np.unwrap(yaw(*odom.q.T)); w = np.abs(np.degrees(np.gradient(yw, ot)))
def motion(ts): k = np.clip(np.searchsorted(odom.ns, ts), 1, len(odom.ns) - 1); return float(np.mean(v[k-2:k+3])), float(np.mean(w[k-2:k+3]))
def build(deskew_on):
    W, meta = [], []
    for ts, xyz, ot_ in scans:
        d = deskew_scan(odom, ts, xyz, ot_, t_bl)
        if not d: continue
        Pb, R0, p0 = d
        if not deskew_on: Pb = xyz + t_bl                          # 점별 시각 무시 (스캔 시작 자세 하나)
        if pp is not None:
            T, ok = pp.at_many(np.array([ts]))
            if not ok[0]: continue
            R0, p0 = T[0, :3, :3], T[0, :3, 3]
        r = np.linalg.norm(Pb - t_bl, axis=1); keep = (r > 1.0) & (r < 30.0); W.append(((Pb[keep][::a.sub]) @ R0.T + p0)); meta.append(motion(ts))
    return W, np.array(meta)
for deskew_on in (True, False):
    W, meta = build(deskew_on); allp = np.concatenate(W); idx = np.repeat(np.arange(len(W)), [len(w_) for w_ in W]); tree = cKDTree(allp)
    per = np.full(len(W), np.nan)
    for k in range(len(W)):
        if len(W[k]) < 30: continue
        d, j = tree.query(W[k][::2], k=6); d = np.where(idx[j] == k, np.inf, d).min(1); d = d[np.isfinite(d)]
        if len(d): per[k] = np.median(d)
    sp, yr = meta[:, 0], meta[:, 1]
    bins = {"정지 (<0.1 m/s)": sp < 0.1, "직진 (>0.4 m/s, <8°/s)": (sp > 0.4) & (yr < 8), "완회전 (8~25°/s)": (yr >= 8) & (yr < 25), "급회전 (>25°/s)": yr >= 25}
    print(f"[deskew {'ON ' if deskew_on else 'OFF'}] log_{a.log:03d} 스캔 {len(W)} · 전체 일관성 중앙 {np.nanmedian(per)*100:.2f} cm")
    for nm, m in bins.items():
        m = m & np.isfinite(per)
        if m.sum(): print(f"   {nm:24s} n={m.sum():3d}  중앙 {np.median(per[m])*100:.2f} cm  p90 {np.percentile(per[m],90)*100:.2f} cm")
