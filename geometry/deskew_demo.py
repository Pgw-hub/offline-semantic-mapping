#!/usr/bin/env python3
"""deskew 효과 데모 — 로그에서 yaw 속도가 가장 큰 구간(회전)을 찾아 스캔 N개를 raw / deskew 로 각각 world 에 누적, 원 점 단위로 나란히 보여준다.
  python geometry/deskew_demo.py <mcap> <calib_dir> <out.rrd> [--n 25 --top 3]
지표: 지면 두께(z std), 수직 구조물 두께(각 점 → 주변 점 PCA 최소 고유값) 대신 간단히 '0.05m 격자 점유 셀 수'(번지면 늘어남) 를 출력."""
import argparse, os, sys, numpy as np, rerun as rr, rerun.blueprint as rrb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deskew import read_mcap, deskew_scan
import yaml
ap = argparse.ArgumentParser(); ap.add_argument("mcap"); ap.add_argument("calib"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=25); ap.add_argument("--top", type=int, default=3)
a = ap.parse_args()
tr = yaml.safe_load(open(os.path.join(a.calib, "livox_lidar_0_lidar.yaml")))["transform"]; t_bl = np.array([tr["translation"][k] for k in "xyz"])
odom, scans = read_mcap(a.mcap)
yaw = np.degrees(np.unwrap(np.arctan2(2 * (odom.q[:, 0] * odom.q[:, 3] + odom.q[:, 1] * odom.q[:, 2]), 1 - 2 * (odom.q[:, 2] ** 2 + odom.q[:, 3] ** 2))))
rate = np.abs(np.gradient(yaw, odom.ns / 1e9))                          # deg/s
stamps = np.array([s[0] for s in scans])
# 스캔 창별 평균 yaw 속도 → 상위 top 개(서로 겹치지 않게)
win = []
for i in range(0, len(scans) - a.n, 5):
    t0, t1 = stamps[i], stamps[i + a.n]; m = (odom.ns >= t0) & (odom.ns <= t1)
    if m.sum() > 3: win.append((float(rate[m].mean()), i))
win.sort(reverse=True); chosen = []
for r_, i in win:
    if all(abs(i - j) > a.n for _, j in chosen): chosen.append((r_, i))
    if len(chosen) >= a.top: break
rr.init("deskew_demo"); rr.save(a.out); views = []
def zcol(P): z = P[:, 2]; lo, hi = np.percentile(z, 2), np.percentile(z, 98); zn = np.clip((z - lo) / max(hi - lo, 1e-3), 0, 1); return (np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], 1) * 255).astype(np.uint8)
def cells(P, v=0.05): return len(np.unique(np.floor(P / v).astype(np.int64), axis=0))
for k, (r_, i) in enumerate(chosen):
    raw, dsk = [], []
    for j in range(i, i + a.n):
        t0, xyz, ot = scans[j]; R, p, ok = odom.at(np.array([t0]))
        if not ok[0]: continue
        rng = np.linalg.norm(xyz, axis=1); m = (rng > 1) & (rng < 30)
        raw.append((xyz[m] + t_bl) @ R[0].T + p[0])
        d = deskew_scan(odom, t0, xyz, ot, t_bl)
        if d: Pb, R0, p0 = d; dsk.append(Pb[m] @ R0.T + p0)
    raw = np.concatenate(raw); dsk = np.concatenate(dsk); c = raw.mean(0)
    name = f"turn{k+1}_{r_:.0f}degps"; print(f"{name}: 스캔 {i}~{i+a.n}, yaw 속도 평균 {r_:.1f}°/s, 점 {len(raw):,} · 5cm 점유 셀 raw {cells(raw):,} → deskew {cells(dsk):,} ({(cells(dsk)/cells(raw)-1)*100:+.1f}%)")
    rr.log(f"{name}/raw", rr.Points3D((raw - c).astype(np.float32), colors=zcol(raw), radii=0.02)); rr.log(f"{name}/deskew", rr.Points3D((dsk - c).astype(np.float32), colors=zcol(dsk), radii=0.02))
    views.append(rrb.Horizontal(rrb.Spatial3DView(origin=f"/{name}/raw", name=f"raw ({r_:.0f}°/s)"), rrb.Spatial3DView(origin=f"/{name}/deskew", name="deskew"), name=name))
rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*views), collapse_panels=True)); print("rrd:", a.out)
