#!/usr/bin/env python3
"""기하 A/B 테스트 — 같은 로그에서 (A) raw /livox/lidar 를 /Odometry/base 자세로 world 에 놓은 것 vs (B) lidar_slam 이 정합해 낸 /cloud_registered.
의미 정보 없이 점 출처만 바꿔 기하 지도의 번짐·일관성을 비교한다. (recorder `pointcloud2` 레이아웃 = geometry/pc2_reader.py)
  python tools/geom_ab_test.py <mcap_dir> <out_dir> [--logs 0,1] [--sub 4] [--rmax 40]
지표
  1) 스캔별 raw↔registered 최근접 거리 (SLAM 이 점을 얼마나 옮겼나; 외부파라미터 관례 자동 판별)
  2) 누적 지도의 0.05 / 0.1 m 점유 셀 수 (같은 점 수 → 셀이 많을수록 번짐)
  3) 스캔별 일관성: 스캔 i 점 → 나머지 스캔 누적 지도 최근접 거리 중앙/p90 (재방문 정합)
  4) 표면 두께: 이웃 0.3 m PCA 최소 고유값 √ (벽·바닥이 얼마나 두꺼워졌나)
산출: <out>/{raw,reg}/lidar_scans/%06d.npz (build_occupancy 입력 형식), <out>/ab_metrics.json, <out>/ab_view.rrd (Side by side / Difference)
"""
import argparse, glob, json, os, struct, sys, time
import numpy as np
from mcap.reader import make_reader
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geometry"))
from pc2_reader import parse_pc2, xyz_of
from deskew import Odom
import yaml

ap = argparse.ArgumentParser(); ap.add_argument("mcap_dir"); ap.add_argument("out"); ap.add_argument("--logs"); ap.add_argument("--sub", type=int, default=4, help="지표 계산용 점 서브샘플")
ap.add_argument("--rmax", type=float, default=40.0); ap.add_argument("--rmin", type=float, default=0.8); ap.add_argument("--cell", type=float, default=0.1)
a = ap.parse_args(); t_all = time.time()
files = sorted(glob.glob(os.path.join(a.mcap_dir, "log_*.mcap")))
if a.logs: lo, hi = (a.logs.split(",") + [None])[:2]; files = [f for f in files if int(os.path.basename(f)[4:7]) in range(int(lo), int(hi or lo) + 1)]
tr = yaml.safe_load(open(os.path.join(a.mcap_dir, "raisin_config/calibration/livox_lidar_0_lidar.yaml")))["transform"]
t_bl = np.array([tr["translation"][k] for k in "xyz"]); q_bl = np.array([tr["rotation"][k] for k in "wxyz"])
from deskew import q2R
R_bl = q2R(q_bl / np.linalg.norm(q_bl))

# ---- 읽기
ns, pos, q = [], [], []; raw, reg = {}, {}
for f in files:
    for sch, ch, m in make_reader(open(f, "rb")).iter_messages(topics=["/Odometry/base", "/livox/lidar", "/cloud_registered"]):
        if ch.topic == "/Odometry/base":
            v = struct.unpack_from("<7d", m.data, 8); sec, nsec = struct.unpack_from("<iI", m.data, 0); ns.append(sec * 10**9 + nsec); pos.append(v[:3]); q.append([v[6], v[3], v[4], v[5]])
        else:
            h, P = parse_pc2(m.data); (raw if ch.topic == "/livox/lidar" else reg)[h["ns"]] = xyz_of(P)
odom = Odom(np.array(ns, np.int64), np.array(pos), np.array(q))
rk = np.array(sorted(raw)); stamps = []
for t in sorted(reg):                                          # registered stamp ↔ 가장 가까운 raw stamp (1 ms 안)
    j = int(np.argmin(np.abs(rk - t)))
    if abs(int(rk[j]) - t) <= 1_000_000: stamps.append((t, int(rk[j])))
print(f"오도메트리 {len(ns)} · raw {len(raw)} · registered {len(reg)} · 짝(≤1ms) {len(stamps)} [{time.time()-t_all:.0f}s]")

def T_at(t):
    R, p, ok = odom.at(np.array([t])); return (R[0], p[0]) if ok[0] else (None, None)

# ---- 외부파라미터 관례 판별: 항등 vs 180° yaw(실로봇 CustomMsg 리더의 -x,-y)
def place(P, R, p, flip): Pl = P.copy(); Pl[:, :2] *= -1 if flip else 1; return (Pl @ R_bl.T + t_bl) @ R.T + p
cand = {}
for flip in (False, True):
    d = []
    for tr_, ts in stamps[len(stamps)//4: len(stamps)//4 + 20]:
        R, p = T_at(ts)
        if R is None: continue
        A = place(raw[ts], R, p, flip); B = reg[tr_]; d.append(np.median(cKDTree(B).query(A[::8])[0]))
    cand[flip] = float(np.median(d))
flip = min(cand, key=cand.get); print(f"raw 배치 관례: 항등 {cand[False]*100:.1f}cm · 180°flip {cand[True]*100:.1f}cm → {'flip' if flip else '항등'} 사용")
# stamp 관계 + 체계적 오프셋 진단: 스캔 몇 개에서 A→B 최근접 대응의 평균 변위 벡터(world) 와 base 프레임 변위
rk = np.array(sorted(raw)); gk = np.array(sorted(reg)); dts = (gk[:, None] - rk[None, :]); nearest = np.abs(dts).min(1); print(f"registered stamp → 최근접 raw stamp 차이: 0 인 것 {int((nearest==0).sum())}/{len(gk)} · 나머지 중앙 {np.median(nearest[nearest>0])/1e6 if (nearest>0).any() else 0:.1f}ms")
offs_w, offs_b = [], []
for tr_, ts in stamps[::max(1, len(stamps)//15)]:
    R, p = T_at(ts)
    if R is None: continue
    A = place(raw[ts], R, p, flip); B = reg[tr_]; d, j = cKDTree(B).query(A[::6]); m = d < 0.3; v = (B[j][m] - A[::6][m]); offs_w.append(v.mean(0)); offs_b.append(R.T @ v.mean(0))
offs_w, offs_b = np.array(offs_w), np.array(offs_b)
print(f"A→B 평균 변위 (world) 중앙 {np.median(offs_w,0).round(3).tolist()} m · (base 프레임) 중앙 {np.median(offs_b,0).round(3).tolist()} m · 스캔 간 표준편차 {offs_b.std(0).round(3).tolist()}")

# ---- 시각 오프셋 스윕: raw 를 자세 시각 t+δ 로 놓았을 때 registered 와의 거리 (상수 오프셋 제거 후). '그때 당시' 자세 시각이 얼마나 지배적인지
off_b = np.median(offs_b, 0); sweep = {}
for dms in (-150, -100, -70, -50, -30, -20, -10, 0, 10, 20, 30, 50, 70, 100, 150):
    d = []
    for tr_, ts in stamps[::max(1, len(stamps)//25)]:
        R, p = T_at(ts + dms * 1_000_000)
        if R is None: continue
        A = place(raw[ts] + off_b @ np.eye(3), R, p, flip) if False else place(raw[ts], R, p, flip) + R @ off_b
        d.append(np.median(cKDTree(reg[tr_]).query(A[::8])[0]))
    sweep[dms] = float(np.median(d)) if d else float("nan")
best_d = min(sweep, key=lambda k: sweep[k]); print("시각 오프셋 스윕 (상수 오프셋 제거 후, 중앙 cm): " + " ".join(f"{k:+d}ms:{v*100:.1f}" for k, v in sweep.items()) + f" → 최적 {best_d:+d} ms")

# ---- 스캔 export + 스캔별 raw↔reg 거리
for nm in ("raw", "reg", "rawfix"): os.makedirs(os.path.join(a.out, nm, "lidar_scans"), exist_ok=True)
per_scan = []; accA, accB, accC = [], [], []
for i, (tr_, ts) in enumerate(stamps):
    R, p = T_at(ts)
    if R is None: continue
    A = place(raw[ts], R, p, flip); B = reg[tr_]
    org = (R @ t_bl + p).astype(np.float32)
    R2, p2 = T_at(ts + best_d * 1_000_000); C = (place(raw[ts], R2, p2, flip) + R2 @ off_b) if R2 is not None else A
    for arr, name in ((A, "raw"), (B, "reg"), (C, "rawfix")):
        r = np.linalg.norm(arr - org, axis=1); keep = (r > a.rmin) & (r < a.rmax); arr = arr[keep]
        np.savez(os.path.join(a.out, name, "lidar_scans", "%06d.npz" % i), xyz=arr.astype(np.float32), origin=org, stamp_ns=np.int64(ts))
        {"raw": accA, "reg": accB, "rawfix": accC}[name].append(arr[::a.sub].astype(np.float32))
    dAB = cKDTree(B).query(A[::8])[0]; per_scan.append((ts, float(np.median(dAB)), float(np.percentile(dAB, 90))))
per_scan = np.array(per_scan); print(f"스캔 {len(per_scan)}개 export · raw↔registered 스캔별 최근접 거리 중앙 {np.median(per_scan[:,1])*100:.1f}cm (p90 스캔 {np.percentile(per_scan[:,1],90)*100:.1f}cm) [{time.time()-t_all:.0f}s]")
A_all = np.concatenate(accA); B_all = np.concatenate(accB); C_all = np.concatenate(accC)

# ---- 지표
def cells(P, v): return int(len(np.unique(np.floor(P / v).astype(np.int64), axis=0)))
def consistency(scans_dir):
    fs = sorted(glob.glob(os.path.join(scans_dir, "*.npz"))); pts = [np.load(f)["xyz"][::a.sub] for f in fs]; allp = np.concatenate(pts); idx = np.repeat(np.arange(len(pts)), [len(p) for p in pts])
    tree = cKDTree(allp); med, p90 = [], []
    for k in range(0, len(pts), max(1, len(pts) // 60)):     # 60 스캔 표본
        d, j = tree.query(pts[k][::4], k=8); own = idx[j] == k; d = np.where(own, np.inf, d).min(1); d = d[np.isfinite(d)]
        if len(d): med.append(np.median(d)); p90.append(np.percentile(d, 90))
    return float(np.median(med)), float(np.median(p90))
def thickness(P, n=4000, r=0.3):
    tree = cKDTree(P); sel = np.random.default_rng(0).choice(len(P), min(n, len(P)), replace=False); th = []
    for i in sel:
        nb = tree.query_ball_point(P[i], r)
        if len(nb) < 12: continue
        Q = P[nb] - P[nb].mean(0); w = np.linalg.eigvalsh(Q.T @ Q / len(nb)); th.append(np.sqrt(max(w[0], 0)))
    th = np.array(th); return float(np.median(th)), float(np.percentile(th, 90))
M = {}
for name, P, d in (("raw", A_all, "raw"), ("registered", B_all, "reg"), ("raw_fix(offset+dt)", C_all, "rawfix")):
    c05, c10 = cells(P, 0.05), cells(P, 0.10); cm, c9 = consistency(os.path.join(a.out, d, "lidar_scans")); tm, t9 = thickness(P)
    M[name] = {"points_sub": int(len(P)), "cells_5cm": c05, "cells_10cm": c10, "scan_consistency_med_m": cm, "scan_consistency_p90_m": c9, "thickness_med_m": tm, "thickness_p90_m": t9}
    print(f"{name:10s} 점(서브) {len(P):,} · 셀 5cm {c05:,} / 10cm {c10:,} · 스캔 일관성 중앙 {cm*100:.1f}cm p90 {c9*100:.1f}cm · 표면 두께 중앙 {tm*100:.1f}cm p90 {t9*100:.1f}cm")
M["raw_vs_registered_per_scan_med_m"] = float(np.median(per_scan[:, 1])); M["const_offset_base_m"] = off_b.round(4).tolist(); M["best_time_offset_ms"] = int(best_d); M["time_sweep_med_m"] = sweep; M["extrinsic_flip"] = bool(flip); M["n_scans"] = int(len(per_scan)); M["mcap_dir"] = a.mcap_dir
json.dump(M, open(os.path.join(a.out, "ab_metrics.json"), "w"), indent=1)

# ---- rrd
import rerun as rr, rerun.blueprint as rrb
rr.init("geom_ab"); rr.save(os.path.join(a.out, "ab_view.rrd"))
def zcol(P): z = P[:, 2]; lo, hi = np.percentile(z, 2), np.percentile(z, 98); zn = np.clip((z - lo) / max(hi - lo, 1e-3), 0, 1); return (np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], 1) * 255).astype(np.uint8)
rr.log("raw/points", rr.Points3D(A_all, colors=zcol(A_all), radii=0.02)); rr.log("registered/points", rr.Points3D(B_all, colors=zcol(B_all), radii=0.02)); rr.log("rawfix/points", rr.Points3D(C_all, colors=zcol(C_all), radii=0.02))
kA = np.unique(np.floor(A_all / a.cell).astype(np.int64), axis=0); kB = np.unique(np.floor(B_all / a.cell).astype(np.int64), axis=0)
sA = {tuple(k) for k in kA}; sB = {tuple(k) for k in kB}; oa = np.array(sorted(sA - sB)); ob = np.array(sorted(sB - sA)); both = np.array(sorted(sA & sB))
c = lambda K: ((K + 0.5) * a.cell).astype(np.float32)
if len(oa): rr.log("difference/raw_only", rr.Points3D(c(oa), colors=[[230, 60, 60]], radii=a.cell * 0.45))
if len(ob): rr.log("difference/registered_only", rr.Points3D(c(ob), colors=[[60, 120, 255]], radii=a.cell * 0.45))
if len(both): rr.log("difference/common", rr.Points3D(c(both), colors=[[150, 150, 150]], radii=a.cell * 0.2))
rr.log("odometry", rr.LineStrips3D([odom.pos.astype(np.float32)], colors=[[255, 200, 40]], radii=0.04))
rr.log("per_scan/raw_vs_registered_cm", rr.SeriesLine(name="median cm"));
for i, (ts, m_, p9) in enumerate(per_scan): rr.set_time_sequence("scan", i); rr.log("per_scan/raw_vs_registered_cm", rr.Scalar(m_ * 100))
rr.send_blueprint(rrb.Blueprint(rrb.Tabs(
    rrb.Horizontal(rrb.Spatial3DView(origin="/raw", name="A: raw + odometry"), rrb.Spatial3DView(origin="/registered", name="B: /cloud_registered"), name="Side by side"),
    rrb.Horizontal(rrb.Spatial3DView(origin="/rawfix", name=f"A': raw + odometry (offset {np.round(off_b*100,1).tolist()}cm, dt {best_d:+d}ms)"), rrb.Spatial3DView(origin="/registered", name="B: /cloud_registered"), name="Side by side (fixed)"),
    rrb.Spatial3DView(origin="/", name=f"Difference {a.cell}m (red raw-only, blue registered-only)", contents=["+ /difference/**", "+ /odometry"]),
    rrb.TimeSeriesView(origin="/per_scan", name="per-scan raw↔registered distance")), collapse_panels=True))
print(f"difference {a.cell}m: 공통 {len(both):,} · raw만 {len(oa):,} · registered만 {len(ob):,}")
print("저장:", a.out, f"[{time.time()-t_all:.0f}s]")
