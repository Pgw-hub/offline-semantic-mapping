#!/usr/bin/env python3
"""voxel3d 생성 과정을 rerun 타임라인으로 재생 — build_voxel3d.py 와 같은 순서(스캔 → 0.1 m 확률 격자 log-odds 순차 갱신 → 2.5 cm 세부 칸)로 쌓아 가며 기록.
  python tools/voxel3d_build_replay.py <map_root> <out.rrd> [--batch 40] [--scan-every 5] [--rays 60]
타임라인 'build_step' (배치 단위 = batch 스캔):
  robot/trajectory · robot/lidar          지금까지의 LiDAR 원점 궤적, 현재 위치
  scan/points · scan/rays                 지금 넣는 스캔(표본) 과 광선 일부
  prob/occupied                           0.1 m 확률 격자 중 log-odds>0 인 칸 (색 = log-odds, 노랑 약함 → 빨강 확신)
  prob/flipped_free                       한때 점유였다가 miss 가 쌓여 빈칸이 된 칸 (회색) — 사람·튀는 점이 지워지는 순간
  fine/new/<k>                            이번 배치에 처음 생긴 2.5 cm 칸 (높이색) — 누적 표시
타임라인 끝 단계 (build_step = 마지막+1..+4): 최종 판정 → 클래스 색 → depth 채움 → 사람 제거·타일 (최종 npz 기준)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # scripts/ (패키지 import)
import argparse, glob, json, os, sys
import numpy as np, rerun as rr, rerun.blueprint as rrb
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--batch", type=int, default=40)
ap.add_argument("--scan-every", type=int, default=5); ap.add_argument("--rays", type=int, default=60); ap.add_argument("--res-xy", type=float, default=0.025); ap.add_argument("--res-z", type=float, default=0.025)
ap.add_argument("--prob-res", type=float, default=0.1); ap.add_argument("--free-range", type=float, default=8.0); ap.add_argument("--max-range", type=float, default=30.0)
a = ap.parse_args()
RX, RZ, RP = a.res_xy, a.res_z, a.prob_res; OFF = 1 << 20; MASK = (1 << 21) - 1
def key3(I): I = I.astype(np.int64) + OFF; return (I[:, 0] << 42) | (I[:, 1] << 21) | I[:, 2]
def unkey(k): return np.stack([(k >> 42) & MASK, (k >> 21) & MASK, k & MASK], -1).astype(np.int64) - OFF
def turbo(z, lo, hi):
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1); r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)
files = sorted(glob.glob(os.path.join(a.root, "dataset", "log_*", "lidar_scans", "*.npz")))
fin = np.load(os.path.join(a.root, "compiled", "session_voxel3d.npz")); zlo, zhi = np.percentile(fin["mean_z"], [2, 98])
lh = np.log(0.85 / 0.15); lm = np.log(0.45 / 0.55); steps = (np.arange(int(np.ceil(a.free_range / (RP * 0.5)))) + 0.5) * RP * 0.5
rr.init("voxel3d_build_replay"); rr.save(a.out)
ck = np.zeros(0, np.int64); clo = np.zeros(0); fine_seen = np.zeros(0, np.int64); ever_occ = np.zeros(0, np.int64)
traj = []; bc = []; bf = []; step = 0
for si, f in enumerate(files):
    d = np.load(f); P = d["xyz"].astype(np.float64); o = d["origin"].astype(np.float64); vec = P - o; rng = np.linalg.norm(vec, axis=1)
    ok = (rng > 0.05) & (rng <= a.max_range); P, vec, rng = P[ok], vec[ok], rng[ok]; traj.append(o)
    hk = np.unique(key3(np.floor(P / RP))); L = np.minimum(rng, a.free_range); dirs = vec / rng[:, None]
    valid = steps[None, :] < L[:, None]; S = o[None, None, :] + dirs[:, None, :] * steps[None, :, None]
    mk = np.unique(key3(np.floor(S.reshape(-1, 3) / RP))[valid.ravel()]); mk = mk[~np.isin(mk, hk, assume_unique=True)]
    bc.append((hk, mk)); fk = np.unique(key3(np.stack([np.floor(P[:, 0] / RX), np.floor(P[:, 1] / RX), np.floor(P[:, 2] / RZ)], -1))); bf.append((fk, P))
    if si % a.scan_every == 0:                                                # 현재 스캔(표본) + 광선 일부
        rr.set_time_sequence("build_step", step); rr.set_time_sequence("scan", si)
        rr.log("scan/points", rr.Points3D(P[::4].astype(np.float32), colors=[60, 160, 255], radii=0.015))
        ri = np.random.default_rng(si).choice(len(P), min(a.rays, len(P)), replace=False)
        rr.log("scan/rays", rr.LineStrips3D([[o, P[i]] for i in ri], colors=[[255, 255, 255, 60]], radii=0.003))
        rr.log("robot/lidar", rr.Points3D([o.astype(np.float32)], colors=[255, 80, 80], radii=0.08))
    if (si + 1) % a.batch == 0 or si == len(files) - 1:
        # 확률 격자 순차 갱신 (build_voxel3d.merge_prob 와 같은 규칙: 배치 합 후 클램프)
        K = np.concatenate([np.concatenate([h, m]) for h, m in bc]); W = np.concatenate([np.r_[np.full(len(h), lh), np.full(len(m), lm)] for h, m in bc])
        u, inv = np.unique(K, return_inverse=True); dl = np.bincount(inv, weights=W, minlength=len(u))
        allk = np.union1d(ck, u); lo_new = np.zeros(len(allk)); lo_new[np.searchsorted(allk, ck)] = clo; lo_new[np.searchsorted(allk, u)] += dl
        ck, clo = allk, np.clip(lo_new, -2.0, 3.5); occ = clo > 0
        flipped = np.setdiff1d(ever_occ, ck[occ], assume_unique=False); ever_occ = np.union1d(ever_occ, ck[occ])
        # 세부 칸: 이번 배치에 처음 생긴 칸
        FK = np.unique(np.concatenate([k for k, _ in bf])); new = np.setdiff1d(FK, fine_seen, assume_unique=True); fine_seen = np.union1d(fine_seen, new)
        Pb = np.concatenate([p for _, p in bf]); kb = key3(np.stack([np.floor(Pb[:, 0] / RX), np.floor(Pb[:, 1] / RX), np.floor(Pb[:, 2] / RZ)], -1))
        sel = np.isin(kb, new); ub, ib = np.unique(kb[sel], return_inverse=True); mz = np.bincount(ib, weights=Pb[sel, 2]) / np.bincount(ib); I = unkey(ub)
        rr.set_time_sequence("build_step", step); rr.set_time_sequence("scan", si)
        rr.log(f"fine/new/{step:04d}", rr.Points3D(np.c_[(I[:, 0] + 0.5) * RX, (I[:, 1] + 0.5) * RX, mz].astype(np.float32), colors=turbo(mz, zlo, zhi), radii=RX * 0.5))
        C = unkey(ck[occ]); rr.log("prob/occupied", rr.Points3D(((C + 0.5) * RP).astype(np.float32), colors=np.c_[np.full(int(occ.sum()), 255), (230 * (1 - clo[occ] / 3.5)).clip(0, 255), np.zeros(int(occ.sum()))].astype(np.uint8), radii=RP * 0.25))
        if len(flipped): rr.log("prob/flipped_free", rr.Points3D(((unkey(flipped) + 0.5) * RP).astype(np.float32), colors=[150, 150, 150], radii=RP * 0.3))
        rr.log("robot/trajectory", rr.LineStrips3D([np.array(traj, np.float32)], colors=[255, 80, 80], radii=0.02))
        rr.log("stats", rr.TextLog(f"스캔 {si+1}/{len(files)} · 확률 칸 {len(ck):,} (점유 {int(occ.sum()):,}) · 세부 칸 {len(fine_seen):,} · 이번에 빈칸으로 뒤집힌 칸 {len(flipped):,}"))
        bc, bf = [], []; step += 1
# 끝 단계: 최종 npz 로 판정·라벨·채움·정리
I = fin["ijk"]; xyz = np.c_[(I[:, 0] + 0.5) * float(fin["res_xy"]), (I[:, 1] + 0.5) * float(fin["res_xy"]), fin["mean_z"]].astype(np.float32); src = fin["source"]; cls = fin["cls"]
classes = [str(c) for c in fin["classes"]]; vs = os.path.join(a.root, "voxel", "session_voxsem.npz")
pal = {}
try:
    from common import classdict as CD
    v = np.load(vs, allow_pickle=True); cd = json.loads(str(v["classdict"])); pal = {i: CD.hex2rgb(c["color"]) for i, c in enumerate(cd["classes"])}; pal[254] = CD.hex2rgb(CD.OTHER_COLOR); pal[255] = CD.hex2rgb(CD.UNKNOWN_COLOR)
except Exception: pass
ccol = np.array([pal.get(int(c), (128, 128, 128)) for c in range(256)], np.uint8)[cls]
def stage(k, title):
    rr.set_time_sequence("build_step", step + k); rr.log("stats", rr.TextLog(title))
stage(0, "① 최종 판정: 확률 격자 log-odds>0 인 칸 안의 세부 칸만 남김 (빨강 = 지워진 칸)")
rr.log("fine/new", rr.Clear(recursive=True)); L = src == 1
rr.log("final/lidar_height", rr.Points3D(xyz[L], colors=turbo(xyz[L, 2], zlo, zhi), radii=RX * 0.5))
R_ = fin["removed_ijk"]; w = fin["removed_why"]
rr.log("final/removed", rr.Points3D(np.c_[(R_[:, 0] + 0.5) * RX, (R_[:, 1] + 0.5) * RX, (R_[:, 2] + 0.5) * RZ][w == 1].astype(np.float32), colors=[255, 50, 50], radii=RX * 0.6))
stage(1, "② 클래스 색: 0.1 m 의미 라벨 복셀에서 칸마다 클래스")
rr.log("final/lidar_height", rr.Clear(recursive=False)); rr.log("final/removed", rr.Clear(recursive=False))
rr.log("final/lidar_class", rr.Points3D(xyz[L], colors=ccol[L], radii=RX * 0.5))
stage(2, "③ depth 채움: LiDAR 가 못 본 칸만 depth 카메라 점으로 (하늘색)")
D = src == 2
if D.any(): rr.log("final/depth_fill", rr.Points3D(xyz[D], colors=[0, 200, 255], radii=RX * 0.6))
stage(3, f"④ 사람 칸 제거({int((w == 2).sum()):,}) · 10 m 타일로 저장 → 로봇은 둘레 3×3 만 적재")
idx = os.path.join(a.root, "compiled", "session_voxel3d_tiles", "index.json")
if os.path.exists(idx):
    ti = json.load(open(idx)); T = float(ti["tile_m"])
    rr.log("final/tiles", rr.LineStrips3D([[[t["x0"], t["y0"], t["z_min"]], [t["x0"] + T, t["y0"], t["z_min"]], [t["x0"] + T, t["y0"] + T, t["z_min"]], [t["x0"], t["y0"] + T, t["z_min"]], [t["x0"], t["y0"], t["z_min"]]] for t in ti["tiles"]], colors=[255, 200, 0], radii=0.02))
rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
    rrb.Tabs(rrb.Spatial3DView(origin="/", name="1 build-up (scans + 2.5 cm cells)", contents=["+ /fine/**", "+ /scan/**", "+ /robot/**", "+ /final/**"]),
             rrb.Spatial3DView(origin="/", name="2 prob grid 0.1 m (occupied / flipped)", contents=["+ /prob/**", "+ /robot/**", "+ /scan/points"]),
             rrb.Spatial3DView(origin="/", name="3 everything", contents="+ /**")),
    rrb.TextLogView(origin="/stats", name="log"), column_shares=[4, 1]), rrb.TimePanel(expanded=True), collapse_panels=False))
n_fin = int((fin["coarse_logodds"].astype(float) > 0).sum())
print(f"스캔 {len(files)} · 배치 {step} 단계 + 끝 단계 4 · 최종 확률 점유 {int((clo > 0).sum()):,} (빌더 npz {n_fin:,}) · 세부 칸 {len(fine_seen):,} · rrd {a.out}")
