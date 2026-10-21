#!/usr/bin/env python3
"""코스트맵 v2 — 확률적 occupancy 레이어(0.2m hits/misses) + semantic 복셀 → nav2 등급 코스트맵.

v1(voxel_postprocess) 문제: 0.1m 표면 복셀 1개(스캔 1회 hit)도 장애물로 취급 + 열 최저점 지면 추정 → 잔디밭·보도까지 lethal 이 덮임.
v2 규칙 (모두 매니페스트 파일만 사용, 재계산 없음):
  지면   : 열(x,y)에서 hits ≥ H_GROUND 인 가장 낮은 셀 (여러 스캔이 확인한 표면). 없으면 255(미관측).
  장애물 : 지면 위 [0.3, clearance] m 밴드의 셀 중 hits ≥ H_OBS 이고 라벨이 grass/person 이 아닌 것.
           tree 또는 높이 > 0.5 m → 254 lethal · 그 외(낮고 라벨 없음/other) → 200 uncertain
  지면 코스트: 지면 셀 라벨 → path 10 / grass 90 / other 150 / 라벨 없음 60
사용: python costmap_from_occupancy.py <manifest> <out_prefix> [--h-ground 2] [--h-obs 3] [--clearance 1.4]
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # scripts/ (패키지 import)
import argparse, json, os, sys, time
import numpy as np
from package.load_map import MapPackage
from common import classdict as CD

ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("out")
ap.add_argument("--h-ground", type=int, default=2); ap.add_argument("--h-obs", type=int, default=3)
ap.add_argument("--clearance", type=float, default=1.4); ap.add_argument("--low", type=float, default=0.5, help="이 높이 이하 라벨 없는 장애물 → 200")
ap.add_argument("--register", action="store_true", help="매니페스트에 costmap_2d_occ 로 등록")
ap.add_argument("--mode", choices=["semantic", "geometric"], default="semantic",
                help="geometric = 라벨 무시: 지면은 전부 free(0), 밴드 안 확인된 점은 전부 lethal (nav2 static/obstacle layer 와 같은 규칙)")
ap.add_argument("--band-min", type=float, default=0.3, help="지면 위 이 높이부터 장애물 밴드 시작 (m)")
ap.add_argument("--ground-label", choices=["ground", "all"], default="ground", help="지면 셀 라벨: ground = 지면 역할(ground_preferred·soft) 클래스 중 argmax(확률<ground-min 이면 other) — 하늘/건물 등 문맥 표가 원거리 지면에 묻는 것 차단 · all = 전체 argmax(구 동작)")
ap.add_argument("--ground-min", type=float, default=0.3)
ap.add_argument("--band-label", choices=["all", "nonground"], default="nonground", help="밴드(지면 위) 복셀 라벨: nonground = 지면 역할 외 클래스 중 argmax(확률<ground-min 이면 미라벨→높이 규칙) · all = 전체 argmax(구 동작)")
a = ap.parse_args(); t0 = time.time()
mp = MapPackage(a.manifest); O = mp.occ; R = O["res"]; ijk = O["ijk"].astype(np.int64); hits = O["hits"]
classes = mp.classes; n_cls = len(classes); cd = mp.classdict
PREF = CD.ids_with_role(cd, "ground_preferred"); SOFT = CD.ids_with_role(cd, "ground_soft"); OBST = CD.ids_with_role(cd, "obstacle"); DYN = CD.ids_with_role(cd, "dynamic")
PASS = set(SOFT + DYN)                      # 밴드에서 장애물로 세지 않는 클래스 (잔디·사람)
print("클래스 사전:", cd["name"], {c["name"]: c["role"] for c in cd["classes"]})

# --- 0.2m 셀 라벨: 0.1m 표면 복셀 다수결 (person·free 안 동적 복셀 제외) ---
ctr = mp.voxel_centers(); lab = mp.labels(); keep = ~mp.surf_removed
K = np.floor(ctr[keep] / R).astype(np.int64); L = lab[keep]
if a.ground_label == "ground":                               # 지면 셀 라벨은 지면 역할 클래스 중 argmax (문맥·장애물 표는 밴드 판정에만)
    GIDX = np.array(PREF + SOFT, np.int64); subp = mp.sem[:, GIDX]; has = mp.n_sem > 0
    labg = np.full(len(lab), n_cls + 1, np.int16); am = subp.argmax(1); mx = subp.max(1)
    labg[has] = np.where(mx[has] >= a.ground_min, GIDX[am[has]], n_cls).astype(np.int16); Lg = labg[keep]
else: Lg = L
if a.band_label == "nonground":                              # 밴드 복셀: 비지면 클래스 중 argmax (hybrid 미세조정 모델이 기둥·나무에 준 지면 확률 무시)
    NG = np.array([i for i in range(n_cls) if i not in set(PREF + SOFT)], np.int64); subn = mp.sem[:, NG]; hasn = mp.n_sem > 0
    labb = np.full(len(lab), n_cls + 1, np.int16); amn = subn.argmax(1); mxn = subn.max(1)
    labb[hasn] = np.where(mxn[hasn] >= a.ground_min, NG[amn[hasn]], n_cls + 1).astype(np.int16); Lb = labb[keep]
else: Lb = L
def pack(K):   # 축당 19비트(±26km@0.2m) → 57비트, ×8(라벨) 해도 int64 안전
    return ((K[:, 0] + (1 << 18)) << 38) | ((K[:, 1] + (1 << 18)) << 19) | (K[:, 2] + (1 << 18))
def vote(L):   # 0.2 m 셀 다수결 → occupancy 셀 라벨
    kp = pack(K); order = np.argsort(kp); kp, L = kp[order], L[order]
    uk, st = np.unique(kp, return_index=True); en = np.r_[st[1:], len(kp)]
    # 다수결(벡터화): 각 (셀,라벨) 쌍 카운트
    assert n_cls + 2 <= 16, "라벨 16개 초과 (4비트 패킹)"                    # 클래스 + other + unknown ≤ 16 (57비트 키 ×16 = 61비트, int64 안전)
    pair = kp * 16 + L.astype(np.int64); up, cnt = np.unique(pair, return_counts=True)
    cell_of = np.searchsorted(uk, up >> 4); labs = (up & 15).astype(np.int16)
    best = np.zeros(len(uk), np.int64); bestlab = np.full(len(uk), n_cls + 1, np.int16)
    o2 = np.lexsort((cnt, cell_of))                            # 셀별 마지막 = 최대 카운트
    cell_of, labs, cnt = cell_of[o2], labs[o2], cnt[o2]
    last = np.r_[cell_of[1:] != cell_of[:-1], True]
    bestlab[cell_of[last]] = labs[last]
    # occupancy 셀에 라벨 붙이기
    ok_p = pack(ijk); pos = np.searchsorted(uk, ok_p); pos_c = np.minimum(pos, len(uk) - 1); found = uk[pos_c] == ok_p
    olab = np.full(len(ijk), n_cls + 1, np.int16); olab[found] = bestlab[pos_c[found]]
    return olab
ok_p = pack(ijk)
olab_full = vote(L); olab_g = vote(Lg) if a.ground_label == "ground" else olab_full; olab = vote(Lb) if a.band_label == "nonground" else olab_full   # olab: 밴드 판정용

# --- 지면: 열별 hits≥H_GROUND 최저 셀 ---
col = (ijk[:, 0] + (1 << 20)) * (1 << 21) + (ijk[:, 1] + (1 << 20))
gm = hits >= a.h_ground
ucol, inv = np.unique(col[gm], return_inverse=True); inv = inv.ravel()
kmin = np.full(len(ucol), 1 << 30, np.int64); np.minimum.at(kmin, inv, ijk[gm, 2])
# 지면 셀 라벨: (col, kmin) 셀 찾기
gcell = pack(np.c_[(ucol // (1 << 21)) - (1 << 20), (ucol % (1 << 21)) - (1 << 20), kmin])
all_p = ok_p; oa = np.argsort(all_p); sp = all_p[oa]
p2 = np.searchsorted(sp, gcell); p2c = np.minimum(p2, len(sp) - 1); f2 = sp[p2c] == gcell
glab = np.full(len(ucol), n_cls + 1, np.int16); glab[f2] = olab_g[oa[p2c[f2]]]      # 지면 라벨 (ground-label 모드)

# --- 격자 ---
i_all = ucol // (1 << 21) - (1 << 20); j_all = ucol % (1 << 21) - (1 << 20)
i0, j0 = int(i_all.min()), int(j_all.min()); H = int(i_all.max() - i0 + 1); W = int(j_all.max() - j0 + 1)
cost = np.full((H, W), 255, np.uint8); gz = np.full((H, W), np.nan, np.float32)
gi, gj = i_all - i0, j_all - j0
ground_cost = np.full(len(ucol), 60, np.uint8)               # 라벨 없는 지면 60
ground_cost[glab == n_cls] = 150                             # other
for ci, c in enumerate(cd["classes"]):                       # 사전의 역할·코스트로 지면값
    ground_cost[glab == ci] = min(int(c["cost"]), 252) if c["role"] in ("ground_preferred", "ground_soft") else 150
if a.mode == "geometric":
    ground_cost[:] = 0                                       # 기하 전용: 지면이면 전부 free
cost[gi, gj] = ground_cost; gz[gi, gj] = (kmin + 0.5) * R

# --- 밴드 장애물 ---
kg_of_cell = np.full(len(ijk), -1, np.int64)
pc = np.searchsorted(ucol, col); pcc = np.minimum(pc, len(ucol) - 1); fc = ucol[pcc] == col
kg_of_cell[fc] = kmin[pcc[fc]]
h = (ijk[:, 2] - kg_of_cell) * R
if a.mode == "geometric":
    obs = (kg_of_cell >= 0) & (hits >= a.h_obs) & (h >= a.band_min - 1e-6) & (h <= a.clearance)   # 라벨 무시
    lethal = obs; unc = np.zeros_like(obs)
else:
    obs = (kg_of_cell >= 0) & (hits >= a.h_obs) & (h >= a.band_min - 1e-6) & (h <= a.clearance) & ~np.isin(olab, list(PASS))
    lethal = obs & (np.isin(olab, OBST) | (h > a.low))
    unc = obs & ~lethal
ci, cj = ijk[:, 0] - i0, ijk[:, 1] - j0
cost[ci[unc], cj[unc]] = np.maximum(cost[ci[unc], cj[unc]], 200)
cost[ci[lethal], cj[lethal]] = 254
# 지면 없는 열의 밴드 장애물은 무시(255 유지) — 위에서 kg_of_cell>=0 조건으로 이미 제외

x0 = i0 * R; y0 = j0 * R
u, c_ = np.unique(cost, return_counts=True); hist = {int(k): int(v) for k, v in zip(u, c_)}
print(f"코스트맵 v2 {cost.shape} res {R} · 원점 ({x0:.1f},{y0:.1f}) · {time.time()-t0:.1f}s")
print("  분포:", hist, f"· 통행가능 {sum(v for k,v in hist.items() if k<=200)*R*R:.0f} m² · lethal {hist.get(254,0)*R*R:.0f} m²")
print("  지면 라벨:", {(classes + ['other', 'unknown'])[k]: int(v) for k, v in zip(*np.unique(glab, return_counts=True))})
from scipy import ndimage
comp, n = ndimage.label(cost <= 200, structure=np.ones((3, 3))); sz = np.bincount(comp.ravel()); sz[0] = 0
print(f"  연결요소 {n} · 상위5(m²) {sorted((sz*R*R).tolist(), reverse=True)[:5]}")
legend = "0-252 cost (path10 grass90 other150 none60), 200 uncertain(low unlabeled obstacle), 254 lethal, 255 unknown"
# 비교용: 밴드 안에서 grass 라벨이라 통과시킨(semantic) / lethal 로 잡았을(geometric) 셀 수
band_all = (kg_of_cell >= 0) & (hits >= a.h_obs) & (h >= a.band_min - 1e-6) & (h <= a.clearance)
print(f"  밴드 안 확인 셀 {int(band_all.sum()):,} · 그중 통과 클래스(soft/dynamic) {int((band_all & np.isin(olab, list(PASS))).sum()):,} · mode={a.mode} band_min={a.band_min}")
np.savez_compressed(a.out + ".npz", cost=cost, ground_z=gz, x0=x0, y0=y0, res=R, axes="row=x,col=y", legend=legend, clearance_m=a.clearance,
                    h_ground=a.h_ground, h_obs=a.h_obs, mode=a.mode, band_min=a.band_min)
print("저장:", a.out + ".npz")
if a.register:
    m = json.load(open(a.manifest)); root = os.path.dirname(os.path.dirname(os.path.abspath(a.manifest)))
    rel = os.path.relpath(a.out + ".npz", root)
    m["files"]["costmap_2d_occ"] = {"path": rel, "bytes": os.path.getsize(a.out + ".npz"),
                                    "desc": f"v2: occupancy 레이어 기반 0.2m 등급 코스트맵 (지면 hits≥{a.h_ground}, 장애물 hits≥{a.h_obs}, clearance {a.clearance})"}
    json.dump(m, open(a.manifest, "w"), ensure_ascii=False, indent=1); print("매니페스트 등록: costmap_2d_occ →", rel)
