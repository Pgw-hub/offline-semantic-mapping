#!/usr/bin/env python3
"""기하 전용 static layer vs semantic(잔디 통행) 코스트맵 비교.

A. geometric_raw : 0.1m 표면 복셀 전부 사용. 열 최저 복셀 = 지면, 지면 위 [0.15, 1.2] m 에 복셀 있으면 lethal (nav2 obstacle layer 처럼 점 하나도 장애물)
B. geometric     : occupancy 레이어(0.2m, hits≥N) 라벨 무시 → 밴드 안 확인 셀 lethal, 지면 free
C. semantic      : B + grass 라벨은 밴드에서 통과, 지면 클래스 등급 (path 10 / grass 90 / other 150 / 미라벨 60), 낮은 미라벨 장애물 200
지표: lethal 면적, 잔디 지면 중 lethal 비율, 통행 면적, 연결요소, 잔디밭 횡단 경로.
출력: <out>/compare.png (3열 크롭), diff.png, stats.json, paths.png
"""
import argparse, json, math, os, sys, heapq
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_map import MapPackage
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("geo_npz"); ap.add_argument("sem_npz"); ap.add_argument("--out", default="/data2/cg/compiled/compare_costmap")
ap.add_argument("--band-min", type=float, default=0.15); ap.add_argument("--clearance", type=float, default=1.2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
mp = MapPackage(a.manifest)
G = np.load(a.geo_npz); S = np.load(a.sem_npz); R = float(S["res"]); x0, y0 = float(S["x0"]), float(S["y0"])
assert G["cost"].shape == S["cost"].shape
cg, cs = G["cost"], S["cost"]; H, W = cs.shape

# ---------- A. 기하 전용 raw (0.1m 복셀 → 0.2m 격자에 투영) ----------
ctr = mp.voxel_centers(); lab = mp.labels(); GRASS = mp.classes.index("grass")
ii = np.floor((ctr[:, 0] - x0) / R).astype(np.int64); jj = np.floor((ctr[:, 1] - y0) / R).astype(np.int64)
ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W); ii, jj, zz, lb = ii[ok], jj[ok], ctr[ok, 2], lab[ok]
flat = ii * W + jj
zmin = np.full(H * W, np.inf, np.float32); np.minimum.at(zmin, flat, zz)
has = np.isfinite(zmin)
dz = zz - zmin[flat]
obs = (dz >= a.band_min) & (dz <= a.clearance)
ca = np.full(H * W, 255, np.uint8); ca[has] = 0
ca[np.unique(flat[obs])] = 254; ca = ca.reshape(H, W)

# 잔디 지면 마스크: semantic 코스트맵에서 지면 라벨 grass(90) 또는 (grass 위 장애물로 덮인 경우 포함) → S 의 지면 라벨을 별도로 복원
# S 는 밴드 장애물이 90 을 덮어쓸 수 있으므로, 잔디 지면 = 0.1m 복셀 열 최저 복셀 라벨이 grass
low_lab = np.full(H * W, -1, np.int16)
order = np.lexsort((zz, flat)); f_sorted = flat[order]; first = np.r_[True, f_sorted[1:] != f_sorted[:-1]]
low_lab[f_sorted[first]] = lb[order][first]; grass_ground = (low_lab == GRASS).reshape(H, W)

def stats(c, name):
    passable = c <= 200; lethal = c == 254
    comp, n = ndimage.label(passable, structure=np.ones((3, 3))); sz = np.bincount(comp.ravel()); sz[0] = 0
    return {"name": name, "passable_m2": round(float(passable.sum()) * R * R, 0), "lethal_m2": round(float(lethal.sum()) * R * R, 0),
            "grass_ground_m2": round(float(grass_ground.sum()) * R * R, 0),
            "grass_blocked_m2": round(float((lethal & grass_ground).sum()) * R * R, 0),
            "grass_blocked_pct": round(100 * float((lethal & grass_ground).sum()) / max(1, grass_ground.sum()), 1),
            "grass_passable_pct": round(100 * float((passable & grass_ground).sum()) / max(1, grass_ground.sum()), 1),
            "components": int(n), "largest_m2": round(float(sz.max()) * R * R, 0)}
ST = [stats(ca, "A geometric_raw(0.1m 점 전부, ≥0.15m)"), stats(cg, f"B geometric(occupancy hits≥{int(G['h_obs'])}, ≥{float(G['band_min'])}m)"), stats(cs, f"C semantic(hits≥{int(S['h_obs'])}, grass 통과)")]
for s in ST: print(s)

# ---------- 렌더 ----------
pal = {0: (200, 200, 200), 10: (170, 140, 110), 60: (150, 150, 150), 90: (95, 150, 95), 150: (120, 130, 150), 200: (220, 120, 40), 254: (200, 40, 40), 255: (25, 25, 25)}
def render(c):
    im = np.zeros(c.shape + (3,), np.uint8)
    for k, v in pal.items(): im[c == k] = v[::-1]
    return im
# 크롭: 잔디 지면 최대 연결 영역 주변
# 크롭: 잔디 밀도(10m 박스 평균)가 최대인 곳 중심 ±25m
dens = ndimage.uniform_filter(grass_ground.astype(np.float32), size=int(10 / R))
ci_, cj_ = np.unravel_index(np.argmax(dens), dens.shape); half = int(25 / R)
i0, i1 = max(ci_ - half, 0), min(ci_ + half, H); j0, j1 = max(cj_ - half, 0), min(cj_ + half, W)
gi, gj = np.where(grass_ground[i0:i1, j0:j1]); gi, gj = gi + i0, gj + j0
print("크롭(월드):", (x0 + i0 * R, y0 + j0 * R), "→", (x0 + i1 * R, y0 + j1 * R), "크롭 내 잔디", len(gi) * R * R, "m² · 밀도", float(dens.max()))
def crop_img(c):
    im = render(c)[i0:i1, j0:j1]; im = np.flipud(im.transpose(1, 0, 2)); return im
S_ = 3
tiles = []
for c, title in [(ca, "A  geometric (raw points, >=0.15 m -> lethal)"), (cg, "B  geometric (multi-scan occupancy)"), (cs, "C  semantic (grass passable, graded)")]:
    im = cv2.resize(crop_img(c), None, fx=S_, fy=S_, interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((36, im.shape[1], 3), np.uint8); cv2.putText(bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(np.vstack([bar, im]))
sep = np.full((tiles[0].shape[0], 6, 3), 60, np.uint8)
cv2.imwrite(os.path.join(a.out, "compare.png"), np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]]))
# diff: A 에서 lethal 이던 잔디 지면이 C 에서 통행 → 초록, 나머지 lethal 공통 → 빨강, 통행 공통 → 회색
d = np.full((H, W, 3), 25, np.uint8)
both_pass = (ca <= 200) & (cs <= 200); d[both_pass] = (120, 120, 120)
freed = (ca == 254) & (cs <= 200); d[freed] = (60, 200, 60)[::-1]
still = (ca == 254) & (cs == 254); d[still] = (200, 40, 40)[::-1]
newl = (ca <= 200) & (cs == 254); d[newl] = (40, 140, 220)[::-1]
dimg = cv2.resize(np.flipud(d[i0:i1, j0:j1].transpose(1, 0, 2)), None, fx=S_, fy=S_, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(a.out, "diff.png"), dimg)
diff_stats = {"freed_by_semantics_m2": round(float(freed.sum()) * R * R, 0), "freed_on_grass_m2": round(float((freed & grass_ground).sum()) * R * R, 0),
              "lethal_in_both_m2": round(float(still.sum()) * R * R, 0), "new_lethal_in_semantic_m2": round(float(newl.sum()) * R * R, 0)}
print("diff:", diff_stats)

# ---------- 잔디밭 횡단 경로 ----------
MULT = {0: 1.0, 10: 1.0, 60: 1.5, 90: 2.0, 150: 3.0, 200: 6.0}
def astar(c, s, g):
    mult = np.full(c.shape, np.inf, np.float32)
    for k, v in MULT.items(): mult[c == k] = v
    if not (np.isfinite(mult[s]) and np.isfinite(mult[g])): return None
    h = lambda p: math.hypot(p[0] - g[0], p[1] - g[1])
    best = {s: 0.0}; par = {}; pq = [(h(s), 0.0, s)]
    while pq:
        f, gc, cur = heapq.heappop(pq)
        if cur == g:
            p = [cur]
            while p[-1] in par: p.append(par[p[-1]])
            return p[::-1]
        if gc > best.get(cur, np.inf): continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0: continue
                n_ = (cur[0] + di, cur[1] + dj)
                if not (0 <= n_[0] < c.shape[0] and 0 <= n_[1] < c.shape[1]): continue
                mm = mult[n_]
                if not np.isfinite(mm): continue
                ng = gc + (1.4142 if di and dj else 1.0) * mm
                if ng < best.get(n_, np.inf): best[n_] = ng; par[n_] = cur; heapq.heappush(pq, (ng + h(n_), ng, n_))
    return None
# 시작/목표: C 의 최대 연결요소 안에서 고른다(C 에서는 반드시 도달 가능). 잔디 질량의 단축 방향 양쪽 끝의 비잔디 셀 = 횡단, 잔디 중심에 가장 가까운 잔디 셀 = 중심 목표
compC, _ = ndimage.label(cs <= 200, structure=np.ones((3, 3))); szC = np.bincount(compC.ravel()); szC[0] = 0; bigC = np.argmax(szC)
pts = np.c_[gi, gj].astype(float); mu = pts.mean(0); u, s_, vt = np.linalg.svd(pts - mu, full_matrices=False); major, minor = vt[0], vt[1]
ri, rj = np.where(compC[i0:i1, j0:j1] == bigC); ri, rj = ri + i0, rj + j0
q = np.c_[ri, rj].astype(float) - mu; pmaj = q @ major; pmin = q @ minor
band = np.abs(pmaj) < 6 / R                                   # 잔디 중심선 근처 ±6m
nong = ~grass_ground[ri, rj]
cand = band & nong
sA = gA = None
if cand.any():
    k1 = np.argmin(np.where(cand, pmin, np.inf)); k2 = np.argmax(np.where(cand, pmin, -np.inf))
    sA = (int(ri[k1]), int(rj[k1])); gA = (int(ri[k2]), int(rj[k2]))
gcand = grass_ground[ri, rj]
mu_cell = None
if gcand.any():
    d2 = (q[:, 0] ** 2 + q[:, 1] ** 2); d2[~gcand] = np.inf; k3 = int(np.argmin(d2)); mu_cell = (int(ri[k3]), int(rj[k3]))
mu = np.array(mu_cell if mu_cell else mu, float)
print("횡단 시나리오 start", sA, "goal", gA, "잔디 중심 목표", mu_cell)
paths = {}
for nm, c in [("A", ca), ("B", cg), ("C", cs)]:
    p1 = astar(c, sA, gA) if sA and gA else None
    p2 = astar(c, sA, mu_cell) if (sA and mu_cell and c[mu_cell] <= 200) else None
    def L(p): return None if p is None else round(sum(math.hypot(p[k+1][0]-p[k][0], p[k+1][1]-p[k][1]) for k in range(len(p)-1)) * R, 1)
    def on_grass(p): return None if p is None else round(100 * np.mean([grass_ground[q] for q in p]), 1)
    paths[nm] = {"cross": {"len_m": L(p1), "on_grass_pct": on_grass(p1)}, "to_center": {"len_m": L(p2), "reachable": p2 is not None}, "_p1": p1, "_p2": p2}
    print(nm, {k: v for k, v in paths[nm].items() if not k.startswith("_")})
straight = math.hypot(gA[0] - sA[0], gA[1] - sA[1]) * R if sA and gA else None
# 경로 그림 (C 배경 위에 A/B/C 경로) — 격자(row=i,col=j) 위에 그리고 마지막에 전치+반전
base = cv2.resize(render(cs)[i0:i1, j0:j1], None, fx=S_, fy=S_, interpolation=cv2.INTER_NEAREST)
def px(q): return (int((q[1] - j0) * S_ + S_ / 2), int((q[0] - i0) * S_ + S_ / 2))
cols = {"A": (60, 60, 255), "B": (255, 200, 40), "C": (255, 255, 0)}
for nm in ("A", "B", "C"):
    for key in ("_p1", "_p2"):
        p = paths[nm][key]
        if p: cv2.polylines(base, [np.array([px(q) for q in p], np.int32)], False, cols[nm], 2 if nm == "C" else 1)
for q, col in [(sA, (0, 255, 0)), (gA, (255, 0, 255)), (mu_cell, (0, 165, 255))]:
    if q: cv2.circle(base, px(q), 7, col, -1)
base = np.ascontiguousarray(np.flipud(base.transpose(1, 0, 2)))
cv2.imwrite(os.path.join(a.out, "paths.png"), base)
json.dump({"stats": ST, "diff": diff_stats, "paths": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in paths.items()}, "straight_m": straight,
           "crop_world": [x0 + i0 * R, y0 + j0 * R, x0 + i1 * R, y0 + j1 * R]}, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(a.out, "costmap_A_geometric_raw.npz"), cost=ca, x0=x0, y0=y0, res=R)
print("저장:", a.out)
