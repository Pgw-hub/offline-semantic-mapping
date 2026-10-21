#!/usr/bin/env python3
"""산책로 준수 정책 검증 — 같은 출발·목표 쌍을 세 코스트맵으로 계획해 경로가 어디로 나는지 비교.

GEO   : 기하 전용 (occupancy, 라벨 무시, 지면 전부 free)          → 최단 경로, 잔디·보도 구분 없음
SEM   : semantic 등급 (path 10 · 미라벨 60 · grass 90 · other 150 · 불확실 200) → 산책로 선호, 필요하면 잔디 통과
STRICT: SEM + grass lethal                                          → 잔디 진입 금지 정책
쌍: 두 코스트맵 모두에서 통행 가능·같은 연결요소인 산책로(path) 셀 두 개, 직선거리 20~80 m.
지표: 산책로 위 비율, 잔디 위 길이(m), 미라벨/불확실 비율, 경로 길이, GEO 대비 길이 증가.
"""
import argparse, heapq, json, math, os, sys, time
import numpy as np, cv2
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("geo_npz"); ap.add_argument("sem_npz"); ap.add_argument("--out", default="/data2/cg/compiled/policy_adherence")
ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--dmin", type=float, default=20); ap.add_argument("--dmax", type=float, default=80)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
G = np.load(a.geo_npz); S = np.load(a.sem_npz); R = float(S["res"]); x0, y0 = float(S["x0"]), float(S["y0"])
cg, cs = G["cost"], S["cost"]; H, W = cs.shape
cstrict = cs.copy(); cstrict[cs == 90] = 254
MULT = {0: 1.0, 10: 1.0, 60: 1.5, 90: 2.0, 150: 3.0, 200: 6.0}
def mult_of(c):
    m = np.full(c.shape, np.inf, np.float32)
    for k, v in MULT.items(): m[c == k] = v
    return m
MAPS = {"GEO": (cg, mult_of(cg)), "SEM": (cs, mult_of(cs)), "STRICT": (cstrict, mult_of(cstrict))}

def astar(mult, s, g, max_expand=4_000_000):
    h = lambda p: math.hypot(p[0] - g[0], p[1] - g[1])
    best = {s: 0.0}; par = {}; pq = [(h(s), 0.0, s)]; n = 0
    while pq:
        f, gc, cur = heapq.heappop(pq)
        if cur == g:
            p = [cur]
            while p[-1] in par: p.append(par[p[-1]])
            return p[::-1]
        if gc > best.get(cur, np.inf): continue
        n += 1
        if n > max_expand: return None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0: continue
                q = (cur[0] + di, cur[1] + dj)
                if not (0 <= q[0] < H and 0 <= q[1] < W): continue
                mm = mult[q]
                if not np.isfinite(mm): continue
                ng = gc + (1.4142 if di and dj else 1.0) * mm
                if ng < best.get(q, np.inf): best[q] = ng; par[q] = cur; heapq.heappush(pq, (ng + h(q), ng, q))
    return None

# 지면 라벨 (SEM 코스트맵의 지면 등급; 장애물이 덮은 셀은 lethal 로 남음)
lab = cs.copy()
def seg_stats(path):
    pts = np.array(path); step = np.r_[0, np.hypot(*np.diff(pts, axis=0).T)] * R      # 각 셀에 도달하는 스텝 길이
    L = float(step.sum()); codes = lab[pts[:, 0], pts[:, 1]]
    def m(code): return float(step[codes == code].sum())
    return {"len_m": round(L, 1), "path_m": round(m(10), 1), "grass_m": round(m(90), 1), "unlabeled_m": round(m(60), 1), "other_m": round(m(150), 1), "uncertain_m": round(m(200), 1),
            "path_pct": round(100 * m(10) / max(L, 1e-6), 1), "grass_pct": round(100 * m(90) / max(L, 1e-6), 1)}

# 쌍 샘플: path 셀, GEO·SEM·STRICT 모두 통행 가능, 세 맵 모두에서 같은 연결요소
comps = {k: ndimage.label(np.isfinite(m), structure=np.ones((3, 3)))[0] for k, (c, m) in MAPS.items()}
cand = (cs == 10) & np.isfinite(MAPS["GEO"][1]) & np.isfinite(MAPS["STRICT"][1])
ci, cj = np.where(cand); rng = np.random.default_rng(a.seed)
pairs = []; tries = 0
while len(pairs) < a.n and tries < 200000:
    tries += 1
    i1, i2 = rng.integers(len(ci), size=2); s = (int(ci[i1]), int(cj[i1])); g = (int(ci[i2]), int(cj[i2]))
    d = math.hypot(s[0] - g[0], s[1] - g[1]) * R
    if not (a.dmin <= d <= a.dmax): continue
    if any(comps[k][s] != comps[k][g] for k in MAPS): continue
    pairs.append((s, g, d))
print(f"path 셀 {len(ci):,} · 쌍 {len(pairs)} (시도 {tries})")

rows = []; allpaths = []
t0 = time.time()
for k, (s, g, d) in enumerate(pairs):
    rec = {"id": k, "start": [round(x0 + (s[0] + .5) * R, 1), round(y0 + (s[1] + .5) * R, 1)], "goal": [round(x0 + (g[0] + .5) * R, 1), round(y0 + (g[1] + .5) * R, 1)], "straight_m": round(d, 1)}
    P = {}
    for nm, (c, m) in MAPS.items():
        t1 = time.time(); p = astar(m, s, g); rec[nm] = None if p is None else {**seg_stats(p), "ms": round((time.time() - t1) * 1000)}
        P[nm] = p
    allpaths.append(P); rows.append(rec)
    if rec["GEO"] and rec["SEM"]:
        print(f"[{k:2d}] {d:5.1f}m  GEO path {rec['GEO']['path_pct']:5.1f}% grass {rec['GEO']['grass_m']:5.1f}m | SEM path {rec['SEM']['path_pct']:5.1f}% grass {rec['SEM']['grass_m']:5.1f}m len +{rec['SEM']['len_m']-rec['GEO']['len_m']:4.1f}m | STRICT " + (f"path {rec['STRICT']['path_pct']:5.1f}% len {rec['STRICT']['len_m']}" if rec["STRICT"] else "불가"))
ok = [r for r in rows if r["GEO"] and r["SEM"]]
def agg(nm, key): 
    v = [r[nm][key] for r in ok if r.get(nm)]; return round(float(np.mean(v)), 1) if v else None
summary = {"pairs": len(rows), "both_ok": len(ok), "strict_ok": sum(1 for r in rows if r["STRICT"]),
           "GEO": {k: agg("GEO", k) for k in ("path_pct", "grass_pct", "grass_m", "unlabeled_m", "uncertain_m", "len_m", "ms")},
           "SEM": {k: agg("SEM", k) for k in ("path_pct", "grass_pct", "grass_m", "unlabeled_m", "uncertain_m", "len_m", "ms")},
           "STRICT": {k: agg("STRICT", k) for k in ("path_pct", "grass_pct", "grass_m", "unlabeled_m", "uncertain_m", "len_m", "ms")},
           "grass_total_m": {nm: round(sum(r[nm]["grass_m"] for r in ok if r.get(nm)), 1) for nm in MAPS},
           "pairs_geo_touch_grass": sum(1 for r in ok if r["GEO"]["grass_m"] > 0.5), "pairs_sem_touch_grass": sum(1 for r in ok if r["SEM"]["grass_m"] > 0.5),
           "len_increase_sem_vs_geo_pct": round(100 * (sum(r["SEM"]["len_m"] for r in ok) / max(1e-6, sum(r["GEO"]["len_m"] for r in ok)) - 1), 1),
           "len_increase_strict_vs_geo_pct": round(100 * (sum(r["STRICT"]["len_m"] for r in ok if r["STRICT"]) / max(1e-6, sum(r["GEO"]["len_m"] for r in ok if r["STRICT"])) - 1), 1)}
print("요약:", json.dumps(summary, ensure_ascii=False)); print(f"[{time.time()-t0:.0f}s]")

# ---- 렌더: 전체 뷰 + 잔디 차이가 큰 6쌍 타일 ----
pal = {0: (200, 200, 200), 10: (170, 140, 110), 60: (150, 150, 150), 90: (95, 150, 95), 150: (120, 130, 150), 200: (220, 120, 40), 254: (200, 40, 40), 255: (25, 25, 25)}
def render(c):
    im = np.zeros(c.shape + (3,), np.uint8)
    for k, v in pal.items(): im[c == k] = v[::-1]
    return im
base = render(cs); COL = {"GEO": (60, 60, 255), "SEM": (255, 255, 0), "STRICT": (255, 0, 255)}
def view(i0, i1, j0, j1, Sc, paths_list, which):
    """격자(row=i=x, col=j=y) 위에 그린 뒤 전치+상하반전 → 화면 x 오른쪽, y 위"""
    img = cv2.resize(base[i0:i1, j0:j1], None, fx=Sc, fy=Sc, interpolation=cv2.INTER_NEAREST)
    for P in paths_list:
        for nm in which:
            p = P.get(nm)
            if p: cv2.polylines(img, [np.array([[int((q[1] - j0) * Sc + Sc / 2), int((q[0] - i0) * Sc + Sc / 2)] for q in p], np.int32)], False, COL[nm], 2)
    return np.ascontiguousarray(np.flipud(img.transpose(1, 0, 2)))
known = np.where(cs < 255); gi0, gi1 = known[0].min(), known[0].max() + 1; gj0, gj1 = known[1].min(), known[1].max() + 1
cv2.imwrite(os.path.join(a.out, "overview.png"), view(gi0, gi1, gj0, gj1, 2, allpaths, ("GEO", "SEM")))
diffs = sorted([(r["GEO"]["grass_m"] - r["SEM"]["grass_m"], r["id"]) for r in ok], reverse=True)[:6]
tiles = []; Wt = 560
for _, k in diffs:
    P = allpaths[k]; pts = np.array(P["GEO"] + P["SEM"] + (P["STRICT"] or [])); m = int(5 / R)
    i0, i1 = max(pts[:, 0].min() - m, 0), min(pts[:, 0].max() + m, H); j0, j1 = max(pts[:, 1].min() - m, 0), min(pts[:, 1].max() + m, W)
    side = max(i1 - i0, j1 - j0); ic, jc = (i0 + i1) // 2, (j0 + j1) // 2          # 정사각 크롭
    i0, i1 = max(ic - side // 2, 0), min(ic - side // 2 + side, H); j0, j1 = max(jc - side // 2, 0), min(jc - side // 2 + side, W)
    Sc = max(1, int(Wt / (i1 - i0)))
    im = view(i0, i1, j0, j1, Sc, [P], ("GEO", "SEM", "STRICT"))
    im = cv2.resize(im, (Wt, int(im.shape[0] * Wt / im.shape[1])), interpolation=cv2.INTER_NEAREST)
    r = rows[k]; bar = np.zeros((48, Wt, 3), np.uint8)
    cv2.putText(bar, f"#{k}  straight {r['straight_m']} m", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, f"GEO grass {r['GEO']['grass_m']} m | SEM grass {r['SEM']['grass_m']} m (+{r['SEM']['len_m']-r['GEO']['len_m']:.1f} m) | STRICT +{(r['STRICT']['len_m']-r['GEO']['len_m']) if r['STRICT'] else float('nan'):.1f} m", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    tiles.append(np.vstack([bar, im]))
hmax = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.full((hmax - t.shape[0], Wt, 3), 25, np.uint8)]) for t in tiles]
grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])]) if len(tiles) >= 6 else np.hstack(tiles)
cv2.imwrite(os.path.join(a.out, "examples.png"), grid)
json.dump({"summary": summary, "rows": rows}, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1)
print("저장:", a.out)
