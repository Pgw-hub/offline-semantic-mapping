#!/usr/bin/env python3
"""voxel3d_tiles v2 → 같은 형식 사본 + 계단 디딤판 채움 셀(src=5). 계단은 맵에서 스스로 찾는다.

그레이팅 계단은 LiDAR 빔이 망 사이로 빠져 디딤판이 듬성듬성 잡히고, 평탄 채움(flat fill)은 0.1 m 발자국 단위로
면을 펴서 단과 단 사이에 가짜 중간 단을 만든다. 여기서는 계단을 '같은 높이 간격으로 같은 방향으로 물러나는
수평 판들의 사슬'로 찾아 모델을 맞추고, 각 디딤판 사각형 안만 그 디딤판 높이로 채운다 → 단 사이에는 아무것도 생기지 않는다.

  1) 윗면 셀: 같은 열에서 바로 위(10 cm 이내)에 다른 셀이 없는 셀 (판 밑면·몸통 내부 제외)
  2) 수평 판: 그레이팅은 점이 듬성듬성하고 위아래로 흩어진 점(halo)이 많아, 먼저 '평면 지지' 셀만 남김
     (반경 15 cm · 높이 차 2 cm 안의 윗면 셀이 SUPPORT_MIN 개 이상) → (x, y, 2.5·z) 12 cm 이웃 연결 성분 중
     z 사분위 폭 ≤ 3 cm, 면적 ≥ 0.035 m²
  3) 계단 사슬: 판 i→j 를 Δz 0.10–0.26 m, 가장 가까운 셀 간 수평 거리 ≤ 0.35 m, 중심 수평 거리 0.12–0.80 m 로 잇고
     가장 긴 사슬(같은 판 수면 면적 큰 쪽). 바닥·플랫폼처럼 넓은 판(> 4 m²)은 사슬 양 끝에만 허용, 채우지 않음
  4) 모델: 계단 축 u = 사슬에서 연이은 디딤판 중심 사이 방향의 중앙값(오르는 쪽 +). 직선(높이·앞모서리 vs 단 번호)에서 벗어난 판은 빼고,
     예측 자리에 맞는 판 조각을 전체 후보에서 다시 모아 2회 재맞춤. 안 잡힌 디딤판은 모델 값으로 채움.
     디딤판 k 의 앞모서리 t_k(축 위치 3 백분위),
     높이 h_k(z 중앙값), 단 높이·단 너비 = h_k, t_k 의 k 에 대한 직선 기울기, 디딤판 깊이 D = max(관측 깊이 중앙값, 단 너비),
     폭 = 디딤판 횡 위치 3–97 백분위의 중앙값
  4b) 판으로 안 잡힌 위·아래 끝 디딤판은 예측 사각형 안 윗면 셀 비율(≥ 15 %)로 확인해 추가(바닥·플랫폼에서 멈춤)
  5) 채움: 디딤판 k 사각형 [t_k, t_k + D] × [l0, l1] 의 2.5 cm 열 중 h_k ±3 cm 에 셀이 없는 열에 h_k 셀 추가
  6b) (--risers) 디딤판 앞모서리 바로 앞 2.5 cm 두께에 아래 디딤판 높이부터 이 디딤판 높이까지 수직 셀 (막힌 단차처럼)
  6) (--clean, 보기용) 디딤판 사각형 안, 그 디딤판 위 3.5–15 cm 의 떠 있는 셀(halo) 제거 — 다른 디딤판 높이 ±3.5 cm 는 남김
출력: <out>/voxel3d_tiles 형식 + <out>_stair.json (계단 모델 · 디딤판별 채움 전후 채워진 비율)
사용: voxel3d_stair_complete.py <in_tiles> <out_tiles> [--roi x0 x1 y0 y1] [--clean]
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_voxel3d_from_pcd import REC, write_tiles

FREE_ABOVE, LINK_R, Z_SCALE = 0.10, 0.12, 2.5
SUPPORT_R, SUPPORT_DZ, SUPPORT_MIN = 0.15, 0.02, 12
SEG_MIN_AREA, SEG_MAX_IQR, WIDE_AREA = 0.035, 0.03, 4.0
RISE = (0.10, 0.26); GAP_MAX, CENTRE = 0.35, (0.12, 0.80)   # 디딤판 하나가 안 잡혀도 잇도록 Δz·거리는 2단까지 허용
TOL_FILL, TOL_KEEP, HALO_TOP = 0.03, 0.035, 0.15


def read_tiles(d):
    idx = json.load(open(os.path.join(d, "index.json")))
    assert idx["format"] == "voxel3d_tiles v2", idx["format"]
    return idx, np.concatenate([np.fromfile(os.path.join(d, t["file"]), REC) for t in idx["tiles"]])


def top_cells(x, y, z, r):
    col = np.floor(x / r).astype(np.int64) * 2000003 + np.floor(y / r).astype(np.int64)
    o = np.lexsort((z, col)); top = np.ones(len(z), bool)
    same = col[o][1:] == col[o][:-1]; close = (z[o][1:] - z[o][:-1]) <= FREE_ABOVE
    top[o[:-1][same & close]] = False
    return top


def planar_support(P):
    sc = SUPPORT_R / SUPPORT_DZ
    return cKDTree(np.c_[P[:, :2], P[:, 2] * sc]).query_ball_point(np.c_[P[:, :2], P[:, 2] * sc], SUPPORT_R, return_length=True) - 1


def segments(P):
    tree = cKDTree(np.c_[P[:, :2], P[:, 2] * Z_SCALE]); pr = tree.query_pairs(LINK_R, output_type="ndarray")
    n = len(P); g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False); segs = []
    for s in np.unique(lab):
        m = np.nonzero(lab == s)[0]
        if len(m) < 20: continue
        z = P[m, 2]; q1, q3 = np.percentile(z, [25, 75]); area = len(m) * CELL_AREA
        if area < SEG_MIN_AREA or q3 - q1 > SEG_MAX_IQR: continue
        segs.append({"idx": m, "z": float(np.median(z)), "area": area, "c": P[m, :2].mean(0), "tree": cKDTree(P[m, :2])})
    return segs


def chain(segs):
    order = sorted(range(len(segs)), key=lambda i: segs[i]["z"]); best = {}
    for a, i in enumerate(order):
        si = segs[i]; cand = [(1, si["area"], [i])]
        for j in order[:a]:
            sj = segs[j]; dz = si["z"] - sj["z"]
            if not (RISE[0] <= dz <= 2 * RISE[1]) or j not in best: continue
            steps = 1 if dz <= RISE[1] else 2
            if sj["area"] > WIDE_AREA and len(best[j][2]) > 1: continue          # 넓은 판은 사슬 맨 아래에만
            dc = np.linalg.norm(si["c"] - sj["c"])
            if not (CENTRE[0] <= dc <= CENTRE[1] * steps or si["area"] > WIDE_AREA or sj["area"] > WIDE_AREA): continue
            sub = si["idx"][:: max(1, len(si["idx"]) // 200)]
            if sj["tree"].query(PTS[sub, :2], distance_upper_bound=GAP_MAX * steps)[0].min() > GAP_MAX * steps: continue
            n, ar, path = best[j]; cand.append((n + 1, ar + si["area"], path + [i]))
        best[i] = max(cand, key=lambda c: (c[0], c[1]))
    return max(best.values(), key=lambda c: (c[0], c[1]))[2]


def split_path(segs, path, max_turn_deg=60.0):
    """사슬을 오르는 방향이 크게 꺾이는 곳(계단참에서 되돌아 오르는 계단 등)에서 나눈다. 각 조각이 직선 계단 후보."""
    P = [i for i in path if segs[i]["area"] <= WIDE_AREA]
    if len(P) < 4: return [path]
    ct = np.cos(np.radians(max_turn_deg)); pieces = [[P[0], P[1]]]
    for b_ in P[2:]:
        q = pieces[-1]; d = segs[b_]["c"] - segs[q[-1]]["c"]; ref = segs[q[-1]]["c"] - segs[q[0]]["c"]
        if len(q) >= 2 and np.linalg.norm(d) > 1e-6 and np.linalg.norm(ref) > 1e-6 and float(d @ ref) / (np.linalg.norm(d) * np.linalg.norm(ref)) < ct:
            pieces.append([b_])
        else: q.append(b_)
    pieces = [q for q in pieces if len(q) >= 3]
    return pieces if len(pieces) > 1 else [path]


def fit_model(segs, path):
    """축을 연이은 디딤판 중심 방향의 중앙값으로 먼저 맞추고, 재맞춤이 실패하면 디딤판 중심 (x, y) 를 높이에 대해 최소제곱 직선으로 맞춘
    방향(디딤판이 좌우로 쪼개져 연이은 중심이 옆으로 튀는 계단에서 안정)으로 한 번 더."""
    try:
        return fit_model_axis(segs, path, "median")
    except SystemExit as ex:
        try: return fit_model_axis(segs, path, "regress")
        except SystemExit as ex2: raise SystemExit(f"{ex} / 회귀 축: {ex2}")


def fit_model_axis(segs, path, axis_mode):
    """사슬로 계단 축·단 높이·단 너비를 잡고, 직선에서 벗어난 판을 빼고, 예측 자리(높이 ±4 cm, 앞모서리 ±8 cm)에 맞는 판을
    전체 후보에서 다시 모아(같은 디딤판 조각은 합침) 다시 맞춤 — 2회."""
    chain_treads = [segs[i] for i in path if segs[i]["area"] <= WIDE_AREA]
    if len(chain_treads) < 3: raise SystemExit(f"계단 디딤판 {len(chain_treads)} 개 — 계단을 찾지 못함 (--roi 확인)")
    # 축 = 연이은 디딤판 중심 사이 방향들의 중앙값 (옆 구조물이 사슬에 몇 개 섞여도 흔들리지 않게)
    C = np.array([s["c"] for s in chain_treads]); Z = np.array([s["z"] for s in chain_treads])
    d = np.diff(C, axis=0); d = d[np.linalg.norm(d, axis=1) > 1e-6]; d /= np.linalg.norm(d, axis=1)[:, None]
    u = np.median(d, axis=0)
    if axis_mode == "regress":                                                        # 중심 xy 를 높이에 대한 직선으로 (오르는 방향)
        u = np.array([np.polyfit(Z, C[:, 0], 1)[0], np.polyfit(Z, C[:, 1], 1)[0]])
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])

    def stats(idx_list):
        q = PTS[np.concatenate(idx_list)]; t = q[:, :2] @ u; l = q[:, :2] @ v
        return dict(h=float(np.median(q[:, 2])), t0=float(np.percentile(t, 3)), t1=float(np.percentile(t, 97)),
                    l0=float(np.percentile(l, 3)), l1=float(np.percentile(l, 97)), idx=idx_list)

    rows = sorted([stats([s["idx"]]) for s in chain_treads], key=lambda r: r["h"])
    dep = np.array([r["t1"] - r["t0"] for r in rows]); wid = np.array([r["l1"] - r["l0"] for r in rows])
    rows = [r for r, d_, w_ in zip(rows, dep, wid) if d_ <= 2 * np.median(dep) and w_ <= 1.5 * np.median(wid)]   # 옆 구조물(길쭉한 판) 제외
    if len(rows) < 3: raise SystemExit("계단 디딤판 3 개 미만 — 계단을 찾지 못함")
    hs = np.array([r["h"] for r in rows]); dz = np.diff(hs); rise0 = float(np.median(dz[dz <= RISE[1]]))
    k = np.round((hs - hs[0]) / rise0).astype(int)
    for _ in range(2):
        rise, h0 = np.polyfit(k, [r["h"] for r in rows], 1); run, t00 = np.polyfit(k, [r["t0"] for r in rows], 1)
        lc = float(np.median([(r["l0"] + r["l1"]) / 2 for r in rows])); lw = float(np.median([r["l1"] - r["l0"] for r in rows]))
        groups = {}
        for s in segs:
            if s["area"] > WIDE_AREA: continue
            st = stats([s["idx"]]); kk = int(round((st["h"] - h0) / rise))
            if abs(st["h"] - (h0 + rise * kk)) > 0.04 or abs(st["t0"] - (t00 + run * kk)) > 0.08: continue
            if st["l1"] < lc - lw / 2 or st["l0"] > lc + lw / 2: continue      # 계단 띠 밖(옆 구조물)
            groups.setdefault(kk, []).append(s["idx"])
        rows = [stats(g) for kk, g in sorted(groups.items())]; k = np.array(sorted(groups))
        if len(rows) < 3: raise SystemExit(f"계단 모델 재맞춤 실패 (맞춘 단 높이 {rise:.3f}, 단 너비 {run:.3f}, 사슬 판 {len(chain_treads)} → 걸러진 판 {len(hs)}, 다시 모은 단 {len(rows)})")
    rise, h0 = np.polyfit(k, [r["h"] for r in rows], 1); run, t00 = np.polyfit(k, [r["t0"] for r in rows], 1)
    depth = max(float(np.median([r["t1"] - r["t0"] for r in rows])), run); depth = min(depth, 1.6 * run)
    l0 = float(np.median([r["l0"] for r in rows])); l1 = float(np.median([r["l1"] for r in rows]))
    res_h = np.array([r["h"] for r in rows]) - (h0 + rise * k); res_t = np.array([r["t0"] for r in rows]) - (t00 + run * k)
    byk = {int(kk): r for kk, r in zip(k, rows)}; k0 = int(k.min())
    treads = [dict(k=kk - k0, h=byk[kk]["h"], nose=byk[kk]["t0"], observed_depth=byk[kk]["t1"] - byk[kk]["t0"], inferred=False) if kk in byk else
              dict(k=kk - k0, h=float(h0 + rise * kk), nose=float(t00 + run * kk), observed_depth=0.0, inferred=True) for kk in range(k0, int(k.max()) + 1)]
    return dict(axis=axis_mode, u=u.tolist(), v=v.tolist(), rise=float(rise), run=float(run), depth=float(depth), l0=l0, l1=l1, treads=treads,
                fit_residual_cm=dict(h=float(np.abs(res_h).max() * 100), nose=float(np.abs(res_t).max() * 100)))


def extend(M, TOP, min_cover=0.15):
    """판으로 안 잡힌 끝 디딤판(점이 아주 성긴 위·아래 단): 모델이 예측한 사각형 안 윗면 셀 중 예측 높이 ±3 cm 비율이
    min_cover 이상이면 디딤판으로 추가. 바깥쪽(위로는 뒤, 아래로는 앞) 0.3 m 에도 같은 높이가 이어지면 바닥·플랫폼이므로 멈춤."""
    u, v = np.array(M["u"]), np.array(M["v"]); t = TOP[:, :2] @ u; l = TOP[:, :2] @ v
    band = (l >= M["l0"]) & (l <= M["l1"]); added = []
    for direction in (+1, -1):
        while True:
            last = M["treads"][-1] if direction > 0 else M["treads"][0]
            kk = last["k"] + direction; h = last["h"] + direction * M["rise"]; nose = last["nose"] + direction * M["run"]
            inr = band & (t >= nose) & (t < nose + M["depth"])
            behind = band & ((t >= nose + M["depth"] + 0.05) & (t < nose + M["depth"] + 0.35) if direction > 0 else (t >= nose - 0.35) & (t < nose - 0.05))
            near = np.abs(TOP[:, 2] - h) <= TOL_FILL
            cols_in = max(int((M["depth"] * (M["l1"] - M["l0"])) / CELL_AREA), 1)
            cover = (inr & near).sum() / cols_in; cont = (behind & near).sum() / max(int(0.3 * (M["l1"] - M["l0"]) / CELL_AREA), 1)
            if cover < min_cover or cont > 0.5: break
            tr = dict(k=kk, h=float(np.median(TOP[inr & near, 2])), nose=float(nose), observed_depth=0.0, inferred=True)
            M["treads"] = M["treads"] + [tr] if direction > 0 else [tr] + M["treads"]; added.append(kk)
    k0 = M["treads"][0]["k"]
    for tr in M["treads"]: tr["k"] -= k0
    M["extended"] = added


def main():
    global PTS, CELL_AREA
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir"); ap.add_argument("out_dir")
    ap.add_argument("--roi", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--clean", action="store_true", help="디딤판 위 떠 있는 셀(halo) 제거 (보기용)")
    ap.add_argument("--risers", action="store_true", help="디딤판 앞모서리 아래에 앞면 판(수직 셀)을 넣어 뚫린 단차를 막힌 계단처럼 만듦")
    ap.add_argument("--maxz-rule", type=float, default=0.03)
    ap.add_argument("--min-treads", type=int, default=5, help="받아들일 계단의 최소 디딤판 수 (가구 오검출 방지; 2–4 단짜리 짧은 계단은 놓침)")
    ap.add_argument("--min-width", type=float, default=0.7, help="받아들일 계단의 최소 폭 (m)")
    ap.add_argument("--max-stairs", type=int, default=30, help="찾을 계단(직선 구간) 최대 수. 한 계단을 찾으면 그 자리의 판을 빼고 다시 찾는다 (다층 건물). 1 = 예전처럼 가장 긴 것 하나")
    a = ap.parse_args()
    idx, R = read_tiles(a.in_dir); r, rz = idx["res_xy"], idx["res_z"]; CELL_AREA = r * r
    x = (R["ix"] + 0.5) * r; y = (R["iy"] + 0.5) * r
    z = np.where(R["max_z"] - R["mean_z"] > a.maxz_rule, R["max_z"], R["mean_z"]).astype(np.float64)
    roi = np.ones(len(R), bool) if a.roi is None else (x > a.roi[0]) & (x < a.roi[1]) & (y > a.roi[2]) & (y < a.roi[3])
    top = top_cells(x, y, z, r) & roi
    TOP = np.c_[x[top], y[top], z[top]]
    sup = planar_support(TOP); PTS = TOP[sup >= SUPPORT_MIN]
    # 계단 찾기: 가장 긴 사슬 → 모델 → 그 계단 자리(축 방향 범위 × 폭 × 높이 범위)의 판을 후보에서 빼고 다시 (다층 건물의 여러 계단)
    segs_all = segments(PTS); alive = np.ones(len(segs_all), bool); models = []; tries = 0
    C_ = np.array([sg["c"] for sg in segs_all]); Z_ = np.array([sg["z"] for sg in segs_all])

    def try_fit(segs, piece):
        """조각 하나 → 받아들일 계단 모델 또는 SystemExit(이유)."""
        M_ = fit_model(segs, piece); extend(M_, TOP)
        # 받아들이는 계단: 디딤판 ≥ min_treads, 폭 ≥ min_width, 단 높이·너비가 건축 계단 범위, 직선 맞춤 잔차 작음 (가구·잡동사니의 층층 판 거름)
        why = [w for w, bad in ((f"디딤판 {len(M_['treads'])} < {a.min_treads}", len(M_["treads"]) < a.min_treads),
                                (f"폭 {M_['l1'] - M_['l0']:.2f} < {a.min_width} m", M_["l1"] - M_["l0"] < a.min_width),
                                (f"단 높이 {M_['rise']:.3f} m 범위 밖", not 0.12 <= M_["rise"] <= 0.23), (f"단 너비 {M_['run']:.3f} m 범위 밖", not 0.20 <= M_["run"] <= 0.40),
                                (f"직선 잔차 {M_['fit_residual_cm']['h']:.1f} cm", M_["fit_residual_cm"]["h"] > 3.0),
                                (f"디딤판 깊이/단 너비 {M_['depth'] / M_['run']:.2f} > 1.3", M_["depth"] > 1.3 * M_["run"])) if bad]
        if why: raise SystemExit(", ".join(why))
        return M_

    def footprint(M_):
        """이 계단 자리(축 방향 범위 × 폭 × 높이 범위)에 있는 판 — 다음 계단 찾기에서 뺀다."""
        u_, v_ = np.array(M_["u"]), np.array(M_["v"]); hs_ = [tr["h"] for tr in M_["treads"]]; ns_ = [tr["nose"] for tr in M_["treads"]]; tc, lc_ = C_ @ u_, C_ @ v_
        return (tc >= min(ns_) - 0.1) & (tc <= max(ns_) + M_["depth"] + 0.1) & (lc_ >= M_["l0"] - 0.1) & (lc_ <= M_["l1"] + 0.1) & (Z_ >= min(hs_) - 0.06) & (Z_ <= max(hs_) + 0.06)

    while len(models) < a.max_stairs and alive.sum() >= 3 and tries < 4 * a.max_stairs:
        tries += 1; ids = np.flatnonzero(alive); segs = [segs_all[i] for i in ids]
        path = chain(segs)
        if len(path) < 3: break
        alive[ids[path]] = False                                                     # 이 사슬은 어떻게 되든 다시 안 씀
        pieces = split_path(segs, path)
        if len(pieces) > 1: print(f"  사슬 {len(path)} 판을 방향이 꺾이는 곳에서 {len(pieces)} 조각으로: {[len(q) for q in pieces]}")
        for piece in pieces:
            try:
                M_ = try_fit(segs, piece)
            except (SystemExit, Exception) as ex:                                     # 맞춤 실패·퇴화 → 계단 아님
                if a.max_stairs == 1 and not models: raise
                zz = [segs[i]["z"] for i in piece]; cc = np.mean([segs[i]["c"] for i in piece], 0)
                print(f"  후보 {len(piece)} 판 (z {min(zz):+.2f}..{max(zz):+.2f}, 중심 {cc[0]:+.1f},{cc[1]:+.1f}) 버림: {ex}"); continue
            alive &= ~footprint(M_); models.append(M_)
            if len(models) >= a.max_stairs: break
    if not models: raise SystemExit("계단을 찾지 못함 (--roi 확인)")
    M = dict(models[0]); M["stairs"] = models                                         # 맨 위 키는 첫 계단 (옛 읽기 코드 호환), stairs = 전부
    for si_, M_ in enumerate(models):
        u_ = np.array(M_["u"])
        print(f"계단 {si_ + 1}: 디딤판 {len(M_['treads'])} 개, 단 높이 {M_['rise'] * 100:.1f} cm, 단 너비 {M_['run'] * 100:.1f} cm, 디딤판 깊이 {M_['depth'] * 100:.1f} cm, "
              f"폭 {(M_['l1'] - M_['l0']) * 100:.0f} cm, 축 ({u_[0]:.3f}, {u_[1]:.3f}), 높이 {M_['treads'][0]['h']:+.2f}–{M_['treads'][-1]['h']:+.2f} m, "
              f"직선 맞춤 잔차 높이 {M_['fit_residual_cm']['h']:.1f} / 앞모서리 {M_['fit_residual_cm']['nose']:.1f} cm")
    # 셀 → 열 키, 열마다 z 목록 (채움 판정)
    col = R["ix"].astype(np.int64) * 2000003 + R["iy"]; o = np.argsort(col); cs = col[o]; zs_sorted = z[o]
    add = []; keep = np.ones(len(R), bool); n_r = 0
    for M_ in models:
      M = M_; u, v = np.array(M["u"]), np.array(M["v"]); t_all = np.c_[x, y] @ u; l_all = np.c_[x, y] @ v
      heights = np.array([tr["h"] for tr in M["treads"]])
      for tr in M["treads"]:
          t0, t1 = tr["nose"], tr["nose"] + M["depth"]
          # 사각형 안 열 중심 격자
          corners = np.array([[t0, M["l0"]], [t0, M["l1"]], [t1, M["l0"]], [t1, M["l1"]]]) @ np.array([u, v])
          ix0, iy0 = np.floor(corners.min(0) / r).astype(int) - 1; ix1, iy1 = np.floor(corners.max(0) / r).astype(int) + 1
          gi, gj = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1), indexing="ij"); cx = (gi.ravel() + 0.5) * r; cy = (gj.ravel() + 0.5) * r
          tt = np.c_[cx, cy] @ u; ll = np.c_[cx, cy] @ v; inside = (tt >= t0) & (tt < t1) & (ll >= M["l0"]) & (ll <= M["l1"])
          gi, gj, cx, cy = gi.ravel()[inside], gj.ravel()[inside], cx[inside], cy[inside]
          q = gi.astype(np.int64) * 2000003 + gj; lo = np.searchsorted(cs, q, "left"); hi = np.searchsorted(cs, q, "right")
          has = np.array([np.any(np.abs(zs_sorted[a_:b_] - tr["h"]) <= TOL_FILL) for a_, b_ in zip(lo, hi)])
          tr["columns"] = int(inside.sum()); tr["covered_before"] = float(has.mean()); tr["added"] = int((~has).sum())
          add += [(i_, j_, tr["h"]) for i_, j_ in zip(gi[~has], gj[~has])]
          if a.clean:
              inr = (t_all >= t0) & (t_all < t1) & (l_all >= M["l0"]) & (l_all <= M["l1"])
              halo = inr & (z > tr["h"] + TOL_KEEP) & (z < tr["h"] + HALO_TOP) & (np.abs(z[:, None] - heights[None, :]).min(1) > TOL_KEEP)
              tr["halo_removed"] = int(halo.sum()); keep &= ~halo
      if a.risers:   # 디딤판 k 앞모서리 바로 앞 한 칸 두께의 수직 판: 아래 디딤판(맨 아래는 h_0 - 단 높이) 위 3 cm 부터 h_k 아래 3 cm 까지
          for i_, tr in enumerate(M["treads"]):
              lo_z = (M["treads"][i_ - 1]["h"] if i_ > 0 else tr["h"] - M["rise"]) + TOL_FILL; hi_z = tr["h"] - TOL_FILL
              t0, t1 = tr["nose"] - r, tr["nose"]
              corners = np.array([[t0, M["l0"]], [t0, M["l1"]], [t1, M["l0"]], [t1, M["l1"]]]) @ np.array([u, v])
              ix0, iy0 = np.floor(corners.min(0) / r).astype(int) - 1; ix1, iy1 = np.floor(corners.max(0) / r).astype(int) + 1
              gi, gj = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1), indexing="ij"); cx = (gi.ravel() + 0.5) * r; cy = (gj.ravel() + 0.5) * r
              tt = np.c_[cx, cy] @ u; ll = np.c_[cx, cy] @ v; ins = (tt >= t0) & (tt < t1) & (ll >= M["l0"]) & (ll <= M["l1"])
              for zz in np.arange(lo_z, hi_z + 1e-9, rz):
                  add += [(i2, j2, zz) for i2, j2 in zip(gi.ravel()[ins], gj.ravel()[ins])]; n_r += int(ins.sum())
    if a.risers: print(f"앞면 판 셀 {n_r:,}")
    A = np.array(add).reshape(-1, 3)
    I = np.c_[A[:, 0], A[:, 1], np.floor(A[:, 2] / rz)].astype(np.int64)
    R2 = R[keep]
    Iall = np.concatenate([np.stack([R2["ix"], R2["iy"], R2["iz"]], -1).astype(np.int64), I])
    cat = lambda f, val, dt: np.concatenate([R2[f].astype(dt), np.full(len(I), val, dt)])
    params = {k: v_ for k, v_ in idx.items() if k not in ("format", "record", "record_bytes", "byte_order", "index_to_world", "tile_of", "n_cells", "total_bytes", "tiles")}
    params["source_codes"] = {**idx.get("source_codes", {}), "5": "stair tread fill (voxel3d_stair_complete.py)"}
    params["derived_from"] = os.path.abspath(a.in_dir); params["stair_complete"] = {"clean": a.clean, "added": int(len(I)), "removed": int((~keep).sum())}
    write_tiles(a.out_dir, Iall, cat("count", 0, np.int64), np.concatenate([R2["mean_z"], A[:, 2]]).astype(np.float32),
                np.concatenate([R2["max_z"], A[:, 2]]).astype(np.float32), cat("cls", 255, np.uint8), cat("src", 5, np.uint8),
                np.concatenate([R2["logodds"].astype(np.float32), np.zeros(len(I), np.float32)]), params)
    M0 = dict(models[0]); M0["stairs"] = models
    json.dump(M0, open(a.out_dir.rstrip("/") + "_stair.json", "w"), indent=1, ensure_ascii=False)
    for tr in (t_ for M_ in models for t_ in M_["treads"]):
        print(f"  디딤판 {tr['k']:2d}{' (모델)' if tr['inferred'] else '       '}  h {tr['h']:+.3f}  앞모서리 {tr['nose']:+.3f}  관측 깊이 {tr['observed_depth'] * 100:4.1f} cm  채워진 칸 {tr['covered_before'] * 100:5.1f} % → 100 %  (+{tr['added']})"
              + (f"  halo 제거 {tr['halo_removed']}" if a.clean else ""))
    print(f"added {len(I):,} cells, removed {int((~keep).sum()):,} → {a.out_dir}")


if __name__ == "__main__":
    main()
