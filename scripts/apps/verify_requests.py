#!/usr/bin/env python3
"""맵 패키지(map_manifest.json)만으로 로봇 요청 검증 — 재계산 없음, 매니페스트의 9개 파일만 사용.

 A) 랜덤 시작 위치 N개 → "벤치 옆으로 가줘" → (텍스트→객체: CLIP) → (옆: 목표 자세 선택, 3D occupancy 로 몸통 공간 확인)
    → 전역 경로 (가중 A*, 패키지 costmap_2d) → 성공률·길이·지형 구성
 B) 지형 코스트맵 요청: grass_only / avoid_grass / grass_layer, n×n 크롭, nav2 map_server (pgm+yaml) 내보내기
출력: <out>/report.json, overview.png, verify.rrd, costmap_*.{npz,pgm,yaml,png}
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # scripts/ (패키지 import)
import argparse, heapq, json, math, os, re, sys, time
import numpy as np
import cv2
from package.load_map import MapPackage

KO2EN = {"벤치": "bench", "자판기": "vending machine", "냉장고": "refrigerator", "자전거": "bicycle", "쓰레기통": "trash bin",
         "화분": "potted plant", "의자": "chair", "문": "door", "티비": "tv", "텔레비전": "tv", "수납함": "storage bin", "모자": "hat",
         "신발": "shoe", "컵": "cup", "테이블": "table", "책상": "desk", "소파": "couch", "차": "car", "자동차": "car", "나무": "tree"}
REL = {"옆": "next_to", "앞": "next_to", "근처": "next_to", "까지": "to", "로": "to", "에": "to"}
STEP_MULT = {10: 1.0, 60: 1.5, 90: 2.0, 150: 3.0, 200: 6.0}      # 코스트 등급 → A* 스텝 가중 (200 = 불확실, 비싸게 통과)


def parse_request(text):
    """'벤치 옆으로 가줘' → (target_en, relation). 영어 태그 직접 입력도 허용."""
    t = text.strip()
    rel = "to"
    for k, v in REL.items():
        if k in t:
            rel = v; break
    words = re.split(r"[\s,]+", re.sub(r"(옆으로|으로|로|에|까지|가줘|가|줘|이동|해줘|near|next to|to|go|the)", " ", t))
    words = [w for w in words if w]
    target = " ".join(KO2EN.get(w, w) for w in words) if words else t
    return target, rel


class Planner:
    def __init__(self, mp, fill_iters=3):
        self.mp = mp; c = mp.costmap.copy(); self.raw = mp.costmap; self.res = mp.cm_res; self.x0 = mp.cm_x0; self.y0 = mp.cm_y0
        from scipy import ndimage; self.nd = ndimage
        self.fill_stats = self.fill_holes(c, fill_iters) if fill_iters else {}
        self.c = c
        self.mult = np.full(c.shape, np.inf, np.float32)
        for k, v in STEP_MULT.items():
            self.mult[c == k] = v
        self.passable = np.isfinite(self.mult)
        self.comp, self.n_comp = ndimage.label(self.passable, structure=np.ones((3, 3)))
        sizes = np.bincount(self.comp.ravel()); sizes[0] = 0; self.comp_sizes = sizes

    def components_of(self, mask):
        comp, n = self.nd.label(mask, structure=np.ones((3, 3))); s = np.bincount(comp.ravel()); s[0] = 0
        return n, sorted((s * self.res ** 2).tolist(), reverse=True)[:5]

    def fill_holes(self, c, iters):
        """희소 LiDAR 로 255(미관측) 가 된 작은 구멍을 통행면으로 메움 (계획용 격자만; 패키지 costmap 은 그대로).
        규칙: 255 셀의 8-이웃 중 통행가능 ≥3 이고 lethal(≥253) 이웃이 없으면 → 이웃 최대 코스트로 채움 (보수적).
        채운 셀은 3D occupancy 로 몸통 밴드(지면+0.4/+0.8m) 가 occupied 가 아닌지 확인 — occupied 면 254."""
        nd = self.nd; mp = self.mp
        before_n, before_top = self.components_of(np.isin(c, list(STEP_MULT)))
        # 지면 z (셀별 최저 복셀) — 3D 확인용
        ctr = mp.voxel_centers(); ii = np.floor((ctr[:, 0] - self.x0) / self.res).astype(np.int64); jj = np.floor((ctr[:, 1] - self.y0) / self.res).astype(np.int64)
        okv = (ii >= 0) & (ii < c.shape[0]) & (jj >= 0) & (jj < c.shape[1]); zg = np.full(c.shape, np.nan, np.float32)
        flat = ii[okv] * c.shape[1] + jj[okv]; zmin = np.full(c.shape[0] * c.shape[1], np.inf, np.float32); np.minimum.at(zmin, flat, ctr[okv, 2])
        zg = zmin.reshape(c.shape); zg[~np.isfinite(zg)] = np.nan
        filled_total = 0; rejected_3d = 0
        k8 = np.ones((3, 3), np.float32); k8[1, 1] = 0
        for _ in range(iters):
            passable = np.isin(c, list(STEP_MULT)); lethal = c >= 253
            n_pass = nd.convolve(passable.astype(np.float32), k8, mode="constant"); n_leth = nd.convolve(lethal.astype(np.float32), k8, mode="constant")
            cand = (c == 255) & (n_pass >= 3) & (n_leth == 0)
            if not cand.any(): break
            # 이웃 최대 통행 코스트
            cm = np.where(passable, c, 0).astype(np.uint8); nb_max = nd.maximum_filter(cm, size=3)
            ci, cj = np.where(cand)
            # 지면 z: 이웃 평균
            zn = nd.generic_filter(np.nan_to_num(zg, nan=0.0), np.mean, size=3, mode="constant") if False else None  # (느려서 생략) → 자기 셀 or 이웃 최소
            zc = zg[ci, cj]; miss = np.isnan(zc)
            if miss.any():
                zmin_nb = nd.minimum_filter(np.nan_to_num(zg, nan=1e6), size=3)[ci, cj]; zc = np.where(miss, zmin_nb, zc)
            x = self.x0 + (ci + 0.5) * self.res; y = self.y0 + (cj + 0.5) * self.res
            body_occ = np.zeros(len(ci), bool)
            for dz in (0.4, 0.8):
                body_occ |= mp.occupancy_state(np.c_[x, y, zc + dz]) == 1
            c[ci[~body_occ], cj[~body_occ]] = nb_max[ci[~body_occ], cj[~body_occ]]
            c[ci[body_occ], cj[body_occ]] = 254
            filled_total += int((~body_occ).sum()); rejected_3d += int(body_occ.sum())
        after_n, after_top = self.components_of(np.isin(c, list(STEP_MULT)))
        return {"iters": iters, "filled_cells": filled_total, "filled_m2": round(filled_total * self.res ** 2, 1), "rejected_3d_cells": rejected_3d,
                "components_before": before_n, "top5_m2_before": [round(v, 1) for v in before_top], "components_after": after_n, "top5_m2_after": [round(v, 1) for v in after_top]}

    def w2c(self, x, y): return (int(math.floor((x - self.x0) / self.res)), int(math.floor((y - self.y0) / self.res)))
    def c2w(self, i, j): return (self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res)
    def inb(self, i, j): return 0 <= i < self.c.shape[0] and 0 <= j < self.c.shape[1]

    def astar(self, s, g, max_expand=3_000_000):
        if s == g: return [s], 0.0
        H, W = self.c.shape; mult = self.mult
        def h(i, j):
            di, dj = abs(i - g[0]), abs(j - g[1]); return (max(di, dj) + (math.sqrt(2) - 1) * min(di, dj))
        best = {s: 0.0}; par = {}; pq = [(h(*s), 0.0, s)]; nexp = 0
        while pq:
            f, gc, cur = heapq.heappop(pq)
            if cur == g:
                p = [cur]
                while p[-1] in par: p.append(par[p[-1]])
                return p[::-1], gc
            if gc > best.get(cur, np.inf): continue
            nexp += 1
            if nexp > max_expand: return None, None
            i, j = cur
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0: continue
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < H and 0 <= nj < W): continue
                    m = mult[ni, nj]
                    if not np.isfinite(m): continue
                    step = (math.sqrt(2) if di and dj else 1.0) * m
                    ng = gc + step
                    if ng < best.get((ni, nj), np.inf):
                        best[(ni, nj)] = ng; par[(ni, nj)] = cur; heapq.heappush(pq, (ng + h(ni, nj), ng, (ni, nj)))
        return None, None

    def path_stats(self, path):
        pts = np.array([self.c2w(i, j) for i, j in path]); L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        codes = np.array([self.c[i, j] for i, j in path]); n = len(codes)
        comp = {"path": int((codes == 10).sum()), "unlabeled": int((codes == 60).sum()), "grass": int((codes == 90).sum()),
                "other": int((codes == 150).sum()), "uncertain": int((codes == 200).sum())}
        return L, {k: round(v / n * 100, 1) for k, v in comp.items()}, pts


class TextQuery:
    """CLIP 텍스트 → 패키지 objects_clip 코사인. (서버/Map Compiler 측; 로봇은 결과만 받음)"""
    def __init__(self, mp):
        import torch, open_clip
        self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        self.model = self.model.to(self.dev).eval(); self.tok = open_clip.get_tokenizer("ViT-H-14")
        F = mp.clip; self.F = F / np.linalg.norm(F, axis=1, keepdims=True)

    def __call__(self, text, topk=10):
        t0 = time.time()
        with self.torch.no_grad():
            f = self.model.encode_text(self.tok([f"a photo of a {text}"]).to(self.dev)).float().cpu().numpy()[0]
        f /= np.linalg.norm(f); s = self.F @ f; idx = np.argsort(-s)[:topk]
        return [(int(i), float(s[i])) for i in idx], time.time() - t0


def pick_goal(mp, pl, start_c, obj, standoff=(0.5, 1.5)):
    """객체 '옆' 목표: 발자국 밖 standoff 링, 통행 가능, 시작과 같은 연결요소, 3D occupancy 로 몸통(0.3~0.8m) 비었는지 확인."""
    cen = np.array(obj["center_w"], float); ext = np.array(obj.get("extent") or [0.5] * 3, float)
    r = float(max(ext[0], ext[1]) / 2); lo, hi = r + standoff[0], r + standoff[1]
    ci, cj = pl.w2c(cen[0], cen[1]); R = int(math.ceil(hi / pl.res)) + 1
    i0, i1 = max(ci - R, 0), min(ci + R + 1, pl.c.shape[0]); j0, j1 = max(cj - R, 0), min(cj + R + 1, pl.c.shape[1])
    ii, jj = np.mgrid[i0:i1, j0:j1]; xs = pl.x0 + (ii + 0.5) * pl.res; ys = pl.y0 + (jj + 0.5) * pl.res
    d = np.hypot(xs - cen[0], ys - cen[1])
    ok = (d >= lo) & (d <= hi) & pl.passable[i0:i1, j0:j1] & (pl.comp[i0:i1, j0:j1] == pl.comp[start_c])
    if not ok.any():
        return None, "no reachable cell in standoff ring"
    # 3D 확인: 지면 z ≈ 객체 바닥; 몸통 높이 0.3/0.6/0.9 m 가 occupied 면 제외 (free 또는 unknown 허용)
    zg = cen[2] - ext[2] / 2
    cand = np.stack([xs[ok], ys[ok]], -1); body_ok = np.ones(len(cand), bool)
    for dz in (0.3, 0.6, 0.9):
        st = mp.occupancy_state(np.c_[cand, np.full(len(cand), zg + dz)])
        body_ok &= st != 1
    n_body_rej = int((~body_ok).sum())
    if not body_ok.any():
        return None, f"all {len(cand)} ring cells occupied in 3D"
    score = pl.mult[i0:i1, j0:j1][ok] + 0.5 * d[ok]; score[~body_ok] = np.inf
    k = int(np.argmin(score)); gi, gj = ii[ok][k], jj[ok][k]
    yaw = math.atan2(cen[1] - ys[ok][k], cen[0] - xs[ok][k])
    return {"cell": (int(gi), int(gj)), "xy": (float(xs[ok][k]), float(ys[ok][k])), "yaw": yaw, "z_ground": float(zg),
            "ring_cells": int(ok.sum()), "rejected_3d": n_body_rej}, None


def semantic_costmap(mp, mode, center=None, size_m=None):
    """지형 코스트맵 요청. 코드: path 10 / unlabeled 60 / grass 90 / other 150 / uncertain 200 / lethal 254 / unknown 255"""
    c = mp.costmap; grass = c == 90
    if mode == "grass_only":       # 잔디만 통행 (잔디 밖 = lethal, 미관측 유지)
        out = np.where(grass, 0, np.where(c == 255, 255, 254)).astype(np.uint8)
    elif mode == "avoid_grass":    # 기본 등급 그대로, 잔디만 lethal
        out = c.copy(); out[grass] = 254
    elif mode == "grass_layer":    # 레이어(마스크)만: 잔디 100 / 아님 0 / 미관측 255
        out = np.where(grass, 100, np.where(c == 255, 255, 0)).astype(np.uint8)
    elif mode == "graded":
        out = c.copy()
    else:
        raise ValueError(mode)
    x0, y0 = mp.cm_x0, mp.cm_y0
    if center is not None and size_m:
        n = int(round(size_m / mp.cm_res)); ci = int((center[0] - x0) / mp.cm_res); cj = int((center[1] - y0) / mp.cm_res)
        i0, j0 = max(ci - n // 2, 0), max(cj - n // 2, 0); out = out[i0:i0 + n, j0:j0 + n]
        x0, y0 = x0 + i0 * mp.cm_res, y0 + j0 * mp.cm_res
    return out, x0, y0


def export_nav2(cost, x0, y0, res, prefix):
    """nav2 map_server 형식: pgm (행=y 위→아래, 열=x) + yaml. 코스트 0→254(흰) … 254→0(검), 255 미관측→205(회)"""
    img = np.where(cost == 255, 205, (254 - cost.astype(np.int32) * 254 // 254)).clip(0, 254).astype(np.uint8)
    img = np.flipud(img.T)                                  # row=x,col=y → rows=y(위가 +y), cols=x
    cv2.imwrite(prefix + ".pgm", img)
    yaml = (f"image: {os.path.basename(prefix)}.pgm\nmode: trinary\nresolution: {res}\norigin: [{x0:.3f}, {y0:.3f}, 0.0]\n"
            f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    open(prefix + ".yaml", "w").write(yaml)
    np.savez_compressed(prefix + ".npz", cost=cost, x0=x0, y0=y0, res=res)
    # 미리보기 png (색)
    pal = {0: (200, 200, 200), 10: (170, 140, 110), 60: (150, 150, 150), 90: (95, 150, 95), 100: (95, 150, 95), 150: (120, 130, 150),
           200: (220, 120, 40), 254: (200, 40, 40), 255: (30, 30, 30)}
    rgb = np.zeros(cost.shape + (3,), np.uint8)
    for k, v in pal.items(): rgb[cost == k] = v[::-1]
    cv2.imwrite(prefix + ".png", np.flipud(rgb.transpose(1, 0, 2)))
    return img.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--out", default="/data2/cg/compiled/verify")
    ap.add_argument("--request", default="벤치 옆으로 가줘"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-score", type=float, default=0.20); ap.add_argument("--exclude-indoor", action="store_true", help="실내 로그(log_000/001) 객체 제외")
    ap.add_argument("--fill", type=int, default=3, help="계획 격자 구멍 메움 반복 (0=끔)")
    ap.add_argument("--costmap-key", default="costmap_2d", help="매니페스트의 코스트맵 항목 (costmap_2d | costmap_2d_occ)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    mp = MapPackage(a.manifest, costmap_key=a.costmap_key); pl = Planner(mp, fill_iters=a.fill)
    print("코스트맵:", mp.costmap_key, mp.costmap.shape, "res", mp.cm_res)
    big = np.argsort(-pl.comp_sizes)[:5]
    print(f"로드 {mp.load_s:.1f}s · 통행가능 셀 {int(pl.passable.sum()):,} · 연결요소 {pl.n_comp} · 상위5 크기(m²) {[round(pl.comp_sizes[k]*pl.res**2,1) for k in big]}")
    print("구멍 메움:", json.dumps(pl.fill_stats, ensure_ascii=False))

    # ---------- A) 텍스트 → 객체 ----------
    target, rel = parse_request(a.request); print(f"요청 '{a.request}' → target='{target}', relation={rel}")
    tq = TextQuery(mp); hits, t_clip = tq(target, topk=15)
    cands = []
    for i, s in hits:
        o = mp.objects[i]
        if s < a.min_score: continue
        if a.exclude_indoor and set(o.get("logs", [])) <= {"log_000", "log_001"}: continue
        cands.append((i, s))
    print(f"CLIP {t_clip*1000:.0f}ms · 후보 {len(cands)}: " + ", ".join(f"{mp.objects[i]['tag']}#{i}({s:.2f})" for i, s in cands[:8]))

    # ---------- A) 랜덤 시작 N개 ----------
    rng = np.random.default_rng(a.seed)
    good = np.argwhere(pl.passable & (mp.costmap <= 90) & np.isin(pl.comp, big))   # 상위 연결요소의 낮은 코스트 셀
    starts = good[rng.choice(len(good), a.n, replace=False)]
    results = []; paths_w = []
    for k, (si, sj) in enumerate(starts):
        t0 = time.time(); sc = (int(si), int(sj)); sx, sy = pl.c2w(*sc); rec = {"start": [round(sx, 2), round(sy, 2)]}
        # 후보 객체를 유클리드 거리로 정렬, 같은 연결요소에 목표 셀이 있는 첫 객체로 계획
        order = sorted(cands, key=lambda t: math.hypot(mp.objects[t[0]]["center_w"][0] - sx, mp.objects[t[0]]["center_w"][1] - sy))
        done = False; reasons = []
        for oi, s in order[:6]:
            goal, why = pick_goal(mp, pl, sc, mp.objects[oi])
            if goal is None:
                reasons.append(f"#{oi}:{why}"); continue
            path, gcost = pl.astar(sc, goal["cell"])
            if path is None:
                reasons.append(f"#{oi}:astar fail"); continue
            L, comp, pts = pl.path_stats(path)
            rec.update({"ok": True, "object": {"idx": oi, "tag": mp.objects[oi]["tag"], "clip": round(s, 3), "center": [round(v, 2) for v in mp.objects[oi]["center_w"]]},
                        "goal": {"xy": [round(v, 2) for v in goal["xy"]], "yaw_deg": round(math.degrees(goal["yaw"]), 1), "ring_cells": goal["ring_cells"], "rejected_3d": goal["rejected_3d"]},
                        "path_m": round(L, 1), "straight_m": round(math.hypot(goal["xy"][0] - sx, goal["xy"][1] - sy), 1), "terrain_pct": comp,
                        "plan_ms": round((time.time() - t0) * 1000), "tried": reasons})
            paths_w.append((pts, goal, mp.objects[oi])); done = True; break
        if not done:
            rec.update({"ok": False, "reasons": reasons, "plan_ms": round((time.time() - t0) * 1000)})
        results.append(rec)
        print(f"  [{k:2d}] start({sx:7.1f},{sy:7.1f}) → " + (f"{rec['object']['tag']}#{rec['object']['idx']} goal({rec['goal']['xy'][0]:.1f},{rec['goal']['xy'][1]:.1f}) path {rec['path_m']}m (직선 {rec['straight_m']}m) 지형 {rec['terrain_pct']} {rec['plan_ms']}ms" if rec["ok"] else f"실패 {rec['reasons'][:3]}"))
    ok = [r for r in results if r["ok"]]
    summary = {"request": a.request, "target": target, "relation": rel, "n": a.n, "success": len(ok), "fill": pl.fill_stats,
               "path_m_mean": round(float(np.mean([r["path_m"] for r in ok])), 1) if ok else None,
               "detour_ratio_mean": round(float(np.mean([r["path_m"] / max(r["straight_m"], 0.1) for r in ok])), 2) if ok else None,
               "plan_ms_mean": round(float(np.mean([r["plan_ms"] for r in results]))), "clip_ms": round(t_clip * 1000),
               "terrain_pct_mean": {kk: round(float(np.mean([r["terrain_pct"][kk] for r in ok])), 1) for kk in ok[0]["terrain_pct"]} if ok else None,
               "goals_rejected_3d_total": int(sum(r["goal"]["rejected_3d"] for r in ok))}
    print("요약:", json.dumps(summary, ensure_ascii=False))

    # ---------- B) 지형 코스트맵 요청 ----------
    cm_out = {}
    for mode in ("grass_only", "avoid_grass", "grass_layer", "graded"):
        cost, x0, y0 = semantic_costmap(mp, mode); shp = export_nav2(cost, x0, y0, mp.cm_res, os.path.join(a.out, f"costmap_{mode}"))
        cm_out[mode] = {"shape": list(cost.shape), "pgm_shape": list(shp), "free_cells": int((cost < 253).sum()), "grass_cells": int((mp.costmap == 90).sum())}
    if ok:   # 첫 성공 목표 주변 20m×20m 크롭 (n×n 요청 예시)
        g = ok[0]["goal"]["xy"]; cost, x0, y0 = semantic_costmap(mp, "graded", center=g, size_m=20.0)
        export_nav2(cost, x0, y0, mp.cm_res, os.path.join(a.out, "costmap_crop20m_goal0")); cm_out["crop20m_goal0"] = {"center": g, "shape": list(cost.shape), "origin": [round(x0, 2), round(y0, 2)]}
    print("코스트맵 요청:", json.dumps(cm_out, ensure_ascii=False))

    # ---------- 출력: overview png + rerun ----------
    c = mp.costmap; ys_, xs_ = np.where(c < 255); i0, i1 = xs_.min(), xs_.max(); j0, j1 = ys_.min(), ys_.max()
    pal = {10: (170, 140, 110), 60: (150, 150, 150), 90: (95, 150, 95), 150: (120, 130, 150), 200: (220, 120, 40), 253: (200, 40, 40), 254: (200, 40, 40), 255: (25, 25, 25)}
    sub = c[i0:i1 + 1, j0:j1 + 1]; img = np.zeros(sub.shape + (3,), np.uint8)
    for k, v in pal.items(): img[sub == k] = v[::-1]
    S = 2; img = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
    def px(x, y): return (int(((x - pl.x0) / pl.res - i0) * S), int(((y - pl.y0) / pl.res - j0) * S))
    for pts, goal, o in paths_w:
        cv2.polylines(img, [np.array([px(*p) for p in pts], np.int32)], False, (255, 255, 0), 2)
        cv2.circle(img, px(*pts[0]), 6, (0, 255, 0), -1); cv2.circle(img, px(*goal["xy"]), 6, (255, 0, 255), -1)
    for r in results:
        if not r["ok"]: cv2.drawMarker(img, px(*r["start"]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    for i, s in cands:
        o = mp.objects[i]; cv2.circle(img, px(o["center_w"][0], o["center_w"][1]), 9, (0, 200, 255), 2)
        cv2.putText(img, f"{o['tag']}#{i}", px(o["center_w"][0] + 0.5, o["center_w"][1] + 0.5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    img = np.flipud(img.transpose(1, 0, 2))     # row=x,col=y → 화면: x 오른쪽, y 위
    cv2.imwrite(os.path.join(a.out, "overview.png"), img)

    import rerun as rr, rerun.blueprint as rrb
    rr.init("verify"); rr.save(os.path.join(a.out, "verify.rrd"))
    bp = rrb.Blueprint(rrb.Tabs(rrb.Spatial3DView(origin="/map", name="Paths on costmap", contents=["+ /map/costmap2d", "+ /map/objects/**", "+ /map/verify/**"]),
                                rrb.Spatial3DView(origin="/map", name="Paths + semantic", contents=["+ /map/**", "- /map/costmap2d", "- /map/voxels/unknown", "- /map/voxels/rgb", "- /map/voxels/other", "- /map/edges/**", "- /map/occupancy/**"])),
                       rrb.TimePanel(expanded=False)); rr.send_blueprint(bp, make_default=True)
    from package.load_map import to_rerun  # noqa — 같은 엔티티 레이아웃 재사용 대신 가벼운 costmap/객체만 직접 로깅
    zg = float(np.percentile(mp.voxel_centers()[:, 2], 5))
    ii, jj = np.where(c < 255); cx = pl.x0 + (ii + 0.5) * pl.res; cy = pl.y0 + (jj + 0.5) * pl.res
    cc = np.array([pal.get(int(v), (255, 255, 255)) for v in c[ii, jj]], np.uint8)
    rr.log("map/costmap2d", rr.Points3D(np.stack([cx, cy, np.full_like(cx, zg)], -1)[::2], colors=cc[::2], radii=pl.res * 0.5))
    for i, s in cands:
        o = mp.objects[i]; rr.log(f"map/objects/{o['tag'].replace(' ', '_')}_{i}", rr.Boxes3D(centers=[o["center_w"]], half_sizes=[np.array(o["extent"]) / 2], labels=[f"{o['tag']} ({s:.2f})"], colors=[[255, 200, 40]]))
    for k, (pts, goal, o) in enumerate(paths_w):
        z = goal["z_ground"] + 0.1
        rr.log(f"map/verify/path_{k}", rr.LineStrips3D([np.c_[pts, np.full(len(pts), z)]], colors=[[0, 255, 255]], radii=0.08))
        rr.log(f"map/verify/start_{k}", rr.Points3D([[pts[0][0], pts[0][1], z]], colors=[[0, 255, 0]], radii=0.3, labels=[f"start {k}"]))
        gx, gy = goal["xy"]; rr.log(f"map/verify/goal_{k}", rr.Arrows3D(origins=[[gx, gy, z]], vectors=[[0.8 * math.cos(goal["yaw"]), 0.8 * math.sin(goal["yaw"]), 0]], colors=[[255, 0, 255]], radii=0.06, labels=[f"goal {k}: next to {o['tag']}"]))
    json.dump({"summary": summary, "runs": results, "costmaps": cm_out, "candidates": [{"idx": i, "tag": mp.objects[i]["tag"], "clip": round(s, 3), "center": mp.objects[i]["center_w"]} for i, s in cands]},
              open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
