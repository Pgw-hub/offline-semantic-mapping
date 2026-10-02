#!/usr/bin/env python3
"""voxel3d 사전맵 hold-out 평가 — 빌드에서 뺀 스캔(i % every == phase)을 '로봇이 나중에 볼 관측' 으로 보고 두 가지를 잰다.
 1) 높이 질의 정확도: 로봇 VoxelMap::getHeight(x, y, max_z) 와 같은 규칙(그 xy 컬럼에서 max_z 아래 가장 높은 점유 셀의 평균 z)으로 사전맵에 질의.
    held-out 점 p 마다 h = getHeight(p.x, p.y, p.z + 0.15) → 오차 h − p.z. 미응답(NaN) 비율 = 사전맵에 그 표면이 없음.
    부분집합: 전체(원점 ≤ 5 m) / 지형(LiDAR 원점보다 낮은 점, ≤ 4 m) / 윗면(컬럼 최고점) / 윗면 중 사전맵 클래스가 stairs 인 컬럼.
    max_z 규칙: std_z > std_thr 인 셀은 max_z 로 답한 경우도 함께 평가 (계단 코에서 '낮게 보는' 오차가 줄어드는지). '3cm 이상 낮게 본 비율' 이 위험한 방향의 오차.
 2) 유령 비율: 사전맵 점유 셀의 부모(확률 격자) 셀을 held-out 스캔이 '통과(miss ≥ 3, hit 0)' 한 비율 — 사전맵에 있지만 실제로는 빈 셀. 확률 판정 on/off 비교용.
사용: python tools/voxel3d_eval.py <prior.npz> --scans dirA [dirB] [--every 10 --phase 0] [--label "prob"] [--out eval.json]   (여러 --prior label=path 로 비교)
"""
import argparse, glob, json, os, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--prior", action="append", required=True, help="label=path.npz"); ap.add_argument("--scans", nargs="+", required=True)
ap.add_argument("--every", type=int, default=10); ap.add_argument("--phase", type=int, default=0); ap.add_argument("--out"); ap.add_argument("--max-pts", type=int, default=3_000_000)
ap.add_argument("--std-thr", type=float, default=0.02, help="max_z 보수 규칙 적용 임계: 셀 z 표준편차(m)")
a = ap.parse_args()
files = []
for d in a.scans: files += sorted(glob.glob(os.path.join(d, "*.npz")))
ho = [f for i, f in enumerate(files) if i % a.every == a.phase]
P, O, Z0 = [], [], []
for f in ho:
    d = np.load(f); x = d["xyz"].astype(np.float64); o = d["origin"].astype(np.float64); r = np.linalg.norm(x - o, axis=1); k = (r > 0.05) & (r <= 5.0)
    P.append(x[k]); O.append(np.repeat(o[None], int(k.sum()), 0))
P = np.concatenate(P); O = np.concatenate(O)
if len(P) > a.max_pts: sel = np.random.default_rng(0).choice(len(P), a.max_pts, replace=False); P, O = P[sel], O[sel]
rng = np.linalg.norm(P - O, axis=1); terrain = (P[:, 2] < O[:, 2]) & (rng <= 4.0)
# 윗면(top-surface) 부분집합: held-out 점들 중 자기 5 cm xy 컬럼에서 가장 높은 점이고 LiDAR 원점보다 낮은 것 (바닥·계단·낮은 물체 윗면). 위에 아무것도 없어야 하므로 getHeight 비교가 공정.
ck_ = np.floor(P[:, 0] / 0.05).astype(np.int64) * 4_000_000 + np.floor(P[:, 1] / 0.05).astype(np.int64); uc, inv = np.unique(ck_, return_inverse=True); zmax = np.full(len(uc), -np.inf); np.maximum.at(zmax, inv, P[:, 2])
top = (P[:, 2] >= zmax[inv] - 1e-9) & terrain
print(f"held-out 스캔 {len(ho)} · 점 {len(P):,} (≤5 m) · 지형 {int(terrain.sum()):,} · 윗면 {int(top.sum()):,}")
JIT = np.random.default_rng(1).uniform(-0.05, 0.05, (len(P), 2))                       # 임의 발 위치 흉내: xy ±5 cm 지터
OFF = 1 << 20
def key3(I): I = I.astype(np.int64) + OFF; return (I[:, 0] << 42) | (I[:, 1] << 21) | I[:, 2]
res = {}
for spec in a.prior:
    label, path = spec.split("=", 1); v = np.load(path); RX, RZ, RP = float(v["res_xy"]), float(v["res_z"]), float(v["prob_res"])
    I = v["ijk"].astype(np.int64); mz = v["mean_z"].astype(np.float64)
    # max_z 보수 규칙: z 표준편차가 std_thr 를 넘는 셀(두 표면이 섞인 계단 코 등)은 평균 대신 최대 z 를 쓴다 — 두 버전을 모두 평가
    mz_rule = mz.copy()
    if "max_z" in v.files: sd_ = v["std_z"].astype(np.float64); mz_rule = np.where(sd_ > a.std_thr, v["max_z"].astype(np.float64), mz)
    cls_all = v["cls"].astype(np.int64); classes_ = [str(c) for c in v["classes"]]; stairs_id = classes_.index("stairs") if "stairs" in classes_ else -1
    # getHeight 벡터화: (컬럼키, z[mm]) 결합키 정렬 → 질의키(컬럼, max_z) 바로 아래 원소가 그 컬럼이면 응답
    col = (I[:, 0] + OFF) * (2 * OFF) + (I[:, 1] + OFF); ZB = 1 << 24; zq = np.round((mz + 1000.0) * 1000).astype(np.int64)
    comb = col * ZB + zq; order = np.argsort(comb); comb_s, col_s, mz_s = comb[order], col[order], mz[order]
    mzr_s = mz_rule[order]; cls_s = cls_all[order]
    def get_height(x, y, max_z, rule=False, want_cls=False):
        qc = (np.floor(x / RX).astype(np.int64) + OFF) * (2 * OFF) + (np.floor(y / RX).astype(np.int64) + OFF); qk = qc * ZB + np.round((max_z + 1000.0) * 1000).astype(np.int64)
        idx = np.searchsorted(comb_s, qk, side="left") - 1; ok = (idx >= 0); idx = np.maximum(idx, 0); ok &= col_s[idx] == qc
        hh = np.where(ok, (mzr_s if rule else mz_s)[idx], np.nan)
        return (hh, np.where(ok, cls_s[idx], -1)) if want_cls else hh
    h, hcls = get_height(P[:, 0], P[:, 1], P[:, 2] + 0.15, want_cls=True); e = h - P[:, 2]
    hr = get_height(P[:, 0], P[:, 1], P[:, 2] + 0.15, rule=True); er = hr - P[:, 2]
    on_stairs = (hcls == stairs_id) if stairs_id >= 0 else np.zeros(len(P), bool)
    m = {"cells": int(len(I)), "mb": round(len(I) * 24 / 1e6, 1)}
    hj = get_height(P[:, 0] + JIT[:, 0], P[:, 1] + JIT[:, 1], P[:, 2] + 0.15); ej = hj - P[:, 2]
    for name, k in (("all", np.ones(len(P), bool)), ("terrain", terrain), ("top_surface", top), ("top_surface_stairs", top & on_stairs)):
        if k.sum() == 0: m[name] = {"n": 0}; continue
        ee = e[k]; nan = np.isnan(ee); ae = np.abs(ee[~nan]); nj = np.isnan(ej[k]); aj = np.abs(ej[k][~nj]); rr_ = er[k]; ar = np.abs(rr_[~np.isnan(rr_)])
        m[name] = {"n": int(k.sum()), "no_answer_pct": round(100 * nan.mean(), 2), "abs_err_med_cm": round(float(np.median(ae)) * 100, 2), "abs_err_p90_cm": round(float(np.percentile(ae, 90)) * 100, 2),
                   "within_5cm_pct": round(100 * float((ae < 0.05).mean()), 1), "within_10cm_pct": round(100 * float((ae < 0.10).mean()), 1), "bias_cm": round(float(np.median(ee[~nan])) * 100, 2),
                   "jitter5cm_no_answer_pct": round(100 * nj.mean(), 2), "jitter5cm_abs_err_med_cm": round(float(np.median(aj)) * 100, 2) if len(aj) else None,
                   "maxz_rule_abs_err_med_cm": round(float(np.median(ar)) * 100, 2) if len(ar) else None, "maxz_rule_p90_cm": round(float(np.percentile(ar, 90)) * 100, 2) if len(ar) else None,
                   "maxz_rule_bias_cm": round(float(np.median(rr_[~np.isnan(rr_)])) * 100, 2) if len(ar) else None, "low_err_pct": round(100 * float((ee[~nan] < -0.03).mean()), 1), "maxz_rule_low_err_pct": round(100 * float((rr_[~np.isnan(rr_)] < -0.03).mean()), 1) if len(ar) else None}
    # 유령 비율: held-out 스캔 레이캐스팅(확률 격자) → 부모 셀이 miss≥3 & hit 0 인 사전맵 셀
    steps = (np.arange(int(np.ceil(6.0 / (RP * 0.5)))) + 0.5) * RP * 0.5; hk_all, mk_all = [], []
    for f in ho:
        d = np.load(f); x = d["xyz"].astype(np.float64); o = d["origin"].astype(np.float64); vec = x - o; r = np.linalg.norm(vec, axis=1); k = (r > 0.05) & (r <= 30); x, vec, r = x[k], vec[k], r[k]
        hk = np.unique(key3(np.floor(x / RP))); L = np.minimum(r, 6.0); S = o[None, None, :] + (vec / r[:, None])[:, None, :] * steps[None, :, None]; valid = steps[None, :] < L[:, None]
        mk = np.unique(key3(np.floor(S.reshape(-1, 3) / RP))[valid.ravel()]); mk = mk[~np.isin(mk, hk, assume_unique=True)]; hk_all.append(hk); mk_all.append(mk)
    hk = np.concatenate(hk_all); mk = np.concatenate(mk_all); uh, ch = np.unique(hk, return_counts=True); um, cm = np.unique(mk, return_counts=True)
    cen = np.stack([(I[:, 0] + 0.5) * RX, (I[:, 1] + 0.5) * RX, (I[:, 2] + 0.5) * RZ], -1); pk = key3(np.floor(cen / RP))
    ph = np.zeros(len(pk)); p = np.minimum(np.searchsorted(uh, pk), len(uh) - 1); f_ = uh[p] == pk; ph[f_] = ch[p[f_]]
    pm = np.zeros(len(pk)); p = np.minimum(np.searchsorted(um, pk), len(um) - 1); f_ = um[p] == pk; pm[f_] = cm[p[f_]]
    seen = (ph > 0) | (pm > 0); ghost = seen & (ph == 0) & (pm >= 3); confirmed = ph > 0
    m["ghost"] = {"cells_seen_by_holdout": int(seen.sum()), "ghost_pct": round(100 * float(ghost.sum() / max(seen.sum(), 1)), 2), "confirmed_pct": round(100 * float(confirmed.sum() / max(seen.sum(), 1)), 2)}
    if "source" in v.files:
        for s_, nm_ in ((1, "lidar"), (2, "depth")):
            k = v["source"] == s_
            if k.any(): m["ghost"][f"{nm_}_ghost_pct"] = round(100 * float((ghost & k).sum() / max((seen & k).sum(), 1)), 2)
    res[label] = m
    print(f"\n[{label}] 셀 {m['cells']:,} ({m['mb']} MB)")
    for name in ("all", "terrain", "top_surface", "top_surface_stairs"):
        q = m[name]
        if not q.get("n"): print(f"  getHeight {name:11s}: 표본 없음"); continue
        print(f"  getHeight {name:11s}: 미응답 {q['no_answer_pct']}% · |오차| 중앙 {q['abs_err_med_cm']} cm · p90 {q['abs_err_p90_cm']} · 5/10 cm 안 {q['within_5cm_pct']}/{q['within_10cm_pct']} % · 편향 {q['bias_cm']} cm · xy±5cm 지터 시 미응답 {q['jitter5cm_no_answer_pct']}% |오차| {q['jitter5cm_abs_err_med_cm']} cm")
        print(f"     ↳ max_z 규칙(std>{a.std_thr*100:.0f}cm): |오차| 중앙 {q['maxz_rule_abs_err_med_cm']} cm · p90 {q['maxz_rule_p90_cm']} · 편향 {q['maxz_rule_bias_cm']} cm · 3cm 이상 낮게 본 비율 {q['low_err_pct']}% → {q['maxz_rule_low_err_pct']}%")
    print(f"  유령: held-out 이 본 셀 {m['ghost']['cells_seen_by_holdout']:,} 중 통과(빈 셀) {m['ghost']['ghost_pct']}% · 재확인 {m['ghost']['confirmed_pct']}%" + (f" · LiDAR {m['ghost'].get('lidar_ghost_pct')}% / depth {m['ghost'].get('depth_ghost_pct')}%" if "depth_ghost_pct" in m["ghost"] else ""))
if a.out: json.dump({"holdout": {"every": a.every, "phase": a.phase, "scans": len(ho), "points": int(len(P))}, "priors": res}, open(a.out, "w"), indent=1, ensure_ascii=False); print("저장:", a.out)
