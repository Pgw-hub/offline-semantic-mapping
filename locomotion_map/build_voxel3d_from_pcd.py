#!/usr/bin/env python3
"""완성된 PCD(x y z [label]) → voxel3d 타일 사전맵 (docs/VOXEL3D_PRIOR.md, 'voxel3d_tiles v2' 와 같은 레코드).
build_voxel3d.py 는 LiDAR 스캔 묶음에서 레이캐스팅으로 만들지만, 이 도구는 이미 정합된 점군 하나를 셀에 모으기만 한다
(확률 격자·depth 채움 없음). 평탄 채움(src=3)은 같은 규칙으로 넣는다 — LiDAR 밀도가 2.5 cm 격자보다 성겨 바닥에 NaN 구멍이 생기므로.
좌표계 = PCD 그대로 (= lidar_slam 이 로드하는 지도 프레임). cls = PCD label 을 uint8 로 (254 이상이면 255).

사용: build_voxel3d_from_pcd.py <map.pcd> <out_dir> [--res-xy 0.025 --res-z 0.1 --tile-m 10 --fill-flat 0.04 --min-count 1 --holdout 0.02]
출력: <out_dir>/{index.json, t_<tx>_<ty>.bin} + <out_dir>_summary.json
"""
import argparse, glob, json, os, time
import numpy as np

OFF = 1 << 20; MASK = (1 << 21) - 1
REC = np.dtype([("ix", "<i4"), ("iy", "<i4"), ("iz", "<i2"), ("cls", "u1"), ("src", "u1"), ("mean_z", "<f4"), ("max_z", "<f4"), ("logodds", "<f2"), ("count", "<u2")])   # 24 B


def key3(I): return ((I[:, 0] + OFF) << 42) | ((I[:, 1] + OFF) << 21) | (I[:, 2] + OFF)
def unkey(k): return np.stack([(k >> 42) & MASK, (k >> 21) & MASK, k & MASK], -1).astype(np.int64) - OFF


def read_pcd(path):
    """binary PCD → (xyz float64 (N,3), label int64 (N,) 또는 None)"""
    with open(path, "rb") as f:
        fields, sizes, types, n, data = None, None, None, None, None
        while True:
            line = f.readline().decode(errors="replace").strip()
            if line.startswith("FIELDS"): fields = line.split()[1:]
            elif line.startswith("SIZE"): sizes = [int(v) for v in line.split()[1:]]
            elif line.startswith("TYPE"): types = line.split()[1:]
            elif line.startswith("POINTS"): n = int(line.split()[1])
            elif line.startswith("DATA"): data = line.split()[1]; break
        if data != "binary": raise SystemExit(f"DATA {data} 미지원 (binary 만)")
        dt = np.dtype([(fn, f"<{ {'F': 'f', 'U': 'u', 'I': 'i'}[t] }{s}") for fn, s, t in zip(fields, sizes, types)])
        P = np.frombuffer(f.read(n * dt.itemsize), dt, n)
    xyz = np.stack([P["x"], P["y"], P["z"]], -1).astype(np.float64)
    ok = np.isfinite(xyz).all(1)
    lab = P["label"].astype(np.int64)[ok] if "label" in fields else None
    return xyz[ok], lab


def mode_per_group(inv, n_groups, labels):
    """그룹별 최빈 라벨 (uint8). labels None 이면 255."""
    if labels is None: return np.full(n_groups, 255, np.uint8)
    lab = np.where(labels >= 254, 255, labels).astype(np.int64)
    pair = inv.astype(np.int64) * 256 + lab
    up, cnt = np.unique(pair, return_counts=True)
    g, l = up // 256, up % 256
    order = np.lexsort((-cnt, g)); g, l = g[order], l[order]          # 그룹 안에서 빈도 내림차순 → 첫 항목이 최빈
    first = np.unique(g, return_index=True)[1]
    out = np.full(n_groups, 255, np.uint8); out[g[first]] = l[first]
    return out


def build_cells(xyz, lab, rx, rz):
    I = np.stack([np.floor(xyz[:, 0] / rx), np.floor(xyz[:, 1] / rx), np.floor(xyz[:, 2] / rz)], -1).astype(np.int64)
    uk, inv = np.unique(key3(I), return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    mz = np.bincount(inv, weights=xyz[:, 2]) / cnt
    sq = np.bincount(inv, weights=xyz[:, 2] ** 2) / cnt
    sd = np.sqrt(np.maximum(sq - mz ** 2, 0.0))
    mx = np.full(len(uk), -np.inf); np.maximum.at(mx, inv, xyz[:, 2])
    cls = mode_per_group(inv, len(uk), lab)
    return unkey(uk), cnt, mz, sd, mx, cls


def flat_fill(I, mz, sd, mx, cls, lo, rx, rz, thr, thick):
    """같은 (0.1 m xy 발자국, iz) 그룹의 z 퍼짐이 thr 이하(국소 평면)면 4×4 발자국의 빈 세부 셀을 그룹 평균 z 로 채움 (src=3).
    구성원 셀이 전부 얇아야(셀 안 max_z-mean_z ≤ thick) 한다 — 벽·라이저 셀은 점이 층 높이(0.1 m)에 걸쳐 있어 평균이 같아도 평면이 아니다.
    이 조건이 없으면 벽을 따라 층마다 10 cm 선반이, 계단 라이저 앞에는 디딤판 위 가짜 셀이 생긴다 (GT 격자 PCD 에서 136만 셀 발생)."""
    nxy = int(round(0.1 / rx))
    g = np.stack([np.floor_divide(I[:, 0], nxy), np.floor_divide(I[:, 1], nxy), I[:, 2]], -1)
    ug, inv = np.unique(key3(g), return_inverse=True); gn = np.bincount(inv)
    gmin = np.full(len(ug), np.inf); np.minimum.at(gmin, inv, mz); gmax = np.full(len(ug), -np.inf); np.maximum.at(gmax, inv, mz)
    gthick = np.zeros(len(ug)); np.maximum.at(gthick, inv, mx - mz)
    gmean = np.bincount(inv, weights=mz) / gn; glo = np.bincount(inv, weights=lo) / gn; gcls = mode_per_group(inv, len(ug), cls.astype(np.int64))
    flat = (gn >= 2) & (gn < nxy * nxy) & ((gmax - gmin) <= thr) & (gthick <= thick)
    if not flat.any(): return None
    G = unkey(ug[flat]); cand = []
    for dx in range(nxy):
        for dy in range(nxy): cand.append(np.stack([G[:, 0] * nxy + dx, G[:, 1] * nxy + dy, G[:, 2]], -1))
    cand = np.concatenate(cand); gi = np.tile(np.flatnonzero(flat), nxy * nxy)
    have = np.isin(key3(cand), key3(I)); cand, gi = cand[~have], gi[~have]
    return cand, gmean[gi], (gmax - gmin)[gi], gmax[gi], gcls[gi], glo[gi]


def write_tiles(out_dir, I, cnt, mz, mx, cls, src, lo, params):
    tile_cells = params["tile_cells"]; tm = params["tile_m"]
    tx = np.floor_divide(I[:, 0], tile_cells); ty = np.floor_divide(I[:, 1], tile_cells)
    tkey = (tx + OFF) * (2 * OFF) + (ty + OFF); order = np.lexsort((I[:, 2], I[:, 1], I[:, 0], tkey))
    I, cnt, mz, mx, cls, src, lo, tkey = I[order], cnt[order], mz[order], mx[order], cls[order], src[order], lo[order], tkey[order]
    ut, tstart = np.unique(tkey, return_index=True)
    rec = np.zeros(len(I), REC); rec["ix"], rec["iy"], rec["iz"] = I[:, 0], I[:, 1], I[:, 2]
    rec["cls"] = cls; rec["src"] = src; rec["mean_z"] = mz; rec["max_z"] = mx; rec["logodds"] = lo; rec["count"] = np.minimum(cnt, 65535)
    os.makedirs(out_dir, exist_ok=True)
    for f_ in glob.glob(os.path.join(out_dir, "t_*.bin")): os.remove(f_)
    tiles = []
    for ti, (s, e) in enumerate(zip(tstart, np.r_[tstart[1:], len(tkey)])):
        txi, tyi = int(ut[ti] // (2 * OFF) - OFF), int(ut[ti] % (2 * OFF) - OFF); fn = f"t_{txi}_{tyi}.bin"
        rec[s:e].tofile(os.path.join(out_dir, fn))
        tiles.append({"tx": txi, "ty": tyi, "x0": txi * tm, "y0": tyi * tm, "start": int(s), "n": int(e - s), "z_min": round(float(mz[s:e].min()), 2), "z_max": round(float(mz[s:e].max()), 2), "file": fn, "bytes": int((e - s) * REC.itemsize)})
    json.dump({"format": "voxel3d_tiles v2", "record": [(n, str(REC.fields[n][0])) for n in REC.names], "record_bytes": REC.itemsize, "byte_order": "little",
               "index_to_world": "x=(ix+0.5)*res_xy, y=(iy+0.5)*res_xy, z=mean_z (voxel center z=(iz+0.5)*res_z); max_z = 셀 안 최대 z — 두 표면이 섞인 셀(계단 코)에서 보수적 높이로 사용 권장",
               "tile_of": "tx=floor(x/tile_m), ty=floor(y/tile_m)", **params, "n_cells": int(len(I)), "total_bytes": int(len(I) * REC.itemsize), "tiles": tiles},
              open(os.path.join(out_dir, "index.json"), "w"), indent=1, ensure_ascii=False)
    return tiles


def holdout_eval(I, mz, mx, rx, rz, Q, maxz_rule_thr):
    """로봇 getHeight 규칙(컬럼에서 max_z 아래 가장 높은 면)으로 hold-out 점 Q 를 질의. max_z = 점 z + res_z/2 (그 점이 속한 면이 보이게)."""
    col = (I[:, 0] + OFF) * (2 * OFF) + (I[:, 1] + OFF); z_mm = np.round((mz + 1000.0) * 1000).astype(np.int64)
    ZB = 1 << 24; k = col * ZB + z_mm; o = np.argsort(k); k, mz_s, mx_s, col_s = k[o], mz[o], mx[o], col[o]
    qc = (np.floor(Q[:, 0] / rx).astype(np.int64) + OFF) * (2 * OFF) + (np.floor(Q[:, 1] / rx).astype(np.int64) + OFF)
    qk = qc * ZB + np.round((Q[:, 2] + rz / 2 + 1000.0) * 1000).astype(np.int64)
    j = np.searchsorted(k, qk, "right") - 1
    ok = (j >= 0); ok[ok] &= col_s[j[ok]] == qc[ok]
    h = np.full(len(Q), np.nan); h[ok] = mz_s[j[ok]]
    rule = np.where((mx_s - mz_s) > maxz_rule_thr, mx_s, mz_s); hr = np.full(len(Q), np.nan); hr[ok] = rule[j[ok]]
    err = h[ok] - Q[ok, 2]; errr = hr[ok] - Q[ok, 2]
    return {"n_query": int(len(Q)), "nan_pct": round(float(100 * (~ok).mean()), 2), "abs_err_med_cm": round(float(np.median(np.abs(err))) * 100, 2), "abs_err_p90_cm": round(float(np.percentile(np.abs(err), 90)) * 100, 2),
            "low_err_3cm_pct": round(float(100 * (err < -0.03).mean()), 2), "maxz_rule_abs_err_med_cm": round(float(np.median(np.abs(errr))) * 100, 2), "maxz_rule_low_err_3cm_pct": round(float(100 * (errr < -0.03).mean()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcd"); ap.add_argument("out_dir")
    ap.add_argument("--res-xy", type=float, default=0.025); ap.add_argument("--res-z", type=float, default=0.1); ap.add_argument("--tile-m", type=float, default=10.0)
    ap.add_argument("--fill-flat", type=float, default=0.04, help="평탄 채움 z 퍼짐 임계 [m], 0 이면 끔")
    ap.add_argument("--fill-thickness", type=float, default=0.01, help="평탄 채움 구성원 셀의 최대 두께(max_z-mean_z) [m] — 벽/라이저 셀 제외")
    ap.add_argument("--min-count", type=int, default=1, help="이 미만 점 수의 셀은 버림")
    ap.add_argument("--holdout", type=float, default=0.02, help="검증용으로 빼 둘 점 비율 (0 이면 검증 안 함)")
    ap.add_argument("--maxz-rule-thr", type=float, default=0.03, help="검증 시 max_z 보수 규칙: max_z-mean_z 가 이보다 크면 max_z")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    xyz, lab = read_pcd(a.pcd); n_all = len(xyz)
    Q = None
    if a.holdout > 0:
        rng = np.random.default_rng(a.seed); m = rng.random(n_all) < a.holdout; Q = xyz[m]; xyz = xyz[~m]; lab = None if lab is None else lab[~m]
    I, cnt, mz, sd, mx, cls = build_cells(xyz, lab, a.res_xy, a.res_z)
    keep = cnt >= a.min_count; I, cnt, mz, sd, mx, cls = I[keep], cnt[keep], mz[keep], sd[keep], mx[keep], cls[keep]
    lo = np.clip(np.log1p(cnt), 0.0, 3.5)                         # 레이캐스팅이 없어 점 수 기반 의사 log-odds (build_voxel3d 의 depth 셀과 같은 취급)
    src = np.ones(len(I), np.uint8); n_lidar = len(I); n_fill = 0
    if a.fill_flat > 0:
        ff = flat_fill(I, mz, sd, mx, cls, lo, a.res_xy, a.res_z, a.fill_flat, a.fill_thickness)
        if ff is not None:
            cand, fz, fsd, fmx, fcls, flo = ff; n_fill = len(cand)
            I = np.concatenate([I, cand]); cnt = np.concatenate([cnt, np.zeros(n_fill)]); mz = np.concatenate([mz, fz]); sd = np.concatenate([sd, fsd]); mx = np.concatenate([mx, fmx])
            cls = np.concatenate([cls, fcls]); lo = np.concatenate([lo, flo]); src = np.concatenate([src, np.full(n_fill, 3, np.uint8)])
    labels = sorted(int(v) for v in np.unique(cls))
    params = {"frame": f"lidar_slam loaded map frame — PCD {os.path.basename(a.pcd)} as-is", "source_pcd": os.path.abspath(a.pcd), "res_xy": a.res_xy, "res_z": a.res_z, "tile_m": a.tile_m, "tile_cells": int(round(a.tile_m / a.res_xy)),
              "n_points": int(n_all), "holdout": a.holdout, "min_count": a.min_count, "fill_flat": a.fill_flat, "fill_thickness": a.fill_thickness, "classes": [], "label_ids": labels,
              "class_codes": {"255": "unlabeled / label>=254", "other": "cls = PCD label id"}, "source_codes": {"1": "pcd points", "3": "flat fill (interpolated within 0.1 m cell)"}}
    tiles = write_tiles(a.out_dir, I, cnt, mz, mx, cls, src, lo, params)
    summ = {"points": int(n_all), "cells": int(len(I)), "cells_pcd": int(n_lidar), "cells_flat_fill": int(n_fill), "total_mb": round(len(I) * REC.itemsize / 1e6, 1), "tiles": len(tiles),
            "tile_mb_max": round(max(t["n"] for t in tiles) * REC.itemsize / 1e6, 1), "std_z_med_cm": round(float(np.median(sd[:n_lidar])) * 100, 2), "z_range": [round(float(mz.min()), 2), round(float(mz.max()), 2)],
            "cells_two_surfaces_pct": round(float(100 * ((mx - mz) > a.maxz_rule_thr).mean()), 2), "by_label": {str(l): int(n) for l, n in zip(*np.unique(cls, return_counts=True))}}
    if Q is not None: summ["holdout"] = holdout_eval(I, mz, mx, a.res_xy, a.res_z, Q, a.maxz_rule_thr)
    summ["seconds"] = round(time.time() - t0, 1)
    json.dump({**summ, "params": params}, open(a.out_dir.rstrip("/") + "_summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"점 {n_all:,} → 세부 셀 {summ['cells']:,} (PCD {n_lidar:,} + 평탄채움 {n_fill:,}) · {summ['total_mb']} MB · 타일 {len(tiles)}개 (최대 {summ['tile_mb_max']} MB) · z {summ['z_range']}")
    print(f"셀 z 표준편차 중앙 {summ['std_z_med_cm']} cm · 두 표면 섞임(max-mean>{a.maxz_rule_thr*100:.0f}cm) {summ['cells_two_surfaces_pct']}% · 라벨 {summ['by_label']}")
    if Q is not None:
        h = summ["holdout"]; print(f"hold-out {h['n_query']:,}점: NaN {h['nan_pct']}% · |오차| 중앙 {h['abs_err_med_cm']} cm · p90 {h['abs_err_p90_cm']} cm · 3cm 이상 낮게 봄 {h['low_err_3cm_pct']}% → max_z 규칙 {h['maxz_rule_low_err_3cm_pct']}%")


if __name__ == "__main__":
    main()
