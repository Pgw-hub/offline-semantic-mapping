#!/usr/bin/env python3
"""보행용 3D 확률 복셀 사전맵 (voxel3d) — OctoMap 식 log-odds 점유 + voxblox 식 복셀 내 평균 z + 클래스, 로봇 VoxelMap 격자(xy 0.025 / z 0.1 m) 호환.

입력: 키프레임 시각 /cloud_registered 스캔(export_registered_scans.py → lidar_scans/*.npz: xyz, origin, stamp_ns; 최종 PGO 지도 프레임)
      + 세션 semantic 복셀 session_voxsem.npz (0.1 m 클래스, depth_xyz = depth 채움 점)
두 격자:
  · 확률 격자(prob_res, 정육면체, 기본 0.1 m): 스캔마다 원점→끝점 레이캐스팅. 끝점 셀 hit, 통과 셀 miss (스캔당 셀 1회, OctoMap insertPointCloud 와 동일).
    log-odds 를 배치(batch 스캔≈4 s)마다 순차 갱신·클램프 [-2, 3.5] (OctoMap 근사: 최근 상태가 이긴다). 점유 = log-odds > 0.  → 한두 번 스친 사람·튀는 점은 miss 가 쌓여 지워진다.
  · 세부 격자(res_xy × res_xy × res_z, 기본 0.025×0.025×0.1 = 로봇 voxel_mapping 격자): hit 점마다 count / mean z / z 표준편차 / 관측 스캔 수.
    세부 셀의 점유 판정은 부모 확률 셀의 log-odds 로 한다 (얇은 표면이 세부 격자에서 miss 에 침식되는 것을 막고, 노이즈 제거는 확률 격자가 담당).
  depth 채움 점(source 2): LiDAR 가 hit 한 적 없고 miss 도 depth_max_miss 이하인 부모 셀에만 넣는다 ("빈 곳만 채우기"). log-odds 는 관측 수 기반 의사값.
  클래스: 세부 셀 중심이 속한 0.1 m voxsem 복셀의 argmax (확신 < 0.3 → other=254, 라벨 없음 → 255). 동적 클래스(person) 셀은 기본 제거.
출력:
  <out>.npz           ijk(int32, 세부 격자 인덱스) · mean_z(f32, 절대 z) · max_z(f32, 셀 안 최대 z; 계단 코 보수 규칙용) · std_z · count · n_scans · logodds(f16, 부모) · cls(u8) · source(u8: 1 LiDAR 2 depth 3 평탄채움) · tile(int32)
                      coarse_ijk / coarse_logodds / coarse_hits / coarse_misses (확률 격자 전체: free 공간 포함) · removed_* (제거된 셀, 검수용) · params(json)
  <out>_tiles/        index.json + t_<tx>_<ty>.bin — tile_m(기본 10 m) 타일별 struct 배열(24 B/셀). 로봇은 자기 창(10 m)을 덮는 3×3 타일만 RAM 에 올리면 된다.
사용: python build_voxel3d.py --scans dirA [dirB] --voxsem session_voxsem.npz --out compiled/session_voxel3d [--res-xy 0.025 --res-z 0.1 --prob-res 0.1]
      [--free-range 8] [--skip-every N --skip-phase k]  ← 평가용 hold-out (i % N == k 스캔 제외)   [--no-prob] ← 확률 판정 끄기(모든 hit 유지; 비교용)
"""
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import classdict as CD

ap = argparse.ArgumentParser()
ap.add_argument("--scans", nargs="+", required=True); ap.add_argument("--voxsem", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--res-xy", type=float, default=0.025); ap.add_argument("--res-z", type=float, default=0.025); ap.add_argument("--prob-res", type=float, default=0.1)
ap.add_argument("--max-range", type=float, default=30.0, help="hit 으로 쓸 최대 거리"); ap.add_argument("--free-range", type=float, default=8.0, help="free 레이 최대 길이")
ap.add_argument("--prob-hit", type=float, default=0.85); ap.add_argument("--prob-miss", type=float, default=0.45)
ap.add_argument("--min-scans", type=int, default=1, help="세부 셀 최소 관측 스캔 수")
ap.add_argument("--no-prob", action="store_true", help="확률 판정 끄기 (hit 이 있는 셀 전부 점유; 비교용)")
ap.add_argument("--no-depth", action="store_true"); ap.add_argument("--depth-max-miss", type=int, default=3, help="depth 채움 허용: 부모 셀 LiDAR miss 수 상한")
ap.add_argument("--keep-dynamic", action="store_true", help="동적 클래스(person) 셀도 유지")
ap.add_argument("--fill-flat", type=float, default=0.0, help="평탄 채움: 확률 셀(0.1 m) 안 LiDAR 세부 셀들의 z 퍼짐이 이 값(m) 이하면 빈 xy 세부 셀을 평균 z 로 채움(source 3). 0 이면 끄기 — 바닥의 2.5 cm 컬럼 빈칸(getHeight NaN) 방지")
ap.add_argument("--tile-m", type=float, default=10.0); ap.add_argument("--no-tiles", action="store_true")
ap.add_argument("--skip-every", type=int, default=0); ap.add_argument("--skip-phase", type=int, default=0)
ap.add_argument("--batch", type=int, default=40)
a = ap.parse_args()

RX, RZ, RP = a.res_xy, a.res_z, a.prob_res; OFF = 1 << 20; MASK = (1 << 21) - 1
def key3(I):
    I = I.astype(np.int64) + OFF; return (I[:, 0] << 42) | (I[:, 1] << 21) | I[:, 2]
def unkey(k): return np.stack([(k >> 42) & MASK, (k >> 21) & MASK, k & MASK], -1).astype(np.int64) - OFF
def fine_idx(P): return np.stack([np.floor(P[:, 0] / RX), np.floor(P[:, 1] / RX), np.floor(P[:, 2] / RZ)], -1).astype(np.int64)
def coarse_key(P): return key3(np.floor(P / RP))
def fine_center(I): return np.stack([(I[:, 0] + 0.5) * RX, (I[:, 1] + 0.5) * RX, (I[:, 2] + 0.5) * RZ], -1)

def reduce_sum(keys, cols):
    """keys(N) 와 열들 → 고유 키별 합 (bincount; add.at 보다 훨씬 빠름)"""
    uk, inv = np.unique(keys, return_inverse=True)
    return uk, [np.bincount(inv, weights=c, minlength=len(uk)) for c in cols]
def merge(acc, uk, cols, mx=None):
    """cols 는 합, mx 는 최대값(있으면) 으로 병합"""
    if acc is None: return (uk, cols) if mx is None else (uk, cols, mx)
    K = np.concatenate([acc[0], uk]); u2, inv = np.unique(K, return_inverse=True)
    c2 = [np.bincount(inv, weights=np.concatenate([x, y]), minlength=len(u2)) for x, y in zip(acc[1], cols)]
    if mx is None: return u2, c2
    m2 = np.full(len(u2), -np.inf); np.maximum.at(m2, inv, np.concatenate([acc[2], mx])); return u2, c2, m2
def merge_prob(acc, uk, h, m, lh, lm, lo_min=-2.0, lo_max=3.5):
    """확률 격자 누적: 총 hits/misses 합 + log-odds 는 배치(≈batch 스캔)마다 순차 클램프 갱신 — OctoMap 의 스캔별 클램프 갱신 근사.
    (합산 후 한 번 클램프하면 오래 free 였다가 나중에 생긴 물체가 영원히 free 로 남는다; 순차 클램프는 '최근 상태가 이긴다')"""
    if acc is None: return uk, h, m, np.clip(h * lh + m * lm, lo_min, lo_max)
    K = np.concatenate([acc[0], uk]); u2, inv = np.unique(K, return_inverse=True); n0 = len(acc[0])
    H = np.bincount(inv, weights=np.concatenate([acc[1], h]), minlength=len(u2)); M = np.bincount(inv, weights=np.concatenate([acc[2], m]), minlength=len(u2))
    lo = np.zeros(len(u2)); lo[inv[:n0]] = acc[3]; lo += np.bincount(inv[n0:], weights=h * lh + m * lm, minlength=len(u2))
    return u2, H, M, np.clip(lo, lo_min, lo_max)

files = []
for d in a.scans: files += sorted(glob.glob(os.path.join(d, "*.npz")))
if a.skip_every > 1: files = [f for i, f in enumerate(files) if i % a.skip_every != a.skip_phase]
print(f"스캔 {len(files)} · 세부 {RX}×{RX}×{RZ} m · 확률 {RP} m · hit {a.prob_hit} miss {a.prob_miss} · free ≤ {a.free_range} m")
lh = np.log(a.prob_hit / (1 - a.prob_hit)); lm = np.log(a.prob_miss / (1 - a.prob_miss))
t0 = time.time(); acc_c = acc_f = None; bc, bf = [], []; n_pts_tot = 0
steps = (np.arange(int(np.ceil(a.free_range / (RP * 0.5)))) + 0.5) * RP * 0.5          # 레이 표본 간격 0.5·prob_res (대각선 셀 누락 방지)
for si, f in enumerate(files):
    d = np.load(f); P = d["xyz"].astype(np.float64); org = d["origin"].astype(np.float64)
    vec = P - org; rng = np.linalg.norm(vec, axis=1); ok = (rng > 0.05) & (rng <= a.max_range); P, vec, rng = P[ok], vec[ok], rng[ok]; n_pts_tot += len(P)
    # 확률 격자: hit 셀(스캔당 1회) + 통과 셀(끝점 셀 제외)
    hk = np.unique(coarse_key(P)); L = np.minimum(rng, a.free_range); dirs = vec / rng[:, None]
    valid = steps[None, :] < L[:, None]; S = org[None, None, :] + dirs[:, None, :] * steps[None, :, None]
    mk = np.unique(coarse_key(S.reshape(-1, 3))[valid.ravel()]); mk = mk[~np.isin(mk, hk, assume_unique=True)]
    bc.append((np.concatenate([hk, mk]), np.r_[np.ones(len(hk)), np.zeros(len(mk))], np.r_[np.zeros(len(hk)), np.ones(len(mk))]))
    # 세부 격자: count / Σz / Σz² / 스캔 수(스캔당 1)
    fi = fine_idx(P); fk = key3(fi); uk, inv_ = np.unique(fk, return_inverse=True); cnt, sz, sz2 = [np.bincount(inv_, weights=c, minlength=len(uk)) for c in (np.ones(len(fk)), P[:, 2], P[:, 2] ** 2)]
    mxz = np.full(len(uk), -np.inf); np.maximum.at(mxz, inv_, P[:, 2])                       # 셀 안 최대 z (계단 코처럼 두 표면이 섞인 셀에서 보수적 높이)
    bf.append((uk, cnt, sz, sz2, np.ones(len(uk)), mxz))
    if (si + 1) % a.batch == 0 or si == len(files) - 1:
        K = np.concatenate([b[0] for b in bc]); u, c = reduce_sum(K, [np.concatenate([b[1] for b in bc]), np.concatenate([b[2] for b in bc])]); acc_c = merge_prob(acc_c, u, c[0], c[1], lh, lm)
        K = np.concatenate([b[0] for b in bf]); u, inv_ = np.unique(K, return_inverse=True); c = [np.bincount(inv_, weights=np.concatenate([b[j] for b in bf]), minlength=len(u)) for j in range(1, 5)]
        mx = np.full(len(u), -np.inf); np.maximum.at(mx, inv_, np.concatenate([b[5] for b in bf])); acc_f = merge(acc_f, u, c, mx)
        bc, bf = [], []
        if (si + 1) % (a.batch * 10) == 0 or si == len(files) - 1: print(f"  {si+1}/{len(files)} · 확률 셀 {len(acc_c[0]):,} · 세부 셀 {len(acc_f[0]):,} · {time.time()-t0:.0f}s", flush=True)

ck, c_hits, c_miss, c_lo = acc_c; fk, (f_cnt, f_sz, f_sz2, f_ns), f_mx = acc_f; c_lo = c_lo.astype(np.float32)
# 세부 셀 → 부모 확률 셀 log-odds
fI = unkey(fk); fc = fine_center(fI); pk = coarse_key(fc); pos = np.searchsorted(ck, pk); pos = np.minimum(pos, len(ck) - 1); assert (ck[pos] == pk).all(), "부모 셀 누락"
f_lo = c_lo[pos]; f_miss = c_miss[pos]
mean_z = f_sz / f_cnt; std_z = np.sqrt(np.maximum(f_sz2 / f_cnt - mean_z ** 2, 0))
occ = (f_lo > 0) if not a.no_prob else np.ones(len(fk), bool); occ &= f_ns >= a.min_scans
n_lidar_all = len(fk); n_prob_removed = int((~occ).sum())

# 클래스 (voxsem 0.1 m)
v = np.load(a.voxsem, allow_pickle=True); V = float(v["vox"]); classes = [str(c) for c in v["classes"]]; NC = len(classes); cd = CD.from_voxsem(v)
sem = v["sem_mean"].astype(np.float32); has = v["n_sem"] > 0; lab = np.where(has, sem.argmax(1), 255); lab = np.where(has & (sem.max(1) < 0.3), 254, lab).astype(np.uint8)
vk = key3(v["ijk"]); order = np.argsort(vk); vk, lab_s = vk[order], lab[order]
def class_of(Pc):
    k = key3(np.floor(Pc / V)); p = np.minimum(np.searchsorted(vk, k), len(vk) - 1); f = vk[p] == k; out = np.full(len(k), 255, np.uint8); out[f] = lab_s[p[f]]; return out
f_cls = class_of(fc)
dyn_ids = CD.ids_with_role(cd, "dynamic"); is_dyn = np.isin(f_cls, dyn_ids)
n_dyn_removed = int((occ & is_dyn).sum()) if not a.keep_dynamic else 0
if not a.keep_dynamic: occ &= ~is_dyn
removed_ijk = fI[~occ].astype(np.int32); removed_why = np.where(is_dyn[~occ], 2, 1).astype(np.uint8)   # 1 확률(free) 2 동적 클래스

# depth 채움 점
dep_n = dep_added = 0; D = None
if not a.no_depth and "depth_xyz" in v.files and len(v["depth_xyz"]):
    Pd = v["depth_xyz"].astype(np.float64); dep_n = len(Pd)
    dk = coarse_key(Pd); p = np.minimum(np.searchsorted(ck, dk), len(ck) - 1); found = ck[p] == dk
    lidar_hit = found & (c_hits[p] > 0); too_free = found & (c_miss[p] > a.depth_max_miss)
    keep = ~lidar_hit & ~too_free; Pd = Pd[keep]
    if len(Pd):
        dI = fine_idx(Pd); dkf = key3(dI); uk, (dc, dz, dz2) = reduce_sum(dkf, [np.ones(len(dkf)), Pd[:, 2], Pd[:, 2] ** 2])
        dI = unkey(uk); dmz = dz / dc; dsd = np.sqrt(np.maximum(dz2 / dc - dmz ** 2, 0)); dcen = fine_center(dI); dcls = class_of(dcen)
        ddyn = np.isin(dcls, dyn_ids) & (not a.keep_dynamic); dI, dc, dmz, dsd, dcls = dI[~ddyn], dc[~ddyn], dmz[~ddyn], dsd[~ddyn], dcls[~ddyn]
        dlo = np.clip(dc * np.log(0.7 / 0.3), 0, 3.5).astype(np.float32)          # 의사 log-odds: depth 관측 수 (hit 0.7)
        dmx = np.full(len(uk), -np.inf); np.maximum.at(dmx, np.unique(dkf, return_inverse=True)[1], Pd[:, 2]); dmx = dmx[~ddyn]
        D = (dI, dc, dmz, dsd, np.ones(len(dI)), dlo, dcls, dmx); dep_added = len(dI)

# 최종 배열
I = fI[occ]; cnt = f_cnt[occ]; mz = mean_z[occ]; sd = std_z[occ]; ns = f_ns[occ]; lo = f_lo[occ]; cls = f_cls[occ]; src = np.ones(len(I), np.uint8); mxz_all = f_mx[occ]
# 평탄 채움: 같은 (확률 셀 xy, iz) 그룹의 LiDAR 세부 셀 z 퍼짐이 작으면(국소 평면) 4×4 xy 발자국의 빈 세부 셀을 그룹 평균 z 로 채움 (source 3).
#   LiDAR 점 밀도가 2.5 cm 격자보다 성겨서 생기는 바닥 빈칸을 메운다. 벽·모서리(z 퍼짐 큼)는 채우지 않는다.
n_fill = 0
if a.fill_flat > 0:
    nxy = int(round(RP / RX)); g = np.stack([np.floor_divide(I[:, 0], nxy), np.floor_divide(I[:, 1], nxy), I[:, 2]], -1); gk = key3(g)
    ug, inv = np.unique(gk, return_inverse=True); gn = np.bincount(inv); gmin = np.full(len(ug), np.inf); np.minimum.at(gmin, inv, mz); gmax = np.full(len(ug), -np.inf); np.maximum.at(gmax, inv, mz)
    gmean = np.bincount(inv, weights=mz) / gn; glo = np.bincount(inv, weights=lo) / gn
    flat = (gn >= 2) & (gn < nxy * nxy) & ((gmax - gmin) <= a.fill_flat)
    if flat.any():
        G = unkey(ug[flat]); cand = []                                                          # 후보: 발자국 nxy×nxy 전체
        for dx in range(nxy):
            for dy in range(nxy): cand.append(np.stack([G[:, 0] * nxy + dx, G[:, 1] * nxy + dy, G[:, 2]], -1))
        cand = np.concatenate(cand); gi = np.tile(np.flatnonzero(flat), nxy * nxy)
        have = np.isin(key3(cand), key3(I)); cand, gi = cand[~have], gi[~have]
        fz = gmean[gi]; fcls = class_of(fine_center(cand)); fdyn = np.isin(fcls, dyn_ids) & (not a.keep_dynamic); cand, gi, fz, fcls = cand[~fdyn], gi[~fdyn], fz[~fdyn], fcls[~fdyn]
        n_fill = len(cand)
        I = np.concatenate([I, cand]); cnt = np.concatenate([cnt, np.zeros(n_fill)]); mz = np.concatenate([mz, fz]); sd = np.concatenate([sd, (gmax - gmin)[gi]]); ns = np.concatenate([ns, np.zeros(n_fill)])
        lo = np.concatenate([lo, glo[gi]]); cls = np.concatenate([cls, fcls]); src = np.concatenate([src, np.full(n_fill, 3, np.uint8)]); mxz_all = np.concatenate([mxz_all, gmax[gi]])
if D is not None:
    I = np.concatenate([I, D[0]]); cnt = np.concatenate([cnt, D[1]]); mz = np.concatenate([mz, D[2]]); sd = np.concatenate([sd, D[3]]); ns = np.concatenate([ns, D[4]]); lo = np.concatenate([lo, D[5]]); cls = np.concatenate([cls, D[6]]); src = np.concatenate([src, np.full(len(D[0]), 2, np.uint8)]); mxz_all = np.concatenate([mxz_all, D[7]])
# 타일
tile_cells = int(round(a.tile_m / RX)); tx = np.floor_divide(I[:, 0], tile_cells); ty = np.floor_divide(I[:, 1], tile_cells)
tkey = (tx + OFF) * (2 * OFF) + (ty + OFF); order = np.lexsort((I[:, 2], I[:, 1], I[:, 0], tkey))
I, cnt, mz, sd, ns, lo, cls, src, tkey, mxz_all = I[order], cnt[order], mz[order], sd[order], ns[order], lo[order], cls[order], src[order], tkey[order], mxz_all[order]
ut, tstart = np.unique(tkey, return_index=True); tile_id = np.repeat(np.arange(len(ut)), np.diff(np.r_[tstart, len(tkey)])).astype(np.int32)
tiles = []
for ti, (s, e) in enumerate(zip(tstart, np.r_[tstart[1:], len(tkey)])):
    txi, tyi = int(ut[ti] // (2 * OFF) - OFF), int(ut[ti] % (2 * OFF) - OFF)
    tiles.append({"tx": txi, "ty": tyi, "x0": txi * a.tile_m, "y0": tyi * a.tile_m, "start": int(s), "n": int(e - s), "z_min": round(float(mz[s:e].min()), 2), "z_max": round(float(mz[s:e].max()), 2)})
params = {"frame": "robot lidar_slam saved map frame (final PGO) — /cloud_registered at keyframe times", "res_xy": RX, "res_z": RZ, "prob_res": RP, "prob_hit": a.prob_hit, "prob_miss": a.prob_miss,
          "free_range": a.free_range, "max_range": a.max_range, "min_scans": a.min_scans, "prob": not a.no_prob, "depth": not a.no_depth, "depth_max_miss": a.depth_max_miss, "drop_dynamic": not a.keep_dynamic,
          "tile_m": a.tile_m, "tile_cells": tile_cells, "n_scans": len(files), "n_points": int(n_pts_tot), "skip_every": a.skip_every, "skip_phase": a.skip_phase, "classes": classes,
          "class_codes": {"254": "other(low confidence)", "255": "unlabeled"}, "source_codes": {"1": "lidar(cloud_registered)", "2": "depth fill", "3": "flat fill (interpolated within 0.1 m cell)"}, "fill_flat": a.fill_flat}
np.savez_compressed(a.out + ".npz", ijk=I.astype(np.int32), mean_z=mz.astype(np.float32), max_z=mxz_all.astype(np.float32), std_z=sd.astype(np.float16), count=np.minimum(cnt, 65535).astype(np.uint16), n_scans=np.minimum(ns, 65535).astype(np.uint16),
                    logodds=lo.astype(np.float16), cls=cls.astype(np.uint8), source=src, tile=tile_id, res_xy=RX, res_z=RZ, prob_res=RP, classes=np.array(classes),
                    coarse_ijk=unkey(ck).astype(np.int32), coarse_logodds=c_lo.astype(np.float16), coarse_hits=np.minimum(c_hits, 65535).astype(np.uint16), coarse_misses=np.minimum(c_miss, 65535).astype(np.uint16),
                    removed_ijk=removed_ijk, removed_why=removed_why, params=np.array(json.dumps(params, ensure_ascii=False)))
# 타일 파일 (로봇용 단순 바이너리)
REC = np.dtype([("ix", "<i4"), ("iy", "<i4"), ("iz", "<i2"), ("cls", "u1"), ("src", "u1"), ("mean_z", "<f4"), ("max_z", "<f4"), ("logodds", "<f2"), ("count", "<u2")])   # 24 B
if not a.no_tiles:
    td = a.out + "_tiles"; os.makedirs(td, exist_ok=True)
    for f_ in glob.glob(os.path.join(td, "t_*.bin")): os.remove(f_)
    rec = np.zeros(len(I), REC); rec["ix"], rec["iy"], rec["iz"] = I[:, 0], I[:, 1], I[:, 2]; rec["cls"] = cls; rec["src"] = src; rec["mean_z"] = mz; rec["max_z"] = mxz_all; rec["logodds"] = lo; rec["count"] = np.minimum(cnt, 65535)
    for t in tiles:
        t["file"] = f"t_{t['tx']}_{t['ty']}.bin"; rec[t["start"]:t["start"] + t["n"]].tofile(os.path.join(td, t["file"])); t["bytes"] = t["n"] * REC.itemsize
    json.dump({"format": "voxel3d_tiles v2", "record": [(n, str(REC.fields[n][0])) for n in REC.names], "record_bytes": REC.itemsize, "byte_order": "little",
               "index_to_world": "x=(ix+0.5)*res_xy, y=(iy+0.5)*res_xy, z=mean_z (voxel center z=(iz+0.5)*res_z); max_z = 셀 안 최대 z — std 큰 셀(두 표면 섞임, 계단 코)에서 보수적 높이로 사용 권장", "tile_of": "tx=floor(x/tile_m), ty=floor(y/tile_m)",
               **params, "n_cells": int(len(I)), "total_bytes": int(len(I) * REC.itemsize), "tiles": tiles}, open(os.path.join(td, "index.json"), "w"), indent=1, ensure_ascii=False)
# 요약
byc = {classes[c] if c < NC else ("other" if c == 254 else "unlabeled"): int(n) for c, n in zip(*np.unique(cls, return_counts=True))}
summ = {"cells": int(len(I)), "cells_lidar": int((src == 1).sum()), "cells_depth": int((src == 2).sum()), "cells_flat_fill": int((src == 3).sum()), "lidar_cells_before_filter": int(n_lidar_all), "removed_prob_free": n_prob_removed, "removed_dynamic": n_dyn_removed,
        "depth_points": int(dep_n), "coarse_cells": int(len(ck)), "coarse_occupied": int((c_lo > 0).sum()), "coarse_free": int((c_lo < 0).sum()), "bytes_per_cell": REC.itemsize, "total_mb": round(len(I) * REC.itemsize / 1e6, 1),
        "tiles": len(tiles), "tile_cells_max": max(t["n"] for t in tiles), "tile_mb_max": round(max(t["n"] for t in tiles) * REC.itemsize / 1e6, 1), "by_class": byc, "std_z_med_cm": round(float(np.median(sd)) * 100, 2), "seconds": round(time.time() - t0)}
json.dump({**summ, "params": params}, open(a.out + "_summary.json", "w"), indent=1, ensure_ascii=False)
print(f"\n세부 셀 {summ['cells']:,} (LiDAR {summ['cells_lidar']:,} + depth {summ['cells_depth']:,} + 평탄채움 {summ['cells_flat_fill']:,}) · LiDAR hit 셀 {n_lidar_all:,} 중 확률 free 로 제거 {n_prob_removed:,} · 동적 제거 {n_dyn_removed:,}")
print(f"확률 격자 {len(ck):,} 셀: 점유 {summ['coarse_occupied']:,} · free {summ['coarse_free']:,} · depth 점 {dep_n:,} → 채움 셀 {dep_added:,}")
print(f"클래스: {byc}"); print(f"메모리: {REC.itemsize} B/셀 → 전체 {summ['total_mb']} MB · 타일 {len(tiles)}개 · 최대 타일 {summ['tile_mb_max']} MB ({summ['tile_cells_max']:,} 셀)")
print("저장:", a.out + ".npz", ("+ " + a.out + "_tiles/") if not a.no_tiles else "", f"[{summ['seconds']}s]")
