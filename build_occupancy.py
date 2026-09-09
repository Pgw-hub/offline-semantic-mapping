#!/usr/bin/env python3
"""확률적 occupancy 레이어 (맵 패키지 내장용) — 외부 OctoMap 없이 numpy 레이캐스팅.

입력: 스캔별 world 점 + 센서 원점 (export_lidar_scans.py 출력 디렉터리, 여러 로그 가능)
      + 세션 semantic 복셀 (session_voxsem.npz, 0.1m) → 하이브리드 판정
출력: <out>.npz
  ijk (M,3) int32 · res · hits · misses (int32) · logodds (f32) · state (int8: 1 occupied / 0 free)
  surf_state (N_surface,) int8 — 우리 0.1m 표면 복셀 각각의 상태 (1 occ / 0 free / -1 미관측)
  surf_removed (N_surface,) bool — 동적 클래스(person)이고 free 공간에 있음 → 표시/코스트맵에서 제외 권장
규칙(하이브리드):
  · log-odds: hit +log(ph/(1-ph)), miss +log(pm/(1-pm)), 클램프 [-2, 3.5]  (기본 ph 0.85 / pm 0.45 — 희소 야외 LiDAR 튠)
  · occupied = logodds>0 ∪ (표면 복셀 있음 & 동적 클래스 아님)   ← 지면 침식·식생 통과로 지워진 표면을 복원
  · free     = logodds<0 ∧ 표면 복셀 없음(또는 동적 클래스만)
  · 스캔당 셀 1회 갱신 (OctoMap insertPointCloud 와 동일), 끝점 셀은 miss 에서 제외
사용: python build_occupancy.py --scans dirA [dirB ...] --voxsem session_voxsem.npz --out session_occupancy [--res 0.2] [--max-range 40]
"""
import argparse, glob, os, time, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import classdict as CD

ap = argparse.ArgumentParser()
ap.add_argument("--scans", nargs="+", required=True)
ap.add_argument("--voxsem", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=float, default=0.2)
ap.add_argument("--max-range", type=float, default=40.0, help="복셀 빌더의 LiDAR 상한과 일치")
ap.add_argument("--free-range", type=float, default=25.0, help="free 광선 최대 길이 (긴 광선은 비용만 크고 정보 적음)")
ap.add_argument("--ray-stride", type=int, default=1, help="free 광선에 쓸 점 간격 (hit 은 전부)")
ap.add_argument("--prob-hit", type=float, default=float(os.environ.get("OCTO_PROB_HIT", 0.85)))
ap.add_argument("--prob-miss", type=float, default=float(os.environ.get("OCTO_PROB_MISS", 0.45)))
ap.add_argument("--dynamic", default=None, help="동적 클래스(쉼표). 생략하면 클래스 사전의 role=dynamic")
ap.add_argument("--batch", type=int, default=20, help="병합 단위 스캔 수")
a = ap.parse_args()

R = a.res; OFF = 1 << 20
def key(P):
    K = np.floor(P / R).astype(np.int64) + OFF
    return (K[:, 0] << 42) | (K[:, 1] << 21) | K[:, 2]
def unkey(k):
    return np.stack([(k >> 42) & 0x1FFFFF, (k >> 21) & 0x1FFFFF, k & 0x1FFFFF], -1).astype(np.int64) - OFF

def merge(acc, k, h, m):
    """acc=(keys,hits,miss) 정렬 배열에 (k,h,m) 합산"""
    if acc is None:
        return k, h, m
    K = np.concatenate([acc[0], k]); uk, inv = np.unique(K, return_inverse=True)
    H = np.zeros(len(uk), np.int64); M = np.zeros(len(uk), np.int64)
    np.add.at(H, inv, np.concatenate([acc[1], h])); np.add.at(M, inv, np.concatenate([acc[2], m]))
    return uk, H, M

files = []
for d in a.scans:
    files += sorted(glob.glob(os.path.join(d, "*.npz")))
print(f"스캔 {len(files)}개 · res {R} m · hit {a.prob_hit} miss {a.prob_miss}")
acc = None; t0 = time.time(); bk, bh, bm = [], [], []
n_steps_max = int(np.ceil(a.free_range / R))
for si, f in enumerate(files):
    d = np.load(f); pts = d["xyz"].astype(np.float64); org = d["origin"].astype(np.float64)
    vec = pts - org; rng = np.linalg.norm(vec, axis=1)
    ok = (rng > 0.05) & (rng <= a.max_range); pts, vec, rng = pts[ok], vec[ok], rng[ok]
    # hits: 끝점 셀 (스캔당 1회)
    hk = np.unique(key(pts))
    # misses: 원점→끝점 광선을 R 간격으로 샘플, 끝점 셀 제외 (긴 광선은 free_range 에서 절단)
    pv, pr = vec[:: a.ray_stride], rng[:: a.ray_stride]
    dirs = pv / pr[:, None]; L = np.minimum(pr, a.free_range)
    ts = (np.arange(n_steps_max) + 0.5) * R                                   # (S,)
    valid = ts[None, :] < L[:, None]                                          # (N,S)
    P = org[None, None, :] + dirs[:, None, :] * ts[None, :, None]             # (N,S,3)
    mk = key(P.reshape(-1, 3))[valid.ravel()]
    mk = np.unique(mk); mk = mk[~np.isin(mk, hk, assume_unique=True)]         # 끝점 셀은 miss 아님
    bk.append(np.concatenate([hk, mk])); bh.append(np.r_[np.ones(len(hk), np.int64), np.zeros(len(mk), np.int64)])
    bm.append(np.r_[np.zeros(len(hk), np.int64), np.ones(len(mk), np.int64)])
    if (si + 1) % a.batch == 0 or si == len(files) - 1:
        K = np.concatenate(bk); uk, inv = np.unique(K, return_inverse=True)
        H = np.zeros(len(uk), np.int64); M = np.zeros(len(uk), np.int64)
        np.add.at(H, inv, np.concatenate(bh)); np.add.at(M, inv, np.concatenate(bm))
        acc = merge(acc, uk, H, M); bk, bh, bm = [], [], []
        if (si + 1) % (a.batch * 10) == 0 or si == len(files) - 1:
            print(f"  {si+1}/{len(files)} 스캔 · 셀 {len(acc[0]):,} · {time.time()-t0:.0f}s")

keys, hits, miss = acc
lh = np.log(a.prob_hit / (1 - a.prob_hit)); lm = np.log(a.prob_miss / (1 - a.prob_miss))
logodds = np.clip(hits * lh + miss * lm, -2.0, 3.5).astype(np.float32)
state = (logodds > 0).astype(np.int8)

# --- 하이브리드: 세션 semantic 표면 복셀과 결합 ---
v = np.load(a.voxsem); V = float(v["vox"]); classes = [str(c) for c in v["classes"]]
sem = v["sem_mean"].astype(np.float32); has = v["n_sem"] > 0
lab = sem.argmax(1); lab = np.where(has & (sem.max(1) < 0.3), len(classes), lab); lab = np.where(has, lab, len(classes) + 1)
cd = CD.from_voxsem(v)
dyn_ids = [classes.index(c) for c in a.dynamic.split(",") if c in classes] if a.dynamic else CD.ids_with_role(cd, "dynamic")
a.dynamic = a.dynamic or ",".join(classes[i] for i in dyn_ids) or "-"
is_dyn = np.isin(lab, dyn_ids)
ctr = (v["ijk"] + 0.5) * V; sk = key(ctr)
pos = np.searchsorted(keys, sk); pos_c = np.minimum(pos, len(keys) - 1); found = keys[pos_c] == sk
surf_state = np.full(len(sk), -1, np.int8); surf_state[found] = state[pos_c[found]]
surf_removed = is_dyn & (surf_state == 0)                    # 동적 클래스가 free 안 → 제거
# 표면 복셀(비동적) 이 있는 셀은 occupied 로 복원 (관측된 셀 + 미관측 셀 모두)
static_surf = ~is_dyn
restore_k = np.unique(sk[static_surf])
new_k = restore_k[~np.isin(restore_k, keys, assume_unique=True)]
keys2 = np.concatenate([keys, new_k]); order = np.argsort(keys2)
hits2 = np.concatenate([hits, np.zeros(len(new_k), np.int64)])[order]
miss2 = np.concatenate([miss, np.zeros(len(new_k), np.int64)])[order]
lo2 = np.concatenate([logodds, np.zeros(len(new_k), np.float32)])[order]; keys2 = keys2[order]
state2 = (lo2 > 0).astype(np.int8)
in_surf = np.isin(keys2, restore_k, assume_unique=True)
n_restored = int((in_surf & (state2 == 0)).sum()); state2[in_surf] = 1
source = np.where(in_surf & (lo2 <= 0), 2, np.where(in_surf, 3, state2)).astype(np.int8)   # 0 free(레이만) 1 occ(레이만) 2 occ(표면 복원) 3 occ(둘 다)

np.savez_compressed(a.out + ".npz", ijk=unkey(keys2).astype(np.int32), res=R, hits=hits2.astype(np.int32), misses=miss2.astype(np.int32),
                    logodds=lo2, state=state2, source=source, prob_hit=a.prob_hit, prob_miss=a.prob_miss,
                    surf_state=surf_state, surf_removed=surf_removed, surf_vox=V, dynamic=np.array(a.dynamic.split(",")))
n = len(keys2)
print(f"\n셀 {n:,} (res {R} m): occupied {int((state2==1).sum()):,} · free {int((state2==0).sum()):,}")
print(f"  occupied 출처: 레이만 {int((source==1).sum()):,} · 표면복원(레이는 free/미관측) {int((source==2).sum()):,} · 둘 다 {int((source==3).sum()):,}")
print(f"  표면 복셀 {len(sk):,}: 레이 occ {int((surf_state==1).sum()):,} · 레이 free {int((surf_state==0).sum()):,} · 미관측 {int((surf_state==-1).sum()):,}")
print(f"  동적({a.dynamic}) 복셀 {int(is_dyn.sum()):,} 중 free 안이라 제거: {int(surf_removed.sum()):,}")
print("저장:", a.out + ".npz", f"[{time.time()-t0:.0f}s]")
