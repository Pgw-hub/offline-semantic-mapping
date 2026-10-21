#!/usr/bin/env python3
"""로그별 복셀 npz 를 세션 하나로 병합 (복셀 키 기준 가중 합산).
first_frame 은 전역 프레임 (log_idx*600 + 로컬) 로 재부여.
사용: python merge_voxels.py <out.npz> <log000.npz> <log001.npz> ...
"""
import sys, re
import numpy as np

out = sys.argv[1]; files = sys.argv[2:]
K, NP, RGBs, SEMs, NS, FF = [], [], [], [], [], []
o = 1 << 20
for f in files:
    d = np.load(f)
    li = int(re.search(r"log_?(\d{3})", f).group(1))
    ijk = d["ijk"].astype(np.int64)
    K.append(((ijk[:, 0]+o) << 42) | ((ijk[:, 1]+o) << 21) | (ijk[:, 2]+o))
    NP.append(d["n_pts"].astype(np.int64))
    ns = d["n_sem"].astype(np.float64); NS.append(ns)
    SEMs.append(d["sem_mean"].astype(np.float64) * ns[:, None])            # 가중 합으로 되돌림
    RGBs.append(d["rgb"].astype(np.float64) * d["n_pts"][:, None])
    FF.append(d["first_frame"].astype(np.int64) + li * 600)
    print(f"{f.split('/')[-1]}: {len(ijk):,}")
keys = np.concatenate(K); uk, inv = np.unique(keys, return_inverse=True); n = len(uk)
n_pts = np.bincount(inv, weights=np.concatenate(NP), minlength=n)
n_sem = np.bincount(inv, weights=np.concatenate(NS), minlength=n)
NCH = SEMs[0].shape[1]
sem = np.stack([np.bincount(inv, weights=np.concatenate(SEMs)[:, c], minlength=n) for c in range(NCH)], -1)
rgb = np.stack([np.bincount(inv, weights=np.concatenate(RGBs)[:, c], minlength=n) for c in range(3)], -1)
first = np.full(n, 10**9, np.int64); np.minimum.at(first, inv, np.concatenate(FF))
m = (1 << 21) - 1
ijk = (np.stack([(uk >> 42) & m, (uk >> 21) & m, uk & m], -1) - o).astype(np.int32)
d0 = np.load(files[0])
np.savez_compressed(out, ijk=ijk, n_pts=n_pts.astype(np.int32),
                    rgb=(rgb / np.maximum(n_pts, 1)[:, None]).astype(np.uint8),
                    sem_mean=(sem / np.maximum(n_sem, 1)[:, None]).astype(np.float16),
                    n_sem=n_sem.astype(np.int32), first_frame=first.astype(np.int32),
                    vox=d0["vox"], classes=d0["classes"], n_frames=len(files) * 600,
                    **({"classdict": d0["classdict"]} if "classdict" in d0.files else {}))
lab = int((n_sem > 0).sum())
print(f"세션 복셀 {n:,} · 라벨 {lab:,} ({100*lab/n:.0f}%) · 저장 {out}")
