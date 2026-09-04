#!/usr/bin/env python3
"""semantic 복셀 맵 → (a) occupancy 복셀(높이 색) (b) n×n 2D costmap 후처리.

(a) occupancy: 점유 복셀 = 관측된 복셀 전부 (semantic 유무 무관). OctoMap 스타일 높이 색.
(b) costmap: xy 컬럼마다 지면 높이 추정(하위 10% z) → 지면 위 [0.15, clearance] m 구간에 복셀이 있으면 장애물(lethal)
             없으면 지면 복셀의 semantic 으로 비용 (path 10 / grass 90 / tree 253 / other 150 / semantic 없음 60)
             컬럼에 복셀이 하나도 없으면 unknown(255).  ROS costmap uint8 규약, 원점·해상도 메타 포함.

사용: python voxel_postprocess.py <voxel.npz> <out_prefix> [--res 0.1] [--clearance 1.2] [--rrd out.rrd]
"""
import argparse, json
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("vox"); ap.add_argument("out")
ap.add_argument("--res", type=float, default=0.1)
ap.add_argument("--clearance", type=float, default=1.2, help="로봇 통과 높이(m)")
ap.add_argument("--min-pts", type=int, default=6, help="컬럼 최소 관측 점 수 (미만이면 unknown)")
ap.add_argument("--rrd", default=None)
ap.add_argument("--objects", default=None, help="ConceptGraphs 객체 json (bbox 오버레이)")
a = ap.parse_args()

d = np.load(a.vox)
V = float(d["vox"]); ijk = d["ijk"].astype(np.int64)
sem = d["sem_mean"].astype(np.float32); n_sem = d["n_sem"]
classes = [str(c) for c in d["classes"]]
lab = sem.argmax(1); has = n_sem > 0; other = has & (sem.max(1) < 0.3)
ctr = (ijk + 0.5) * V

# ---------- (a) occupancy 복셀: 높이 색 ----------
z = ctr[:, 2]
zlo, zhi = np.percentile(z, 2), np.percentile(z, 98)
t = np.clip((z - zlo) / max(zhi - zlo, 1e-6), 0, 1)
occ_col = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, 0, ::-1]  # BGR→RGB
np.savez_compressed(a.out + "_occupancy.npz", ijk=d["ijk"], vox=V,
                    note="occupied voxels; free/unknown 미구분(ray-cast 정보 없음)")
print(f"(a) occupancy 복셀 {len(ijk):,}개, 높이 {zlo:.1f}~{zhi:.1f}m")

# ---------- (b) 2D costmap ----------
f = int(round(a.res / V))                       # 복셀→costmap 셀 배수
gi = ijk[:, 0] // f; gj = ijk[:, 1] // f
i0, j0 = gi.min(), gj.min()
H, W = int(gi.max() - i0 + 1), int(gj.max() - j0 + 1)
ci = (gi - i0).astype(np.int64); cj = (gj - j0).astype(np.int64)
cell = ci * W + cj
order = np.argsort(cell, kind="stable")
cell_s = cell[order]; z_s = z[order]
uniq, start = np.unique(cell_s, return_index=True)
end = np.r_[start[1:], len(cell_s)]

ground = np.full(H * W, np.nan, np.float32)
lethal = np.zeros(H * W, bool)
cost_sem = np.full(H * W, 60, np.uint8)         # 기하는 있으나 semantic 없음 → 중간 비용
BASE = {"path": 10, "grass": 90, "tree": 253, "person": 60}
lab_s = lab[order]; grass_s = (has[order]) & (lab_s == classes.index("grass")) & ~other[order]
npts_s = d["n_pts"][order]
# 1차: 컬럼별 지면(하위 10% z) + 관측 밀도
col_pts = np.zeros(H * W, np.int64)
for u, s0, s1 in zip(uniq, start, end):
    ground[u] = np.percentile(z_s[s0:s1], 10)
    col_pts[u] = npts_s[s0:s1].sum()
sparse = col_pts < a.min_pts                     # 원거리 희소 LiDAR 링 → 판정 보류(unknown)
# 지면을 3×3 이웃 최소값으로 평활화 (경사·노이즈로 인한 가짜 장애물 억제)
g2 = ground.reshape(H, W).copy(); g2[np.isnan(g2)] = np.inf
gmin = cv2.erode(g2.astype(np.float32), np.ones((3, 3), np.uint8))
gmin[~np.isfinite(gmin)] = np.nan
ground_s = np.where(np.isfinite(gmin), gmin, ground.reshape(H, W)).reshape(-1)
has_s = has[order]; hard_s = has_s & ~other[order] & (lab_s == classes.index("tree"))   # 확인된 단단한 식생/장애물
other_s = other[order]
uncertain = np.zeros(H * W, bool)                # 라벨 없는 낮은 물체 (LiDAR 만 봄) → 고비용
for u, s0, s1 in zip(uniq, start, end):
    if sparse[u]:
        continue
    zz = z_s[s0:s1]; g = ground_s[u]
    band = (zz > g + 0.15) & (zz < g + a.clearance)
    if not band.any():
        pass
    else:
        b_grass = band & grass_s[s0:s1]           # 키 큰 풀 → 통행 가능 (semantic 덕분에 가능한 판단)
        b_hard = band & (hard_s[s0:s1] | other_s[s0:s1])
        b_unl = band & ~has_s[s0:s1]              # 라벨 없음
        if b_hard.any() or (b_unl.any() and (zz[b_unl] - g).max() > 0.5):
            lethal[u] = True                      # 확인된 장애물 또는 0.5m 넘는 미확인 물체
            continue
        if b_unl.any():
            uncertain[u] = True                   # 낮은 미확인 물체: 통과 가능성 있지만 비싸게
    # 지면 복셀들(g±0.15)의 semantic 지배 클래스
    idx = order[s0:s1][np.abs(zz - g) <= 0.15]
    idx = idx[has[idx]]
    if len(idx) == 0:
        continue
    if np.mean(other[idx]) > 0.5:
        cost_sem[u] = 150
    else:
        idx = idx[~other[idx]]
        lab_dom = np.bincount(lab[idx], minlength=4).argmax() if len(idx) else None
        if lab_dom is not None:
            cost_sem[u] = BASE[classes[lab_dom]]
cost = np.full(H * W, 255, np.uint8)             # unknown
known = ~np.isnan(ground) & ~sparse              # 희소 컬럼은 unknown 으로 남김
cost[known] = cost_sem[known]
cost[uncertain & known] = 200
cost[lethal] = 254
cost = cost.reshape(H, W); ground = ground.reshape(H, W); known = known.reshape(H, W); lethal = lethal.reshape(H, W)
np.savez_compressed(a.out + "_costmap.npz", cost=cost, ground_z=ground,
                    x0=float(i0 * f * V), y0=float(j0 * f * V), res=a.res, axes="row=x,col=y",
                    legend="0-252 cost, 254 lethal(obstacle in [0.15,clearance]m), 255 unknown",
                    clearance_m=a.clearance)
kn = int(known.sum())
print(f"(b) costmap {H}×{W} @ {a.res}m — 관측 {kn:,}셀 | 장애물 {int(lethal.sum()):,} | "
      f"free path {(cost==10).sum():,} / grass {(cost==90).sum():,} / other {(cost==150).sum():,} / 기하만 {(cost==60).sum():,}")

# PNG 미리보기 (가로=x, 세로=y 위+)
vis = np.zeros((H, W, 3), np.uint8); vis[:] = (40, 40, 40)
pal = {10: (110, 140, 170), 90: (95, 150, 95), 150: (150, 130, 120), 60: (120, 120, 120), 200: (40, 120, 220), 253: (60, 95, 60), 254: (40, 40, 200)}
for v, c in pal.items():
    vis[cost == v] = c
cv2.imwrite(a.out + "_costmap.png", cv2.resize(vis.transpose(1, 0, 2)[::-1], None, fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST))

# ---------- rerun ----------
if a.rrd:
    import rerun as rr
    rr.init("voxel_postprocess"); rr.save(a.rrd)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ijk), min(len(ijk), 3_000_000), replace=False)
    rr.log("world/occupancy_height", rr.Points3D(ctr[sub], colors=occ_col[sub], radii=V * 0.45))
    # semantic 클래스 색 / 실제 RGB 색 복셀 (같은 서브샘플) — 클래스색 = 인식, 청회색 = 봤지만 어휘 밖, 회색 = LiDAR 만
    CC = {"grass": (95, 170, 95), "path": (200, 165, 120), "tree": (40, 110, 60), "person": (220, 60, 60)}
    pal_sem = np.array([CC[c] for c in classes] + [(120, 135, 160), (110, 110, 110)], np.uint8)
    lab_v = np.where(other, len(classes), lab); lab_v = np.where(has, lab_v, len(classes) + 1)
    rr.log("world/occupancy_semantic", rr.Points3D(ctr[sub], colors=pal_sem[lab_v[sub]], radii=V * 0.45))
    rr.log("world/occupancy_rgb", rr.Points3D(ctr[sub], colors=d["rgb"][sub], radii=V * 0.45))
    if a.objects:
        raw = json.load(open(a.objects)); objs = list(raw.values()) if isinstance(raw, dict) else raw
        for i, o in enumerate(objs):
            tag = o.get("object_tag") or o.get("tag")
            if tag == "person":
                continue
            c = np.array(o.get("bbox_center") or o["center_w"], float)
            e = np.array(o.get("bbox_extent") or o.get("extent") or [0.5, 0.5, 0.5], float)
            rr.log(f"world/objects/{tag.replace(' ', '_')}_{i}",
                   rr.Boxes3D(centers=[c], half_sizes=[e / 2], labels=[tag], colors=[[255, 200, 40]]))
    # costmap 을 지면 높이에 색 점으로
    ii, jj = np.where(known)
    cx = (ii + i0 * 0 + 0.5) * a.res + i0 * f * V; cy = (jj + 0.5) * a.res + j0 * f * V
    cz = ground[ii, jj] + 0.05
    cc = np.array([pal.get(int(v), (255, 255, 255)) for v in cost[ii, jj]], np.uint8)
    rr.log("world/costmap2d", rr.Points3D(np.stack([cx, cy, cz], -1), colors=cc, radii=a.res * 0.5))
    print("rrd:", a.rrd)
json.dump({"H": H, "W": W, "res": a.res, "known_cells": kn, "lethal": int(lethal.sum())}, open(a.out + "_costmap_meta.json", "w"))
