#!/usr/bin/env python3
"""voxel3d 단면 검사 뷰 — rerun 0.16 에는 클리핑 평면이 없으므로 타임라인을 슬라이스 슬라이더로 쓴다.
  python voxel3d_slice_view.py <prior.npz> <out.rrd> [--region auto|X0 X1 Y0 Y1] [--step 0.25] [--pad 2.0] [--voxsem ...]
기록:
  region/lidar · region/depth · region/fill  정적, 높이색 (영역 전체 — 토글용 배경)
  region/stairs_cells                          정적, class=stairs 셀만 (노랑)
  타임라인 slice_x / slice_y / slice_z        인덱스 i 마다 sweep/x 등 같은 엔티티에 [a_i, a_i+step) 슬래브만 기록 → 슬라이더를 끌면 단면이 지나감
                                               (시간 패널에서 타임라인을 slice_x ↔ slice_y ↔ slice_z 로 바꿔 축 선택)
  slabs_x/<a> · slabs_y/<a> · slabs_z/<a>      정적, 슬래브별 개별 엔티티 (여러 개를 동시에 켜서 비교)
영역: auto = class 'stairs' LiDAR 셀의 가장 큰 덩어리 bbox ± pad. 영역 안은 표본 없이 전부 넣는다.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # 저장소 루트 (패키지 import)
import argparse, json, os, sys
import numpy as np, rerun as rr, rerun.blueprint as rrb
from scipy import ndimage
from common import classdict as CD
ap = argparse.ArgumentParser(); ap.add_argument("prior"); ap.add_argument("out"); ap.add_argument("--region", nargs="+", default=["auto"]); ap.add_argument("--step", type=float, default=0.25)
ap.add_argument("--pad", type=float, default=2.0); ap.add_argument("--voxsem"); ap.add_argument("--radius", type=float, default=None)
a = ap.parse_args()
v = np.load(a.prior); RX, RZ = float(v["res_xy"]), float(v["res_z"]); I = v["ijk"].astype(np.int64); cls = v["cls"]; src = v["source"]; classes = [str(c) for c in v["classes"]]
xyz = np.stack([(I[:, 0] + 0.5) * RX, (I[:, 1] + 0.5) * RX, v["mean_z"].astype(np.float32)], -1).astype(np.float32)
st_id = classes.index("stairs") if "stairs" in classes else -1
if a.region[0] == "auto":
    k = (cls == st_id) & (src == 1)
    if k.sum() < 50: sys.exit("stairs 셀이 거의 없음 → --region X0 X1 Y0 Y1 로 지정")
    g = 0.5; gx = np.floor((xyz[k, 0] - xyz[k, 0].min()) / g).astype(int); gy = np.floor((xyz[k, 1] - xyz[k, 1].min()) / g).astype(int)
    H = np.zeros((gx.max() + 1, gy.max() + 1), int); np.add.at(H, (gx, gy), 1); lab, n = ndimage.label(H > 3); sizes = ndimage.sum(H, lab, range(1, n + 1)); best = int(np.argmax(sizes)) + 1
    m = lab[gx, gy] == best; sx, sy = xyz[k][m, 0], xyz[k][m, 1]
    x0, x1, y0, y1 = sx.min() - a.pad, sx.max() + a.pad, sy.min() - a.pad, sy.max() + a.pad
    print(f"stairs 덩어리 {n}개 중 최대({int(m.sum())} 셀) bbox x {sx.min():.1f}~{sx.max():.1f} y {sy.min():.1f}~{sy.max():.1f} → 영역 ±{a.pad} m")
else:
    x0, x1, y0, y1 = map(float, a.region)
R = (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
P = xyz[R]; C = cls[R]; S = src[R]; print(f"영역 셀 {len(P):,} (LiDAR {int((S==1).sum()):,} · depth {int((S==2).sum()):,} · 채움 {int((S==3).sum()):,}) · 영역 x {x0:.2f}~{x1:.2f} y {y0:.2f}~{y1:.2f}")
def turbo(z, lo=None, hi=None):
    lo = np.percentile(z, 1) if lo is None else lo; hi = np.percentile(z, 99) if hi is None else hi; t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1); return (np.stack([r, g, b], -1) * 255).astype(np.uint8)
zlo, zhi = np.percentile(P[:, 2], 1), np.percentile(P[:, 2], 99); col = turbo(P[:, 2], zlo, zhi); rad = a.radius or RX * 0.5
rr.init("voxel3d_slice"); rr.save(a.out)
for s_, nm in ((1, "lidar"), (2, "depth"), (3, "fill")):
    k = S == s_
    if k.any(): rr.log(f"region/{nm}", rr.Points3D(P[k], colors=col[k], radii=rad), static=True)
if st_id >= 0 and (C == st_id).any(): rr.log("region/stairs_cells", rr.Points3D(P[C == st_id], colors=[255, 220, 0], radii=rad * 1.2), static=True)
rr.log("region/box", rr.Boxes3D(centers=[[(x0 + x1) / 2, (y0 + y1) / 2, (zlo + zhi) / 2]], half_sizes=[[(x1 - x0) / 2, (y1 - y0) / 2, (zhi - zlo) / 2]], colors=[[255, 200, 0]]), static=True)
# 슬라이스: 축별 타임라인 + 슬래브 엔티티
for ax, name, lo, hi in ((0, "x", x0, x1), (1, "y", y0, y1), (2, "z", zlo - 0.2, zhi + 0.2)):
    edges = np.arange(lo, hi + a.step, a.step); n_sl = 0
    for i in range(len(edges) - 1):
        k = (P[:, ax] >= edges[i]) & (P[:, ax] < edges[i + 1])
        rr.set_time_sequence(f"slice_{name}", i)
        rr.log(f"sweep/{name}", rr.Points3D(P[k], colors=col[k], radii=rad * 1.1) if k.any() else rr.Clear(recursive=False))
        rr.log(f"sweep/{name}_label", rr.TextLog(f"{name} ∈ [{edges[i]:.2f}, {edges[i+1]:.2f}) m · {int(k.sum()):,} 셀"))
        if k.any(): rr.log(f"slabs_{name}/{edges[i]:+.2f}", rr.Points3D(P[k], colors=col[k], radii=rad), static=True); n_sl += 1
    rr.reset_time()
    print(f"{name} 슬라이스 {n_sl}개 (두께 {a.step} m)")
rr.send_blueprint(rrb.Blueprint(rrb.Tabs(
    rrb.Spatial3DView(origin="/", name="sweep (타임라인 = 슬라이스)", contents=["+ /sweep/**", "+ /region/box"]),
    rrb.Spatial3DView(origin="/", name="sweep + 배경(연함)", contents=["+ /sweep/**", "+ /region/**"]),
    rrb.Spatial3DView(origin="/", name="영역 전체 (LiDAR/depth/fill 토글)", contents=["+ /region/**"]),
    rrb.Spatial3DView(origin="/", name="slabs x (개별 토글)", contents=["+ /slabs_x/**", "+ /region/box"]),
    rrb.Spatial3DView(origin="/", name="slabs y (개별 토글)", contents=["+ /slabs_y/**", "+ /region/box"]),
    rrb.Spatial3DView(origin="/", name="slabs z (개별 토글)", contents=["+ /slabs_z/**", "+ /region/box"])),
    rrb.TimePanel(expanded=True), collapse_panels=False))
print("rrd", a.out)
