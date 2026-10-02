#!/usr/bin/env python3
"""voxel3d 사전맵 rerun 뷰 + BEV 높이맵 PNG.
  python voxel3d_view.py compiled/session_voxel3d.npz compiled/voxel3d_view.rrd [--voxsem voxel/session_voxsem.npz] [--sub 1] [--bev compiled/voxel3d_bev.png]
엔티티: prior/lidar/<class>(클래스 색, 클래스별 토글) · prior/lidar_height(높이 색) · prior/depth_fill · prior/flat_fill(평탄 채움, 연한 클래스 색) · removed/free(확률 free 로 제거) · removed/dynamic · free_space(확률 격자 free, 1/30 표본) · tiles(경계)
"""
import argparse, json, os, sys
import numpy as np, rerun as rr, rerun.blueprint as rrb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import classdict as CD
ap = argparse.ArgumentParser(); ap.add_argument("prior"); ap.add_argument("out"); ap.add_argument("--voxsem"); ap.add_argument("--sub", type=int, default=1); ap.add_argument("--bev"); ap.add_argument("--max-pts", type=int, default=6_000_000)
a = ap.parse_args()
v = np.load(a.prior); RX, RZ, RP = float(v["res_xy"]), float(v["res_z"]), float(v["prob_res"]); prm = json.loads(str(v["params"])); classes = [str(c) for c in v["classes"]]
I = v["ijk"]; xyz = np.stack([(I[:, 0] + 0.5) * RX, (I[:, 1] + 0.5) * RX, v["mean_z"]], -1).astype(np.float32); cls = v["cls"]; src = v["source"]
pal = None
if a.voxsem:
    vs = np.load(a.voxsem, allow_pickle=True)
    if "classdict" in vs.files: cd = json.loads(str(vs["classdict"])); pal = {i: CD.hex2rgb(c["color"]) for i, c in enumerate(cd["classes"])}
if pal is None: pal = {i: tuple(int(x) for x in np.array([[46,204,113],[241,196,15],[52,152,219],[155,89,182],[230,126,34],[26,188,156],[231,76,60],[149,165,166]][i % 8])) for i in range(len(classes))}
pal[254] = CD.hex2rgb(CD.OTHER_COLOR); pal[255] = CD.hex2rgb(CD.UNKNOWN_COLOR)
col_cls = np.array([pal.get(int(c), (128, 128, 128)) for c in range(256)], np.uint8)[cls]
def turbo(z):
    t = np.clip((z - np.percentile(z, 2)) / max(np.percentile(z, 98) - np.percentile(z, 2), 1e-6), 0, 1)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1); return (np.stack([r, g, b], -1) * 255).astype(np.uint8)
sub = max(a.sub, int(np.ceil(len(xyz) / a.max_pts)))
rr.init("voxel3d"); rr.save(a.out); rad = RX * 0.5
L = src == 1; D = src == 2; FF = src == 3
def cname(c): return classes[c] if c < len(classes) else ("other" if c == 254 else "unlabeled")
# 클래스별 엔티티 — rerun 왼쪽 패널에서 클래스 하나씩 켜고 끌 수 있게 (prior/lidar/<class>, prior/flat_fill/<class>)
for c in np.unique(cls):
    kc = L & (cls == c)
    if kc.any(): rr.log(f"prior/lidar/{cname(int(c))}", rr.Points3D(xyz[kc][::sub], colors=col_cls[kc][::sub], radii=rad), static=True)
    kf = FF & (cls == c)
    if kf.any(): rr.log(f"prior/flat_fill/{cname(int(c))}", rr.Points3D(xyz[kf][::sub], colors=col_cls[kf][::sub] // 2 + 100, radii=rad * 0.8), static=True)   # 평탄 채움: 클래스 색을 연하게
rr.log("prior/lidar_height", rr.Points3D(xyz[L][::sub], colors=turbo(xyz[L, 2])[::sub], radii=rad), static=True)
if D.any(): rr.log("prior/depth_fill", rr.Points3D(xyz[D][::sub], colors=[0, 200, 255], radii=rad), static=True)
if "removed_ijk" in v.files and len(v["removed_ijk"]):
    R = v["removed_ijk"]; rx = np.stack([(R[:, 0] + 0.5) * RX, (R[:, 1] + 0.5) * RX, (R[:, 2] + 0.5) * RZ], -1).astype(np.float32); w = v["removed_why"]
    rr.log("removed/free", rr.Points3D(rx[w == 1][::sub], colors=[255, 60, 60], radii=rad), static=True)
    if (w == 2).any(): rr.log("removed/dynamic", rr.Points3D(rx[w == 2][::sub], colors=[255, 0, 255], radii=rad), static=True)
if "coarse_ijk" in v.files:
    C = v["coarse_ijk"]; lo = v["coarse_logodds"].astype(np.float32); fr = lo < 0
    cx = ((C[fr] + 0.5) * RP).astype(np.float32); s2 = max(30, int(np.ceil(len(cx) / 300000)))
    rr.log("free_space", rr.Points3D(cx[::s2], colors=[170, 170, 170], radii=RP * 0.15), static=True)
# 타일 경계
idx_path = a.prior.replace(".npz", "_tiles/index.json")
if os.path.exists(idx_path):
    ti = json.load(open(idx_path)); T = float(ti["tile_m"]); strips = []
    for t in ti["tiles"]:
        x0, y0, z = t["x0"], t["y0"], t["z_min"]; strips.append([[x0, y0, z], [x0 + T, y0, z], [x0 + T, y0 + T, z], [x0, y0 + T, z], [x0, y0, z]])
    rr.log("tiles", rr.LineStrips3D(strips, colors=[255, 200, 0], radii=0.01), static=True)
# BEV 높이맵: 컬럼별 최상단(mean_z) — LiDAR 원점 높이 아래 지형만 (max_z = 지형 상한: 전체 z 5퍼센타일 + 1.5 m)
zl = np.percentile(xyz[:, 2], 3); maxz = zl + 1.5; k = xyz[:, 2] < maxz
res = 0.05; x0, y0 = xyz[:, 0].min(), xyz[:, 1].min(); W = int((xyz[:, 0].max() - x0) / res) + 1; H = int((xyz[:, 1].max() - y0) / res) + 1
u = np.clip(((xyz[k, 0] - x0) / res).astype(int), 0, W - 1); vv = np.clip(((xyz[k, 1] - y0) / res).astype(int), 0, H - 1); hm = np.full((H, W), -np.inf, np.float32); np.maximum.at(hm, (vv, u), xyz[k, 2])
img = np.full((H, W, 3), 255, np.uint8); ok = np.isfinite(hm); img[ok] = turbo(hm[ok]); img = img[::-1]
rr.log("bev_height", rr.Image(img), static=True)
if a.bev:
    import cv2; cv2.imwrite(a.bev, img[:, :, ::-1])
rr.send_blueprint(rrb.Blueprint(rrb.Tabs(
    rrb.Spatial3DView(origin="/", name="3D prior (class colors)", contents=["+ /prior/lidar/**", "+ /prior/depth_fill", "+ /prior/flat_fill/**", "+ /tiles"]),
    rrb.Spatial3DView(origin="/", name="height colors", contents=["+ /prior/lidar_height", "+ /prior/depth_fill", "+ /tiles"]),
    rrb.Spatial3DView(origin="/", name="removed (red) + free space", contents=["+ /removed/**", "+ /free_space", "+ /prior/lidar/**"]),
    rrb.Spatial2DView(origin="/bev_height", name="BEV height")), collapse_panels=False))   # 왼쪽 패널 열어둠: 클래스별 토글용
n = len(xyz); print(f"셀 {n:,} (표본 1/{sub}) · LiDAR {int(L.sum()):,} · depth {int(D.sum()):,} · 평탄채움 {int(FF.sum()):,} · rrd {a.out}" + (f" · bev {a.bev}" if a.bev else ""))
