#!/usr/bin/env python3
"""트랙 2 기하 변형 3종 비교 — <prefix>_pcd / <prefix>_keyframe_scans / <prefix>_all_scans 의 robot_map_semantic.pcd 를 BEV PNG(A|B|C) + rrd 로.
  python tools/t2_compare_variants.py <prefix> [--out output/<prefix>_pcd/compiled]
"""
import argparse, json, os
import numpy as np, open3d as o3d, cv2, rerun as rr, rerun.blueprint as rrb
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--out"); ap.add_argument("--zmin", type=float, default=-0.3); ap.add_argument("--zmax", type=float, default=2.2)
a = ap.parse_args(); out = a.out or f"{REPO}/output/{a.prefix}_pcd/compiled"
variants = [("A: robot saved map", "pcd"), ("B: keyframe-time scans", "keyframe_scans"), ("C: all scans", "all_scans")]
def load(p): pc = o3d.io.read_point_cloud(p); return np.asarray(pc.points).astype(np.float32), (np.asarray(pc.colors) * 255).astype(np.uint8)
data = []
for n, g in variants:
    p = f"{REPO}/output/{a.prefix}_{g}/compiled/robot_map_semantic.pcd"
    if not os.path.exists(p): print("없음:", p); continue
    P, C = load(p); s = json.load(open(f"{REPO}/output/{a.prefix}_{g}/compiled/summary.json")); lab = sum(v for k, v in s["voxel_labels"].items() if not k.startswith(("unknown", "other")))
    data.append((f"{n} ({len(P)/1000:.1f}k pts, labeled {100*lab/max(s['voxels'],1):.0f}%)", g, P, C)); print(data[-1][0], "| occupied", s["occupancy"]["occupied"], "| costmap free", s["costmap"]["free_path"], "| objects", s["objects"])
allp = np.concatenate([d[2] for d in data]); x0, y0 = np.percentile(allp[:, 0], 0.5) - 1, np.percentile(allp[:, 1], 0.5) - 1; x1, y1 = np.percentile(allp[:, 0], 99.5) + 1, np.percentile(allp[:, 1], 99.5) + 1
zl = np.percentile(allp[:, 2], 3)
def bev(P, C, res=0.05):
    m = (P[:, 2] > zl + a.zmin) & (P[:, 2] < zl + a.zmax + 0.3); P, C = P[m], C[m]; W, H = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    img = np.full((H, W, 3), 255, np.uint8); o = np.argsort(P[:, 2]); u = ((P[o, 0] - x0) / res).astype(int); v = ((y1 - P[o, 1]) / res).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H); img[v[ok], u[ok]] = C[o][ok][:, ::-1]; return img
tiles = []
for n, g, P, C in data:
    im = bev(P, C); cv2.putText(im, n, (10, im.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1); tiles.append(im)
sep = np.full((tiles[0].shape[0], 8, 3), 100, np.uint8); img = tiles[0]
for t in tiles[1:]: img = np.concatenate([img, sep, t], 1)
cd = json.load(open(f"{REPO}/output/{a.prefix}_pcd/sem/classes.json")); y = 20
for c in cd["classes"]:
    col = tuple(int(c["color"][i:i + 2], 16) for i in (5, 3, 1)); cv2.rectangle(img, (10, y - 12), (30, y + 2), col, -1); cv2.putText(img, c["name"], (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1); y += 18
png = f"{out}/semantic_bev_A_B_C.png"; cv2.imwrite(png, img); print("png", img.shape, png)
rr.init("t2_variants"); rr.save(f"{out}/track2_variants.rrd")
for n, g, P, C in data: rr.log(g, rr.Points3D(P, colors=C, radii=0.05))
T = np.loadtxt(json.load(open(f"{REPO}/sessions/session_{a.prefix}_pcd.yaml".replace(".yaml", ".yaml")))["options"]["robot_map"]["traj"], comments="#") if False else None
rr.log("bev", rr.Image(img[:, :, ::-1]))
rr.send_blueprint(rrb.Blueprint(rrb.Tabs(
    rrb.Horizontal(*[rrb.Spatial3DView(origin="/", name=n, contents=[f"+ /{g}"]) for n, g, _, _ in data], name="A | B | C"),
    rrb.Spatial2DView(origin="/bev", name="BEV A | B | C")), collapse_panels=True)); print("rrd", f"{out}/track2_variants.rrd")
