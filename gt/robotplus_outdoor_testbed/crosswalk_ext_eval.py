#!/usr/bin/env python3
"""횡단보도 연장 전/후 GT 비교 + 탑뷰. 사용: python crosswalk_ext_eval.py <map_root> <out_png> [--max-ext 1.0]"""
import os, sys, json, argparse
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, HERE)
import crosswalk_extend as CX, gt_eval as GE
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--max-ext", type=float, default=1.0); ap.add_argument("--fill-interior", action="store_true")
a = ap.parse_args()
z = dict(np.load(os.path.join(a.root, "compiled", "session_costmap_occ.npz"), allow_pickle=True)); cost = z["cost"]; res = float(z["res"])
gt = np.load(os.path.join(a.root, "compiled", "gt_map_frame.npz"), allow_pickle=True)["label"]; assert gt.shape == cost.shape
new, rep = CX.extend(cost, res, 20, 10, 230, max_ext=a.max_ext, fill_interior=a.fill_interior)
CW, SW, RD = GE.CLASSES.index("crosswalk"), GE.CLASSES.index("sidewalk"), GE.CLASSES.index("road")
def metrics(c):
    obs = c != 255; out = {}
    for name, gi, cv in [("sidewalk", SW, 10), ("crosswalk", CW, 20), ("road", RD, 230)]:
        g = (gt == gi) & obs; p = (c == cv) & (gt != 0); inter = (g & p).sum(); out[name] = {"iou": round(float(inter / max((g | p).sum(), 1)), 3), "recall": round(float(inter / max(g.sum(), 1)), 3), "precision": round(float(inter / max(p.sum(), 1)), 3)}
    gcw = (gt == CW) & obs; out["gt_crosswalk_as_road"] = round(float(((c == 230) & gcw).sum() / max(gcw.sum(), 1)), 4)
    grd = (gt == RD) & obs; out["gt_road_as_crosswalk"] = round(float(((c == 20) & grd).sum() / max(grd.sum(), 1)), 4)
    out["gt_crosswalk_as_road_cells"] = int(((c == 230) & gcw).sum()); return out
mb, ma = metrics(cost), metrics(new)
print(json.dumps({"before": mb, "after": ma, "blobs": rep, "changed_cells": int((new != cost).sum())}, ensure_ascii=False, indent=1))
np.savez_compressed(os.path.join(a.root, "compiled", "session_costmap_occ_cwext.npz"), **{**z, "cost": new})
COL = {10: (255, 255, 255), 20: (240, 210, 74), 60: (46, 139, 87), 90: (95, 170, 95), 150: (170, 170, 170), 200: (255, 140, 0), 230: (128, 64, 128), 254: (0, 0, 0), 255: (70, 70, 70)}
def img(c, hl=None):
    im = np.full(c.shape + (3,), 60, np.uint8)
    for v, k in COL.items(): im[c == v] = k
    if hl is not None: im[hl] = (255, 0, 0)
    return im
def gtimg():
    im = np.zeros(gt.shape + (3,), np.uint8)
    for k, n in enumerate(GE.CLASSES): im[gt == k] = GE.COLORS[n]
    return im
chg = new != cost
# 전체 + 횡단보도 확대
cw_cells = np.argwhere((gt == CW))
r0, c0 = cw_cells.min(0) - 12; r1, c1 = cw_cells.max(0) + 12; r0, c0 = max(r0, 0), max(c0, 0)
def tile(im, crop, s):
    if crop: im = im[r0:r1, c0:c1]
    return Image.fromarray(np.flipud(im.transpose(1, 0, 2))).resize((im.shape[0] * s, im.shape[1] * s), Image.NEAREST)
rows = [("GT", gtimg(), None), ("Before", img(cost), mb), ("After (crosswalk extended)", img(new), ma), ("Changed cells (red)", img(new, chg), None)]
def label(t, txt, m):
    d = ImageDraw.Draw(t); d.rectangle((0, 0, t.width, 32 if m else 18), fill=(0, 0, 0)); d.text((5, 3), txt, fill=(255, 255, 0))
    if m: d.text((5, 17), f"crosswalk IoU {m['crosswalk']['iou']:.3f}  GT crosswalk→road {m['gt_crosswalk_as_road']:.1%} ({m['gt_crosswalk_as_road_cells']} cells)", fill=(255, 255, 255))
    return t
full = [label(tile(im, False, 2), t, m) for t, im, m in rows]; zoom = [label(tile(im, True, 5), t, m) for t, im, m in rows]
W = sum(t.width for t in full) + 30; Z = sum(t.width for t in zoom) + 30
sheet = Image.new("RGB", (max(W, Z), full[0].height + zoom[0].height + 50), (255, 255, 255)); d = ImageDraw.Draw(sheet)
x = 0
for t in full: sheet.paste(t, (x, 20)); x += t.width + 10
x = 0
for t in zoom: sheet.paste(t, (x, full[0].height + 40)); x += t.width + 10
d.text((4, 4), "top: full map   bottom: crosswalk area zoom   white=sidewalk yellow=crosswalk purple=road green=unlabeled gray=context black=obstacle", fill=(0, 0, 0))
sheet.save(a.out); print("saved", a.out, sheet.size)
