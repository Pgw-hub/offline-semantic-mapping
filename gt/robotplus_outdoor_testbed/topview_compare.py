#!/usr/bin/env python3
"""탑뷰 비교 그림: GT | 설정별 코스트맵 (같은 맵 프레임). 사용: python topview_compare.py <out.png> <gt_root> <label>=<root> [<label>=<root> ...]
색: 흰=인도10 노랑=횡단보도20 초록=미라벨60 연두=잔디90 회색=문맥150 주황=uncertain200 보라=차도230 검정=lethal 짙은회=미관측"""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw
out, gt_root, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import gt_eval as GE
g = np.load(os.path.join(gt_root, "compiled", "gt_map_frame.npz"), allow_pickle=True); gt = g["label"]; H, W = gt.shape
COL = {10: (255, 255, 255), 20: (240, 210, 74), 60: (46, 139, 87), 90: (95, 170, 95), 150: (170, 170, 170), 200: (255, 140, 0), 230: (128, 64, 128), 254: (0, 0, 0), 255: (70, 70, 70)}
def cost_img(root):
    c = np.load(os.path.join(root, "compiled", "session_costmap_occ.npz"), allow_pickle=True)["cost"]; img = np.full(c.shape + (3,), 60, np.uint8)
    for v, k in COL.items(): img[c == v] = k
    return img, json.load(open(os.path.join(root, "compiled", "gt_eval.json")))
def gt_img():
    img = np.zeros(gt.shape + (3,), np.uint8)
    for k, name in enumerate(GE.CLASSES): img[gt == k] = GE.COLORS[name]
    return img
S = 2; panels = [("GT (testbed design)", gt_img(), None)]
for sp in specs:
    lab, root = sp.split("=", 1); img, ev = cost_img(root); panels.append((lab, img, ev))
tiles = []
for lab, img, ev in panels:
    im = Image.fromarray(np.flipud(img.transpose(1, 0, 2))).resize((H * S, W * S), Image.NEAREST); d = ImageDraw.Draw(im)
    d.rectangle((0, 0, H * S, 34), fill=(0, 0, 0)); d.text((6, 4), lab, fill=(255, 255, 0))
    if ev: d.text((6, 18), f"sidewalk IoU {ev['iou']['sidewalk']['iou']:.2f}  crosswalk {ev['iou']['crosswalk']['iou']:.2f}  road {ev['iou']['road']['iou']:.2f}  sidewalk→road {ev['key']['sidewalk_as_road']:.1%}", fill=(255, 255, 255))
    tiles.append(im)
sheet = Image.new("RGB", (sum(t.width for t in tiles) + 10 * (len(tiles) - 1), tiles[0].height + 22), (255, 255, 255)); x = 0
for t in tiles: sheet.paste(t, (x, 22)); x += t.width + 10
d = ImageDraw.Draw(sheet); d.text((4, 4), "white=sidewalk(10)  yellow=crosswalk(20)  purple=road(230)  green=unlabeled ground(60)  gray=context(150)  black=lethal  dark=unobserved", fill=(0, 0, 0))
sheet.save(out); print("saved", out, sheet.size)
