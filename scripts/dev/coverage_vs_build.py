#!/usr/bin/env python3
"""커버리지 모니터(recording/coverage_monitor.py) 판정이 실제 빌드 결과와 맞는지 검증 — 0.1 m 위에서 본 격자에서 칸 단위로 비교.
  LiDAR : 빌드 voxel3d 의 LiDAR 셀(source 1) 이 있는 칸  vs  모니터 '보행용(near)' 칸
  카메라 : 빌드 voxsem 의 라벨 받은 LiDAR 복셀(n_sem > 0) 칸  vs  모니터 '카메라' 칸 (보행용 칸 안에서, 모니터 표시와 같은 판정)
  재현율(빌드 칸 중 모니터도 본 비율)이 낮으면 모니터 기준이 빌더와 다르다는 뜻. 반대 방향(모니터만 본 칸)은 확률 판정으로 지워진 칸 등이라 참고값.
사용: python coverage_vs_build.py <coverage_grid.npz> <빌드 root (output/<이름>)> [--out report.json] [--png diff.png]
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # scripts/ (패키지 import)
import argparse, json, os
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("root"); ap.add_argument("--voxel3d", default="compiled/session_voxel3d.npz")
ap.add_argument("--out"); ap.add_argument("--png"); a = ap.parse_args()
G = np.load(a.grid); res = float(G["res"]); x0, y0 = int(G["x0"]), int(G["y0"]); near, far, cam = G["near"], G["far"], G["cam"]; H, W = near.shape

def footprint(xy):
    """(N,2) m → 모니터 격자 위 bool 지도 (격자 밖은 따로 셈)"""
    k = np.floor(xy / res).astype(np.int64); ix, iy = k[:, 0] - x0, k[:, 1] - y0; inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    m = np.zeros((H, W), bool); m[iy[inb], ix[inb]] = True; return m, int((~inb).sum())
def dil(m, r=1): return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
def rates(build, mon, tol):
    mt = dil(mon, tol) if tol else mon; bt = dil(build, tol) if tol else build
    return {"build_cells": int(build.sum()), "monitor_cells": int(mon.sum()), "recall_build_in_monitor": round(float((build & mt).sum() / max(build.sum(), 1)), 4),
            "monitor_in_build": round(float((mon & bt).sum() / max(mon.sum(), 1)), 4)}

rep = {"grid": a.grid, "root": a.root}
v = np.load(os.path.join(a.root, a.voxel3d)); rxy = float(v["res_xy"]); src = v["source"]
lid_xy = (v["ijk"][src == 1][:, :2] + 0.5) * rxy; B_lidar, out_l = footprint(lid_xy)
rep["voxel3d"] = {"params": json.loads(str(v["params"])).get("max_range"), "outside_grid": out_l, "exact": rates(B_lidar, near > 0, 0), "tol1": rates(B_lidar, near > 0, 1)}
vs = os.path.join(a.root, "voxel", "session_voxsem.npz")
if os.path.exists(vs):
    s = np.load(vs); vx = float(s["vox"]); l1 = s["source"] == 1 if "source" in s.files else np.ones(len(s["ijk"]), bool)
    lab = l1 & (s["n_sem"] > 0); B_lab, out_s = footprint((s["ijk"][lab][:, :2] + 0.5) * vx); B_all, _ = footprint((s["ijk"][l1][:, :2] + 0.5) * vx)
    B_unlab = B_all & ~B_lab                                                                   # 칸 안 LiDAR 복셀이 하나도 라벨을 못 받은 칸
    camd = cam > 0; nearm = near > 0
    rep["voxsem"] = {"outside_grid": out_s, "labelled_cells": int(B_lab.sum()), "recall_labelled_in_camera": round(float((B_lab & camd).sum() / max(B_lab.sum(), 1)), 4),
                     "near_cells_build_unlabelled": int((nearm & B_unlab).sum()), "near_cells_monitor_no_camera": int((nearm & ~camd).sum()),
                     "agree_on_near_cells": round(float(((nearm & B_lab & camd) | (nearm & B_unlab & ~camd)).sum() / max((nearm & B_all).sum(), 1)), 4),
                     "near_labelled_recall": round(float((nearm & B_lab & camd).sum() / max((nearm & B_lab).sum(), 1)), 4),                 # 보행용 칸 중 빌드가 라벨 붙인 칸을 모니터도 '카메라 봄' 으로
                     "purple_precision": round(float((nearm & B_unlab & ~camd).sum() / max((nearm & B_all & ~camd).sum(), 1)), 4),           # 모니터 보라 칸이 실제로 라벨 없는 비율
                     "purple_recall": round(float((nearm & B_unlab & ~camd).sum() / max((nearm & B_unlab).sum(), 1)), 4)}                     # 실제 라벨 없는 칸 중 보라로 잡은 비율
print(json.dumps(rep, ensure_ascii=False, indent=1))
if a.out: json.dump(rep, open(a.out, "w"), ensure_ascii=False, indent=1)
if a.png:                                                                                       # 초록 = 둘 다 · 빨강 = 빌드에만 (모니터가 놓침) · 파랑 = 모니터에만
    m = near > 0; img = np.zeros((H, W, 3), np.uint8); img[m & B_lidar] = (60, 175, 85); img[B_lidar & ~dil(m)] = (230, 50, 50); img[m & ~dil(B_lidar)] = (60, 120, 230)
    cv2.imwrite(a.png, img[::-1, :, ::-1])
