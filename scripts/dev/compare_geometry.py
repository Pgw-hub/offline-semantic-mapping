#!/usr/bin/env python3
"""두 빌드의 기하 비교 — 세션의 lidar_scans (단계 S 산출물, deskew+자세 적용된 world 점) 로 번짐·일관성 지표를 계산하고, occupancy 차이를 센다.
  python tools/compare_geometry.py <root_A> <root_B> [--label-a old --label-b new] [--sub 4] [--out compare_geometry.json]
지표 (geom_ab_test.py 와 동일 정의)
  cells_5cm / cells_10cm   누적 점(같은 서브샘플)의 점유 셀 수 — 같은 점 수에서 셀이 많을수록 번짐
  scan_consistency         스캔 i 점 → 나머지 스캔 최근접 거리 중앙/p90 (재방문 정합)
  thickness                이웃 0.3 m PCA 최소 고유값 √ 의 중앙/p90 (벽·바닥 두께)
  occupancy                compiled/session_occupancy.npz 의 occupied 셀 수, 한쪽에만 있는 셀 수
+ 벽 단면 확대 PNG (점 밀도 최대 지점 3×3 m, 1 cm) A | B
"""
import argparse, glob, json, os
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--label-a", default="old"); ap.add_argument("--label-b", default="new")
ap.add_argument("--sub", type=int, default=4); ap.add_argument("--out"); ap.add_argument("--zoom-center", help="x,y (m) 확대 중심; 생략하면 B 점 밀도 최대 지점")
a = ap.parse_args()

def scans_of(root): return sorted(glob.glob(os.path.join(root, "dataset", "log_*", "lidar_scans", "*.npz")))
def load(files, sub): return [np.load(f)["xyz"][::sub].astype(np.float64) for f in files]
def cells(P, v): return int(len(np.unique(np.floor(P / v).astype(np.int64), axis=0)))
def consistency(pts):
    allp = np.concatenate(pts); idx = np.repeat(np.arange(len(pts)), [len(p) for p in pts]); tree = cKDTree(allp); med, p90 = [], []
    for k in range(0, len(pts), max(1, len(pts) // 80)):
        if len(pts[k]) < 50: continue
        d, j = tree.query(pts[k][::4], k=8); own = idx[j] == k; d = np.where(own, np.inf, d).min(1); d = d[np.isfinite(d)]
        if len(d): med.append(np.median(d)); p90.append(np.percentile(d, 90))
    return float(np.median(med)), float(np.median(p90))
def thickness(P, n=5000, r=0.3):
    tree = cKDTree(P); sel = np.random.default_rng(0).choice(len(P), min(n, len(P)), replace=False); th = []
    for i in sel:
        nb = tree.query_ball_point(P[i], r)
        if len(nb) < 12: continue
        Q = P[nb] - P[nb].mean(0); w = np.linalg.eigvalsh(Q.T @ Q / len(nb)); th.append(np.sqrt(max(w[0], 0)))
    th = np.array(th); return float(np.median(th)), float(np.percentile(th, 90))
def occ_keys(root):
    f = os.path.join(root, "compiled", "session_occupancy.npz")
    if not os.path.exists(f): return None, None
    o = np.load(f); I = o["ijk"][o["state"] == 1].astype(np.int64) + (1 << 20); return (I[:, 0] << 42) | (I[:, 1] << 21) | I[:, 2], float(o["res"])

R = {}; clouds = {}
for lab, root in ((a.label_a, a.a), (a.label_b, a.b)):
    fs = scans_of(root); pts = load(fs, a.sub); P = np.concatenate(pts); clouds[lab] = P
    cm, c9 = consistency(pts); tm, t9 = thickness(P)
    R[lab] = {"root": root, "scans": len(fs), "points_sub": int(len(P)), "cells_5cm": cells(P, 0.05), "cells_10cm": cells(P, 0.10),
              "scan_consistency_med_cm": round(cm * 100, 2), "scan_consistency_p90_cm": round(c9 * 100, 2), "thickness_med_cm": round(tm * 100, 2), "thickness_p90_cm": round(t9 * 100, 2)}
    k, res = occ_keys(root)
    if k is not None: R[lab]["occupied_cells"] = int(len(k)); R[lab]["occ_res"] = res; R[lab]["_keys"] = k
    print(f"{lab:8s} 스캔 {len(fs)} · 점 {len(P):,} · 셀 5cm {R[lab]['cells_5cm']:,} / 10cm {R[lab]['cells_10cm']:,} · 스캔 일관성 {cm*100:.2f}/{c9*100:.2f} cm · 두께 {tm*100:.2f}/{t9*100:.2f} cm" + (f" · occupied {R[lab]['occupied_cells']:,}" if k is not None else ""))
ka, kb = R[a.label_a].pop("_keys", None), R[a.label_b].pop("_keys", None)
if ka is not None and kb is not None:
    common = len(np.intersect1d(ka, kb)); R["occupancy_diff"] = {"common": int(common), f"only_{a.label_a}": int(len(ka) - common), f"only_{a.label_b}": int(len(kb) - common)}
    print(f"occupancy: 공통 {common:,} · {a.label_a}만 {len(ka)-common:,} · {a.label_b}만 {len(kb)-common:,}")
A, B = clouds[a.label_a], clouds[a.label_b]
d = cKDTree(B).query(A[::10])[0]; R["A_to_B_nn_cm"] = {"median": round(float(np.median(d)) * 100, 2), "p90": round(float(np.percentile(d, 90)) * 100, 2)}
print(f"{a.label_a}→{a.label_b} 최근접 거리 중앙 {np.median(d)*100:.1f} cm p90 {np.percentile(d,90)*100:.1f} cm")
# 벽 단면 확대 PNG
try:
    import cv2
    allp = np.concatenate([A, B]); zlo, zhi = np.percentile(allp[:, 2], 5), np.percentile(allp[:, 2], 95); band = lambda P: P[(P[:, 2] > zlo + 0.3 * (zhi - zlo)) & (P[:, 2] < zlo + 0.9 * (zhi - zlo))]
    def bev(P, x0, y0, x1, y1, res):
        W, H = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res)); u = ((P[:, 0] - x0) / res).astype(int); v = ((y1 - P[:, 1]) / res).astype(int)
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H); img = np.zeros((H, W), np.int32); np.add.at(img, (v[m], u[m]), 1); return img
    gray = lambda img: (255 - np.log1p(img) / max(np.log1p(img).max(), 1e-6) * 255).astype(np.uint8)
    if a.zoom_center: cx, cy = map(float, a.zoom_center.split(","))
    else:
        x0, y0 = allp[:, :2].min(0) - 1; x1, y1 = allp[:, :2].max(0) + 1; IB = bev(band(B), x0, y0, x1, y1, 0.2); vm = np.unravel_index(np.argmax(IB), IB.shape); cx, cy = x0 + vm[1] * 0.2, y1 - vm[0] * 0.2
    tiles = [gray(bev(band(P), cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5, 0.01)) for P in (A, B)]; sep = np.full((tiles[0].shape[0], 6), 128, np.uint8)
    out_png = (a.out or "compare_geometry.json").replace(".json", "_wall_zoom.png"); cv2.imwrite(out_png, np.concatenate([tiles[0], sep, tiles[1]], 1)); R["wall_zoom_png"] = out_png; R["wall_zoom_center"] = [round(cx, 2), round(cy, 2)]
    print(f"벽 단면 확대 ({cx:.1f},{cy:.1f}) → {out_png}")
except Exception as e: print("PNG 생략:", e)
if a.out: json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False); print("저장:", a.out)
