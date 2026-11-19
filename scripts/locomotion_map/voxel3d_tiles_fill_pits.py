#!/usr/bin/env python3
"""voxel3d_tiles v2 → 같은 형식 사본 + 계단 구멍 채움 셀(src=4).

단차 앞이 뚫린 계단(그레이팅)은 LiDAR 가 디딤판 밑 바닥까지 찍고 디딤판 뒤쪽이 덜 잡혀서,
컨트롤러 규칙(몸통 아래 가장 높은 면)으로 읽으면 디딤판 뒤쪽 칸이 계단 밑바닥(한 단보다 깊은 구멍)으로 읽힌다.
처리 영역(계단 경사 구간, yaw 로 돌린 상자) 안에서:
  몸통 높이 = 열마다 최고 면에 RANSAC 평면(내점 폭 0.15 m — 디딤판 톱니 ±반 단, 구멍·난간은 이상치) + 0.45 m
  (1) 그 아래 가장 높은 면이 주변 5×5 최고면보다 0.25 m 넘게 낮은 칸 → 13×13 이웃 중 구멍 아닌 면의 가장 낮은 무리(+5 cm) 중앙값, 2회
  (2) 빈 칸 → 3×3 유효 칸(≥4) 중앙값, 2회
채운 높이에 셀을 추가만 한다(원래 셀은 그대로 — 컨트롤러는 몸통 아래 가장 높은 면을 읽으므로 추가 셀이 구멍 바닥을 가린다).
시뮬 검증(output/_grating_1023, 실제 수준 결함 맵 타일): 계단 칸이 깨끗한 맵과 2 cm 이내 72.8 % → 88.2 %, 15 cm 넘게 낮은 칸 15.1 % → 4.8 %, 빈 칸 11.1 % → 0.4 %.

사용: voxel3d_tiles_fill_pits.py <in_tiles> <out_tiles> --region u0 u1 v0 v1 [--yaw DEG]
  u, v = 맵 좌표를 yaw 만큼 돌린 축 (u = 계단 오르는 방향). 영역은 계단 경사 구간만(아래 바닥·위 플랫폼 제외).
"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_voxel3d_from_pcd import REC, write_tiles

DEEP, BODY = 0.25, 0.45


def read_tiles(d):
    idx = json.load(open(os.path.join(d, "index.json")))
    assert idx["format"] == "voxel3d_tiles v2", idx["format"]
    R = np.concatenate([np.fromfile(os.path.join(d, t["file"]), REC) for t in idx["tiles"]])
    return idx, R


def fit_plane(x, y, z, band=0.15, iters=500, seed=0):
    rng = np.random.default_rng(seed); A = np.stack([x, y, np.ones_like(x)], -1); best = None
    for _ in range(iters):
        i = rng.choice(len(z), 3, replace=False)
        try:
            c = np.linalg.solve(A[i], z[i])
        except np.linalg.LinAlgError:
            continue
        n = (np.abs(A @ c - z) < band).sum()
        if best is None or n > best[0]: best = (n, c)
    inl = np.abs(A @ best[1] - z) < band
    return np.linalg.lstsq(A[inl], z[inl], rcond=None)[0], float(inl.mean())


def fill_pits(P, r):
    """P (N,3) 맵 좌표 셀 높이 (처리 영역만) → 추가할 (x, y, z) (M,3), 로그 문자열들"""
    log = []
    k = np.floor(P[:, :2] / r).astype(int); kx0, ky0 = k.min(0); shape = tuple(k.max(0) - k.min(0) + 1)
    t = np.full(shape, -np.inf); np.maximum.at(t, (k[:, 0] - kx0, k[:, 1] - ky0), P[:, 2])
    ii, jj = np.nonzero(np.isfinite(t))
    c, frac = fit_plane((ii + kx0 + 0.5) * r, (jj + ky0 + 0.5) * r, t[ii, jj]); log.append(f"plane z = {c[0]:.3f} x + {c[1]:.3f} y + {c[2]:.3f} (inliers {frac:.2f})")
    gi, gj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    B = c[0] * (gi + kx0 + 0.5) * r + c[1] * (gj + ky0 + 0.5) * r + c[2] + BODY
    i, j = k[:, 0] - kx0, k[:, 1] - ky0; sel = P[:, 2] < B[i, j]
    T = np.full(shape, -9.0); np.maximum.at(T, (i[sel], j[sel]), P[sel, 2]); add = []
    for it in range(2):
        valid = T > -9; loc5 = maximum_filter(np.where(valid, T, -9), size=5); fills = []
        for a, b in zip(*np.nonzero(valid & (T < loc5 - DEEP))):
            w = T[max(a - 6, 0):a + 7, max(b - 6, 0):b + 7]; cand = w[(w > -9) & (w >= loc5[a, b] - DEEP)]
            if len(cand) >= 3: fills.append((a, b, np.median(cand[cand <= cand.min() + 0.05])))
        for a, b, z in fills: T[a, b] = z; add.append(((a + kx0 + 0.5) * r, (b + ky0 + 0.5) * r, z))
        log.append(f"pits iter {it}: {len(fills)}")
    for it in range(2):
        valid = T > -9; fills = []
        for a, b in zip(*np.nonzero(~valid)):
            w = T[max(a - 1, 0):a + 2, max(b - 1, 0):b + 2]; v = w[w > -9]
            if len(v) >= 4: fills.append((a, b, np.median(v)))
        for a, b, z in fills: T[a, b] = z; add.append(((a + kx0 + 0.5) * r, (b + ky0 + 0.5) * r, z))
        log.append(f"holes iter {it}: {len(fills)}")
    return np.array(add).reshape(-1, 3), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir"); ap.add_argument("out_dir")
    ap.add_argument("--region", type=float, nargs=4, required=True, metavar=("U0", "U1", "V0", "V1"))
    ap.add_argument("--yaw", type=float, default=0.0, help="u 축(계단 오르는 방향)의 맵 yaw [deg]")
    ap.add_argument("--maxz-rule", type=float, default=0.03, help="컨트롤러 prior_maxz_rule_threshold 와 같게")
    a = ap.parse_args()
    idx, R = read_tiles(a.in_dir); r, rz = idx["res_xy"], idx["res_z"]
    x = (R["ix"] + 0.5) * r; y = (R["iy"] + 0.5) * r
    z = np.where(R["max_z"] - R["mean_z"] > a.maxz_rule, R["max_z"], R["mean_z"]).astype(np.float64)
    yaw = np.radians(a.yaw); u = np.cos(yaw) * x + np.sin(yaw) * y; v = -np.sin(yaw) * x + np.cos(yaw) * y
    m = (u > a.region[0]) & (u < a.region[1]) & (v > a.region[2]) & (v < a.region[3])
    if m.sum() < 100: raise SystemExit(f"처리 영역 셀 {m.sum()} 개 — 영역/yaw 확인")
    A, log = fill_pits(np.stack([x[m], y[m], z[m]], -1), r); print("\n".join(log))
    I = np.stack([np.floor(A[:, 0] / r), np.floor(A[:, 1] / r), np.floor(A[:, 2] / rz)], -1).astype(np.int64)
    old = set(zip(R["ix"].tolist(), R["iy"].tolist(), R["iz"].tolist()))
    new = np.array([tuple(q) not in old for q in I.tolist()], bool); I, A = I[new], A[new]
    Iall = np.concatenate([np.stack([R["ix"], R["iy"], R["iz"]], -1).astype(np.int64), I])
    cat = lambda f, v, dt: np.concatenate([R[f].astype(dt), np.full(len(I), v, dt)])
    params = {k: v for k, v in idx.items() if k not in ("format", "record", "record_bytes", "byte_order", "index_to_world", "tile_of", "n_cells", "total_bytes", "tiles")}
    params["source_codes"] = {**idx.get("source_codes", {}), "4": "stair pit fill (voxel3d_tiles_fill_pits.py)"}
    params["derived_from"] = os.path.abspath(a.in_dir); params["fill_pits"] = {"region_uv": a.region, "yaw_deg": a.yaw, "deep": DEEP, "body": BODY, "added": int(len(I))}
    write_tiles(a.out_dir, Iall, cat("count", 0, np.int64), np.concatenate([R["mean_z"], A[:, 2]]).astype(np.float32),
                np.concatenate([R["max_z"], A[:, 2]]).astype(np.float32), cat("cls", 255, np.uint8), cat("src", 4, np.uint8),
                np.concatenate([R["logodds"].astype(np.float32), np.zeros(len(I), np.float32)]), params)
    print(f"added {len(I):,} cells (skipped {int((~new).sum())} already present) → {a.out_dir}")


if __name__ == "__main__":
    main()
