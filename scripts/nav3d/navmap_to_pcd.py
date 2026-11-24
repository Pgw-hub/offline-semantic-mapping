#!/usr/bin/env python3
"""voxel3d npz (locomotion_map/build_voxel3d.py 출력) → 3D 내비 그래프용 navigation_map.pcd (빌드 N 단계).

셀마다 점 하나: 셀 중심 xy + 셀 안 최고 z (발 딛는 면의 윗면). 0.1 m 격자에서 이웃 점유 칸이 적은 외톨이 칸(허공의 잡음 점)은 지운다.
위치추정 지도(<맵>.pcd, 0.25 m)는 스캔 정합용이라 성기고, 이 파일은 같은 좌표계에서 계단 디딤판·문틀을 그래프가 구분할 만큼 촘촘하다.
로봇 raisin_navigation3d_plugin 은 log/map/<맵>/navigation_map.pcd 가 있으면 그것으로 그래프를 만든다.

사용: navmap_to_pcd.py <in.npz> <out.pcd> [--min-scans N] [--ror VOXEL MIN_NEIGHBOURS]
"""
import argparse, os
import numpy as np


def write_xyz_pcd(path, P):
    """x y z float32 binary PCD (raisin_navigation3d_plugin TraversabilityGraph::loadPointCloudFile 가 읽는 형식)."""
    P = np.ascontiguousarray(P, dtype=np.float32)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write((f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                 f"WIDTH {len(P)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(P)}\nDATA binary\n").encode())
        f.write(P.tobytes())
    os.replace(tmp, path)


def drop_isolated(P, vox=0.1, min_nb=5):
    """vox 격자에서 26-이웃 중 점유 칸이 min_nb 개 미만인 칸의 점을 지운다."""
    k = np.floor(P / vox).astype(np.int64) + 100000
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    u, inv = np.unique(key, return_inverse=True); inv = inv.ravel()
    ku = np.c_[u >> 42, (u >> 21) & ((1 << 21) - 1), u & ((1 << 21) - 1)]
    nb = np.zeros(len(u), np.int32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0: continue
                q = ((ku[:, 0] + dx) << 42) | ((ku[:, 1] + dy) << 21) | (ku[:, 2] + dz)
                i = np.searchsorted(u, q); i[i >= len(u)] = 0
                nb += (u[i] == q)
    keep = nb[inv] >= min_nb
    return P[keep], int((nb < min_nb).sum()), int((~keep).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz"); ap.add_argument("out")
    ap.add_argument("--min-scans", type=int, default=0, help="이 수 미만의 스캔에서만 보인 셀은 버림 (0 = 전부)")
    ap.add_argument("--ror", type=float, nargs=2, default=None, metavar=("VOXEL", "MIN_NB"), help="외톨이 칸 지우기 (예: 0.1 5)")
    a = ap.parse_args()
    z = np.load(a.npz); r = float(z["res_xy"]); ijk = z["ijk"]; mz = z["max_z"]
    keep = z["n_scans"] >= a.min_scans if a.min_scans else np.ones(len(mz), bool)
    P = np.c_[(ijk[keep, 0] + 0.5) * r, (ijk[keep, 1] + 0.5) * r, mz[keep]].astype(np.float32)
    msg = ""
    if a.ror:
        P, nv, npnt = drop_isolated(P, a.ror[0], int(a.ror[1])); msg = f", 외톨이 칸 {nv:,} 개({npnt:,} 점) 제거"
    write_xyz_pcd(a.out, P)
    print(f"{a.out}: {len(P):,} 점 (셀 {r} m{msg})")
