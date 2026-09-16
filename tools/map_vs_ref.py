#!/usr/bin/env python3
"""지도 정합 지표 — 로봇 저장 지도(.pcd, 최종 PGO 프레임)를 기준(GT)으로 후보 지도들이 얼마나 어긋났는지.
점 수가 달라도 되는 지표만 쓴다:
  · ref→cand 최근접 거리: 기준 점마다 후보 지도에서 가장 가까운 점까지 (기준 표면이 후보에 '그 자리에' 있는가). 중앙/p90/p95, 10/20/50 cm 안 비율.
    후보가 촘촘할수록 유리하지 않다(가장 가까운 한 점만 보므로). 후보가 성기면 불리 → cand→ref 는 쓰지 않음(기준이 0.5 m 격자라 밀도 차이에 지배됨).
  · 강체 오프셋: cand→ref ICP(point-to-point, 0.5 m) 로 후보 전체가 기준 대비 얼마나 밀려/돌아가 있나(이동 cm, yaw°). 잔차(ICP 후 ref→cand 중앙)와 비교하면
    '전체가 밀린 것' 과 '국소 번짐' 을 구분할 수 있다.
  · 표면 두께: 기준 점 주변 0.3 m 후보 점들의 PCA 최소 고유값 √ 중앙 — 벽이 두 겹이면 커짐(번짐).
  · 기준 원점 기준 거리대별 ref→cand 중앙 (드리프트가 거리 따라 자라는지).
사용: python tools/map_vs_ref.py <ref.pcd> --cand "B keyframe scans=path.pcd" --cand "D track1=output/x/voxel/session_voxsem.npz" [--sub 1] [--out metrics.json]
"""
import argparse, json, os
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("--cand", action="append", required=True); ap.add_argument("--sub", type=int, default=1); ap.add_argument("--out")
ap.add_argument("--zband", type=float, nargs=2, default=None, help="기준 점 z 범위 제한 (예: 바닥 제외)")
a = ap.parse_args()
def load(p):
    if p.endswith(".npz"): v = np.load(p); return ((v["ijk"] + 0.5) * float(v["vox"])).astype(np.float64)
    return np.asarray(o3d.io.read_point_cloud(p).points).astype(np.float64)
R = load(a.ref)
if a.zband: R = R[(R[:, 2] > a.zband[0]) & (R[:, 2] < a.zband[1])]
print(f"기준(GT): {a.ref} · {len(R):,} 점 · bbox {R.min(0).round(1)}..{R.max(0).round(1)}")
tree_r = cKDTree(R); res = {}
def thickness(P, tree_c, n=3000, r=0.3, min_nb=12):
    sel = np.random.default_rng(0).choice(len(R), min(n, len(R)), replace=False); th = []
    for i in sel:
        nb = tree_c.query_ball_point(R[i], r)
        if len(nb) < min_nb: continue
        Q = P[nb] - P[nb].mean(0); w = np.linalg.eigvalsh(Q.T @ Q / len(nb)); th.append(np.sqrt(max(w[0], 0)))
    return float(np.median(th)) if th else float("nan"), len(th)
# 기준(저장 지도) 자체의 표면 두께 — 0.5 m 격자라 0.3 m 이웃이 부족하므로 0.8 m 반경도 함께 (후보들도 같은 반경으로 재측정해 비교)
th03, n03 = thickness(R, tree_r, r=0.3, min_nb=6); th08, n08 = thickness(R, tree_r, r=0.8, min_nb=8)
res["_ref_self"] = {"thickness_r0.3_cm": round(th03 * 100, 1) if th03 == th03 else None, "n_r0.3": n03, "thickness_r0.8_cm": round(th08 * 100, 1) if th08 == th08 else None, "n_r0.8": n08}
print(f"기준 자체 표면 두께: r=0.3 m {th03*100:.1f} cm (n={n03}) · r=0.8 m {th08*100:.1f} cm (n={n08})")
for c in a.cand:
    name, path = c.split("=", 1); P = load(path)[::a.sub]; tree_c = cKDTree(P)
    d = tree_c.query(R)[0]; m = {"points": int(len(P)), "ref2cand_med_cm": round(float(np.median(d)) * 100, 1), "ref2cand_p90_cm": round(float(np.percentile(d, 90)) * 100, 1), "ref2cand_p95_cm": round(float(np.percentile(d, 95)) * 100, 1),
                                   "within_10cm": round(float((d < 0.10).mean()) * 100, 1), "within_20cm": round(float((d < 0.20).mean()) * 100, 1), "within_50cm": round(float((d < 0.50).mean()) * 100, 1)}
    # 강체 오프셋: 후보(서브샘플) → 기준 ICP
    src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(P[::max(1, len(P) // 200000)]); tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(R)
    icp = o3d.pipelines.registration.registration_icp(src, tgt, 0.5, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    T = icp.transformation; yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0])); P2 = P @ T[:3, :3].T + T[:3, 3]; d2 = cKDTree(P2).query(R)[0]
    m.update({"rigid_offset_cm": round(float(np.linalg.norm(T[:3, 3])) * 100, 1), "rigid_offset_xyz_cm": np.round(T[:3, 3] * 100, 1).tolist(), "rigid_yaw_deg": round(float(yaw), 2), "after_icp_med_cm": round(float(np.median(d2)) * 100, 1), "after_icp_p90_cm": round(float(np.percentile(d2, 90)) * 100, 1)})
    th, nth = thickness(P, tree_c); m["thickness_med_cm"] = round(th * 100, 1); m["thickness_n"] = nth
    th8, n8 = thickness(P, tree_c, r=0.8, min_nb=8); m["thickness_r0.8_cm"] = round(th8 * 100, 1)
    # 원점 거리대별
    rr_ = np.linalg.norm(R[:, :2], axis=1); bins = [0, 10, 20, 40, 80, 1e9]; byr = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (rr_ >= lo) & (rr_ < hi)
        if k.sum() > 50: byr[f"{lo}-{hi if hi < 1e9 else 'inf'}m"] = round(float(np.median(d[k])) * 100, 1)
    m["ref2cand_med_by_range_cm"] = byr; res[name] = m
    print(f"\n[{name}] 후보 {len(P):,} 점")
    print(f"  ref→cand  중앙 {m['ref2cand_med_cm']} cm · p90 {m['ref2cand_p90_cm']} · p95 {m['ref2cand_p95_cm']} · 10/20/50 cm 안 {m['within_10cm']}/{m['within_20cm']}/{m['within_50cm']} %")
    print(f"  강체 오프셋(ICP) 이동 {m['rigid_offset_cm']} cm {m['rigid_offset_xyz_cm']} · yaw {m['rigid_yaw_deg']}° → ICP 후 중앙 {m['after_icp_med_cm']} cm p90 {m['after_icp_p90_cm']}")
    print(f"  표면 두께(기준점 주변) r=0.3 m {m['thickness_med_cm']} cm (n={nth}) · r=0.8 m {m['thickness_r0.8_cm']} cm · 원점 거리대별 중앙 {byr}")
if a.out: json.dump({"ref": a.ref, "ref_points": int(len(R)), "candidates": res}, open(a.out, "w"), indent=1, ensure_ascii=False); print("\n저장:", a.out)
