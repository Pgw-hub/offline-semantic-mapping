#!/usr/bin/env python3
"""초기 위치 잡기 검증 — 대략적인 (x, y, yaw) 에서 시작해 LiDAR 스캔을 Map Package 의 정적 점군에 정합.

정합용 지도 (--map-source):
  voxels      terrain_voxels(0.1m) 중심점, 제외 클래스(기본 person,car) 제거
  voxels_occ  위 + occupancy 가 free 로 본 복셀 제거 (다중 스캔 확인된 표면만)
  occupancy   occupancy occupied 셀(0.2m) 중심점, 동적 복셀만 있는 셀 제거
정합: 다중 스케일 point-to-plane ICP (Open3D). 스캔은 world 좌표 npz(lidar_scans) 를 --accum 개 이어 붙여 사용.
평가(--sweep): 정답=항등(스캔은 SLAM world 좌표) → 초기 오차(반경 r, 각도 dyaw)를 주고 정합 → 남은 오차로 성공 판정 → 수렴 반경 표.
단일 실행(--init x y yaw): 로봇 프레임 스캔(--scan-frame robot)이나 world 스캔을 초기 자세로 놓고 정합 → rrd.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # 저장소 루트 (패키지 import)
import argparse, glob, json, math, os, sys, time
import numpy as np
import open3d as o3d
from package.load_map import MapPackage
from common import classdict as CD


def yaw_T(x, y, yaw, z=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], float)


def pose_err(T):
    """항등 대비 오차: 평면 거리(m), yaw(deg)"""
    return float(np.hypot(T[0, 3], T[1, 3])), float(abs(math.degrees(math.atan2(T[1, 0], T[0, 0]))))


def map_points(mp, source="voxels_occ", exclude=("person", "car")):
    """정합용 정적 점군 (N,3) + 설명"""
    names = mp.classes; lab = mp.labels(); n_cls = len(names)
    ex_ids = [names.index(c) for c in exclude if c in names]
    keep = ~np.isin(lab, ex_ids)                                   # 제외 클래스 (unknown/other 는 유지 — 기하는 정적일 확률 높음)
    if source == "voxels":
        pts = mp.voxel_centers()[keep]
    elif source == "voxels_occ":
        if mp.occ is None: raise RuntimeError("occupancy 레이어 없음")
        keep &= (mp.surf_state != 0) & ~mp.surf_removed             # occupancy free 판정·동적 제거 복셀 제외
        pts = mp.voxel_centers()[keep]
    elif source == "occupancy":
        if mp.occ is None: raise RuntimeError("occupancy 레이어 없음")
        O = mp.occ; R = O["res"]; occ_c = (O["ijk"] + 0.5) * R; occm = O["state"] == 1
        # 제외 클래스 복셀만 들어 있는 셀 찾기: 셀 키별 (전체 복셀 수, 제외 복셀 수)
        K = np.floor(mp.voxel_centers() / R).astype(np.int64); key = lambda k: (k[:, 0] + (1 << 20)) * (1 << 42) // (1 << 21) * 0 + ((k[:, 0] + (1 << 20)) << 42 | (k[:, 1] + (1 << 20)) << 21 | (k[:, 2] + (1 << 20)))
        kv = key(K); uk, inv = np.unique(kv, return_inverse=True); inv = inv.ravel()
        tot = np.bincount(inv, minlength=len(uk)); exc = np.bincount(inv, weights=(~keep).astype(float), minlength=len(uk))
        bad = uk[(exc >= 0.8 * tot) & (tot > 0)]
        ko = key(O["ijk"].astype(np.int64))
        pts = occ_c[occm & ~np.isin(ko, bad)]
    else:
        raise ValueError(source)
    return pts.astype(np.float64)


def load_scans(files, accum=1, start=0):
    """world 좌표 스캔 npz 여러 개 이어붙임 → (pts, origin)"""
    P, O = [], []
    for f in files[start:start + accum]:
        d = np.load(f); P.append(d["xyz"].astype(np.float64)); O.append(d["origin"].astype(np.float64))
    return np.concatenate(P), np.mean(O, 0)


class Localizer:
    def __init__(self, map_pts, voxel=0.1):
        self.pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(map_pts))
        self.levels = []                                          # (다운샘플 지도, 대응거리)
        for vs, dist in ((0.4, 2.5), (0.2, 1.0), (0.1, 0.4)):
            m = self.pcd.voxel_down_sample(vs) if vs > voxel else self.pcd
            m.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(3 * vs, 0.5), max_nn=30))
            self.levels.append((m, dist, vs))
        self.tree = o3d.geometry.KDTreeFlann(self.levels[-1][0])

    def register(self, scan_pts, T_init, max_iter=(40, 40, 60)):
        """scan_pts: 로봇(또는 임의) 프레임 점, T_init: 그 프레임 → world 초기 추정. 반환 (T, fitness, rmse)"""
        T = T_init.copy(); src_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts))
        for (m, dist, vs), it in zip(self.levels, max_iter):
            src = src_full.voxel_down_sample(vs)
            res = o3d.pipelines.registration.registration_icp(src, m, dist, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
            T = res.transformation
        return np.asarray(T), float(res.fitness), float(res.inlier_rmse)

    def register_multi_yaw(self, scan_pts, T_init, yaw_offsets_deg=(0, -30, 30, -60, 60, -90, 90, 180)):
        best = None
        for d in yaw_offsets_deg:
            Ti = T_init @ yaw_T(0, 0, math.radians(d))
            T, fit, rmse = self.register(scan_pts, Ti)
            score = fit - 0.5 * rmse                              # 대응 비율 우선, 잔차로 보정
            if best is None or score > best[3]: best = (T, fit, rmse, score, d)
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--map-source", default="voxels_occ", choices=["voxels", "voxels_occ", "occupancy"])
    ap.add_argument("--exclude", default="person,car"); ap.add_argument("--scans", default="/data2/cg/data/raibo", help="lidar_scans 루트 (…/log_XXX/lidar_scans)")
    ap.add_argument("--logs", default="6,7,8,9,10"); ap.add_argument("--accum", type=int, default=5, help="이어붙일 스캔 수 (10Hz → 0.5s)")
    ap.add_argument("--n-scans", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-overlap", type=float, default=0.5, help="정답 자세에서 스캔 점 중 0.3m 안에 지도 점이 있는 비율 최소값 (지도 커버리지 필터)")
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--radii", default="0,0.5,1,2,3,5"); ap.add_argument("--yaws", default="0,10,20,30,45,60,90")
    ap.add_argument("--multi-yaw", action="store_true", help="yaw 다중 시작 사용")
    ap.add_argument("--succ-m", type=float, default=0.3); ap.add_argument("--succ-deg", type=float, default=3.0)
    ap.add_argument("--init", nargs=3, type=float, default=None, metavar=("X", "Y", "YAW_DEG"), help="단일 실행 초기 자세")
    ap.add_argument("--scan-file", default=None); ap.add_argument("--out", default=None); ap.add_argument("--rrd", default=None)
    a = ap.parse_args(); t0 = time.time()
    mp = MapPackage(a.manifest); ex = tuple(x for x in a.exclude.split(",") if x)
    pts = map_points(mp, a.map_source, ex)
    print(f"지도 점군 [{a.map_source}, 제외 {ex}]: {len(pts):,} 점  (로드 {time.time()-t0:.1f}s)")
    t1 = time.time(); loc = Localizer(pts); print(f"kd-tree/normal 준비 {time.time()-t1:.1f}s")
    bb_lo, bb_hi = pts.min(0), pts.max(0)

    # ---- 스캔 선택: 보류 로그 중 지도 범위 안(원점 주변 20m 안에 지도 점 충분) ----
    rng = np.random.default_rng(a.seed); cand = []
    for lg in [int(x) for x in a.logs.split(",")]:
        files = sorted(glob.glob(os.path.join(a.scans, f"log_{lg:03d}", "lidar_scans", "*.npz")))
        for i in range(0, len(files) - a.accum, 20):
            org = np.load(files[i])["origin"]
            if np.all(org[:2] > bb_lo[:2] + 5) and np.all(org[:2] < bb_hi[:2] - 5):
                k, idx, _ = loc.tree.search_radius_vector_3d(org.astype(np.float64), 20.0)
                if k > 3000:
                    P, _ = load_scans(files, a.accum, i); P = P[:: max(1, len(P) // 3000)]
                    near = sum(1 for q in P if loc.tree.search_radius_vector_3d(q, 0.3)[0] > 0); ov = near / len(P)
                    if ov >= a.min_overlap: cand.append((files, i, lg, ov))
    print(f"지도 범위 안·커버리지 ≥{a.min_overlap:.0%} 후보 스캔 {len(cand)}개" + (f" (overlap 중앙값 {np.median([c[3] for c in cand]):.2f})" if cand else ""))
    if a.init is not None:
        files, i, lg, _ov = cand[0] if not a.scan_file else (sorted(glob.glob(os.path.join(os.path.dirname(a.scan_file), "*.npz"))), sorted(glob.glob(os.path.join(os.path.dirname(a.scan_file), "*.npz"))).index(a.scan_file), -1, 1.0)
        P, org = load_scans(files, a.accum, i)
        T_init = yaw_T(a.init[0], a.init[1], math.radians(a.init[2]), z=org[2])
        # world 스캔을 로봇 프레임처럼 만들기: 원점 기준 평행이동만 (yaw 는 정답 0 → 초기 yaw 오차 = 준 값)
        P_local = P - org
        T, fit, rmse = (loc.register_multi_yaw(P_local, T_init)[:3] if a.multi_yaw else loc.register(P_local, T_init))
        err_m, err_deg = pose_err(np.linalg.inv(yaw_T(org[0], org[1], 0, org[2])) @ T)
        print(f"단일 정합: init ({a.init[0]:.1f},{a.init[1]:.1f},{a.init[2]:.0f}°) → 오차 {err_m:.2f} m / {err_deg:.1f}° · fitness {fit:.2f} rmse {rmse:.3f}")
        if a.rrd: write_rrd(a.rrd, mp, pts, P_local, T_init, T, yaw_T(org[0], org[1], 0, org[2]))
        return
    if not a.sweep: return
    picks = [cand[j] for j in rng.choice(len(cand), min(a.n_scans, len(cand)), replace=False)]
    radii = [float(x) for x in a.radii.split(",")]; yaws = [float(x) for x in a.yaws.split(",")]
    table = {}; runs = []; t2 = time.time()
    for (files, i, lg, ov) in picks:
        P, org = load_scans(files, a.accum, i); P_local = P - org; T_gt = yaw_T(org[0], org[1], 0, org[2])
        for r in radii:
            for yd in yaws:
                ang = rng.uniform(0, 2 * math.pi); sgn = rng.choice([-1, 1])
                T_init = T_gt @ yaw_T(r * math.cos(ang), r * math.sin(ang), math.radians(sgn * yd))
                ts = time.time()
                T, fit, rmse = (loc.register_multi_yaw(P_local, T_init)[:3] if a.multi_yaw else loc.register(P_local, T_init))
                em, ed = pose_err(np.linalg.inv(T_gt) @ T); ok = em < a.succ_m and ed < a.succ_deg
                table.setdefault((r, yd), []).append(ok); runs.append({"log": lg, "scan": i, "overlap": round(ov, 2), "r": r, "yaw": yd, "err_m": round(em, 3), "err_deg": round(ed, 2), "fitness": round(fit, 3), "rmse": round(rmse, 4), "ok": bool(ok), "ms": round((time.time() - ts) * 1000)})
        okr = [x["ok"] for x in runs if x["log"] == lg and x["scan"] == i]
        print(f"  log_{lg:03d} scan {i} (overlap {ov:.2f}): 성공 {sum(okr)}/{len(okr)} ({time.time()-t2:.0f}s)", flush=True)
    print(f"\n수렴 성공률 [{a.map_source}] (성공 = 오차 <{a.succ_m} m & <{a.succ_deg}°), 스캔 {len(picks)}개 × 반경 × yaw")
    print("r(m)\\yaw " + " ".join(f"{y:>5.0f}°" for y in yaws))
    grid = {}
    for r in radii:
        row = []
        for yd in yaws:
            v = table[(r, yd)]; p = 100 * sum(v) / len(v); row.append(p)
        grid[r] = row; print(f"{r:>7.1f}  " + " ".join(f"{p:>5.0f}%" for p in row))
    oks = [x for x in runs if x["ok"]]
    print(f"성공 시 잔여 오차 중앙값: {np.median([x['err_m'] for x in oks]):.3f} m / {np.median([x['err_deg'] for x in oks]):.2f}° · 정합 시간 중앙값 {np.median([x['ms'] for x in runs]):.0f} ms")
    if a.out:
        json.dump({"map_source": a.map_source, "exclude": ex, "map_points": int(len(pts)), "accum": a.accum, "n_scans": len(picks), "radii": radii, "yaws": yaws,
                   "success_grid_pct": {str(r): grid[r] for r in radii}, "runs": runs, "multi_yaw": a.multi_yaw}, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("저장:", a.out)


def write_rrd(path, mp, map_pts, P_local, T_init, T_est, T_gt):
    import rerun as rr, rerun.blueprint as rrb
    rr.init("localize"); rr.save(path)
    sub = max(1, len(map_pts) // 800000)
    rr.log("loc/map", rr.Points3D(map_pts[::sub], colors=[[150, 150, 150]], radii=0.04))
    tf = lambda T: (P_local @ T[:3, :3].T + T[:3, 3])
    rr.log("loc/scan_init", rr.Points3D(tf(T_init), colors=[[255, 120, 40]], radii=0.06))
    rr.log("loc/scan_registered", rr.Points3D(tf(T_est), colors=[[40, 200, 255]], radii=0.06))
    rr.log("loc/scan_gt", rr.Points3D(tf(T_gt), colors=[[80, 255, 80]], radii=0.05))
    for nm, T, col in (("init", T_init, [255, 120, 40]), ("registered", T_est, [40, 200, 255]), ("gt", T_gt, [80, 255, 80])):
        rr.log(f"loc/pose_{nm}", rr.Arrows3D(origins=[T[:3, 3]], vectors=[T[:3, 0] * 2.0], colors=[col], radii=0.08, labels=[nm]))
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="/loc", name="localize"), rrb.TimePanel(expanded=False)), make_default=True)
    print("rrd:", path)


if __name__ == "__main__":
    main()
