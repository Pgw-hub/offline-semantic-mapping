#!/usr/bin/env python3
"""전역 초기화 벤치 — 지도 × 질의 스캔(raw / deskew) 조건별로 held-out 스캔을 정답 주변 (r, dyaw) 오차에서 정합해 잔차 통계.
  python geometry/loc_bench.py --map old=output/A/compiled/map_manifest.json --scans raw=output/A/dataset/log_003/lidar_scans --scans dsk=output/B/dataset/log_003/lidar_scans
     [--source voxels_occ --accum 5 --stride 10 --max-scans 25 --r 1.5 --dyaw 20 --inits 2]
질의 스캔은 localize_server 와 같은 방식(누적 5, 원점 평균, world 방향 그대로) → 정답 yaw = 0. 출력: 조건별 위치/yaw 잔차 중앙·p90, 성공률(<0.3m·<3°), json 저장.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))  # scripts/ (패키지 import)
import argparse, glob, json, math, os, sys, time
import numpy as np, open3d as o3d
CODE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); REPO = os.path.dirname(CODE); sys.path.insert(0, CODE)
from package.load_map import MapPackage
from apps.localize import map_points, load_scans, Localizer, yaw_T, pose_err

ap = argparse.ArgumentParser(); ap.add_argument("--map", action="append", required=True); ap.add_argument("--scans", action="append", required=True)
ap.add_argument("--source", default="voxels_occ"); ap.add_argument("--accum", type=int, default=5); ap.add_argument("--stride", type=int, default=10); ap.add_argument("--max-scans", type=int, default=25)
ap.add_argument("--r", type=float, default=1.5); ap.add_argument("--dyaw", type=float, default=20.0); ap.add_argument("--inits", type=int, default=2); ap.add_argument("--out")
a = ap.parse_args(); rng = np.random.default_rng(0)
maps = dict(m.split("=", 1) for m in a.map); scans = dict(s.split("=", 1) for s in a.scans); results = {}
for mname, man in maps.items():
    t0 = time.time(); mp = MapPackage(os.path.join(REPO, man) if not os.path.isabs(man) else man); pts = map_points(mp, a.source); desc = a.source
    loc = Localizer(pts); tree = loc.tree; print(f"[{mname}] {desc} · {len(pts):,} pts · Localizer {time.time()-t0:.0f}s", flush=True)
    for sname, sdir in scans.items():
        files = sorted(glob.glob(os.path.join(REPO if not os.path.isabs(sdir) else "/", sdir, "*.npz"))); rows = []
        for i in range(0, len(files) - a.accum, a.stride):
            P, org = load_scans(files, a.accum, i); Q = P[:: max(1, len(P) // 1500)]
            ov = sum(1 for q in Q if tree.search_radius_vector_3d(q, 0.3)[0] > 0) / len(Q)
            if ov < 0.4: continue
            P_local = P - org; T_gt = yaw_T(org[0], org[1], 0, org[2])
            for k in range(a.inits):
                th = rng.uniform(0, 2 * math.pi); s = 1 if k % 2 == 0 else -1
                T_init = yaw_T(org[0] + a.r * math.cos(th), org[1] + a.r * math.sin(th), math.radians(s * a.dyaw), org[2])
                T, fit, rmse = loc.register(P_local, T_init); e_pos, e_yaw = pose_err(np.linalg.inv(T_gt) @ T)
                rows.append({"scan": i, "overlap": round(ov, 2), "pos_m": e_pos, "yaw_deg": e_yaw, "fitness": fit, "rmse": rmse})
            if len(rows) >= a.max_scans * a.inits: break
        E = np.array([[r["pos_m"], r["yaw_deg"]] for r in rows]); ok = ((E[:, 0] < 0.3) & (E[:, 1] < 3)).mean() if len(E) else 0
        summ = {"n": len(rows), "pos_med_cm": float(np.median(E[:, 0]) * 100), "pos_p90_cm": float(np.percentile(E[:, 0], 90) * 100), "yaw_med_deg": float(np.median(E[:, 1])), "yaw_p90_deg": float(np.percentile(E[:, 1], 90)), "yaw_max_deg": float(E[:, 1].max()), "success": float(ok)}
        results[f"{mname}|{sname}"] = {"summary": summ, "rows": rows}
        print(f"  map={mname:8s} query={sname:6s} n={summ['n']:3d}  pos {summ['pos_med_cm']:5.1f}/{summ['pos_p90_cm']:5.1f} cm (med/p90)  yaw {summ['yaw_med_deg']:.2f}/{summ['yaw_p90_deg']:.2f}° (max {summ['yaw_max_deg']:.2f})  성공 {ok*100:.0f}%", flush=True)
if a.out: json.dump(results, open(a.out, "w"), indent=1); print("저장:", a.out)
