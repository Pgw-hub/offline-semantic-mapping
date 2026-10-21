#!/usr/bin/env python3
"""그래프 평가 — 최적화에 쓰지 않은 held-out 재방문 쌍(i%5≠0)에 대해 odom 자세 vs 보정 자세로 초기화한 GICP 보정량 비교 + rerun 시각화.
  python graph_eval.py <graph_dir> [--rrd out.rrd] [--max-pairs 60]
보정량이 작을수록 자세가 서로 일관됨(지도가 덜 겹침/어긋남)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))  # 저장소 루트 (패키지 import)
import argparse, json, os, sys, time
import numpy as np, open3d as o3d
from preprocess.geometry.build_graph import register, inv, rot_deg, pcd

ap = argparse.ArgumentParser(); ap.add_argument("graph_dir"); ap.add_argument("--rrd"); ap.add_argument("--max-pairs", type=int, default=60); ap.add_argument("--radius", type=float, default=8.0); ap.add_argument("--gap", type=int, default=40)
a = ap.parse_args(); g = json.load(open(os.path.join(a.graph_dir, "graph.json"))); nodes = g["nodes"]; N = len(nodes)
To = np.array([n["T_odom"] for n in nodes]); Tc = np.array([n["T_corr"] for n in nodes])
clouds = [np.load(os.path.join(a.graph_dir, n["cloud"]))["xyz"] for n in nodes]
used = {(e["to"], e["from"]) for e in g["edges"] if e["type"] == "loop"}
pos = Tc[:, :3, 3]; tree = o3d.geometry.KDTreeFlann(pcd(pos)); pairs = []
for i in range(N):
    if i % 5 == 0: continue                       # 최적화에 쓴 후보 집합 제외
    _, idx, _ = tree.search_radius_vector_3d(pos[i], a.radius)
    js = [j for j in idx if abs(j - i) >= a.gap and (i, j) not in used]
    if js: pairs.append((i, min(js, key=lambda j: np.linalg.norm(pos[j] - pos[i]))))
rng = np.random.default_rng(0); pairs = [pairs[k] for k in sorted(rng.choice(len(pairs), min(a.max_pairs, len(pairs)), replace=False))]
res = {"odom": [], "corr": []}; t0 = time.time()
for i, j in pairs:
    for name, T in (("odom", To), ("corr", Tc)):
        init = inv(T[j]) @ T[i]; Tr, fit, rmse = register(clouds[i], clouds[j], init)
        if fit < 0.3: continue
        c = inv(init) @ Tr; res[name].append((np.linalg.norm(c[:3, 3]), rot_deg(c), fit, rmse))
print(f"held-out 재방문 쌍 {len(pairs)} [{time.time()-t0:.0f}s]  — 자세로 초기화했을 때 GICP 가 더 움직여야 한 양(작을수록 일관)")
for k, v in res.items():
    v = np.array(v); print(f"  {k:5s} n={len(v)}  이동 중앙 {np.median(v[:,0])*100:5.1f} cm  p90 {np.percentile(v[:,0],90)*100:5.1f} cm  최대 {v[:,0].max()*100:5.1f} cm | 회전 중앙 {np.median(v[:,1]):.2f}°  p90 {np.percentile(v[:,1],90):.2f}°")
json.dump({"pairs": len(pairs), **{k: {"trans_med_cm": float(np.median(np.array(v)[:, 0]) * 100), "trans_p90_cm": float(np.percentile(np.array(v)[:, 0], 90) * 100), "rot_med_deg": float(np.median(np.array(v)[:, 1]))} for k, v in res.items()}},
          open(os.path.join(a.graph_dir, "eval_heldout.json"), "w"), indent=1)
if a.rrd:
    import rerun as rr
    rr.init("graph_eval"); rr.save(a.rrd)
    def zcol(P): z = np.clip((P[:, 2] + 3) / 8, 0, 1); return (np.stack([z, 1 - np.abs(z - 0.5) * 2, 1 - z], 1) * 255).astype(np.uint8)
    for name, T, off in (("odom", To, 0.0), ("corr", Tc, 0.0)):
        allp = []
        for k in range(0, N, 2):
            P = np.asarray(pcd(clouds[k]).voxel_down_sample(0.15).points); allp.append(P @ T[k][:3, :3].T + T[k][:3, 3])
        P = np.concatenate(allp); rr.log(f"{name}/points", rr.Points3D(P, colors=zcol(P), radii=0.05))
        rr.log(f"{name}/trajectory", rr.LineStrips3D([T[:, :3, 3]], colors=[[255, 255, 255]] if name == "corr" else [[255, 128, 0]], radii=0.08))
    L = [e for e in g["edges"] if e["type"] == "loop"]
    rr.log("corr/loops", rr.LineStrips3D([[Tc[e["from"], :3, 3], Tc[e["to"], :3, 3]] for e in L], colors=[[0, 255, 0]], radii=0.05))
    print("rrd:", a.rrd)
