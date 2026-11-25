#!/usr/bin/env python3
"""빌드 결과(메인 맵 + 보행 맵) → 포털 카드용 compiled/summary.json + compiled/bev.png (빌드 H 단계).

summary.json 은 옛 키(voxels · voxel_labels · objects · edges · classdict · build_seconds_total)를 메인 맵 기준으로 채워 포털이 그대로 읽고,
새 키 main_map · locomotion · nav3d · traversable_m2 · outputs(출력별 파일 형식·크기)를 더한다.
bev.png = 메인 맵 윗면도 (의미 색, 기하 전용이면 높이 색).

사용: map_summary.py <빌드 폴더> [--build-seconds N]
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # scripts/
import argparse, json, os
import numpy as np
from common import classdict as CD
from package.robot_map_package import read_pcd_xyz


def jload(p, default=None):
    try: return json.load(open(p))
    except Exception: return default


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--build-seconds", type=float, default=None); a = ap.parse_args()
    comp = os.path.join(a.root, "compiled"); man = jload(os.path.join(comp, "map_manifest.json"), {}) or {}
    S = {"layout": "main_map + locomotion_map"}
    if a.build_seconds is not None: S["build_seconds_total"] = int(a.build_seconds)
    # 메인 맵 기하·의미
    mp = os.path.join(comp, "main_map", "main_map.pcd"); sem = os.path.join(comp, "main_map", "main_map_semantic.npz")
    P = lab = None; classes = []; cdict = man.get("classdict")
    if os.path.exists(sem):
        z = np.load(sem, allow_pickle=True); P = z["xyz"].astype(np.float64); lab = z["label"]; classes = [str(c) for c in z["classes"]]
        if "classdict" in z.files and not cdict: cdict = json.loads(str(z["classdict"]))
    elif os.path.exists(mp):
        P = read_pcd_xyz(mp)
    ms = jload(os.path.join(comp, "main_map", "main_map_summary.json"), {}) or {}
    lj = jload(os.path.join(comp, "main_map", "localization_map.pcd.json"), {}) or {}
    if P is not None:
        S["voxels"] = int(len(P))
        S["main_map"] = {"points": int(len(P)), "res_m": (ms.get("params") or {}).get("res_xy"), "max_range_m": (ms.get("params") or {}).get("max_range"),
                         "free_range_m": (ms.get("params") or {}).get("free_range"), "removed_prob_free": ms.get("removed_prob_free"), "seconds": ms.get("seconds"),
                         "bbox": [P.min(0).round(2).tolist(), P.max(0).round(2).tolist()], "localization_points": lj.get("points"), "localization_voxel_m": lj.get("voxel_m")}
    if lab is not None:
        names = {i: c for i, c in enumerate(classes)}; names[254] = "other"; names[255] = "unknown(LiDAR only)"
        u, n = np.unique(lab, return_counts=True); S["voxel_labels"] = {names.get(int(k), str(k)): int(c) for k, c in zip(u, n)}
    if cdict: S["classdict"] = cdict
    # 3D scene graph
    obj = jload(os.path.join(comp, "session_objects.json"))
    if obj is not None:
        S["objects"] = len(obj)
        S["edges"] = {k: len(jload(os.path.join(comp, f), []) or []) for k, f in (("heuristic", "session_edges.json"), ("object_terrain", "voxel_session_relations.json"), ("vlm", "session_edges_vlm.json"))}
    # 통행성 그래프·토폴로지
    nv = jload(os.path.join(comp, "nav3d", "nav3d_graph_summary.json"))
    if nv:
        pl = nv.get("places", [])
        S["nav3d"] = {"nodes": nv.get("nodes"), "rooms": sum(p["type"] == "room" for p in pl), "stairs": sum(p["type"] == "stair" for p in pl), "portals": len(nv.get("portals", [])),
                      "build_ms": nv.get("build_ms"), "load_ms": nv.get("load_ms")}
        S["traversable_m2"] = round(sum(p.get("nodes", 0) for p in pl) * 0.01)       # 방·계단에 든 노드(0.1 m 칸) — 벽 윗면·좁은 틈 노드 제외
    # 보행 맵
    li = jload(os.path.join(comp, "locomotion", "tiles", "index.json"))
    if li:
        st = jload(os.path.join(comp, "locomotion", "tiles_stair.json"), {}) or {}
        bs = jload(os.path.join(comp, "locomotion", "base_summary.json"), {}) or {}
        S["locomotion"] = {"cells": li.get("n_cells"), "tiles": len(li.get("tiles", [])), "mb": round((li.get("total_bytes") or 0) / 1e6, 1), "res_m": li.get("res_xy"),
                           "max_range_m": li.get("max_range"), "base_cells": bs.get("cells"), "removed_prob_free": bs.get("removed_prob_free"), "seconds": bs.get("seconds"),
                           "stair": ({"flights": len(st.get("stairs") or [st]), "treads": sum(len(m["treads"]) for m in (st.get("stairs") or [st])),
                                      "rise_cm": round(float(np.median([m["rise"] for m in (st.get("stairs") or [st])])) * 100, 1),
                                      "run_cm": round(float(np.median([m["run"] for m in (st.get("stairs") or [st])])) * 100, 1),
                                      "added": (li.get("stair_complete") or {}).get("added"), "removed_halo": (li.get("stair_complete") or {}).get("removed"),
                                      "per_flight": [{"treads": len(m["treads"]), "z": [round(m["treads"][0]["h"], 2), round(m["treads"][-1]["h"], 2)], "rise_cm": round(m["rise"] * 100, 1),
                                                      "run_cm": round(m["run"] * 100, 1), "width_cm": round((m["l1"] - m["l0"]) * 100)} for m in (st.get("stairs") or [st])]}
                                     if st.get("treads") else None)}
    # 출력별 파일 형식·크기 (매니페스트)
    F = man.get("files", {})
    S["outputs"] = {g: {k: {"path": F[k]["path"], "mb": round(F[k].get("dir_bytes", F[k]["bytes"]) / 1e6, 2), "desc": F[k]["desc"]} for k in keys if k in F}
                    for g, keys in (man.get("maps") or {}).items()}
    json.dump(S, open(os.path.join(comp, "summary.json"), "w"), ensure_ascii=False, indent=1)
    # BEV: 메인 맵 윗면 (칸마다 가장 높은 점의 색 — 의미 색, 없으면 높이)
    if P is not None and len(P):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        r = 0.1; ij = np.floor(P[:, :2] / r).astype(np.int64); ij -= ij.min(0); W, H = ij.max(0) + 1
        o = np.lexsort((P[:, 2], ij[:, 1], ij[:, 0])); key = ij[o, 0] * H + ij[o, 1]; last = np.r_[key[1:] != key[:-1], True]; sel = o[last]
        img = np.full((H, W, 3), 245, np.uint8)
        if lab is not None and cdict:
            lut = np.full((256, 3), 160, np.uint8)
            for i, c in enumerate(cdict["classes"]): lut[i] = CD.hex2rgb(c["color"])
            lut[254] = CD.hex2rgb(CD.OTHER_COLOR); lut[255] = CD.hex2rgb(CD.UNKNOWN_COLOR); col = lut[lab[sel]]
        else:
            zz = P[sel, 2]; lo, hi = np.percentile(zz, [2, 98]); col = (plt.get_cmap("viridis")((zz - lo) / max(hi - lo, 1e-6))[:, :3] * 255).astype(np.uint8)
        img[ij[sel, 1], ij[sel, 0]] = col
        img = img[::-1]
        s = 900 / max(W, H); plt.figure(figsize=(max(W * s, 200) / 100, max(H * s, 200) / 100), dpi=100); plt.imshow(img, interpolation="nearest"); plt.axis("off")
        plt.savefig(os.path.join(comp, "bev.png"), bbox_inches="tight", pad_inches=0); plt.close()
    print(json.dumps({k: v for k, v in S.items() if k not in ("classdict", "outputs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
