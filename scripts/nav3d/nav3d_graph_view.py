#!/usr/bin/env python3
"""3D 내비 그래프 확인 그림 (빌드 N 단계에서 자동 생성, 포털 'Nav3d graph').

raisin_navigation3d_plugin 의 navigation3d_build_graph 가 쓴 nav3d_nodes.csv(노드별 x y z 여유 장소 연결성분)와
nav3d_graph_summary.json(장소·포털)을 읽어 층마다 윗면도 한 칸 + 전체 측면도 한 칸을 그린다.
  색 = 장소(방마다 다른 색, 계단 = 빨강), 회색 = 벽에 너무 붙어 어느 장소에도 안 든 노드, 검은 표시 = 포털(D 문 · S 계단 입구 · O 트인 곳).
로봇에 넣기 전에 층이 다 잡혔는지, 계단이 이어졌는지, 천장·가구 윗면이 바닥으로 잡히지 않았는지 본다. 글자는 영어(노트북 이미지에 한글 글꼴 없음).

사용: nav3d_graph_view.py <nav3d 폴더> <out.png> [--name 세션] [--floor-nodes 1500]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PORTAL_MARK = {"door": "D", "stair_entry": "S", "opening": "O"}


def floor_levels(places, min_nodes):
    """방(계단참 포함) 중 노드가 min_nodes 이상인 것의 높이를 1 m 안쪽끼리 묶어 층 높이 목록 (낮은 것부터)."""
    zs = sorted(p["centroid"][2] for p in places if p["type"] != "stair" and p["nodes"] >= min_nodes)
    levels = []
    for z in zs:
        if levels and z - levels[-1][-1] < 1.0: levels[-1].append(z)
        else: levels.append([z])
    return [float(np.median(l)) for l in levels]


def place_colors(places):
    cmap = plt.get_cmap("tab20"); col = {}
    k = 0
    for i, p in enumerate(places):
        if p["type"] == "stair": col[i] = (0.85, 0.1, 0.1)
        elif p["type"] == "landing": col[i] = (1.0, 0.62, 0.24)
        else: col[i] = cmap(k % 20)[:3]; k += 1
    return col


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder"); ap.add_argument("out"); ap.add_argument("--name", default=None); ap.add_argument("--floor-nodes", type=int, default=1500)
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.folder, "nav3d_graph_summary.json")))
    N = np.genfromtxt(os.path.join(a.folder, "nav3d_nodes.csv"), delimiter=",", names=True)
    x, y, z, place = N["x"], N["y"], N["z"], N["place"].astype(int)
    places, portals = S["places"], S["portals"]
    col = place_colors(places)
    rgb = np.array([col.get(p, (0.75, 0.75, 0.75)) for p in range(len(places))] + [(0.75, 0.75, 0.75)])
    c = rgb[np.where(place >= 0, place, len(places))]
    levels = floor_levels(places, a.floor_nodes) or [float(np.median(z))]
    levels = levels[:8]
    # 노드 → 그 아래(0.5 m 여유) 가장 높은 층
    lv = np.searchsorted(np.array(levels) - 0.5, z, side="right") - 1
    n = len(levels) + 1; ncol = min(3, n); nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5.2 * nrow), squeeze=False)
    xlim = (x.min() - 1, x.max() + 1); ylim = (y.min() - 1, y.max() + 1)
    for k, zl in enumerate(levels):
        ax = axs[k // ncol][k % ncol]; m = lv == k
        ax.scatter(x[m], y[m], c=c[m], s=0.6, marker="s", linewidths=0)
        top = levels[k + 1] - 0.5 if k + 1 < len(levels) else np.inf
        for q in portals:
            qx, qy, qz = q["centroid"]
            if zl - 0.5 <= qz < top:
                ax.text(qx, qy, PORTAL_MARK.get(q["type"], "?"), fontsize=8, ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="black", lw=0.5))
        nroom = sum(1 for p in places if p["type"] == "room" and zl - 0.5 <= p["centroid"][2] < top)
        nst = sum(1 for p in places if p["type"] == "stair" and zl - 0.5 <= p["centroid"][2] < top)
        nld = sum(1 for p in places if p["type"] == "landing" and zl - 0.5 <= p["centroid"][2] < top)
        ax.set_title(f"level {k + 1}: z {zl:.2f} m  |  {int(m.sum()):,} nodes, {nroom} rooms, {nld} landings, {nst} stair runs", fontsize=9)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    ax = axs[(n - 1) // ncol][(n - 1) % ncol]
    span_x = x.max() - x.min(); span_y = y.max() - y.min()
    h = x if span_x >= span_y else y
    ax.scatter(h, z, c=c, s=0.4, marker="s", linewidths=0)
    for zl in levels: ax.axhline(zl, color="k", lw=0.4, ls=":")
    ax.set_title(f"side view ({'x' if span_x >= span_y else 'y'} vs z), red = stairs, orange = landings, grey = no place", fontsize=9)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    for k in range(n, nrow * ncol): axs[k // ncol][k % ncol].axis("off")
    rooms = sum(p["type"] == "room" for p in places); stairs = sum(p["type"] == "stair" for p in places); landings = sum(p["type"] == "landing" for p in places)
    fig.suptitle(f"nav3d graph {a.name or S.get('map', '')} ({S.get('source', '')}): {S['nodes']:,} nodes, {rooms} rooms, {landings} landings, {stairs} stair runs, "
                 f"{len(portals)} portals, built in {S.get('build_ms', 0) / 1000:.1f} s", fontsize=11)
    fig.tight_layout(); fig.savefig(a.out, dpi=110); plt.close(fig)
    print(f"{a.out}: {len(levels)} levels")


if __name__ == "__main__":
    main()
