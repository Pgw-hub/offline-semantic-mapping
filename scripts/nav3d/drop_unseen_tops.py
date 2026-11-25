#!/usr/bin/env python3
"""메인 맵 → 3D 내비 입력 navigation_map.pcd: 위에서 한 번도 본 적 없는 '윗면 판'(천장·지붕 윗면)을 뺀다 (빌드 N 단계).

왜: nav3d 그래프는 점 위 0.03–0.75 m 가 비어 있는 면을 전부 걸을 수 있는 면으로 본다. 천장(특히 반자 위 공간·최상층 천장·지붕)은
아래에서만 찍혀서 그 위에 점이 없으니 그대로 '바닥'이 되고, 토폴로지에 방·계단으로 잡힌다. 바닥·계단 디딤판은 센서가 그보다
높은 곳에서 본다 — 서울·그레이팅 맵에서 바닥 장소는 노드의 ~100 % 가, 천장 장소는 0 % 가 위에서 관측됐다.

무엇을 빼나 (기둥 = 5 cm xy 칸, 위에서부터 한 겹씩 반복):
  기둥의 점을 z 로 정렬해 gap m 이하 간격으로 이어진 '덩어리'로 나눈다. 덩어리가
  ① 전부 위에서 본 적 없음 (그 점을 찍은 스캔 중 센서가 점보다 tau m 이상 높았던 적이 없음 — navmap_to_pcd --elev-out)
  ② 맨 위 점이 그래프가 면으로 쓸 점 (같은 기둥 위 low ~ high m 사이에 다른 점 없음 — 플러그인 clearance_low/high 기준)
  ③ 두께 max_run m 이하 (천장 판·지붕 판·반자). 벽·기둥은 세로로 길게 이어져서 남는다
  이면 덩어리째 뺀다. 빼고 나서 아래 덩어리가 새로 윗면이 되면 다음 겹에서 다시 본다 (지붕 판 아래 보·반자).
천장은 아래에서 찍히므로 바로 밑이 비어 있고(그래서 짧은 덩어리), 벽은 아래로 이어진다 — 판 모양 검사보다 성긴 지붕도 빠지고 벽은 안 깎인다.
그래프의 구멍 메우기(hole_fill_radius)가 남은 조각 사이를 다시 이어 붙이지 않도록 가장자리까지 뺀다.
슬래브 밑면처럼 바로 위(0.75 m 안)에 위층 바닥이 있는 점은 ②에서 걸러져 남는다 — 로봇의 isKnownStructure(동적 장애물 거르기)용 구조는 그대로.

사용: drop_unseen_tops.py <main_map.pcd> <elev.npy> <out.pcd> [--tau 0.2] [--max-run 0.3] [--gap 0.1] [--low 0.03] [--high 0.75] [--res 0.05] [--summary out.json]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav3d.navmap_to_pcd import write_xyz_pcd
from package.robot_map_package import read_pcd_xyz

OFF = 1 << 20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcd"); ap.add_argument("elev"); ap.add_argument("out")
    ap.add_argument("--tau", type=float, default=0.2, help="센서가 점보다 이만큼(m) 높았던 적이 있으면 '위에서 본 점'")
    ap.add_argument("--max-run", type=float, default=0.3, help="이보다 두꺼운 세로 덩어리(벽·기둥)는 안 뺀다 (m)")
    ap.add_argument("--gap", type=float, default=0.1, help="기둥에서 이 간격(m) 이하로 이어진 점은 한 덩어리")
    ap.add_argument("--low", type=float, default=0.03); ap.add_argument("--high", type=float, default=0.75)
    ap.add_argument("--res", type=float, default=0.05); ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    P = read_pcd_xyz(a.pcd); E = np.load(a.elev).astype(np.float32)
    assert len(P) == len(E), f"점 수 다름: pcd {len(P)} vs elev {len(E)} (M 단계 다시)"
    ix = np.floor(P[:, 0] / a.res).astype(np.int64); iy = np.floor(P[:, 1] / a.res).astype(np.int64)
    col = ((ix + OFF) << 21) | (iy + OFF)
    unseen = E <= a.tau
    alive = np.ones(len(P), bool); per_iter = []
    for it in range(a.max_iter):
        idx = np.flatnonzero(alive)
        o = idx[np.lexsort((P[idx, 2], col[idx]))]                     # 기둥별 z 오름차순
        c, z = col[o], P[o, 2].astype(np.float64)
        new_col = np.r_[True, c[1:] != c[:-1]]
        new_run = new_col | np.r_[True, np.diff(z) > a.gap]
        rid = np.cumsum(new_run) - 1; nr = rid[-1] + 1 if len(rid) else 0
        r_top_i = np.r_[np.flatnonzero(new_run)[1:] - 1, len(o) - 1]   # 덩어리 맨 위 (정렬 위치)
        r_bot = z[new_run]; r_top = z[r_top_i]
        r_unseen = np.bincount(rid, weights=(~unseen[o]).astype(float), minlength=nr) == 0
        # 맨 위 점 바로 위 (top+low, top+high] 에 같은 기둥의 다른 점이 있으면 면이 아님
        nxt = r_top_i + 1; has = nxt < len(o); nn = np.minimum(nxt, len(o) - 1)
        blocked = has & (c[nn] == c[r_top_i]) & (z[nn] > r_top + a.low) & (z[nn] <= r_top + a.high)
        rm_run = r_unseen & ~blocked & (r_top - r_bot <= a.max_run)
        rm = o[rm_run[rid]]
        if not len(rm): break
        alive[rm] = False; per_iter.append(int(len(rm)))
    write_xyz_pcd(a.out, P[alive])
    S = {"points_in": int(len(P)), "points_out": int(alive.sum()), "removed": int((~alive).sum()), "iterations": per_iter,
         "unseen_from_above": int(unseen.sum()), "params": {"tau": a.tau, "max_run": a.max_run, "gap": a.gap, "low": a.low, "high": a.high, "res": a.res}}
    if a.summary:
        json.dump(S, open(a.summary, "w"), indent=1)
    print(f"{a.out}: {S['points_in']:,} → {S['points_out']:,} 점 (위에서 본 적 없는 얇은 윗면 {S['removed']:,} 점 제거, {len(per_iter)} 겹)")


if __name__ == "__main__":
    main()
