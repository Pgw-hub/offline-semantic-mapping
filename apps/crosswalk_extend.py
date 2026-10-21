#!/usr/bin/env python3
"""횡단보도 연장 후처리 — 줄무늬 끝과 연석 사이의 좁은 '차도' 띠를 횡단보도로 바꾼다.
코스트 격자(0.2 m, ROS 코스트값)에서:
  1. crosswalk 셀 마스크 → closing(0.6 m) → 연결 요소 → 작은 것(<min_area) 버림
  2. 덩어리마다 최소 외접 회전 사각형(minAreaRect)
  3. 두 축 각각 양 끝 바깥을 probe(기본 2 m)까지 띠 단위로 탐침 → 띠에서 '보행 지면'(인도·횡단보도·라벨 없는 지면) 비율 ≥ 0.5 가 처음 나오는 거리
     양 끝 모두 인도가 나오는 축 = 건너는 축 (둘 다면 더 가까운 쪽, 없으면 그 덩어리는 건드리지 않음)
  4. 건너는 축의 각 끝에서 인도가 나오는 거리까지(≤ max_ext) 띠 안의 차도 셀만 횡단보도로 바꿈. max_ext 안에 인도가 없으면 그 끝은 그대로
  (옵션) fill_interior: 사각형 내부(가장자리 1셀 안쪽)의 차도 셀도 횡단보도로
사용(모듈): extend(cost, res, c_cross, c_side, c_road, walk_costs=(…)) → (new_cost, report)
사용(CLI):  python tools/crosswalk_extend.py <in_costmap.npz> <out_costmap.npz> --cross 20 --side 10 --road 230 [--max-ext 1.0]"""
import argparse, json, math
import numpy as np, cv2


def extend(cost, res, c_cross, c_side, c_road, walk_costs=None, max_ext=1.0, probe=2.0, min_area=2.0, close_m=0.6, fill_interior=False):
    walk = set(walk_costs if walk_costs is not None else (c_side, c_cross, 60))
    out = cost.copy(); H, W = cost.shape; rep = []
    cw = (cost == c_cross).astype(np.uint8); k = max(1, int(round(close_m / res))) | 1
    cw = cv2.morphologyEx(cw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cw, connectivity=8)
    step = 0.5                                                        # 셀 단위 샘플 간격 (반 셀) — 회전된 띠의 빈틈 방지
    def lab_at(pr, pc):
        r = np.round(pr).astype(int); c = np.round(pc).astype(int); ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        return r[ok], c[ok]
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] * res * res < min_area: continue
        pts = np.argwhere(lab == i).astype(np.float32)                   # (row, col) — minAreaRect 에는 좌표 쌍이면 됨
        (cr, cc), (w, h), ang = cv2.minAreaRect(pts[:, None, :])
        a = math.radians(ang); u = np.array([math.cos(a), math.sin(a)]); v = np.array([-math.sin(a), math.cos(a)])   # (row, col) 공간의 두 축
        axes = [(u, w / 2, v, h / 2), (v, h / 2, u, w / 2)]
        probes = []
        for ax, half, bx, half_b in axes:
            ends = []
            for side in (1, -1):
                hit = None; s_b = np.arange(-half_b + 1, half_b - 1 + 1e-6, step)
                for t in np.arange(half + step, half + probe / res + 1e-6, step):
                    pr = cr + side * t * ax[0] + s_b * bx[0]; pc = cc + side * t * ax[1] + s_b * bx[1]; r, c = lab_at(pr, pc)
                    if len(r) == 0: break
                    vals = cost[r, c]
                    if np.isin(vals, list(walk - {c_cross})).mean() >= 0.5 or ((vals == c_cross).mean() >= 0.5 and t > half + 1): hit = t; break   # 인도 도달 (또는 다른 횡단보도)
                ends.append(hit)
            probes.append(ends)
        cand = [(j, max(e)) for j, e in enumerate(probes) if all(x is not None for x in e)]
        info = {"id": i, "center_rc": [round(float(cr), 1), round(float(cc), 1)], "size_m": [round(w * res, 2), round(h * res, 2)], "angle": round(ang, 1), "probe_m": [[None if x is None else round((x - axes[j][1]) * res, 2) for x in probes[j]] for j in range(2)]}
        if not cand: info["action"] = "skip (양 끝 인도 없음)"; rep.append(info); continue
        j = min(cand, key=lambda t: t[1])[0]; ax, half, bx, half_b = axes[j]; changed = 0
        info["cross_axis_len_m"] = round(2 * half * res, 2)
        for side, hit in zip((1, -1), probes[j]):
            ext = (hit - half) * res
            if ext > max_ext: continue
            s_b = np.arange(-half_b, half_b + 1e-6, step)
            for t in np.arange(half - 1, hit + 1e-6, step):
                pr = cr + side * t * ax[0] + s_b * bx[0]; pc = cc + side * t * ax[1] + s_b * bx[1]; r, c = lab_at(pr, pc)
                m = out[r, c] == c_road; changed += int(m.sum()); out[r[m], c[m]] = c_cross
        if fill_interior:
            sa = np.arange(-half + 1, half - 1 + 1e-6, step); sb = np.arange(-half_b + 1, half_b - 1 + 1e-6, step); A, B = np.meshgrid(sa, sb)
            pr = cr + A.ravel() * ax[0] + B.ravel() * bx[0]; pc = cc + A.ravel() * ax[1] + B.ravel() * bx[1]; r, c = lab_at(pr, pc)
            m = out[r, c] == c_road; changed += int(m.sum()); out[r[m], c[m]] = c_cross
        info["action"] = f"extended axis {j}"; info["cells_changed"] = changed; rep.append(info)
    return out, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--cross", type=int, default=20); ap.add_argument("--side", type=int, default=10); ap.add_argument("--road", type=int, default=230)
    ap.add_argument("--max-ext", type=float, default=1.0); ap.add_argument("--probe", type=float, default=2.0); ap.add_argument("--fill-interior", action="store_true")
    a = ap.parse_args(); z = dict(np.load(a.inp, allow_pickle=True)); cost = z["cost"]; res = float(z["res"])
    new, rep = extend(cost, res, a.cross, a.side, a.road, max_ext=a.max_ext, probe=a.probe, fill_interior=a.fill_interior)
    z["cost"] = new; np.savez_compressed(a.out, **z); print(json.dumps(rep, ensure_ascii=False, indent=1)); print("changed cells:", int((new != cost).sum()))
