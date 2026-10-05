#!/usr/bin/env python3
"""의미 코스트맵 → raisin_navigation_plugin(nav2 KeepoutFilter) 용 파일 묶음.
입력: <map_root>/compiled/session_costmap_occ.npz (G 단계, 0.2 m, row=x col=y, ROS 코스트 0~255)
출력 <out_dir>/semantic/:
  semantic_mask.png + semantic_mask.yaml   nav2 map_server 규약 이미지(mode: raw) — 픽셀 값 = OccupancyGrid 값 0~100, 255 = -1(정보 없음)
  semantic_mask.npy                       OccupancyGrid data 와 같은 순서(row-major, 행 = y, 0행 = 맵 아래쪽) int8, -1 포함
  geofence_block_road.json                차도 − 횡단보도 영역을 폴리곤으로 (planning/set_geofence BLOCK 시험용, 맵 프레임 m)
  semantic_info.json                      값 대응표·원점·출처
  topview.png                             사람이 보는 미리보기
마스크 값은 KeepoutFilter 가 cost = round(v·254/100) 로 바꾸므로: 인도 4(→10) · 횡단보도 8(→20) · 잔디 35(→89) · 차도 --road 90(→229).
라벨 없는 지면·문맥·장애물·미관측은 -1 — 실시간 Obstacle/HeightMap 레이어가 판단 (정적 장애물을 중복으로 박지 않음).
사용: python tools/export_nav_semantic.py <map_root> <out_dir> [--road 90] [--map-name <로봇 SLAM 맵 이름>] [--frame <frame_id>]"""
import argparse, json, os, sys, time
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--road", type=int, default=90, help="차도 마스크 값 (0~100; 100 = lethal)")
ap.add_argument("--sidewalk", type=int, default=4); ap.add_argument("--crosswalk", type=int, default=8); ap.add_argument("--grass", type=int, default=35)
ap.add_argument("--map-name", default="", help="로봇이 Load Map 할 SLAM 맵 이름 (= frame_id). 비우면 빌드 맵 frame 설명만 기록")
ap.add_argument("--frame", default="", help="OccupancyGrid header.frame_id (기본 = --map-name)")
ap.add_argument("--crosswalk-extend", action="store_true", help="(옵션, 기본 끔) 횡단보도 끝~인도 사이 좁은 차도 띠를 횡단보도로 (tools/crosswalk_extend.py). sim 고원식 횡단보도 끝 여백 대응")
ap.add_argument("--crosswalk-max-ext", type=float, default=1.0, help="연장 최대 거리 m")
ap.add_argument("--min-poly-area", type=float, default=0.5, help="geofence 폴리곤 최소 면적 m²"); ap.add_argument("--simplify", type=float, default=0.15, help="폴리곤 단순화 허용 오차 m")
a = ap.parse_args(); root = os.path.abspath(a.root); D = os.path.join(a.out, "semantic"); os.makedirs(D, exist_ok=True)
z = np.load(os.path.join(root, "compiled", "session_costmap_occ.npz"), allow_pickle=True); cost = z["cost"]; x0, y0, R = float(z["x0"]), float(z["y0"]), float(z["res"])
man = json.load(open(os.path.join(root, "compiled", "map_manifest.json"))); cd = man.get("classdict", {}); classes = {c["name"]: c for c in cd.get("classes", [])}
def cost_of(name): return int(classes[name]["cost"]) if name in classes else None
C_SW, C_CW, C_GR, C_RD = cost_of("sidewalk"), cost_of("crosswalk"), cost_of("grass"), cost_of("road")
assert C_SW is not None and C_RD is not None, f"사전에 sidewalk/road 클래스가 있어야 함 (사전 {cd.get('name')})"
# cost grid (row=x, col=y) → OccupancyGrid 배열 (row=y, col=x)
cw_report = None
if a.crosswalk_extend and C_CW is not None:                          # 옵션: 횡단보도 연장 후처리
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import crosswalk_extend as CX
    before = cost; cost, cw_report = CX.extend(cost, R, C_CW, C_SW, C_RD, max_ext=a.crosswalk_max_ext); print(f"crosswalk_extend: {int((cost != before).sum())} cells road→crosswalk")
m = np.full(cost.shape, -1, np.int8)
for c_val, v in [(C_SW, a.sidewalk), (C_CW, a.crosswalk), (C_GR, a.grass), (C_RD, a.road)]:
    if c_val is not None: m[cost == c_val] = v
occ = m.T.copy()                                                    # (H=y, W=x), 0행 = y 최소 (OccupancyGrid 규약: origin 이 좌하단)
np.save(os.path.join(D, "semantic_mask.npy"), occ)
img = np.where(occ < 0, 255, occ).astype(np.uint8); cv2.imwrite(os.path.join(D, "semantic_mask.png"), np.flipud(img))   # map_server 이미지는 0행 = 위쪽(y 최대)
frame = a.frame or a.map_name or "<SLAM map frame of the build session>"
open(os.path.join(D, "semantic_mask.yaml"), "w").write(
    f"image: semantic_mask.png\nmode: raw          # 픽셀 값 = OccupancyGrid 값 그대로 (0~100), 255 = -1\nresolution: {R}\norigin: [{x0:.4f}, {y0:.4f}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nframe_id: {frame}\n")
# geofence BLOCK 폴리곤: 차도 셀(횡단보도 제외), 맵 좌표
road = (cost == C_RD).astype(np.uint8); road = cv2.morphologyEx(road, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
cnts, hier = cv2.findContours(road, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE); polys = []
for k, c in enumerate(cnts):
    if hier is not None and hier[0][k][3] != -1: continue           # 구멍(내부 윤곽)은 건너뜀 — 지오펜스는 단순 폴리곤
    area = cv2.contourArea(c) * R * R
    if area < a.min_poly_area: continue
    c = cv2.approxPolyDP(c, a.simplify / R, True)[:, 0, :].astype(float)   # (col=y_idx, row=x_idx) — cv2 는 (x=col, y=row)
    pts = [[round(x0 + (r + 0.5) * R, 3), round(y0 + (q + 0.5) * R, 3)] for q, r in c]   # q=col→y, r=row→x
    polys.append({"area_m2": round(area, 1), "points_xy": pts})
json.dump({"frame": frame, "mode": "BLOCK", "note": "차도(코스트 %d) − 횡단보도. planning/set_geofence {zones: Polygon[], frame, mode: BLOCK} 시험용. 값이 100(lethal)이라 차도가 완전히 막힘" % C_RD,
           "zones": sorted(polys, key=lambda p: -p["area_m2"])}, open(os.path.join(D, "geofence_block_road.json"), "w"), indent=1, ensure_ascii=False)
cnt = {name: int((cost == v).sum()) for name, v in [("sidewalk", C_SW), ("crosswalk", C_CW), ("grass", C_GR), ("road", C_RD)] if v is not None}
info = {"created": time.strftime("%Y-%m-%d %H:%M"), "source_map_root": root, "map_id": man.get("map_id"), "build_frame": man.get("frame"), "classdict": cd.get("name"), "model": cd.get("model"),
        "frame_id": frame, "resolution_m": R, "origin_xy": [x0, y0], "width_cells_x": int(occ.shape[1]), "height_cells_y": int(occ.shape[0]),
        "extent_m": [round(occ.shape[1] * R, 1), round(occ.shape[0] * R, 1)],
        "value_map": {"sidewalk": [C_SW, a.sidewalk, round(a.sidewalk * 254 / 100)], "crosswalk": [C_CW, a.crosswalk, round(a.crosswalk * 254 / 100)], "grass": [C_GR, a.grass, round(a.grass * 254 / 100)],
                      "road": [C_RD, a.road, round(a.road * 254 / 100)], "_columns": ["build costmap value", "mask value 0-100", "nav2 cost after KeepoutFilter"], "everything_else": -1},
        "cells": cnt, "geofence_polygons": len(polys), "crosswalk_extend": ({"max_ext_m": a.crosswalk_max_ext, "blobs": cw_report} if cw_report is not None else False)}
json.dump(info, open(os.path.join(D, "semantic_info.json"), "w"), indent=1, ensure_ascii=False)
col = np.full(occ.shape + (3,), 70, np.uint8)                      # BGR: 인도 흰 · 횡단보도 노랑 · 잔디 초록 · 차도 보라
col[occ == a.sidewalk] = (255, 255, 255); col[occ == a.crosswalk] = (0, 220, 255); col[occ == a.grass] = (60, 176, 67); col[occ == a.road] = (173, 13, 106); cv2.imwrite(os.path.join(D, "topview.png"), cv2.resize(np.flipud(col), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
print(json.dumps(info, ensure_ascii=False, indent=1))
