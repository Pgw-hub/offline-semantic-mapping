"""보행 맵 블록 (L): 2.5 cm 확률 복셀 지도 (LiDAR 만, 5 m 이내) → 계단 인식·디딤판/앞면 채움·halo 제거 → 로봇 타일."""
import json, os, shutil

import numpy as np

from pipeline.env import CODE, PY


class LocomotionMapSteps:
    # 기본값 (세션 options.locomotion 으로 덮어씀)
    #   res          세부 격자 = 확률 격자, xy·z 모두 (m). 로봇 prior map controller 가 읽는 voxel3d_tiles v2 형식
    #   max_range    hit 으로 쓸 최대 거리 — 먼 거리 스치는 빔의 혼합 픽셀 제외 (2026-10-20 결정, 대전 11/03 실기 조합)
    #   free_range   빈 공간 레이 최대 길이 (움직인 사람·물체 지우기)
    #   tile_m       타일 한 변 (로봇은 주변 3×3 타일만 적재)
    #   stair        계단 보완: {risers: 앞면 판, clean: 디딤판 위 떠 있는 셀(halo) 제거, roi: [x0,x1,y0,y1] (계단이 여럿이라 엉뚱한 것을 잡을 때만)}
    #                false 면 계단 보완 없이 복셀 지도 그대로
    opt_default_locomotion = {"res": 0.025, "max_range": 5.0, "free_range": 5.0, "tile_m": 10.0, "stair": {"risers": True, "clean": True, "roi": None}}

    def loco_opts(self):
        o = self.opt.get("locomotion", True)
        if o is False: return None
        o = dict(self.opt_default_locomotion, **(o if isinstance(o, dict) else {}))
        if o["stair"] is True or o["stair"] is None: o["stair"] = dict(self.opt_default_locomotion["stair"])
        elif isinstance(o["stair"], dict): o["stair"] = dict(self.opt_default_locomotion["stair"], **o["stair"])
        return o

    def loco_dir(self): return os.path.join(self.D["compiled"], "locomotion")

    # ───────────────────────────────────────────────────────────────────────────────────────────
    # L (locomotion map) — 보행 컨트롤러용 사전맵. 의미 정보 없음(cls = 255), depth 카메라 채움 없음, LiDAR 만.
    #   ① locomotion_map/build_voxel3d.py: 2.5 cm 정육면체 확률 복셀 (OctoMap 식 log-odds, 5 m 이내 점) → 세부 셀 mean_z·max_z·count
    #      → compiled/locomotion/base.npz + base_tiles/  (노이즈 제거 ①: 한두 번 스친 사람·튀는 점은 miss 가 쌓여 지워짐)
    #   ② locomotion_map/voxel3d_stair_complete.py: 계단을 스스로 찾아(같은 간격으로 물러나는 수평 판 사슬, 가장 긴 것 하나) 전역 직선 모델로
    #      디딤판 사각형을 그 높이로 채움(src 5) + 앞면 판(risers) + 디딤판 위 떠 있는 셀 제거(clean, 노이즈 제거 ②)
    #      → compiled/locomotion/tiles/ (+ tiles_stair.json 계단 모델, stair_check.png). 계단을 못 찾으면 base_tiles 를 그대로 복사.
    #   ③ tiles → locomotion.npz (Potree 레이어·PCD 내보내기용)
    #   최종 보행 맵 = compiled/locomotion/tiles/ — 로봇 맵 패키지의 voxel3d_tiles/ 로 들어간다.
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_L(self):
        o = self.loco_opts()
        if not o: self.say("  options.locomotion: false → 생략"); return
        d = self.loco_dir(); base = os.path.join(d, "base"); tiles = os.path.join(d, "tiles"); r = str(o["res"])
        if not self.dry: os.makedirs(d, exist_ok=True)
        scans = [os.path.join(self.ds(l), "lidar_scans") for l in self.logs]
        if not self.done("L", base + ".npz", lambda: os.path.exists(os.path.join(base + "_tiles", "index.json"))):
            self.run([PY, f"{CODE}/locomotion_map/build_voxel3d.py", "--scans", *scans, "--out", base, "--res-xy", r, "--res-z", r, "--prob-res", r,
                      "--max-range", str(o["max_range"]), "--free-range", str(o["free_range"]), "--tile-m", str(o["tile_m"]), "--only-hit-cells",
                      "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"])], env={"CUDA_VISIBLE_DEVICES": ""}, logname="L_locomotion.log")
            if not self.dry and os.path.exists(tiles): shutil.rmtree(tiles)          # 바탕이 바뀌면 계단 보완도 다시
        sc = o["stair"]; sj = tiles + "_stair.json"; png = os.path.join(d, "stair_check.png")
        if not self.done("L", tiles, lambda: os.path.exists(os.path.join(tiles, "index.json"))):
            for f in (sj, png):
                if os.path.exists(f) and not self.dry: os.remove(f)
            ok = False
            if sc:
                try:
                    self.run([PY, f"{CODE}/locomotion_map/voxel3d_stair_complete.py", base + "_tiles", tiles] + (["--risers"] if sc.get("risers") else [])
                             + (["--clean"] if sc.get("clean") else []) + (["--roi", *map(str, sc["roi"])] if sc.get("roi") else []), env={"CUDA_VISIBLE_DEVICES": ""}, logname="L_stair.log")
                    ok = True
                except Exception as ex:
                    self.say(f"  계단 보완 실패 또는 계단 없음 (로그 logs/L_stair.log): {ex} → 복셀 지도 그대로")
            if not ok and not self.dry:
                if os.path.exists(tiles): shutil.rmtree(tiles)
                shutil.copytree(base + "_tiles", tiles)
                json.dump({"stair": None, "note": "계단 보완 없음 (options.locomotion.stair false 또는 계단을 못 찾음) — base_tiles 와 같음"}, open(sj, "w"), ensure_ascii=False)
        if not self.dry and os.path.exists(sj):
            M = json.load(open(sj))
            if M.get("treads"):
                if not os.path.exists(png) or os.path.getmtime(png) < os.path.getmtime(sj):
                    try:
                        self.run([PY, f"{CODE}/locomotion_map/voxel3d_stair_view.py", sj, png, base + "_tiles", tiles], env={"CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": "/tmp/mpl"}, logname="L_stair.log")
                    except Exception as ex: self.say(f"  계단 확인 그림 실패(무시): {ex}")
                FL = M.get("stairs") or [M]; T = [t for m in FL for t in m["treads"]]
                self.say(f"  계단 {len(FL)} 개 (직선 구간): 디딤판 {len(T)} 개 (모델로 추가 {sum(t['inferred'] for t in T)}), 높이 {min(m['treads'][0]['h'] for m in FL):+.2f}..{max(m['treads'][-1]['h'] for m in FL):+.2f} m, "
                         f"단 높이 {np.median([m['rise'] for m in FL]) * 100:.1f} cm, 단 너비 {np.median([m['run'] for m in FL]) * 100:.1f} cm, halo 제거 {sum(t.get('halo_removed', 0) for t in T):,} 셀")
        npz = os.path.join(d, "locomotion.npz")
        if not self.done("L", npz, lambda: os.path.getmtime(npz) >= os.path.getmtime(os.path.join(tiles, "index.json"))):
            self.run([PY, f"{CODE}/locomotion_map/voxel3d_tiles_to_npz.py", tiles, npz, "--base", base + ".npz"], logname="L_locomotion.log")

    # 옛 단계 이름 F (0.2 m occupancy + voxel3d) — 이제 보행 맵은 L. 옛 명령(--steps …,F,…)이 그대로 돌게 L 로 넘긴다.
    def step_F(self):
        self.say("  F 단계는 L(보행 맵)로 바뀜 → L 실행 (0.2 m occupancy 는 기본에서 빠짐, 필요하면 --steps O)")
        self.step_L()
