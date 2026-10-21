"""보행용 voxel map 블록 (F3d): locomotion_map/build_voxel3d.py 로 확률 복셀 사전맵·타일·뷰."""
import glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import CODE, REPO, PY, CGREPO, HF_HOME, CD


class LocomotionMapSteps:
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # F3d (voxel3d) — 보행 컨트롤러용 "3D 확률 복셀 사전맵". F 와 목적이 다르다: F 는 nav 코스트맵용 0.2 m, 이것은 로봇
    #   voxel_mapping 의 VoxelMap 격자(0.025×0.025×0.1 m) 로 getHeight(x,y) 가 바로 읽을 표면 높이(mean_z)를 준다.
    #   스크립트: build_voxel3d.py (S 의 스캔 + C 의 복셀) → voxel3d_view.py (rerun·BEV)
    #   하는 일: 0.1 m 확률 격자 레이캐스팅(점유 판정) + 세부 격자 count/mean_z/std_z + depth 채움(빈 곳만) + 평탄 채움(바닥 빈칸)
    #            + 클래스 id + 10 m 타일 바이너리(로봇은 주변 3×3 타일만 적재). 규격: docs/VOXEL3D_PRIOR.md
    #   산출: compiled/session_voxel3d.npz, session_voxel3d_tiles/{index.json, t_<tx>_<ty>.bin}, voxel3d_view.rrd, voxel3d_bev.png
    #   options.voxel3d: false 면 생략. 확률 임계(prob_hit/miss) 는 F 와 공유.
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_F3d(self):
        """보행용 3D 확률 복셀 사전맵 (voxel3d). 확률 판정은 prob_hit/prob_miss(occupancy 와 공유)."""
        v3 = self.opt.get("voxel3d")
        if not v3: return
        v3 = dict(self.opt_default_voxel3d, **(v3 if isinstance(v3, dict) else {}))
        out = os.path.join(self.D["compiled"], "session_voxel3d")
        scans = [os.path.join(self.ds(l), "lidar_scans") for l in self.logs]
        if not self.done("F", out + ".npz", lambda: os.path.exists(os.path.join(out + "_tiles", "index.json"))):
            self.run([PY, f"{CODE}/locomotion_map/build_voxel3d.py", "--scans", *scans, *self.voxsem_args(), "--out", out,
                      "--res-xy", str(v3["res_xy"]), "--res-z", str(v3["res_z"]), "--prob-res", str(v3["prob_res"]), "--max-range", str(v3.get("max_range", 30.0)), "--free-range", str(v3["free_range"]), "--tile-m", str(v3["tile_m"]),
                      "--fill-flat", str(v3["fill_flat"]), "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"])] + ([] if v3.get("depth", True) else ["--no-depth"]),
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="F_voxel3d.log")
        rrd = os.path.join(self.D["compiled"], "voxel3d_view.rrd")
        if not self.done("F", rrd):
            self.run([PY, f"{CODE}/locomotion_map/voxel3d_view.py", out + ".npz", rrd, *self.voxsem_args(), "--bev", os.path.join(self.D["compiled"], "voxel3d_bev.png")] + (["--sub", "2"] if len(self.logs) > 2 else []),
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="F_voxel3d_view.log")
        # 변형: iso025 = 0.025 m 정육면체 세부 격자 (확률 격자는 동일 0.1 m). 산출 session_voxel3d_<variant>.npz/_tiles/, 뷰 voxel3d_<variant>_view.rrd (포털이 자동 인식)
        for var in v3.get("variants", []):
            spec = {"iso025": {"res_xy": 0.025, "res_z": 0.025}, "z01": {"res_xy": 0.025, "res_z": 0.1},
                    "bonxai": {"res_xy": v3["res_xy"], "res_z": v3["res_z"], "occupancy": "bonxai"}}.get(var)   # bonxai = 본 맵과 같은 격자·채움, 점유 판정만 Bonxai

            if spec is None: self.say(f"  voxel3d 변형 {var}: 정의 없음 → 생략"); continue
            vout = os.path.join(self.D["compiled"], f"session_voxel3d_{var}")
            if not self.done("F", vout + ".npz", lambda: os.path.exists(os.path.join(vout + "_tiles", "index.json"))):
                self.run([PY, f"{CODE}/locomotion_map/build_voxel3d.py", "--scans", *scans, *self.voxsem_args(), "--out", vout,
                          "--res-xy", str(spec["res_xy"]), "--res-z", str(spec["res_z"]), "--prob-res", str(v3["prob_res"]), "--max-range", str(v3.get("max_range", 30.0)), "--free-range", str(v3["free_range"]), "--tile-m", str(v3["tile_m"]),
                          "--fill-flat", str(v3["fill_flat"]), "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"]), "--occupancy", spec.get("occupancy", "own")] + ([] if v3.get("depth", True) else ["--no-depth"]),
                         env={"CUDA_VISIBLE_DEVICES": ""}, logname=f"F_voxel3d_{var}.log")
            vrrd = os.path.join(self.D["compiled"], f"voxel3d_{var}_view.rrd")
            if not self.done("F", vrrd):
                self.run([PY, f"{CODE}/locomotion_map/voxel3d_view.py", vout + ".npz", vrrd, *self.voxsem_args(), "--bev", os.path.join(self.D["compiled"], f"voxel3d_{var}_bev.png"), "--sub", "4"],
                         env={"CUDA_VISIBLE_DEVICES": ""}, logname=f"F_voxel3d_{var}_view.log")
    opt_default_voxel3d = {"res_xy": 0.025, "res_z": 0.025, "prob_res": 0.025, "max_range": 5.0, "free_range": 5.0, "tile_m": 10.0, "fill_flat": 0.0, "depth": True, "variants": []}
