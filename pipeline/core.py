"""빌드 공통: 세션 설정 해석, 단계 실행 유틸(run · done · 경로), 단계 순서 실행(go), 산출물 소유권 정리."""
import glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import REPO, PY, CGREPO, HF_HOME, CD


class BuildCore:
    def __init__(self, cfg, dry=False, force=()):
        self.c = cfg; self.dry = dry; self.force = set(force)
        root = cfg.get("root") or cfg.get("name")
        self.root = root if os.path.isabs(root) else os.path.join(REPO, "output", root)   # 상대경로/이름 → <repo>/output/<name>
        self.logs = [f"log_{int(i):03d}" for i in cfg["logs"]]
        self.mcap_dir = cfg["mcap_dir"]
        self.calib = cfg.get("calib") or next((c for c in (os.path.join(self.mcap_dir, "raisin_config", "calibration"),                 # 옛 레코더
                                                            os.path.join(self.mcap_dir, "config_snapshot", "raisin_root", "calibration"))     # 레코더 v2 스냅샷
                                               if os.path.isdir(c)), os.path.join(self.mcap_dir, "raisin_config", "calibration"))
        self.D = {k: os.path.join(self.root, k) for k in ("dataset", "sem", "voxel", "compiled", "logs")}
        self.opt = {"lidar_fusion": True, "mask_bottom_px": 60, "drop_classes": "person", "scene_cloud": 0, "vlm_edges": True, "vlm_max_pairs": 30,
                    "vlm_model": "Qwen/Qwen2.5-VL-32B-Instruct", "occupancy_res": 0.2, "prob_hit": 0.85, "prob_miss": 0.45,
                    "costmap": {"h_ground": 2, "h_obs": 3, "clearance": 1.4, "band_min": 0.3}, "view_coarse": 0.3, "view_cubes": False, "view_sub": 1, "video_speed": 15, "video_width": 960, "video_pipeline": True, "quality_img_stride": 15, "classdict": "default", "label_edge_px": 4, "require_depth": True, "map_id": os.path.basename(self.root),
                    "frame": "SLAM world (/Odometry/base)", "pose_graph": False, "lidar_extrinsic": "calib", "livox_flip": True, "depth_fill_range": 4.0, "depth_fill_gap": 0.15,
                    "voxel3d": {"res_xy": 0.025, "res_z": 0.025, "prob_res": 0.025, "max_range": 5.0, "free_range": 5.0, "tile_m": 10.0, "fill_flat": 0.0, "depth": True, "variants": []}}
        self.opt.update(cfg.get("options", {}))
        self.graph = os.path.join(self.root, "graph")
        self.geo_only = bool(self.opt.get("geometry_only"))                     # 카메라 없는 기하 전용 빌드 (보행 맵만)
        self.with_objects = bool(self.opt.get("objects", True)) and not self.geo_only   # false: 3D scene graph(D·E) 없이 semantic 맵까지만 (GPU 메모리 작은 PC용)
        if self.opt["pose_graph"] and "frame" not in cfg.get("options", {}): self.opt["frame"] = "SLAM world, PGO-corrected (preprocess/geometry/build_graph.py)"
        if self.opt.get("robot_map"): self.opt["pose_graph"] = True; self.opt.setdefault("frame", "robot lidar_slam saved map frame (final PGO)"); self.opt["frame"] = "robot lidar_slam saved map frame (final PGO)"
        for d in self.D.values():
            if not dry: os.makedirs(d, exist_ok=True)
        self.t0 = time.time(); self.log_f = None if dry else open(os.path.join(self.D["logs"], "build.log"), "a")
    # ---- 유틸 ----
    def say(self, msg):
        line = f"[{time.time()-self.t0:7.0f}s] {msg}"; print(line, flush=True)
        if self.log_f: self.log_f.write(line + "\n"); self.log_f.flush()
    def run(self, cmd, cwd=None, env=None, logname=None):
        self.say("$ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
        if self.dry: return 0
        e = dict(os.environ); e["LIDAR_EXTRINSIC"] = str(self.opt.get("lidar_extrinsic", "calib")); e["LIVOX_FLIP"] = "1" if self.opt.get("livox_flip", True) else "0"; e.update(env or {})
        lf = open(os.path.join(self.D["logs"], logname), "a") if logname else None
        p = subprocess.run(cmd, cwd=cwd, env=e, shell=isinstance(cmd, str), stdout=lf or None, stderr=subprocess.STDOUT if lf else None)
        if lf: lf.close()
        if p.returncode != 0:
            raise RuntimeError(f"실패 (exit {p.returncode}): {cmd if isinstance(cmd, str) else ' '.join(cmd)}" + (f" — 로그 {self.D['logs']}/{logname}" if logname else ""))
        return p.returncode
    def done(self, step, path, check=None):
        """산출물이 있으면 True (건너뜀). --force 면 지우고 False."""
        if step in self.force and os.path.exists(path) and not self.dry:
            self.say(f"  --force {step}: {path} 삭제"); shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            if step == "B" and os.path.exists(os.path.join(self.D["sem"], "classes.json")): os.remove(os.path.join(self.D["sem"], "classes.json"))
            return False
        ok = os.path.exists(path) and (check() if check else True)
        if ok: self.say(f"  건너뜀 (있음): {path}")
        return ok
    def mcap(self, log): return os.path.join(self.mcap_dir, f"{log}.mcap")
    def voxsem_args(self):
        """voxel3d 계열에 넘길 semantic 복셀. 기하 전용이면 없음(클래스 unlabeled)."""
        return [] if self.geo_only else ["--voxsem", os.path.join(self.D["voxel"], "session_voxsem.npz")]
    def pg(self): return ["--pose-graph", self.graph] if self.opt["pose_graph"] else []
    def ds(self, log): return os.path.join(self.D["dataset"], log)
    # ---- 단계 ----
    def rd_arg(self):
        rd = self.opt.get("require_depth", True)                      # true/false 또는 'lidar'(=2: depth 없으면 LiDAR 자체 z-buffer)
        return "2" if rd in (2, "2", "lidar", "zbuf") else ("1" if rd and rd not in ("false", "0") else "0")
    @property
    def robot_map(self): return self.opt.get("robot_map") or None          # {"pcd": ..., "traj": ...} → 트랙 2 (로봇 저장 지도에 의미 입히기)
    def chown_outputs(self):
        """컨테이너(root)가 만든 산출물을 output 폴더 소유자(호스트 사용자)에게 넘김"""
        if self.dry or os.geteuid() != 0: return
        try:
            st = os.stat(os.path.dirname(self.root)); uid, gid = st.st_uid, st.st_gid
            if uid == 0: return
            for dp, dn, fn in os.walk(self.root):
                os.chown(dp, uid, gid)
                for f in fn: os.chown(os.path.join(dp, f), uid, gid)
            self.say(f"  소유권 → uid {uid}")
        except Exception as e:
            self.say(f"  chown 실패(무시): {e}")
    # go — 단계 문자열 순서대로 step_<X> 를 호출. 각 단계는 산출물이 있으면 건너뛰므로(멱등) 중단된 빌드를 같은 명령으로 재개할 수 있다.
    #      --force X 는 그 단계 산출물만 지우고 다시 만든다. 끝나면 컨테이너(root)가 만든 파일 소유권을 호스트 사용자로 돌린다.
    GEOMETRY_STEPS = ("Q", "P", "S", "F", "G", "H")
    OBJECT_STEPS = ("D", "E")                                               # 3D Scene Graph builder (ConceptGraphs, GPU 메모리 ~10 GB)
    def go(self, steps):
        if self.geo_only:
            skipped = [s for s in steps if s not in self.GEOMETRY_STEPS]; steps = [s for s in steps if s in self.GEOMETRY_STEPS]
            if skipped: self.say(f"기하 전용 → 카메라 단계 생략: {','.join(skipped)}")
        elif not self.with_objects:
            skipped = [s for s in steps if s in self.OBJECT_STEPS]; steps = [s for s in steps if s not in self.OBJECT_STEPS]
            if skipped: self.say(f"objects: false → 3D scene graph 단계 생략: {','.join(skipped)}")
        try:
            for s in steps:
                self.say(f"===== 단계 {s} =====")
                getattr(self, f"step_{s}")()
            self.say("완료")
        finally:
            self.chown_outputs()
