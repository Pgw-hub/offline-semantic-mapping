#!/usr/bin/env python3
"""build_map_package.py — mcap 세션 → Map Package(map_manifest.json) 를 한 번에 만드는 오케스트레이터.

컨테이너 offline_3d_semantic_map 안에서 실행 (docker exec 없음, 단일 프로세스가 단계별 스크립트를 subprocess 로 호출).
사용:
  python build_map_package.py <session.yaml>              # 전부 (이미 있는 산출물은 건너뜀). root 가 이름만이면 <repo>/output/<이름>
  python build_map_package.py <session.yaml> --steps D,E  # 일부 단계만
  python build_map_package.py <session.yaml> --force D    # 해당 단계 산출물 지우고 다시
  python build_map_package.py <session.yaml> --dry-run    # 명령만 출력
단계:
  A  dataset    mcap → dataset/<log>/ (results/, traj.txt, stamps.npz, lidar_world/)   [로그별]
  S  scans      mcap → dataset/<log>/lidar_scans/ (스캔별 world 점+원점)                [로그별]
  R  rgb        mcap → sem/rgb/ (LiDAR 시각 10Hz RGB) + rgb_index.npz                  [세션]
  B  semantic   sem/rgb → sem/sem/ (SegFormer-B4, grass/path/tree/person)              [세션, GPU]
  C  voxels     dataset+sem → voxel/<log>_voxsem.npz → voxel/session_voxsem.npz        [로그별+병합]
  D  objects    ConceptGraphs → dataset/<log>/exps/r_mapping_stride10/                 [로그별, GPU]
  E  graph      객체 병합 → compiled/session_objects.json(+clip), 휴리스틱 엣지, 객체-지형 엣지, (옵션) VLM 엣지
  F  occupancy  lidar_scans + session_voxsem → compiled/session_occupancy.npz
  G  package    map_manifest.json 작성 + 코스트맵(costmap_from_occupancy) 등록
  H  view       compiled/map_package.rrd — 기본 0.1m 점 표현 (options.view_cubes: true 면 view_coarse m 큐브)
  V  video      RGB 배속 영상 compiled/session_rgb_x<speed>.mp4 + 3분할(RGB|검출|세그) session_pipeline_x<speed>.mp4 (options.video_speed 0 이면 생략)
레이아웃: <root>/{dataset,sem,voxel,compiled}/ ; 매니페스트 경로는 <root> 기준 상대경로 (load_map 규약).
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # 이 저장소 (우리 코드)
CG = REPO; TOOLS = f"{REPO}/tools/preprocess"
CGREPO = os.environ.get("CG_REPO", "/data2/cg/concept-graphs") + "/conceptgraph"   # ConceptGraphs 설치 위치 (컨테이너에 editable 설치됨)
HF_HOME = os.environ.get("HF_HOME", "/data2/cg/hf_cache")                        # 모델 캐시 (수십 GB, 데이터 디스크에 둠)
PY = sys.executable

class Build:
    def __init__(self, cfg, dry=False, force=()):
        self.c = cfg; self.dry = dry; self.force = set(force)
        root = cfg.get("root") or cfg.get("name")
        self.root = root if os.path.isabs(root) else os.path.join(REPO, "output", root)   # 상대경로/이름 → <repo>/output/<name>
        self.logs = [f"log_{int(i):03d}" for i in cfg["logs"]]
        self.mcap_dir = cfg["mcap_dir"]; self.calib = cfg.get("calib") or os.path.join(self.mcap_dir, "raisin_config", "calibration")
        self.D = {k: os.path.join(self.root, k) for k in ("dataset", "sem", "voxel", "compiled", "logs")}
        self.opt = {"lidar_fusion": True, "mask_bottom_px": 60, "drop_classes": "person", "scene_cloud": 0, "vlm_edges": True, "vlm_max_pairs": 30,
                    "vlm_model": "Qwen/Qwen2.5-VL-32B-Instruct", "occupancy_res": 0.2, "prob_hit": 0.85, "prob_miss": 0.45,
                    "costmap": {"h_ground": 2, "h_obs": 3, "clearance": 1.4, "band_min": 0.3}, "view_coarse": 0.3, "view_cubes": False, "view_sub": 1, "video_speed": 15, "video_width": 960, "video_pipeline": True, "map_id": os.path.basename(self.root),
                    "frame": "SLAM world (/Odometry/base)"}
        self.opt.update(cfg.get("options", {}))
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
        e = dict(os.environ); e.update(env or {})
        lf = open(os.path.join(self.D["logs"], logname), "a") if logname else None
        p = subprocess.run(cmd, cwd=cwd, env=e, shell=isinstance(cmd, str), stdout=lf or None, stderr=subprocess.STDOUT if lf else None)
        if lf: lf.close()
        if p.returncode != 0:
            raise RuntimeError(f"실패 (exit {p.returncode}): {cmd if isinstance(cmd, str) else ' '.join(cmd)}" + (f" — 로그 {self.D['logs']}/{logname}" if logname else ""))
        return p.returncode
    def done(self, step, path, check=None):
        """산출물이 있으면 True (건너뜀). --force 면 지우고 False."""
        if step in self.force and os.path.exists(path) and not self.dry:
            self.say(f"  --force {step}: {path} 삭제"); shutil.rmtree(path) if os.path.isdir(path) else os.remove(path); return False
        ok = os.path.exists(path) and (check() if check else True)
        if ok: self.say(f"  건너뜀 (있음): {path}")
        return ok
    def mcap(self, log): return os.path.join(self.mcap_dir, f"{log}.mcap")
    def ds(self, log): return os.path.join(self.D["dataset"], log)

    # ---- 단계 ----
    def step_A(self):
        for log in self.logs:
            out = self.ds(log)
            if self.done("A", os.path.join(out, "stamps.npz"), lambda: len(glob.glob(os.path.join(out, "results", "frame*.jpg"))) > 100): continue
            cmd = [PY, f"{TOOLS}/mcap_to_cg.py", self.mcap(log), self.calib, out] + (["--lidar"] if self.opt["lidar_fusion"] else [])
            self.run(cmd, logname=f"A_{log}.log")
    def step_S(self):
        for log in self.logs:
            out = os.path.join(self.ds(log), "lidar_scans")
            if self.done("S", out, lambda: len(glob.glob(os.path.join(out, "*.npz"))) > 100): continue
            self.run([PY, f"{TOOLS}/export_lidar_scans.py", self.mcap(log), self.calib, out], logname=f"S_{log}.log")
    def step_R(self):
        idx = os.path.join(self.D["sem"], "rgb_index.npz")
        if self.done("R", idx): return
        nums = [int(l[4:]) for l in self.logs]
        if nums != list(range(min(nums), max(nums) + 1)):
            raise RuntimeError("extract_rgb_session 은 연속 로그 범위만 지원 (logs 를 a..b 로 주세요)")
        self.run([PY, f"{TOOLS}/extract_rgb_session.py", self.mcap_dir, self.D["sem"], f"{min(nums)}-{max(nums)}"], logname="R.log")
    def step_B(self):
        n_rgb = len(glob.glob(os.path.join(self.D["sem"], "rgb", "*.png")))
        semdir = os.path.join(self.D["sem"], "sem")
        if self.done("B", semdir, lambda: len(glob.glob(os.path.join(semdir, "*.npz"))) >= n_rgb > 0): return
        self.run([PY, f"{TOOLS}/precompute_semantic.py", self.D["sem"], "--skip-existing"], logname="B.log")
    def step_C(self):
        outs = []
        for log in self.logs:
            out = os.path.join(self.D["voxel"], f"{log}_voxsem.npz"); outs.append(out)
            if self.done("C", out): continue
            self.run([PY, f"{TOOLS}/build_voxel_sem_map.py", self.mcap(log), self.calib, self.ds(log), self.D["sem"], out, "--mask-bottom", str(self.opt["mask_bottom_px"])], logname=f"C_{log}.log")
        sess = os.path.join(self.D["voxel"], "session_voxsem.npz")
        if not self.done("C", sess):
            self.run([PY, f"{CG}/merge_voxels.py", sess] + outs, logname="C_merge.log")
    def step_D(self):
        env = {"YOLO_CONFIG_DIR": "/root/.yolo", "CG_MASK_BOTTOM_PX": str(self.opt["mask_bottom_px"]), "CG_DROP_CLASSES": self.opt["drop_classes"],
               "CG_LOG_SCENE_CLOUD": str(self.opt["scene_cloud"])}
        for log in self.logs:
            exp = os.path.join(self.ds(log), "exps", "r_mapping_stride10")
            if self.done("D", exp, lambda: bool(glob.glob(os.path.join(exp, "obj_json_*.json"))) and bool(glob.glob(os.path.join(exp, "pcd_*.pkl.gz")))): continue
            if os.path.isdir(os.path.join(self.ds(log), "exps")) and not self.dry:
                shutil.rmtree(os.path.join(self.ds(log), "exps"))          # 불완전 detections 재사용 방지
            env["CG_HEADLESS_RRD"] = os.path.join(self.ds(log), f"rerun_{log}.rrd")
            self.run([PY, "slam/rerun_realtime_mapping.py", f"scene_id={log}", f"dataset_root={self.D['dataset']}"], cwd=CGREPO, env=env, logname=f"D_{log}.log")
    def step_E(self):
        env = {"CG_RAIBO_ROOT": self.D["dataset"], "CG_OUT_DIR": self.D["compiled"]}
        so = os.path.join(self.D["compiled"], "session_objects.json")
        if not self.done("E", so):
            self.run([PY, f"{CG}/map_compiler.py", "merge"], env=env, logname="E_merge.log")
        if not self.done("E", os.path.join(self.D["compiled"], "session_edges.json")):
            self.run([PY, f"{CG}/map_compiler.py", "edges"], env=env, logname="E_edges.log")
        rel = os.path.join(self.D["compiled"], "voxel_session_relations.json")
        if not self.done("E", rel):
            self.run([PY, f"{CG}/voxel_rerun_v2.py", os.path.join(self.D["voxel"], "session_voxsem.npz"), so, self.D["dataset"], os.path.join(self.D["compiled"], "voxel_session.rrd"), "--unknown-sub", "4"],
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="E_terrain.log")
        vlm = os.path.join(self.D["compiled"], "session_edges_vlm.json")
        if self.opt["vlm_edges"]:
            if not self.done("E", vlm):
                for log in self.logs:
                    out = os.path.join(self.D["compiled"], f"vlm_edges_{log}_v2")
                    if os.path.exists(os.path.join(out, "edges_claude.json")): self.say(f"  건너뜀 (있음): {out}"); continue
                    self.run([PY, f"{CG}/vlm_edges.py", "--scene", log, "--dataset-root", self.D["dataset"], "--out", out, "--backend", "local",
                              "--local-model", self.opt["vlm_model"], "--max-pairs", str(self.opt["vlm_max_pairs"])], env={"HF_HOME": HF_HOME}, logname=f"E_vlm_{log}.log")
                self.run([PY, f"{CG}/edges_session_rerun.py", self.D["dataset"], self.D["compiled"], os.path.join(self.D["compiled"], "vlm_edges_session.rrd")], logname="E_vlm_session.log")
        elif not os.path.exists(vlm) and not self.dry:
            json.dump([], open(vlm, "w")); self.say("  VLM 엣지 비활성 → 빈 목록 저장")
    def step_F(self):
        out = os.path.join(self.D["compiled"], "session_occupancy")
        if self.done("F", out + ".npz"): return
        scans = [os.path.join(self.ds(l), "lidar_scans") for l in self.logs]
        self.run([PY, f"{CG}/build_occupancy.py", "--scans", *scans, "--voxsem", os.path.join(self.D["voxel"], "session_voxsem.npz"), "--out", out,
                  "--res", str(self.opt["occupancy_res"]), "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"]), "--dynamic", self.opt["drop_classes"]],
                 env={"CUDA_VISIBLE_DEVICES": ""}, logname="F.log")
    def step_G(self):
        man = os.path.join(self.D["compiled"], "map_manifest.json")
        rel = lambda p: os.path.relpath(p, self.root)
        files = {"terrain_voxels": ("voxel/session_voxsem.npz", "0.1m semantic 복셀 (ijk, n_pts, rgb, sem_mean[grass,path,tree,person], n_sem)"),
                 "occupancy_probabilistic": ("compiled/session_occupancy.npz", f"{self.opt['occupancy_res']}m 레이캐스팅 log-odds + 표면 하이브리드 (state/source/surf_removed)"),
                 "objects": ("compiled/session_objects.json", "ConceptGraphs 객체 (로그 간 병합), bbox center/extent/tag/num_det"),
                 "objects_clip": ("compiled/session_objects_clip.npz", "객체 CLIP ViT-H/14 임베딩 1024D (objects 와 같은 순서)"),
                 "edges_heuristic": ("compiled/session_edges.json", "기하 휴리스틱 엣지 (on top of / next to)"),
                 "edges_object_terrain": ("compiled/voxel_session_relations.json", "객체-지형 관계 (bbox 발자국 아래 지배 지형)"),
                 "edges_vlm": ("compiled/session_edges_vlm.json", "VLM 판정 엣지 (Qwen2.5-VL)")}
        m = {"map_id": self.opt["map_id"], "frame": self.opt["frame"], "voxel_m": 0.1, "grid_axes": "row=x, col=y", "occupancy_res_m": self.opt["occupancy_res"],
             "built": time.strftime("%Y-%m-%d %H:%M"), "source": {"mcap_dir": self.mcap_dir, "logs": self.logs, "calib": self.calib}, "options": self.opt, "files": {}}
        for k, (p, desc) in files.items():
            full = os.path.join(self.root, p)
            if not os.path.exists(full):
                if self.dry: self.say(f"  (dry) {k} ← {p}"); continue
                raise RuntimeError(f"매니페스트 항목 {k} 의 파일이 없음: {full}")
            m["files"][k] = {"path": p, "bytes": os.path.getsize(full), "desc": desc}
        if not self.dry:
            json.dump(m, open(man, "w"), ensure_ascii=False, indent=1); self.say(f"  매니페스트 작성: {man} ({len(m['files'])} 항목)")
        cm = self.opt["costmap"]; out = os.path.join(self.D["compiled"], "session_costmap_occ")
        if not self.done("G", out + ".npz"):
            self.run([PY, f"{CG}/costmap_from_occupancy.py", man, out, "--register", "--h-ground", str(cm["h_ground"]), "--h-obs", str(cm["h_obs"]),
                      "--clearance", str(cm["clearance"]), "--band-min", str(cm["band_min"])], env={"CUDA_VISIBLE_DEVICES": ""}, logname="G_costmap.log")
        if not self.dry:   # 로더 기본 키(costmap_2d)도 같은 파일을 가리키게
            m = json.load(open(man))
            if "costmap_2d_occ" in m["files"] and "costmap_2d" not in m["files"]:
                m["files"]["costmap_2d"] = dict(m["files"]["costmap_2d_occ"]); json.dump(m, open(man, "w"), ensure_ascii=False, indent=1)
            tot = sum(v["bytes"] for v in m["files"].values()); self.say(f"  패키지 {len(m['files'])} 파일 · {tot/1e6:.1f} MB")
    def step_H(self):
        man = os.path.join(self.D["compiled"], "map_manifest.json"); rrd = os.path.join(self.D["compiled"], "map_package.rrd")
        if not self.done("H", rrd):
            cmd = [PY, f"{CG}/load_map.py", man, "--rrd", rrd, "--sub", str(self.opt.get("view_sub", 1))]
            if self.opt.get("view_cubes", False):                  # 기본은 0.1m 점(구) 표현; 큐브(OctoMap 풍)는 옵션
                cmd += ["--coarse", str(self.opt["view_coarse"]), "--cubes"]
            self.run(cmd, env={"CUDA_VISIBLE_DEVICES": ""}, logname="H.log")
        # 요약 JSON (웹 포털이 읽음) — 항상 갱신
        if not self.dry:
            e = dict(os.environ); e["CUDA_VISIBLE_DEVICES"] = ""
            p = subprocess.run([PY, f"{CG}/load_map.py", man], env=e, capture_output=True, text=True)
            out = p.stdout; j = out[out.index("{"):] if "{" in out else "{}"
            try:
                summ = json.loads(j)
            except Exception:
                summ = {"error": "summary parse failed"}
            summ["build_seconds_total"] = round(time.time() - self.t0)
            json.dump(summ, open(os.path.join(self.D["compiled"], "summary.json"), "w"), ensure_ascii=False, indent=1)
            self.say("  summary.json 갱신")

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

    def step_V(self):
        sp = int(self.opt.get("video_speed", 0) or 0)
        if sp <= 0: self.say("  video_speed 0 → 생략"); return
        out = os.path.join(self.D["compiled"], f"session_rgb_x{sp}.mp4")
        if not self.done("V", out):
            self.run([PY, f"{REPO}/tools/make_rgb_video.py", self.D["dataset"], out, "--speed", str(sp), "--width", str(self.opt.get("video_width", 960)), "--logs", *self.logs], logname="V.log")
        if self.opt.get("video_pipeline", True):                 # RGB | 검출 | 세그멘테이션 3분할
            out2 = os.path.join(self.D["compiled"], f"session_pipeline_x{sp}.mp4")
            if not self.done("V", out2):
                self.run([PY, f"{REPO}/tools/make_pipeline_video.py", self.root, out2, "--speed", str(sp), "--logs", *self.logs], logname="V_pipeline.log")

    def go(self, steps):
        try:
            for s in steps:
                self.say(f"===== 단계 {s} =====")
                getattr(self, f"step_{s}")()
            self.say("완료")
        finally:
            self.chown_outputs()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("session_yaml"); ap.add_argument("--steps", default="A,S,R,B,C,D,E,F,G,H,V")
    ap.add_argument("--force", default="", help="다시 만들 단계 (쉼표)"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.session_yaml))
    b = Build(cfg, dry=a.dry_run, force=[s for s in a.force.split(",") if s])
    b.go([s.strip() for s in a.steps.split(",") if s.strip()])
