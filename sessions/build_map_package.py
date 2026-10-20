#!/usr/bin/env python3
"""build_map_package.py — mcap 세션 → Map Package(map_manifest.json) 를 한 번에 만드는 오케스트레이터.

컨테이너 offline_3d_semantic_map 안에서 실행 (docker exec 없음, 단일 프로세스가 단계별 스크립트를 subprocess 로 호출).
사용:
  python build_map_package.py <session.yaml>              # 전부 (이미 있는 산출물은 건너뜀). root 가 이름만이면 <repo>/output/<이름>
  python build_map_package.py <session.yaml> --steps D,E  # 일부 단계만
  python build_map_package.py <session.yaml> --force D    # 해당 단계 산출물 지우고 다시
  python build_map_package.py <session.yaml> --dry-run    # 명령만 출력
단계:
  ★ 기본 파이프라인 = 트랙 2 (options.robot_map: {pcd, traj, geometry: keyframe_scans}) — 2026-09-16 실로봇 검증 후 채택.
     기하 = 로봇 lidar_slam 이 저장한 지도 프레임(최종 PGO). 자세 = <name>.txt 키프레임 + /Odometry/base 보간(Δ(t)). 점 = 키프레임 시각의 /cloud_registered.
     options.robot_map 이 없으면 옛 트랙 1(raw /livox/lidar + /Odometry/base [+ build_graph PGO]) 로 동작.
  P  posegraph  robot_map: geometry/robot_graph.py --traj (로봇 키프레임 자세 → graph/) | 트랙1: deskew 키프레임 + 루프클로저 + GTSAM PGO (geometry/build_graph.py)
  A  dataset    mcap → dataset/<log>/ (results/, traj.txt, stamps.npz, lidar_world/)   [로그별]
  S  scans      robot_map: /cloud_registered → Δ(t) 최종 프레임 (export_registered_scans.py) | 트랙1: raw deskew (export_lidar_scans.py) → dataset/<log>/lidar_scans/
  R  rgb        mcap → sem/rgb/ (LiDAR 시각 10Hz RGB) + rgb_index.npz                  [세션]
  B  semantic   sem/rgb → sem/sem/ (사전의 세그 모델: SegFormer ADE / Mask2Former Mapillary / EoMT, N채널) [세션, GPU]
  C  voxels     robot_map: 기하 pcd(keyframe_scans|all_scans: scans_to_pcd.py / pcd: 저장 지도) 점에 세그 확률 투영 (label_map_points.py) → voxel/session_voxsem.npz + compiled/robot_map_semantic.pcd
                트랙1: dataset+sem → voxel/<log>_voxsem.npz → voxel/session_voxsem.npz        [로그별+병합]
  D  objects    ConceptGraphs → dataset/<log>/exps/r_mapping_stride10/                 [로그별, GPU]
  E  graph      객체 병합 → compiled/session_objects.json(+clip), 휴리스틱 엣지, 객체-지형 엣지, (옵션) VLM 엣지
  F  occupancy  lidar_scans + session_voxsem → compiled/session_occupancy.npz (0.2 m 코스트맵용)
                + options.voxel3d.variants 에 bonxai: 같은 격자·채움에 점유 판정만 Bonxai ProbabilisticMap(third_party/bonxai, tools/bonxai) → session_voxel3d_bonxai_tiles/
                + 보행용 3D 확률 복셀 사전맵 (build_voxel3d.py: OctoMap 식 log-odds + 복셀 평균 z, 로봇 VoxelMap 격자 0.025×0.025×0.1 m) → compiled/session_voxel3d.npz
                  + session_voxel3d_tiles/(10 m 타일 바이너리, 로봇 적재용) + voxel3d_view.rrd/voxel3d_bev.png (기본 0.025 정육면체, 2026-10-15~) + 선택 변형 z01(로봇 VoxelMap 격자 0.025×0.025×0.1, options.voxel3d.variants). options.voxel3d: false 면 생략
  G  package    map_manifest.json 작성 + 코스트맵(costmap_from_occupancy) 등록
  H  view       compiled/map_package.rrd — 기본 0.1m 점 표현 (options.view_cubes: true 면 view_coarse m 큐브)
                + compiled/potree/ Potree 웹 뷰어 (tools/potree/potree_export.py; options.potree: false 면 생략)
  Q  quality    mcap 데이터 품질(토픽별 dt/rate/size, 유실·커버리지 이벤트) + 재생 rerun → compiled/data_quality.{rrd,json}
  V  video      RGB 배속 영상 compiled/session_rgb_x<speed>.mp4 + 3분할(RGB|검출|세그) session_pipeline_x<speed>.mp4 (options.video_speed 0 이면 생략)
기하 전용 (options.geometry_only: true): 카메라 없이 LiDAR 만으로 보행 맵(voxel3d 타일)을 만든다. Q,P,S,F,G,H 만 돌고(A,R,B,C,D,E,V 생략),
                F 는 0.2 m 코스트맵(라벨 필요)을 건너뛰고 voxel3d 만 --voxsem 없이(클래스 unlabeled) 만든다. 의미는 나중에 별도 세션으로 입힌다.
레이아웃: <root>/{dataset,sem,voxel,compiled}/ ; 매니페스트 경로는 <root> 기준 상대경로 (load_map 규약).
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # 이 저장소 (우리 코드)
sys.path.insert(0, REPO); import classdict as CD                               # 사전·세그 모델 레지스트리 (step B 가드)
CG = REPO; TOOLS = f"{REPO}/tools/preprocess"
CGREPO = os.environ.get("CG_REPO", "/data2/cg/concept-graphs") + "/conceptgraph"   # ConceptGraphs 설치 위치 (컨테이너에 editable 설치됨)
HF_HOME = os.environ.get("HF_HOME", "/data2/cg/hf_cache")                        # 모델 캐시 (수십 GB, 데이터 디스크에 둠)
PY = sys.executable

# ─── 단계 사이 데이터 흐름 (한 세션 기준) ────────────────────────────────────────────────────────────
#
#   mcap ─┬─ Q 품질 점검
#         ├─ P 자세 그래프 graph/ ──────────────┬───────────────┐
#         ├─ A 카메라 데이터셋 dataset/<log>/ ◄─┘(자세 보간)     │
#         ├─ S LiDAR 스캔 lidar_scans/ ◄────────────────────────┘(최종 프레임 보정)
#         └─ R 세션 RGB sem/rgb/ ─► B 세그 모델 sem/sem/ (GPU)
#
#   A + B (+S 기하) ─► C 0.1 m semantic 복셀 voxel/session_voxsem.npz          ← 지형(stuff) 레이어
#   A ───────────────► D ConceptGraphs 객체 (GPU) ─► E 세션 scene graph (노드·엣지) ← 객체(things) 레이어
#   S + C ───────────► F 확률 occupancy 0.2 m  +  F3d 보행용 voxel3d (0.025/0.1 m 타일)
#   C,E,F,F3d ───────► G map_manifest.json + 코스트맵   ─► H rerun·summary   ─► V 영상
#
#   각 step_X 메서드 위 주석에 그 단계의 목적·스크립트·산출물을 적어 두었다.
# ────────────────────────────────────────────────────────────────────────────────────────────────────
class Build:
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
                    "voxel3d": {"res_xy": 0.025, "res_z": 0.025, "prob_res": 0.1, "free_range": 8.0, "tile_m": 10.0, "fill_flat": 0.0, "depth": True, "variants": []}}
        self.opt.update(cfg.get("options", {}))
        self.graph = os.path.join(self.root, "graph")
        self.geo_only = bool(self.opt.get("geometry_only"))                     # 카메라 없는 기하 전용 빌드 (보행 맵만)
        if self.opt["pose_graph"] and "frame" not in cfg.get("options", {}): self.opt["frame"] = "SLAM world, PGO-corrected (geometry/build_graph.py)"
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
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # P (posegraph) — "어느 시각에 로봇이 어디 있었나"를 정하는 기하의 뼈대. 뒤 단계(A/S/C)가 모두 이 자세를 쓴다.
    #   왜 필요한가: mcap 의 /Odometry/base 는 실시간 SLAM 자세라 루프클로저 전 드리프트가 남아 있다.
    #                최종 보정(PGO) 자세로 바꿔 줘야 지도 내부가 어긋나지 않는다.
    #   트랙 2(robot_map 있음): geometry/robot_graph.py — 로봇 lidar_slam 이 저장한 키프레임 자세(<name>.txt, 없으면 /path/pgo)
    #                            를 읽어 graph/graph.json (T_odom ↔ T_corr 쌍) 으로 변환. 서버는 최적화를 안 한다.
    #   트랙 1(robot_map 없음): geometry/build_graph.py — 서버가 직접 deskew → 키프레임 → 루프클로저(GICP) → GTSAM PGO.
    #   산출: <root>/graph/{graph.json, odom_log_XXX.npz, *.rrd}. 소비: geometry/pose_provider.py 가 임의 시각 t 의 보정 자세를
    #         T(t) = T_corr(k)·T_odom(k)⁻¹·T_odom(t) 로 보간해 준다 (--pose-graph 옵션으로 각 스크립트에 전달).
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_P(self):
        if self.robot_map:                                                  # 로봇 lidar_slam 최종 키프레임 자세(<name>.txt) → graph/
            if self.done("P", os.path.join(self.graph, "graph.json")): return
            self.run([PY, f"{REPO}/geometry/robot_graph.py", self.mcap_dir, self.graph] + (["--traj", self.robot_map["traj"]] if self.robot_map.get("traj") else []) + ["--rrd"], env={"CUDA_VISIBLE_DEVICES": ""}, logname="P.log"); return   # traj 없으면 /path/pgo 마지막 메시지 + 오도메트리 매칭
        if not self.opt["pose_graph"]: self.say("  pose_graph 꺼짐 → 생략"); return
        if self.done("P", os.path.join(self.graph, "graph.json")): return
        nums = ",".join(str(int(l[4:])) for l in self.logs)
        self.run([PY, f"{REPO}/geometry/build_graph.py", self.mcap_dir, "--logs", nums, "--out", self.graph, "--rrd"], env={"CUDA_VISIBLE_DEVICES": ""}, logname="P.log")

    # ───────────────────────────────────────────────────────────────────────────────────────────
    # A (dataset) — mcap 을 "카메라 프레임 데이터셋" 으로 푼다. ConceptGraphs(D) 와 라벨 투영(C) 의 공통 입력.
    #   스크립트: tools/preprocess/mcap_to_cg.py  (로그별 실행)
    #   하는 일: rgb/depth 메시지를 시각으로 짝지어(pair_ms) 이미지 파일로 저장, 각 프레임의 카메라 자세(traj.txt, P 의
    #            보정 자세로 보간), 시각표(stamps.npz). --lidar 면 depth 의 빈 픽셀을 ±0.4 s LiDAR 누적으로 채운다(lidar_fusion).
    #   산출: dataset/<log>/{results/frame*.jpg, depth*.png, traj.txt, stamps.npz, lidar_world/}  (Replica 데이터셋 레이아웃)
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_A(self):
        for log in self.logs:
            out = self.ds(log)
            if self.done("A", os.path.join(out, "stamps.npz"), lambda: len(glob.glob(os.path.join(out, "results", "frame*.jpg"))) > 100): continue
            cmd = [PY, f"{TOOLS}/mcap_to_cg.py", self.mcap(log), self.calib, out, "--pair-ms", str(self.opt.get("pair_ms", 34))] + (["--lidar"] if self.opt["lidar_fusion"] else []) + self.pg()   # pair_ms: rgb-depth 최대 시차 (sim 카메라는 비동기 → 250)
            self.run(cmd, logname=f"A_{log}.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # S (scans) — LiDAR 스캔을 "world 좌표 점 + 그 시각 센서 원점" 으로 스캔별 npz 저장. occupancy(F)·voxel3d(F3d)·
    #   트랙 2 기하 pcd(C) 의 입력이다. 센서 원점이 같이 있어야 레이캐스팅(어디가 비었는지)이 가능하다.
    #   트랙 2: tools/preprocess/export_registered_scans.py — lidar_slam 이 이미 deskew·정합해 발행한 /cloud_registered 를 쓰되,
    #           발행 시점 실시간 프레임이므로 P 의 Δ(t)=T_corr·T_odom⁻¹ 로 최종 PGO 프레임에 맞춘다.
    #   트랙 1: tools/preprocess/export_lidar_scans.py — raw /livox/lidar 를 점별 시각으로 deskew 해 world 로.
    #   산출: dataset/<log>/lidar_scans/%06d.npz {xyz, origin, stamp_ns}
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_S(self):
        for log in self.logs:
            out = os.path.join(self.ds(log), "lidar_scans")
            if self.done("S", out, lambda: len(glob.glob(os.path.join(out, "*.npz"))) > 100): continue
            if self.robot_map: self.run([PY, f"{TOOLS}/export_registered_scans.py", self.mcap(log), self.calib, out, "--aux-mode", {False: "off", True: "auto"}.get(self.opt.get("aux_lidar", "off"), str(self.opt.get("aux_lidar", "off")))] + self.pg(), logname=f"S_{log}.log")   # /cloud_registered → 최종 PGO 프레임 (+ 보조 LiDAR 채움)
            else: self.run([PY, f"{TOOLS}/export_lidar_scans.py", self.mcap(log), self.calib, out] + self.pg(), logname=f"S_{log}.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # R (rgb) — semantic 추론에 넣을 RGB 를 세션 단위로 뽑는다 (LiDAR 스캔 시각 기준 10 Hz).
    #   스크립트: tools/preprocess/extract_rgb_session.py  (연속 로그 범위 a-b 만 지원)
    #   왜 세션 단위인가: 세그 추론(B) 은 GPU 가 비싼 단계라 로그 간 중복 없이 한 번만 돌리기 위해.
    #   산출: sem/rgb/%06d.png + sem/rgb_index.npz (각 이미지가 어느 로그·어느 시각인지)
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_R(self):
        idx = os.path.join(self.D["sem"], "rgb_index.npz")
        if self.done("R", idx): return
        nums = [int(l[4:]) for l in self.logs]
        if nums != list(range(min(nums), max(nums) + 1)):
            raise RuntimeError("extract_rgb_session 은 연속 로그 범위만 지원 (logs 를 a..b 로 주세요)")
        self.run([PY, f"{TOOLS}/extract_rgb_session.py", self.mcap_dir, self.D["sem"], f"{min(nums)}-{max(nums)}"], logname="R.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # B (semantic) — RGB 마다 픽셀 단위 클래스 확률을 만든다. 지형(stuff) 의미의 유일한 원천. [GPU]
    #   스크립트: tools/preprocess/precompute_semantic.py --classdict <이름>
    #   모델: 사전(classdicts/<이름>.json)의 model 키가 정함 (기본 SegFormer-B4 ADE20K) → 사전 규칙으로 우리 클래스(N채널)로 묶는다.
    #   sem/classes.json 에 어떤 사전으로 만들었는지 기록. 다른 사전으로 만든 sem/ 이 있으면 중단(섞이면 라벨 의미가 어긋남).
    #   산출: sem/sem/%06d.npz (프레임별 N채널 확률) + sem/classes.json
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_B(self):
        n_rgb = len(glob.glob(os.path.join(self.D["sem"], "rgb", "*.png")))
        semdir = os.path.join(self.D["sem"], "sem")
        cj = os.path.join(self.D["sem"], "classes.json")
        if os.path.exists(cj) and not self.dry and "B" not in self.force:          # --force B 면 재계산이므로 가드 생략
            cur = json.load(open(cj)); want = CD.load(self.opt["classdict"])        # 사전 이름뿐 아니라 사전에 묶인 세그 모델까지 같아야 재사용
            if cur.get("name") != want["name"] or (cur.get("model") or CD.DEFAULT_MODEL) != want["model"]:
                raise RuntimeError(f"sem/ 이 다른 클래스 사전({cur.get('name')} / {cur.get('model') or CD.DEFAULT_MODEL})으로 만들어짐 → 새 root 를 쓰거나 --force B (sem 재계산)")
        if self.done("B", semdir, lambda: len(glob.glob(os.path.join(semdir, "*.npz"))) >= n_rgb > 0 and os.path.exists(cj)): return
        self.run([PY, f"{TOOLS}/precompute_semantic.py", self.D["sem"], "--skip-existing", "--classdict", str(self.opt["classdict"])] + (["--size", str(self.opt["seg_size"])] if self.opt.get("seg_size") else []) + (["--tta-flip"] if self.opt.get("seg_tta_flip") else []), logname="B.log")   # seg_size: 모델 입력 긴 변(기본 레지스트리 값)
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # C (voxels) — 3D 기하 점에 B 의 2D 라벨을 투영해 "0.1 m semantic 복셀" 을 만든다. 지형 레이어의 마스터 표현.
    #   트랙 2: ① 기하 pcd 를 정한다 — robot_map.geometry
    #              pcd            : 로봇이 저장한 지도 그대로 (sparse, 다운샘플됨)
    #              keyframe_scans : 키프레임 시각의 /cloud_registered 만 모음 (기본값; 저장 지도의 ~4배 밀도, GT 와 2 cm 내 일치)
    #              all_scans      : S 의 전체 스캔 (가장 조밀하지만 로봇 몸·정지 구간 잔상 포함)
    #              every_N        : N 스캔마다
    #           ② tools/preprocess/label_map_points.py — 프레임별 카메라 자세로 지도 점을 투영, depth z-buffer 로 가림 판정,
    #              클래스 경계 edge_px 안 픽셀 제외, require_depth 면 depth 없는 픽셀엔 라벨 안 줌. depth 카메라 점으로 LiDAR 가
    #              못 본 빈 곳 채움(depth_fill_range/gap, source=2). 여러 로그를 한 번에 → 세션 파일 직접 생성.
    #   트랙 1: tools/preprocess/build_voxel_sem_map.py 로 로그별 복셀 → merge_voxels.py 로 병합.
    #   산출: voxel/session_voxsem.npz {ijk, n_pts, rgb, sem_mean(N채널), n_sem, classdict …} + compiled/robot_map_semantic.pcd
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_C(self):
        if self.robot_map:                                                  # 로봇 지도 점에 라벨 투영 → session_voxsem 직접
            sess = os.path.join(self.D["voxel"], "session_voxsem.npz")
            if self.done("C", sess): return
            geom = self.robot_map.get("geometry", "keyframe_scans"); pcd = self.robot_map["pcd"] if geom == "pcd" else os.path.join(self.D["compiled"], f"geometry_{geom}.pcd")
            if geom != "pcd":                                               # keyframe_scans | all_scans | every_N : S 단계 /cloud_registered 에서 기하 pcd 생성
                sel = ["--keyframes", self.robot_map.get("traj") or os.path.join(self.graph, "keyframes_tum.txt")] if geom == "keyframe_scans" else (["--every", geom.split("_")[1]] if geom.startswith("every_") else [])
                self.run([PY, f"{TOOLS}/scans_to_pcd.py", pcd, "--scans", *[os.path.join(self.ds(l), "lidar_scans") for l in self.logs]] + sel, logname="C_geometry.log")
            self.run([PY, f"{TOOLS}/label_map_points.py", pcd, self.calib, self.D["sem"], sess, "--datasets", *[self.ds(l) for l in self.logs],
                      "--mask-bottom", str(self.opt["mask_bottom_px"]), "--edge-px", str(self.opt["label_edge_px"]), "--require-depth", self.rd_arg(),
                      "--pcd-out", os.path.join(self.D["compiled"], "robot_map_semantic.pcd"), "--depth-fill-range", str(self.opt.get("depth_fill_range", 5.0)), "--depth-gap", str(self.opt.get("depth_fill_gap", 0.15))], logname="C_robot_map.log"); return
        outs = []
        for log in self.logs:
            out = os.path.join(self.D["voxel"], f"{log}_voxsem.npz"); outs.append(out)
            if self.done("C", out): continue
            self.run([PY, f"{TOOLS}/build_voxel_sem_map.py", self.mcap(log), self.calib, self.ds(log), self.D["sem"], out, "--mask-bottom", str(self.opt["mask_bottom_px"]),
                      "--edge-px", str(self.opt["label_edge_px"]), "--require-depth", self.rd_arg()] + self.pg(), logname=f"C_{log}.log")
        sess = os.path.join(self.D["voxel"], "session_voxsem.npz")
        if not self.done("C", sess):
            self.run([PY, f"{CG}/merge_voxels.py", sess] + outs, logname="C_merge.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # D (objects) — 객체(things) 레이어. ConceptGraphs 를 A 의 데이터셋 위에 돌려 객체 인스턴스를 뽑는다. [GPU, 로그별]
    #   실행: /data2/cg/concept-graphs 의 slam/rerun_realtime_mapping.py (우리 패치판; 환경변수로 옵션 전달)
    #   하는 일: 프레임마다 검출(YOLO) + 분할(SAM) + CLIP 임베딩 → 3D 로 역투영 → 프레임 간 같은 객체끼리 병합.
    #   CG_MASK_BOTTOM_PX: 화면 아래 로봇 몸 가림, CG_DROP_CLASSES: 동적 클래스(person) 제외.
    #   주의: 기존 exps/ 가 있으면 지우고 다시 (불완전 detections 재사용 사고 이력).
    #   산출: dataset/<log>/exps/r_mapping_stride10/{obj_json_*.json(bbox·태그), pcd_*.pkl.gz(점군·CLIP)}
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_D(self):
        env = {"YOLO_CONFIG_DIR": "/root/.yolo", "CG_MASK_BOTTOM_PX": str(self.opt["mask_bottom_px"]), "CG_DROP_CLASSES": self.opt["drop_classes"],
               "CG_LOG_SCENE_CLOUD": str(self.opt["scene_cloud"])}
        # 카메라 설정은 세션 캘리브레이션에서 (CG 기본 raibo.yaml 은 옛 로봇 640x480 fx 384.8 고정 → 다른 카메라·해상도면 마스크 크기 불일치로 죽음)
        dcfg = os.path.join(self.D["dataset"], "cg_dataconfig.yaml")
        if not self.dry:
            ci = yaml.safe_load(open(os.path.join(self.calib, "d455_front_color_caminfo.yaml")))["camera_info"]; K = ci["K"]
            yaml.safe_dump({"dataset_name": "replica", "camera_params": {"image_height": int(ci["height"]), "image_width": int(ci["width"]), "fx": float(K[0]), "fy": float(K[4]),
                            "cx": float(K[2]), "cy": float(K[5]), "png_depth_scale": 1000.0, "crop_edge": 0}}, open(dcfg, "w"), sort_keys=False)
        for log in self.logs:
            exp = os.path.join(self.ds(log), "exps", "r_mapping_stride10")
            if self.done("D", exp, lambda: bool(glob.glob(os.path.join(exp, "obj_json_*.json"))) and bool(glob.glob(os.path.join(exp, "pcd_*.pkl.gz")))): continue
            if os.path.isdir(os.path.join(self.ds(log), "exps")) and not self.dry:
                shutil.rmtree(os.path.join(self.ds(log), "exps"))          # 불완전 detections 재사용 방지
            env["CG_HEADLESS_RRD"] = os.path.join(self.ds(log), f"rerun_{log}.rrd")
            self.run([PY, "slam/rerun_realtime_mapping.py", f"scene_id={log}", f"dataset_root={self.D['dataset']}", f"dataset_config={dcfg}"], cwd=CGREPO, env=env, logname=f"D_{log}.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # E (graph) — D 의 로그별 객체를 세션 하나의 "3D scene graph" 로 묶는다. 노드 = 객체, 엣지 = 관계.
    #   ① map_compiler.py merge : 로그 간 같은 객체 병합 → compiled/session_objects.json (+ _clip.npz, CLIP 1024D)
    #   ② map_compiler.py edges : 기하 휴리스틱 엣지 (on top of / next to) → compiled/session_edges.json
    #   ③ voxel_rerun_v2.py     : 객체 bbox 발자국 아래의 지배 지형(C) 조회 → compiled/voxel_session_relations.json (객체-지형 엣지)
    #   ④ (옵션 vlm_edges) vlm_edges.py : Qwen2.5-VL 에 객체 쌍 이미지를 보여 관계 판정 → compiled/session_edges_vlm.json
    #      꺼져 있으면 빈 목록을 저장해 매니페스트 항목은 항상 존재하게 한다.
    # ───────────────────────────────────────────────────────────────────────────────────────────
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
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # F (occupancy) — "무엇이 있다" 뿐 아니라 "어디가 비었다" 를 확률로 아는 레이어. 코스트맵(G)·초기 위치 정합의 근거.
    #   스크립트: build_occupancy.py  (S 의 스캔 + C 의 복셀)
    #   하는 일: 스캔마다 센서 원점→점 레이캐스팅. 끝점 셀 hit(prob_hit), 통과 셀 miss(prob_miss) 를 log-odds 로 누적 (OctoMap 식,
    #            외부 라이브러리 없이 numpy). 한두 번 스친 사람·튀는 점은 miss 가 쌓여 free 가 된다.
    #            C 의 0.1 m 표면 복셀 각각에도 occ/free/미관측 상태를 붙이고, 동적 클래스인데 free 공간이면 surf_removed 표시.
    #   산출: compiled/session_occupancy.npz {ijk(0.2 m), hits, misses, logodds, state, surf_state, surf_removed}
    #   이어서 step_F3d (보행용 3D 사전맵) 를 호출한다.
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_F(self):
        out = os.path.join(self.D["compiled"], "session_occupancy")
        if self.geo_only: self.say("  기하 전용 → 0.2 m 코스트맵용 occupancy 생략 (라벨 필요)")
        elif not self.done("F", out + ".npz"):
            scans = [os.path.join(self.ds(l), "lidar_scans") for l in self.logs]
            self.run([PY, f"{CG}/build_occupancy.py", "--scans", *scans, "--voxsem", os.path.join(self.D["voxel"], "session_voxsem.npz"), "--out", out,
                      "--res", str(self.opt["occupancy_res"]), "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"])],
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="F.log")
        self.step_F3d()                                                   # occupancy 가 이미 있어도 voxel3d/변형은 각자 done 검사
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
            self.run([PY, f"{CG}/build_voxel3d.py", "--scans", *scans, *self.voxsem_args(), "--out", out,
                      "--res-xy", str(v3["res_xy"]), "--res-z", str(v3["res_z"]), "--prob-res", str(v3["prob_res"]), "--free-range", str(v3["free_range"]), "--tile-m", str(v3["tile_m"]),
                      "--fill-flat", str(v3["fill_flat"]), "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"])] + ([] if v3.get("depth", True) else ["--no-depth"]),
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="F_voxel3d.log")
        rrd = os.path.join(self.D["compiled"], "voxel3d_view.rrd")
        if not self.done("F", rrd):
            self.run([PY, f"{CG}/voxel3d_view.py", out + ".npz", rrd, *self.voxsem_args(), "--bev", os.path.join(self.D["compiled"], "voxel3d_bev.png")] + (["--sub", "2"] if len(self.logs) > 2 else []),
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="F_voxel3d_view.log")
        # 변형: iso025 = 0.025 m 정육면체 세부 격자 (확률 격자는 동일 0.1 m). 산출 session_voxel3d_<variant>.npz/_tiles/, 뷰 voxel3d_<variant>_view.rrd (포털이 자동 인식)
        for var in v3.get("variants", []):
            spec = {"iso025": {"res_xy": 0.025, "res_z": 0.025}, "z01": {"res_xy": 0.025, "res_z": 0.1},
                    "bonxai": {"res_xy": v3["res_xy"], "res_z": v3["res_z"], "occupancy": "bonxai"}}.get(var)   # bonxai = 본 맵과 같은 격자·채움, 점유 판정만 Bonxai

            if spec is None: self.say(f"  voxel3d 변형 {var}: 정의 없음 → 생략"); continue
            vout = os.path.join(self.D["compiled"], f"session_voxel3d_{var}")
            if not self.done("F", vout + ".npz", lambda: os.path.exists(os.path.join(vout + "_tiles", "index.json"))):
                self.run([PY, f"{CG}/build_voxel3d.py", "--scans", *scans, *self.voxsem_args(), "--out", vout,
                          "--res-xy", str(spec["res_xy"]), "--res-z", str(spec["res_z"]), "--prob-res", str(v3["prob_res"]), "--free-range", str(v3["free_range"]), "--tile-m", str(v3["tile_m"]),
                          "--fill-flat", str(v3["fill_flat"]), "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"]), "--occupancy", spec.get("occupancy", "own")] + ([] if v3.get("depth", True) else ["--no-depth"]),
                         env={"CUDA_VISIBLE_DEVICES": ""}, logname=f"F_voxel3d_{var}.log")
            vrrd = os.path.join(self.D["compiled"], f"voxel3d_{var}_view.rrd")
            if not self.done("F", vrrd):
                self.run([PY, f"{CG}/voxel3d_view.py", vout + ".npz", vrrd, *self.voxsem_args(), "--bev", os.path.join(self.D["compiled"], f"voxel3d_{var}_bev.png"), "--sub", "4"],
                         env={"CUDA_VISIBLE_DEVICES": ""}, logname=f"F_voxel3d_{var}_view.log")
    opt_default_voxel3d = {"res_xy": 0.025, "res_z": 0.025, "prob_res": 0.1, "free_range": 8.0, "tile_m": 10.0, "fill_flat": 0.0, "depth": True, "variants": []}
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # G (package) — 지금까지의 산출물을 "맵 패키지" 하나로 묶는 매니페스트를 쓴다. 이후 소비자(load_map, verify_requests,
    #   localize, export_pcd, 포털)는 재계산 없이 이 JSON 만 읽는다.
    #   ① compiled/map_manifest.json : map_id, frame(좌표계 설명), geometry_source, classdict 전체, 빌드 옵션, files{키: 경로·크기·설명}
    #      files 키: terrain_voxels(C) / occupancy_probabilistic(F) / objects, objects_clip, edges_*(E) / voxel3d_prior, voxel3d_tiles(F3d)
    #      경로는 <root> 기준 상대경로 (load_map 규약). 파일이 하나라도 없으면 실패시킨다.
    #   ② costmap_from_occupancy.py --register : F 의 occupancy + C 의 라벨 → nav2 등급 코스트맵.
    #      지면 = hits ≥ h_ground 인 가장 낮은 셀, 장애물 = 지면 위 [band_min, clearance] 밴드의 hits ≥ h_obs 셀,
    #      지면 코스트 = 클래스 사전의 role/cost. → compiled/session_costmap_occ.npz, 매니페스트에 costmap_2d(_occ) 등록.
    # ───────────────────────────────────────────────────────────────────────────────────────────
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
        if self.geo_only: files = {}                                       # 기하 전용: 보행 맵(voxel3d)만
        if self.opt.get("voxel3d") and os.path.exists(os.path.join(self.root, "compiled", "session_voxel3d.npz")):
            files["voxel3d_prior"] = ("compiled/session_voxel3d.npz", "보행용 3D 확률 복셀 사전맵: 로봇 VoxelMap 격자(0.025×0.025×0.1 m) ijk/mean_z/logodds/cls/source + 확률 격자(0.1 m) 전체")
            files["voxel3d_tiles"] = ("compiled/session_voxel3d_tiles/index.json", "voxel3d 타일 바이너리(10 m, 20 B/셀) 인덱스 — 로봇은 창 주변 3×3 타일만 적재")
        cdj = os.path.join(self.D["sem"], "classes.json"); cdict = json.load(open(cdj)) if os.path.exists(cdj) else None
        m = {"map_id": self.opt["map_id"], "frame": self.opt["frame"], "pose_graph": ("graph/graph.json" if self.opt["pose_graph"] else None), "geometry_source": ((f"robot_map:{self.robot_map.get('geometry','keyframe_scans')}:" + os.path.basename(self.robot_map.get("pcd") or "path_pgo")) if self.robot_map else "raw scans"), "voxel_m": 0.1, "grid_axes": "row=x, col=y", "occupancy_res_m": self.opt["occupancy_res"], "classdict": cdict,
             "built": time.strftime("%Y-%m-%d %H:%M"), "source": {"mcap_dir": self.mcap_dir, "logs": self.logs, "calib": self.calib}, "options": self.opt, "files": {}}
        v3s = os.path.join(self.root, "compiled", "session_voxel3d_summary.json")
        if os.path.exists(v3s):
            try: js = json.load(open(v3s)); m["voxel3d"] = {k: js.get(k) for k in ("cells", "cells_lidar", "cells_depth", "cells_flat_fill", "total_mb", "tiles", "tile_mb_max", "removed_prob_free", "removed_dynamic")}; m["voxel3d"]["params"] = {k: js["params"].get(k) for k in ("res_xy", "res_z", "prob_res", "tile_m", "frame")}
            except Exception as ex: self.say(f"  voxel3d 요약 읽기 실패: {ex}")
        for k, (p, desc) in files.items():
            full = os.path.join(self.root, p)
            if not os.path.exists(full):
                if self.dry: self.say(f"  (dry) {k} ← {p}"); continue
                raise RuntimeError(f"매니페스트 항목 {k} 의 파일이 없음: {full}")
            m["files"][k] = {"path": p, "bytes": os.path.getsize(full), "desc": desc}
        if not self.dry:
            json.dump(m, open(man, "w"), ensure_ascii=False, indent=1); self.say(f"  매니페스트 작성: {man} ({len(m['files'])} 항목)")
        cm = self.opt["costmap"]; out = os.path.join(self.D["compiled"], "session_costmap_occ")
        if self.geo_only: self.say("  기하 전용 → 코스트맵 생략")
        elif not self.done("G", out + ".npz"):
            self.run([PY, f"{CG}/costmap_from_occupancy.py", man, out, "--register", "--h-ground", str(cm["h_ground"]), "--h-obs", str(cm["h_obs"]),
                      "--clearance", str(cm["clearance"]), "--band-min", str(cm["band_min"]), "--ground-label", str(cm.get("ground_label", "ground")), "--ground-min", str(cm.get("ground_min", 0.3))],
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="G_costmap.log")   # ground_label/ground_min: 지면 셀 라벨 = 지면 역할 클래스 argmax (costmap_from_occupancy)
        if not self.dry:   # 로더 기본 키(costmap_2d)도 같은 파일을 가리키게
            m = json.load(open(man))
            if "costmap_2d_occ" in m["files"] and "costmap_2d" not in m["files"]:
                m["files"]["costmap_2d"] = dict(m["files"]["costmap_2d_occ"]); json.dump(m, open(man, "w"), ensure_ascii=False, indent=1)
            tot = sum(v["bytes"] for v in m["files"].values()); self.say(f"  패키지 {len(m['files'])} 파일 · {tot/1e6:.1f} MB")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # H (view) — 사람이 볼 것. 패키지를 load_map.py 로 읽어 rerun 파일과 요약을 만든다 (CPU).
    #   ① compiled/map_package.rrd : 복셀·객체·엣지·occupancy 를 한 장면에 (기본 0.1 m 점, view_cubes 면 큐브)
    #   ② compiled/summary.json + bev.png : 셀 수·객체 수·크기 등 통계와 상단 뷰 썸네일 — 웹 포털 카드가 읽는다. 항상 갱신.
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_H(self):
        if self.geo_only: self.step_H_geometry()
        else: self.step_H_package()
        self.step_H_potree()

    def step_H_package(self):
        man = os.path.join(self.D["compiled"], "map_manifest.json"); rrd = os.path.join(self.D["compiled"], "map_package.rrd")
        if not self.done("H", rrd):
            cmd = [PY, f"{CG}/load_map.py", man, "--rrd", rrd, "--sub", str(self.opt.get("view_sub", 1))]
            if self.opt.get("view_cubes", False):                  # 기본은 0.1m 점(구) 표현; 큐브(OctoMap 풍)는 옵션
                cmd += ["--coarse", str(self.opt["view_coarse"]), "--cubes"]
            self.run(cmd, env={"CUDA_VISIBLE_DEVICES": ""}, logname="H.log")
        # 요약 JSON + BEV 썸네일 (웹 포털이 읽음) — 항상 갱신
        if not self.dry:
            e = dict(os.environ); e["CUDA_VISIBLE_DEVICES"] = ""
            p = subprocess.run([PY, f"{CG}/load_map.py", man, "--bev", os.path.join(self.D["compiled"], "bev.png")], env=e, capture_output=True, text=True)
            out = p.stdout
            try:                                   # JSON 뒤에 다른 출력(bev …)이 붙어도 첫 JSON 객체만 읽는다
                summ = json.JSONDecoder().raw_decode(out[out.index("{"):])[0]
            except Exception:
                summ = {"error": "summary parse failed", "stdout_tail": out[-300:]}
            summ["build_seconds_total"] = round(time.time() - self.t0)
            json.dump(summ, open(os.path.join(self.D["compiled"], "summary.json"), "w"), ensure_ascii=False, indent=1)
            self.say("  summary.json 갱신")

    def step_H_geometry(self):
        """기하 전용: map_package.rrd(semantic 복셀·객체용) 대신 voxel3d 요약·BEV 를 포털 카드용 summary.json/bev.png 로."""
        if self.dry: return
        comp = self.D["compiled"]; summ = {"geometry_only": True, "build_seconds_total": round(time.time() - self.t0)}
        v3s = os.path.join(comp, "session_voxel3d_summary.json")
        if os.path.exists(v3s):
            js = json.load(open(v3s)); summ["voxel3d"] = {k: js.get(k) for k in ("cells", "cells_lidar", "cells_flat_fill", "total_mb", "tiles", "tile_mb_max", "removed_prob_free")}
        if os.path.exists(os.path.join(comp, "voxel3d_bev.png")): shutil.copyfile(os.path.join(comp, "voxel3d_bev.png"), os.path.join(comp, "bev.png"))
        json.dump(summ, open(os.path.join(comp, "summary.json"), "w"), ensure_ascii=False, indent=1); self.say("  summary.json 갱신 (기하 전용)")

    # ③ compiled/potree/ : Potree 웹 뷰어 (보행 맵 voxel3d·변형 + 정합된 원시 LiDAR 점) — 포털 'Potree' 버튼. options.potree: false 면 생략.
    #    보행 맵이 뷰어보다 새로우면 다시 만든다. 실패해도 빌드는 계속 (뷰어는 검수용).
    def step_H_potree(self):
        if not self.opt.get("potree", True) or self.dry: return
        comp = self.D["compiled"]; idx = os.path.join(comp, "potree", "index.html")
        srcs = [f for f in glob.glob(os.path.join(comp, "session_voxel3d*.npz")) if not f.endswith("_summary.npz")]
        if not srcs: self.say("  Potree: voxel3d 없음 → 생략"); return
        stale = not os.path.exists(idx) or max(os.path.getmtime(f) for f in srcs) > os.path.getmtime(idx)
        if not stale and "H" not in self.force: self.say(f"  건너뜀 (있음): {idx}"); return
        try:
            self.run([PY, f"{REPO}/tools/potree/potree_export.py", self.root, "--force"], env={"CUDA_VISIBLE_DEVICES": ""}, logname="H_potree.log")
        except Exception as ex:
            self.say(f"  Potree 생성 실패(빌드는 계속): {ex}")

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

    # ───────────────────────────────────────────────────────────────────────────────────────────
    # Q (quality) — 빌드 맨 앞에서 mcap 자체의 상태를 점검한다 (토픽별 주기·크기, 유실 구간, 카메라 커버리지).
    #   스크립트: tools/mcap_quality.py. 결과가 나쁘면 뒤 단계 결과를 해석할 때 참고 (빌드를 막지는 않음).
    #   산출: compiled/data_quality.{json, rrd}
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_Q(self):
        out = os.path.join(self.D["compiled"], "data_quality")
        if self.done("Q", out + ".rrd"): return
        nums = [int(l[4:]) for l in self.logs]
        self.run([PY, f"{REPO}/tools/mcap_quality.py", self.mcap_dir, out, "--logs", *map(str, nums), "--img-stride", str(self.opt.get("quality_img_stride", 15))],
                 env={"CUDA_VISIBLE_DEVICES": ""}, logname="Q.log")

    # ───────────────────────────────────────────────────────────────────────────────────────────
    # V (video) — 검수용 배속 영상. ① RGB 만 (session_rgb_x<speed>.mp4) ② RGB | 객체 검출 | 세그멘테이션 3분할
    #   (session_pipeline_x<speed>.mp4). 포털에서 자동재생. video_speed 0 이면 생략.
    # ───────────────────────────────────────────────────────────────────────────────────────────
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

    # go — 단계 문자열 순서대로 step_<X> 를 호출. 각 단계는 산출물이 있으면 건너뛰므로(멱등) 중단된 빌드를 같은 명령으로 재개할 수 있다.
    #      --force X 는 그 단계 산출물만 지우고 다시 만든다. 끝나면 컨테이너(root)가 만든 파일 소유권을 호스트 사용자로 돌린다.
    GEOMETRY_STEPS = ("Q", "P", "S", "F", "G", "H")
    def go(self, steps):
        if self.geo_only:
            skipped = [s for s in steps if s not in self.GEOMETRY_STEPS]; steps = [s for s in steps if s in self.GEOMETRY_STEPS]
            if skipped: self.say(f"기하 전용 → 카메라 단계 생략: {','.join(skipped)}")
        try:
            for s in steps:
                self.say(f"===== 단계 {s} =====")
                getattr(self, f"step_{s}")()
            self.say("완료")
        finally:
            self.chown_outputs()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("session_yaml"); ap.add_argument("--steps", default="Q,P,A,S,R,B,C,D,E,F,G,H,V")
    ap.add_argument("--force", default="", help="다시 만들 단계 (쉼표)"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.session_yaml))
    b = Build(cfg, dry=a.dry_run, force=[s for s in a.force.split(",") if s])
    b.go([s.strip() for s in a.steps.split(",") if s.strip()])
