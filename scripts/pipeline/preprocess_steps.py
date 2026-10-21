"""전처리 블록: Q 데이터 품질 · P 자세 정렬 · A 카메라 데이터셋 · S LiDAR 스캔 정리 · R RGB 추출 · B 세그멘테이션."""
import glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import CODE, REPO, PY, CGREPO, HF_HOME, CD


class PreprocessSteps:
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # Q (quality) — 빌드 맨 앞에서 mcap 자체의 상태를 점검한다 (토픽별 주기·크기, 유실 구간, 카메라 커버리지).
    #   스크립트: preprocess/mcap_quality.py. 결과가 나쁘면 뒤 단계 결과를 해석할 때 참고 (빌드를 막지는 않음).
    #   산출: compiled/data_quality.{json, rrd}
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_Q(self):
        out = os.path.join(self.D["compiled"], "data_quality")
        if self.done("Q", out + ".rrd"): return
        nums = [int(l[4:]) for l in self.logs]
        self.run([PY, f"{CODE}/preprocess/mcap_quality.py", self.mcap_dir, out, "--logs", *map(str, nums), "--img-stride", str(self.opt.get("quality_img_stride", 15))],
                 env={"CUDA_VISIBLE_DEVICES": ""}, logname="Q.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # P (posegraph) — "어느 시각에 로봇이 어디 있었나"를 정하는 기하의 뼈대. 뒤 단계(A/S/C)가 모두 이 자세를 쓴다.
    #   왜 필요한가: mcap 의 /Odometry/base 는 실시간 SLAM 자세라 루프클로저 전 드리프트가 남아 있다.
    #                최종 보정(PGO) 자세로 바꿔 줘야 지도 내부가 어긋나지 않는다.
    #   트랙 2(robot_map 있음): preprocess/geometry/robot_graph.py — 로봇 lidar_slam 이 저장한 키프레임 자세(<name>.txt, 없으면 /path/pgo)
    #                            를 읽어 graph/graph.json (T_odom ↔ T_corr 쌍) 으로 변환. 서버는 최적화를 안 한다.
    #   트랙 1(robot_map 없음): preprocess/geometry/build_graph.py — 서버가 직접 deskew → 키프레임 → 루프클로저(GICP) → GTSAM PGO.
    #   산출: <root>/graph/{graph.json, odom_log_XXX.npz, *.rrd}. 소비: preprocess/geometry/pose_provider.py 가 임의 시각 t 의 보정 자세를
    #         T(t) = T_corr(k)·T_odom(k)⁻¹·T_odom(t) 로 보간해 준다 (--pose-graph 옵션으로 각 스크립트에 전달).
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_P(self):
        if self.robot_map:                                                  # 로봇 lidar_slam 최종 키프레임 자세(<name>.txt) → graph/
            if self.done("P", os.path.join(self.graph, "graph.json")): return
            self.run([PY, f"{CODE}/preprocess/geometry/robot_graph.py", self.mcap_dir, self.graph] + (["--traj", self.robot_map["traj"]] if self.robot_map.get("traj") else []) + ["--rrd"], env={"CUDA_VISIBLE_DEVICES": ""}, logname="P.log"); return   # traj 없으면 /path/pgo 마지막 메시지 + 오도메트리 매칭
        if not self.opt["pose_graph"]: self.say("  pose_graph 꺼짐 → 생략"); return
        if self.done("P", os.path.join(self.graph, "graph.json")): return
        nums = ",".join(str(int(l[4:])) for l in self.logs)
        self.run([PY, f"{CODE}/preprocess/geometry/build_graph.py", self.mcap_dir, "--logs", nums, "--out", self.graph, "--rrd"], env={"CUDA_VISIBLE_DEVICES": ""}, logname="P.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # A (dataset) — mcap 을 "카메라 프레임 데이터셋" 으로 푼다. ConceptGraphs(D) 와 라벨 투영(C) 의 공통 입력.
    #   스크립트: preprocess/mcap_to_cg.py  (로그별 실행)
    #   하는 일: rgb/depth 메시지를 시각으로 짝지어(pair_ms) 이미지 파일로 저장, 각 프레임의 카메라 자세(traj.txt, P 의
    #            보정 자세로 보간), 시각표(stamps.npz). --lidar 면 depth 의 빈 픽셀을 ±0.4 s LiDAR 누적으로 채운다(lidar_fusion).
    #   산출: dataset/<log>/{results/frame*.jpg, depth*.png, traj.txt, stamps.npz, lidar_world/}  (Replica 데이터셋 레이아웃)
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_A(self):
        for log in self.logs:
            out = self.ds(log)
            if self.done("A", os.path.join(out, "stamps.npz"), lambda: len(glob.glob(os.path.join(out, "results", "frame*.jpg"))) > 100): continue
            cmd = [PY, f"{CODE}/preprocess/mcap_to_cg.py", self.mcap(log), self.calib, out, "--pair-ms", str(self.opt.get("pair_ms", 34))] + (["--lidar"] if self.opt["lidar_fusion"] else []) + self.pg()   # pair_ms: rgb-depth 최대 시차 (sim 카메라는 비동기 → 250)
            self.run(cmd, logname=f"A_{log}.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # S (scans) — LiDAR 스캔을 "world 좌표 점 + 그 시각 센서 원점" 으로 스캔별 npz 저장. occupancy(F)·voxel3d(F3d)·
    #   트랙 2 기하 pcd(C) 의 입력이다. 센서 원점이 같이 있어야 레이캐스팅(어디가 비었는지)이 가능하다.
    #   트랙 2: preprocess/export_registered_scans.py — lidar_slam 이 이미 deskew·정합해 발행한 /cloud_registered 를 쓰되,
    #           발행 시점 실시간 프레임이므로 P 의 Δ(t)=T_corr·T_odom⁻¹ 로 최종 PGO 프레임에 맞춘다.
    #   트랙 1: preprocess/export_lidar_scans.py — raw /livox/lidar 를 점별 시각으로 deskew 해 world 로.
    #   산출: dataset/<log>/lidar_scans/%06d.npz {xyz, origin, stamp_ns}
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_S(self):
        for log in self.logs:
            out = os.path.join(self.ds(log), "lidar_scans")
            if self.done("S", out, lambda: len(glob.glob(os.path.join(out, "*.npz"))) > 100): continue
            if self.robot_map: self.run([PY, f"{CODE}/preprocess/export_registered_scans.py", self.mcap(log), self.calib, out, "--aux-mode", {False: "off", True: "auto"}.get(self.opt.get("aux_lidar", "off"), str(self.opt.get("aux_lidar", "off")))] + self.pg(), logname=f"S_{log}.log")   # /cloud_registered → 최종 PGO 프레임 (+ 보조 LiDAR 채움)
            else: self.run([PY, f"{CODE}/preprocess/export_lidar_scans.py", self.mcap(log), self.calib, out] + self.pg(), logname=f"S_{log}.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # R (rgb) — semantic 추론에 넣을 RGB 를 세션 단위로 뽑는다 (LiDAR 스캔 시각 기준 10 Hz).
    #   스크립트: preprocess/extract_rgb_session.py  (연속 로그 범위 a-b 만 지원)
    #   왜 세션 단위인가: 세그 추론(B) 은 GPU 가 비싼 단계라 로그 간 중복 없이 한 번만 돌리기 위해.
    #   산출: sem/rgb/%06d.png + sem/rgb_index.npz (각 이미지가 어느 로그·어느 시각인지)
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_R(self):
        idx = os.path.join(self.D["sem"], "rgb_index.npz")
        if self.done("R", idx): return
        nums = [int(l[4:]) for l in self.logs]
        if nums != list(range(min(nums), max(nums) + 1)):
            raise RuntimeError("extract_rgb_session 은 연속 로그 범위만 지원 (logs 를 a..b 로 주세요)")
        self.run([PY, f"{CODE}/preprocess/extract_rgb_session.py", self.mcap_dir, self.D["sem"], f"{min(nums)}-{max(nums)}"], logname="R.log")
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # B (semantic) — RGB 마다 픽셀 단위 클래스 확률을 만든다. 지형(stuff) 의미의 유일한 원천. [GPU]
    #   스크립트: preprocess/precompute_semantic.py --classdict <이름>
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
        self.run([PY, f"{CODE}/preprocess/precompute_semantic.py", self.D["sem"], "--skip-existing", "--classdict", str(self.opt["classdict"])] + (["--size", str(self.opt["seg_size"])] if self.opt.get("seg_size") else []) + (["--batch", str(self.opt["seg_batch"])] if self.opt.get("seg_batch") else []) + (["--tta-flip"] if self.opt.get("seg_tta_flip") else []), logname="B.log")   # seg_size: 모델 입력 긴 변(기본 레지스트리 값)
