"""Semantic 블록: C Semantic 3D voxel builder · D/E 3D Scene Graph builder · F 점유 레이어 0.2 m (끝에 보행용 맵 F3d 호출)."""
import glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import CODE, REPO, PY, CGREPO, HF_HOME, CD


class SemanticSteps:
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # C (voxels) — 3D 기하 점에 B 의 2D 라벨을 투영해 "0.1 m semantic 복셀" 을 만든다. 지형 레이어의 마스터 표현.
    #   트랙 2: ① 기하 pcd 를 정한다 — robot_map.geometry
    #              pcd            : 로봇이 저장한 지도 그대로 (sparse, 다운샘플됨)
    #              keyframe_scans : 키프레임 시각의 /cloud_registered 만 모음 (기본값; 저장 지도의 ~4배 밀도, GT 와 2 cm 내 일치)
    #              all_scans      : S 의 전체 스캔 (가장 조밀하지만 로봇 몸·정지 구간 잔상 포함)
    #              every_N        : N 스캔마다
    #           ② semantic/voxel/label_map_points.py — 프레임별 카메라 자세로 지도 점을 투영, depth z-buffer 로 가림 판정,
    #              클래스 경계 edge_px 안 픽셀 제외, require_depth 면 depth 없는 픽셀엔 라벨 안 줌. depth 카메라 점으로 LiDAR 가
    #              못 본 빈 곳 채움(depth_fill_range/gap, source=2). 여러 로그를 한 번에 → 세션 파일 직접 생성.
    #   트랙 1: semantic/voxel/build_voxel_sem_map.py 로 로그별 복셀 → merge_voxels.py 로 병합.
    #   산출: voxel/session_voxsem.npz {ijk, n_pts, rgb, sem_mean(N채널), n_sem, classdict …} + compiled/robot_map_semantic.pcd
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_C(self):
        if self.robot_map:                                                  # 로봇 지도 점에 라벨 투영 → session_voxsem 직접
            sess = os.path.join(self.D["voxel"], "session_voxsem.npz")
            if self.done("C", sess): return
            geom = self.robot_map.get("geometry", "keyframe_scans"); pcd = self.robot_map["pcd"] if geom == "pcd" else os.path.join(self.D["compiled"], f"geometry_{geom}.pcd")
            if geom != "pcd":                                               # keyframe_scans | all_scans | every_N : S 단계 /cloud_registered 에서 기하 pcd 생성
                sel = ["--keyframes", self.robot_map.get("traj") or os.path.join(self.graph, "keyframes_tum.txt")] if geom == "keyframe_scans" else (["--every", geom.split("_")[1]] if geom.startswith("every_") else [])
                self.run([PY, f"{CODE}/semantic/voxel/scans_to_pcd.py", pcd, "--scans", *[os.path.join(self.ds(l), "lidar_scans") for l in self.logs]] + sel, logname="C_geometry.log")
            self.run([PY, f"{CODE}/semantic/voxel/label_map_points.py", pcd, self.calib, self.D["sem"], sess, "--datasets", *[self.ds(l) for l in self.logs],
                      "--mask-bottom", str(self.opt["mask_bottom_px"]), "--edge-px", str(self.opt["label_edge_px"]), "--require-depth", self.rd_arg(),
                      "--pcd-out", os.path.join(self.D["compiled"], "robot_map_semantic.pcd"), "--depth-fill-range", str(self.opt.get("depth_fill_range", 5.0)), "--depth-gap", str(self.opt.get("depth_fill_gap", 0.15))], logname="C_robot_map.log"); return
        outs = []
        for log in self.logs:
            out = os.path.join(self.D["voxel"], f"{log}_voxsem.npz"); outs.append(out)
            if self.done("C", out): continue
            self.run([PY, f"{CODE}/semantic/voxel/build_voxel_sem_map.py", self.mcap(log), self.calib, self.ds(log), self.D["sem"], out, "--mask-bottom", str(self.opt["mask_bottom_px"]),
                      "--edge-px", str(self.opt["label_edge_px"]), "--require-depth", self.rd_arg()] + self.pg(), logname=f"C_{log}.log")
        sess = os.path.join(self.D["voxel"], "session_voxsem.npz")
        if not self.done("C", sess):
            self.run([PY, f"{CODE}/semantic/voxel/merge_voxels.py", sess] + outs, logname="C_merge.log")
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
            self.run([PY, f"{CODE}/semantic/scene_graph/map_compiler.py", "merge"], env=env, logname="E_merge.log")
        if not self.done("E", os.path.join(self.D["compiled"], "session_edges.json")):
            self.run([PY, f"{CODE}/semantic/scene_graph/map_compiler.py", "edges"], env=env, logname="E_edges.log")
        rel = os.path.join(self.D["compiled"], "voxel_session_relations.json")
        if not self.done("E", rel):
            self.run([PY, f"{CODE}/semantic/scene_graph/voxel_rerun_v2.py", os.path.join(self.D["voxel"], "session_voxsem.npz"), so, self.D["dataset"], os.path.join(self.D["compiled"], "voxel_session.rrd"), "--unknown-sub", "4"],
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="E_terrain.log")
        vlm = os.path.join(self.D["compiled"], "session_edges_vlm.json")
        if self.opt["vlm_edges"]:
            if not self.done("E", vlm):
                for log in self.logs:
                    out = os.path.join(self.D["compiled"], f"vlm_edges_{log}_v2")
                    if os.path.exists(os.path.join(out, "edges_claude.json")): self.say(f"  건너뜀 (있음): {out}"); continue
                    self.run([PY, f"{CODE}/semantic/scene_graph/vlm_edges.py", "--scene", log, "--dataset-root", self.D["dataset"], "--out", out, "--backend", "local",
                              "--local-model", self.opt["vlm_model"], "--max-pairs", str(self.opt["vlm_max_pairs"])], env={"HF_HOME": HF_HOME}, logname=f"E_vlm_{log}.log")
                self.run([PY, f"{CODE}/semantic/scene_graph/edges_session_rerun.py", self.D["dataset"], self.D["compiled"], os.path.join(self.D["compiled"], "vlm_edges_session.rrd")], logname="E_vlm_session.log")
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
            self.run([PY, f"{CODE}/semantic/occupancy/build_occupancy.py", "--scans", *scans, "--voxsem", os.path.join(self.D["voxel"], "session_voxsem.npz"), "--out", out,
                      "--res", str(self.opt["occupancy_res"]), "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"])],
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="F.log")
        self.step_F3d()                                                   # occupancy 가 이미 있어도 voxel3d/변형은 각자 done 검사
