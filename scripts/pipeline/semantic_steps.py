"""3D Scene Graph 블록: D 객체(ConceptGraphs) · E 세션 scene graph. (+ 옛 0.2 m occupancy 레이어 O — 기본 단계 아님)
의미 라벨 투영(C)은 메인 맵 블록(main_map_steps)으로 옮겼다."""
import glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import CODE, REPO, PY, CGREPO, HF_HOME, CD


class SemanticSteps:
    # 객체 융합(E ①): vlm_model = 라벨을 고를 비전 LLM (none 이면 검출 투표). 서울 cut715 점검 표본 라벨 정확도 34 % → 83 %
    opt_default_objects = {"vlm_model": "Qwen/Qwen2.5-VL-32B-Instruct"}
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
    #   ① object_fusion.py : 로그별 객체를 점구름 겹침 + 의미로 한꺼번에 군집화 → Qwen2.5-VL 이 여러 장면 crop 을 보고 라벨
    #      (구조물이면 객체 아님) → 같은 라벨끼리 2차 병합 → compiled/session_objects.json (+ _clip.npz, _rejected.json)
    #      options.objects.vlm_model (기본 Qwen2.5-VL-32B, GPU ~70 GB) · none 이면 검출 투표 라벨. 옛 병합은 map_compiler.py merge
    #   ② map_compiler.py edges : 기하 휴리스틱 엣지 (on top of / next to) → compiled/session_edges.json
    #   ③ voxel_rerun_v2.py     : 객체 bbox 발자국 아래의 지배 지형(C) 조회 → compiled/voxel_session_relations.json (객체-지형 엣지)
    #   ④ (옵션 vlm_edges) vlm_edges.py : Qwen2.5-VL 에 객체 쌍 이미지를 보여 관계 판정 → compiled/session_edges_vlm.json
    #      꺼져 있으면 빈 목록을 저장해 매니페스트 항목은 항상 존재하게 한다.
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_E(self):
        env = {"CG_RAIBO_ROOT": self.D["dataset"], "CG_OUT_DIR": self.D["compiled"]}
        so = os.path.join(self.D["compiled"], "session_objects.json")
        if not self.done("E", so):
            o = dict(self.opt_default_objects, **(self.opt.get("objects") if isinstance(self.opt.get("objects"), dict) else {}))
            self.run([PY, f"{CODE}/semantic/scene_graph/object_fusion.py", "--root", self.D["dataset"], "--out", self.D["compiled"], "--vlm-model", str(o["vlm_model"] or "none")],
                     env=dict(env, HF_HOME=HF_HOME, HF_HUB_OFFLINE="1"), logname="E_merge.log")
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
    # O (occupancy, 옛 기능 — 기본 단계 아님) — 0.2 m 확률 점유 격자. nav2 2D 코스트맵(G 의 Application)의 입력이었다.
    #   3D 내비(N)·메인 맵으로 바뀌면서 기본 빌드에서 뺐다. 필요하면 --steps ...,O,G 로 (G 가 occupancy 가 있으면 코스트맵을 만든다).
    #   스크립트: semantic/occupancy/build_occupancy.py (S 의 스캔 + C 의 0.1 m semantic 복셀) → compiled/session_occupancy.npz
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_O(self):
        out = os.path.join(self.D["compiled"], "session_occupancy")
        if self.geo_only: self.say("  기하 전용 → occupancy 생략 (라벨 필요)"); return
        if self.done("O", out + ".npz"): return
        scans = [os.path.join(self.ds(l), "lidar_scans") for l in self.logs]
        self.run([PY, f"{CODE}/semantic/occupancy/build_occupancy.py", "--scans", *scans, "--voxsem", os.path.join(self.D["voxel"], "session_voxsem.npz"), "--out", out,
                  "--res", str(self.opt.get("occupancy_res", 0.2)), "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"])],
                 env={"CUDA_VISIBLE_DEVICES": ""}, logname="O.log")
