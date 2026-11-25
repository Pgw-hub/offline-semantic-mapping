"""산출물 블록: G 매니페스트(메인 맵 · 보행 맵) · H 뷰어(Potree · summary) · V 영상."""
import glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import CODE, REPO, PY, CGREPO, HF_HOME, CD


class PackageSteps:
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # G (package) — 두 출력(메인 맵 · 보행 맵)의 파일 목록을 매니페스트 하나로. 소비자(포털, 로봇 맵 패키지, load_map, export_pcd)는 이 JSON 만 읽는다.
    #   compiled/map_manifest.json : map_id, frame, classdict, 빌드 옵션,
    #     files{키: 경로(<root> 기준)·크기·형식 설명}  — 필수 파일이 없으면 실패
    #     maps{main_map: [키…], locomotion_map: [키…]} — 어느 파일이 어느 출력에 속하는지
    #   옛 0.2 m occupancy(O 단계)가 있으면 nav2 코스트맵(Application, app_steps)도 만들어 등록한다 (기본 빌드에는 없음).
    # ───────────────────────────────────────────────────────────────────────────────────────────
    MAIN_FILES = [   # (키, 경로, 필수 조건, 설명)
        ("main_map", "compiled/main_map/main_map.pcd", "always", "메인 맵 기하: PCD x y z (float32 binary). 5 cm 확률 복셀 점유 칸마다 점 하나(칸 중심 xy + 칸 안 최고 z)"),
        ("localization_map", "compiled/main_map/localization_map.pcd", "always", "위치추정 지도: 메인 맵 0.25 m 칸 평균, PCD x y z intensity (lidar_slam 이 읽는 필드)"),
        ("main_map_semantic", "compiled/main_map/main_map_semantic.pcd", "semantic", "점별 의미: PCD x y z rgb(의미 색) label(u32 클래스 id, 254 other, 255 없음) — main_map 과 같은 점·순서"),
        ("main_map_semantic_points", "compiled/main_map/main_map_semantic.npz", "semantic", "점별 label·확신도·카메라 색·관측 수 (npz)"),
        ("terrain_voxels", "voxel/session_voxsem.npz", "semantic", "0.1 m semantic 복셀 (ijk, sem_mean, n_sem …) — 객체-지형 관계·옛 도구용"),
        ("objects", "compiled/session_objects.json", "objects", "3D scene graph 노드: 객체 bbox center/extent/tag/num_det (JSON)"),
        ("objects_clip", "compiled/session_objects_clip.npz", "objects", "객체 CLIP ViT-H/14 임베딩 1024D (objects 와 같은 순서)"),
        ("edges_heuristic", "compiled/session_edges.json", "objects", "3D scene graph 엣지: 기하 휴리스틱 (on top of / next to)"),
        ("edges_object_terrain", "compiled/voxel_session_relations.json", "objects", "3D scene graph 엣지: 객체-지형 (bbox 발자국 아래 지배 지형)"),
        ("edges_vlm", "compiled/session_edges_vlm.json", "objects", "3D scene graph 엣지: VLM 판정 (Qwen2.5-VL, 꺼져 있으면 빈 목록)"),
        ("nav3d_navigation_map", "compiled/nav3d/navigation_map.pcd", "optional", "3D 내비 입력 = main_map.pcd − 위에서 본 적 없는 천장·지붕 윗면 (로봇 log/map/<맵>/ 에 이 이름으로)"),
        ("nav3d_graph", "compiled/nav3d/nav3d_graph.bin", "optional", "통행성 그래프 + 토폴로지 (raisin_navigation3d_plugin GraphFile, 원본 pcd 해시·파라미터 포함)"),
        ("nav3d_summary", "compiled/nav3d/nav3d_graph_summary.json", "optional", "통행성 그래프 요약: 노드 수, 장소(방·계단), 포털"),
        ("nav3d_nodes", "compiled/nav3d/nav3d_nodes.csv", "optional", "통행성 그래프 노드 x y z clearance place component (CSV)"),
    ]
    LOCO_FILES = [
        ("locomotion_tiles", "compiled/locomotion/tiles/index.json", "always", "보행 맵 (최종): voxel3d_tiles v2 — 10 m 타일, 24 B/셀, 2.5 cm 정육면체, 계단 보완 포함. 로봇 voxel3d_tiles/"),
        ("locomotion_stair_model", "compiled/locomotion/tiles_stair.json", "always", "계단 모델 (디딤판 높이·앞모서리·채움 수; 계단 없으면 stair: null)"),
        ("locomotion_base_tiles", "compiled/locomotion/base_tiles/index.json", "always", "보행 맵 바탕: 2.5 cm 확률 복셀 그대로 (계단 보완 전)"),
        ("locomotion_points", "compiled/locomotion/locomotion.npz", "always", "최종 보행 맵 셀 배열 (npz: ijk, mean_z, max_z, source …) — Potree·PCD 내보내기"),
    ]

    def step_G(self):
        man = os.path.join(self.D["compiled"], "map_manifest.json")
        need = {"always": True, "optional": False, "semantic": not self.geo_only, "objects": self.with_objects}
        groups = {"main_map": self.MAIN_FILES, "locomotion_map": self.LOCO_FILES if self.loco_opts() else []}
        cdj = os.path.join(self.D["sem"], "classes.json"); cdict = json.load(open(cdj)) if os.path.exists(cdj) else None
        m = {"map_id": self.opt["map_id"], "frame": self.opt["frame"], "pose_graph": ("graph/graph.json" if self.opt["pose_graph"] else None),
             "geometry_source": "main_map (5 cm LiDAR 확률 복셀, compiled/main_map/main_map.pcd)", "layout": "2026-11-24 main_map + locomotion_map",
             "voxel_m": 0.1, "grid_axes": "row=x, col=y", "classdict": cdict, "built": time.strftime("%Y-%m-%d %H:%M"),
             "source": {"mcap_dir": self.mcap_dir, "logs": self.logs, "calib": self.calib}, "options": self.opt, "files": {}, "maps": {}}
        missing = []
        for g, lst in groups.items():
            m["maps"][g] = []
            for k, p, cond, desc in lst:
                full = os.path.join(self.root, p)
                if not os.path.exists(full):
                    if need.get(cond, False): missing.append(p)
                    continue
                e = {"path": p, "bytes": os.path.getsize(full), "desc": desc}
                if p.endswith("index.json"):                                     # 타일 폴더: 폴더 전체 크기
                    td = os.path.dirname(full); e["dir_bytes"] = sum(os.path.getsize(os.path.join(td, f)) for f in os.listdir(td)); e["tiles"] = len(glob.glob(os.path.join(td, "t_*.bin")))
                m["files"][k] = e; m["maps"][g].append(k)
        occ = os.path.join(self.D["compiled"], "session_occupancy.npz")
        if os.path.exists(occ):
            m["files"]["occupancy_probabilistic"] = {"path": "compiled/session_occupancy.npz", "bytes": os.path.getsize(occ), "desc": "(옛) 0.2 m 확률 점유 — O 단계"}
        for f, key in (("main_map/main_map_summary.json", "main_map"), ("locomotion/base_summary.json", "locomotion")):
            fp = os.path.join(self.D["compiled"], f)
            if os.path.exists(fp):
                js = json.load(open(fp)); m.setdefault("stats", {})[key] = {k: js.get(k) for k in ("cells", "removed_prob_free", "n_points", "seconds")} | {"params": {k: js["params"].get(k) for k in ("res_xy", "res_z", "max_range", "free_range")}}
        if missing:
            if self.dry: self.say(f"  (dry) 아직 없는 파일: {missing}")
            else: raise RuntimeError(f"매니페스트 필수 파일 없음: {missing}")
        if self.dry: return
        json.dump(m, open(man, "w"), ensure_ascii=False, indent=1)
        sz = lambda g: sum(m["files"][k].get("dir_bytes", m["files"][k]["bytes"]) for k in m["maps"][g])
        self.say(f"  매니페스트: 메인 맵 {len(m['maps']['main_map'])} 파일 {sz('main_map') / 1e6:.1f} MB · 보행 맵 {len(m['maps']['locomotion_map'])} 파일 {sz('locomotion_map') / 1e6:.1f} MB")
        if os.path.exists(occ):
            self.app_costmap(man)                                              # (옛) Application: 코스트맵 변환
            m = json.load(open(man))
            if "costmap_2d_occ" in m["files"] and "costmap_2d" not in m["files"]:
                m["files"]["costmap_2d"] = dict(m["files"]["costmap_2d_occ"]); json.dump(m, open(man, "w"), ensure_ascii=False, indent=1)

    # ───────────────────────────────────────────────────────────────────────────────────────────
    # H (view) — 사람이 볼 것.
    #   ① compiled/potree/ : Potree 웹 뷰어 (package/potree_export.py) — 포털의 기본 보기. 메인 맵(의미 색·카메라 색·높이)·위치추정 지도·
    #      보행 맵·통행성 그래프 노드를 점 레이어로, 토폴로지(장소·포털)와 3D scene graph(객체 상자·라벨·엣지)를 겹쳐 그리는 선·라벨로. 패널에서 클릭으로 켜고 끔.
    #   ② compiled/summary.json + bev.png : 포털 카드용 통계·윗면 썸네일 (package/map_summary.py). 항상 갱신.
    #   ③ (options.rerun: true) compiled/map_package.rrd — 옛 rerun 보기 (load_map.py)
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_H(self):
        if self.dry: return
        comp = self.D["compiled"]
        try:
            self.run([PY, f"{CODE}/package/map_summary.py", self.root, "--build-seconds", str(round(time.time() - self.t0))], env={"CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": "/tmp/mpl"}, logname="H.log")
        except Exception as ex: self.say(f"  요약 실패(빌드는 계속): {ex}")
        if self.opt.get("rerun") and not self.geo_only:
            rrd = os.path.join(comp, "map_package.rrd")
            if not self.done("H", rrd):
                try: self.run([PY, f"{CODE}/package/load_map.py", os.path.join(comp, "map_manifest.json"), "--rrd", rrd, "--sub", str(self.opt.get("view_sub", 1))], env={"CUDA_VISIBLE_DEVICES": ""}, logname="H.log")
                except Exception as ex: self.say(f"  rerun 실패(빌드는 계속): {ex}")
        self.step_H_potree()

    def step_H_potree(self):
        po = self.opt.get("potree", True)
        if not po: return
        comp = self.D["compiled"]; idx = os.path.join(comp, "potree", "index.html")
        srcs = [os.path.join(comp, f) for f in ("main_map/main_map.pcd", "main_map/main_map_semantic.npz", "locomotion/locomotion.npz", "nav3d/nav3d_nodes.csv", "session_objects.json", "map_manifest.json")]
        srcs = [f for f in srcs if os.path.exists(f)]
        if not srcs: self.say("  Potree: 메인 맵 없음 → 생략"); return
        stale = not os.path.exists(idx) or max(os.path.getmtime(f) for f in srcs) > os.path.getmtime(idx)
        if not stale and "H" not in self.force: self.say(f"  건너뜀 (있음): {idx}"); return
        try:
            self.run([PY, f"{CODE}/package/potree_export.py", self.root, "--force"] + (["--layers", "main,loco,nav,loc,raw"] if isinstance(po, dict) and po.get("raw") else []),
                     env={"CUDA_VISIBLE_DEVICES": ""}, logname="H_potree.log")
        except Exception as ex:
            self.say(f"  Potree 생성 실패(빌드는 계속): {ex}")

    # ───────────────────────────────────────────────────────────────────────────────────────────
    # V (video) — 검수용 배속 영상. ① RGB 만 (session_rgb_x<speed>.mp4) ② RGB | 객체 검출 | 세그멘테이션 3분할
    #   (session_pipeline_x<speed>.mp4). 포털에서 자동재생. video_speed 0 이면 생략.
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_V(self):
        sp = int(self.opt.get("video_speed", 0) or 0)
        if sp <= 0: self.say("  video_speed 0 → 생략"); return
        out = os.path.join(self.D["compiled"], f"session_rgb_x{sp}.mp4")
        if not self.done("V", out):
            self.run([PY, f"{CODE}/package/make_rgb_video.py", self.D["dataset"], out, "--speed", str(sp), "--width", str(self.opt.get("video_width", 960)), "--logs", *self.logs], logname="V.log")
        if self.opt.get("video_pipeline", True) and self.with_objects:   # RGB | 검출 | 세그멘테이션 3분할 (검출 = D 단계 결과)
            out2 = os.path.join(self.D["compiled"], f"session_pipeline_x{sp}.mp4")
            if not self.done("V", out2):
                self.run([PY, f"{CODE}/package/make_pipeline_video.py", self.root, out2, "--speed", str(sp), "--logs", *self.logs], logname="V_pipeline.log")
