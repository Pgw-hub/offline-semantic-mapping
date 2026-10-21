"""산출물 블록: G Map package(매니페스트) · H 뷰어(rerun · summary · Potree) · V 영상."""
import glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import CODE, REPO, PY, CGREPO, HF_HOME, CD


class PackageSteps:
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # G (package) — 지금까지의 산출물을 "맵 패키지" 하나로 묶는 매니페스트를 쓴다. 이후 소비자(load_map, verify_requests,
    #   localize, export_pcd, 포털)는 재계산 없이 이 JSON 만 읽는다.
    #   ① compiled/map_manifest.json : map_id, frame(좌표계 설명), geometry_source, classdict 전체, 빌드 옵션, files{키: 경로·크기·설명}
    #      files 키: terrain_voxels(C) / occupancy_probabilistic(F) / objects, objects_clip, edges_*(E) / voxel3d_prior, voxel3d_tiles(F3d)
    #      경로는 <root> 기준 상대경로 (load_map 규약). 파일이 하나라도 없으면 실패시킨다.
    #   ② Application 코스트맵 변환 (pipeline/app_steps.py · apps/costmap_from_occupancy.py --register): F 의 occupancy + C 의 라벨 → nav2 등급 코스트맵.
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
        elif not self.with_objects:                                        # objects: false — semantic 맵까지만 (객체·엣지 항목 없음)
            files = {k: v for k, v in files.items() if k in ("terrain_voxels", "occupancy_probabilistic")}
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
        self.app_costmap(man)                                              # Application: 코스트맵 변환 (pipeline/app_steps.py)
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
            cmd = [PY, f"{CODE}/package/load_map.py", man, "--rrd", rrd, "--sub", str(self.opt.get("view_sub", 1))]
            if self.opt.get("view_cubes", False):                  # 기본은 0.1m 점(구) 표현; 큐브(OctoMap 풍)는 옵션
                cmd += ["--coarse", str(self.opt["view_coarse"]), "--cubes"]
            self.run(cmd, env={"CUDA_VISIBLE_DEVICES": ""}, logname="H.log")
        # 요약 JSON + BEV 썸네일 (웹 포털이 읽음) — 항상 갱신
        if not self.dry:
            e = dict(os.environ); e["CUDA_VISIBLE_DEVICES"] = ""
            p = subprocess.run([PY, f"{CODE}/package/load_map.py", man, "--bev", os.path.join(self.D["compiled"], "bev.png")], env=e, capture_output=True, text=True)
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
            self.run([PY, f"{CODE}/package/potree_export.py", self.root, "--force"], env={"CUDA_VISIBLE_DEVICES": ""}, logname="H_potree.log")
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
