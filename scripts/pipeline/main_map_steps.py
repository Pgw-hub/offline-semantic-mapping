"""메인 맵 블록 (M · C): 5 cm 확률 복셀 지도 하나 → 기하 pcd · 위치추정 지도 · 점별 의미 라벨.

메인 맵이 빌드의 첫 번째 산출물이다. 같은 점들 위에 의미 라벨(C), 3D scene graph(D·E), 통행성 그래프·토폴로지(N) 가 얹히고,
이것을 성기게 줄인 것이 위치추정 지도다. 보행 맵(L, locomotion_map_steps)만 따로 2.5 cm 로 다시 만든다.
"""
import os

from pipeline.env import CODE, PY


class MainMapSteps:
    # 기본값 (세션 options.main_map 으로 덮어씀)
    #   res          확률 격자 = 세부 격자 = 출력 격자 (m)
    #   max_range    hit 으로 쓸 최대 거리 — 사실상 제한 없음 (Mid-360 최대 ~40 m). 보행 맵의 5 m 제한과 다르다
    #   free_range   빈 공간 레이(miss) 최대 길이. 사람·문처럼 움직인 것을 지우는 거리. 길수록 깨끗하지만 시간은 길이에 비례
    #   ror          외톨이 칸 지우기 [격자 m, 26-이웃 중 최소 점유 칸 수] (허공의 잡음 점)
    #   loc_voxel    위치추정 지도 격자 (m) — 칸마다 점 평균 하나 (로봇 relocalization.globalmap_voxel_size 이하면 로봇이 다시 줄이지 않음)
    #   sparse_far   [N, R]: R m 보다 가까이서 본 적 없고 N 스캔 이하에서만 맞은 칸 제거 — 유리창 너머·유리 반사 고스트(건물 밖 점).
    #                벽에 막혀 다른 스캔의 빈 공간 레이가 닿지 않으니 확률 판정으로는 안 지워진다 (서울 cut715: 건물 밖 점 87 % 제거, 안쪽 0.5 %). null 이면 끔
    opt_default_main_map = {"res": 0.05, "max_range": 100.0, "free_range": 40.0, "ror": [0.1, 5], "loc_voxel": 0.25, "sparse_far": [2, 10.0]}

    def main_opts(self):
        m = self.opt.get("main_map") or {}
        return dict(self.opt_default_main_map, **(m if isinstance(m, dict) else {}))

    def main_dir(self): return os.path.join(self.D["compiled"], "main_map")
    def main_pcd(self): return os.path.join(self.main_dir(), "main_map.pcd")
    def main_elev(self): return os.path.join(self.main_dir(), "main_map_elev.npy")

    # ───────────────────────────────────────────────────────────────────────────────────────────
    # M (main map) — S 의 LiDAR 스캔 전부로 5 cm 확률 복셀 지도 (OctoMap 식 log-odds, locomotion_map/build_voxel3d.py 를 5 cm 정육면체로).
    #   스캔마다 센서 원점→점 레이캐스팅: 끝점 칸 hit, 지나간 칸 miss. 한두 번 스친 사람·튀는 점은 miss 가 쌓여 지워진다.
    #   ① compiled/main_map/main_map.pcd : 점유 칸마다 점 하나 (칸 중심 xy + 칸 안 최고 z), 외톨이 칸 제거 (nav3d/navmap_to_pcd.py).
    #      이 파일이 메인 맵의 기하 — 의미 라벨(C)·3D 내비 그래프(N)·위치추정 지도가 모두 이 점들에서 나온다.
    #   ② compiled/main_map/localization_map.pcd : main_map 을 loc_voxel(0.25 m) 칸 평균으로 (x y z intensity) — 로봇 lidar_slam 위치추정용
    #   ③ main_map_summary.json : 칸 수·제거 수·시간 (확률 격자 npz 는 수백 MB 라 지운다)
    #   ④ main_map_elev.npy : 점마다 '센서가 그 점보다 가장 높았을 때의 높이'(m, main_map.pcd 와 같은 순서) — N 단계가 천장 윗면을 빼는 데 씀
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_M(self):
        o = self.main_opts(); d = self.main_dir(); pcd = self.main_pcd()
        if not self.dry: os.makedirs(d, exist_ok=True)
        if not self.done("M", pcd):
            npz = os.path.join(d, "main_map_voxels"); r = str(o["res"])
            scans = [os.path.join(self.ds(l), "lidar_scans") for l in self.logs]
            self.run([PY, f"{CODE}/locomotion_map/build_voxel3d.py", "--scans", *scans, "--out", npz, "--res-xy", r, "--res-z", r, "--prob-res", r,
                      "--max-range", str(o["max_range"]), "--free-range", str(o["free_range"]), "--no-tiles", "--only-hit-cells",
                      *(["--sparse-far", *map(str, o["sparse_far"])] if o.get("sparse_far") else []),
                      "--prob-hit", str(self.opt["prob_hit"]), "--prob-miss", str(self.opt["prob_miss"])], env={"CUDA_VISIBLE_DEVICES": ""}, logname="M_main_map.log")
            self.run([PY, f"{CODE}/nav3d/navmap_to_pcd.py", npz + ".npz", pcd, "--elev-out", self.main_elev()] + (["--ror", *map(str, o["ror"])] if o["ror"] else []), logname="M_main_map.log")
            if not self.dry:
                if os.path.exists(npz + "_summary.json"): os.replace(npz + "_summary.json", os.path.join(d, "main_map_summary.json"))
                if os.path.exists(npz + ".npz"): os.remove(npz + ".npz")
        loc = os.path.join(d, "localization_map.pcd")
        if not self.done("M", loc, lambda: os.path.getmtime(loc) >= os.path.getmtime(pcd)):
            self.run([PY, f"{CODE}/package/robot_map_package.py", "--localization-only", pcd, loc, "--loc-voxel", str(o["loc_voxel"])], logname="M_main_map.log")

    # ───────────────────────────────────────────────────────────────────────────────────────────
    # C (semantic labels) — 메인 맵 점 하나하나에 B 의 2D 라벨을 투영한다 (semantic/voxel/label_map_points.py).
    #   프레임별 카메라 자세(A 의 traj.txt) → 지도 점을 카메라에 투영 → depth z-buffer 로 가림 판정 → 클래스 경계 edge_px 안 픽셀 제외
    #   → 보이는 점에 픽셀 semantic 누적. 점을 더하지 않는다(depth 채움 없음) — 메인 맵 기하는 LiDAR 복셀 지도 그대로.
    #   산출: compiled/main_map/main_map_semantic.pcd (x y z rgb label — 점별 클래스, main_map.pcd 와 같은 점·순서)
    #         compiled/main_map/main_map_semantic.npz (점별 label·확신도·카메라 색 — Potree 레이어)
    #         voxel/session_voxsem.npz (0.1 m 로 묶은 semantic 복셀 — E 의 객체-지형 관계가 읽음)
    # ───────────────────────────────────────────────────────────────────────────────────────────
    def step_C(self):
        sess = os.path.join(self.D["voxel"], "session_voxsem.npz"); spcd = os.path.join(self.main_dir(), "main_map_semantic.pcd")
        if self.done("C", sess, lambda: os.path.exists(spcd) and os.path.getmtime(spcd) >= os.path.getmtime(self.main_pcd())): return
        self.run([PY, f"{CODE}/semantic/voxel/label_map_points.py", self.main_pcd(), self.calib, self.D["sem"], sess, "--datasets", *[self.ds(l) for l in self.logs],
                  "--mask-bottom", str(self.opt["mask_bottom_px"]), "--edge-px", str(self.opt["label_edge_px"]), "--require-depth", self.rd_arg(),
                  "--pcd-out", spcd, "--points-out", os.path.join(self.main_dir(), "main_map_semantic.npz")], logname="C_labels.log")
