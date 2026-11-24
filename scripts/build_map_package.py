#!/usr/bin/env python3
"""build_map_package.py — mcap 세션 → Map Package(map_manifest.json) 를 한 번에 만드는 오케스트레이터.

컨테이너 offline_3d_semantic_map 안에서 실행 (docker exec 없음, 단일 프로세스가 단계별 스크립트를 subprocess 로 호출).
사용:
  python build_map_package.py <session.yaml>              # 전부 (이미 있는 산출물은 건너뜀). root 가 이름만이면 <repo>/output/<이름>
  python build_map_package.py <session.yaml> --steps D,E  # 일부 단계만
  python build_map_package.py <session.yaml> --force D    # 해당 단계 산출물 지우고 다시
  python build_map_package.py <session.yaml> --dry-run    # 명령만 출력
출력은 두 개 (2026-11-24 구조):
  ① 메인 맵   compiled/main_map/ — 5 cm 확률 복셀 지도(LiDAR 전부, 거리 제한 없음) 하나 위에
              점별 의미 라벨(main_map_semantic.pcd) · 3D scene graph(session_objects·edges) · 통행성 그래프·토폴로지(nav3d/) ·
              위치추정 지도(localization_map.pcd = 메인 맵을 0.25 m 로 줄인 것)
  ② 보행 맵   compiled/locomotion/tiles/ — 2.5 cm 확률 복셀 지도(LiDAR, 5 m 이내) + 계단 인식·디딤판/앞면 채움·halo 제거. 의미 정보 없음
  보기: compiled/potree/index.html (Potree, 위 레이어를 클릭으로 켜고 끔) — 포털의 기본 보기.
단계 (기본 순서 Q,P,A,S,M,R,B,C,D,E,N,L,G,H,V):
  Q  quality    mcap 데이터 품질(토픽별 dt/rate/size, 유실·커버리지 이벤트) → compiled/data_quality.{rrd,json}
  P  posegraph  robot_map: preprocess/geometry/robot_graph.py --traj (로봇 키프레임 자세 → graph/) | 트랙1: deskew 키프레임 + 루프클로저 + GTSAM PGO (build_graph.py)
  A  dataset    mcap → dataset/<log>/ (results/, traj.txt, stamps.npz, lidar_world/)   [로그별]
  S  scans      robot_map: /cloud_registered → 최종 PGO 프레임 (export_registered_scans.py) | 트랙1: raw deskew (export_lidar_scans.py) → dataset/<log>/lidar_scans/
  M  main map   [메인 맵] 스캔 전부 → 5 cm 확률 복셀 → main_map/main_map.pcd (+ localization_map.pcd 0.25 m)     (pipeline/main_map_steps.py)
  R  rgb        mcap → sem/rgb/ (LiDAR 시각 10Hz RGB) + rgb_index.npz                  [세션]
  B  semantic   sem/rgb → sem/sem/ (사전의 세그 모델, N채널) [세션, GPU]
  C  labels     [메인 맵] 메인 맵 점마다 세그 확률 투영 (label_map_points.py) → main_map/main_map_semantic.{pcd,npz} + voxel/session_voxsem.npz(0.1 m)
  D  objects    [메인 맵] ConceptGraphs → dataset/<log>/exps/r_mapping_stride10/      [로그별, GPU]
  E  graph      [메인 맵] 객체 병합 → compiled/session_objects.json(+clip), 휴리스틱 엣지, 객체-지형 엣지, (옵션) VLM 엣지
  N  nav3d      [메인 맵] main_map.pcd 위 통행성 그래프·토폴로지 → compiled/nav3d/ (raisin_navigation3d_plugin 실행 파일, 없으면 생략)
  L  locomotion [보행 맵] 2.5 cm 확률 복셀(5 m) → 계단 보완 → compiled/locomotion/tiles/          (pipeline/locomotion_map_steps.py)
  G  package    map_manifest.json (두 맵의 파일 목록·크기)
  H  view       compiled/potree/ Potree 뷰어 + summary.json·bev.png (포털 카드). options.rerun: true 면 map_package.rrd 도
  V  video      RGB 배속 영상 + 3분할(RGB|검출|세그) (options.video_speed 0 이면 생략)
  옛 단계: F → L 로 넘김. O = 0.2 m occupancy(기본 아님, 있으면 G 가 nav2 코스트맵도 만듦).
기하 전용 (options.geometry_only: true): 카메라 없이 Q,P,S,M,N,L,G,H 만 (메인 맵은 라벨·scene graph 없이 기하만).
레이아웃: <root>/{dataset,sem,voxel,compiled}/ ; 매니페스트 경로는 <root> 기준 상대경로 (load_map 규약).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # scripts/ (패키지 import)
import argparse, glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import CODE, REPO, PY, CGREPO, HF_HOME, CD            # 공통 상수 (pipeline/env.py)
from pipeline.core import BuildCore
from pipeline.preprocess_steps import PreprocessSteps
from pipeline.main_map_steps import MainMapSteps
from pipeline.semantic_steps import SemanticSteps
from pipeline.locomotion_map_steps import LocomotionMapSteps
from pipeline.nav3d_steps import Nav3dSteps
from pipeline.package_steps import PackageSteps
from pipeline.app_steps import AppSteps


# ─── 단계 사이 데이터 흐름 (한 세션 기준) ────────────────────────────────────────────────────────────
#
#   mcap ─┬─ Q 품질 점검
#         ├─ P 자세 그래프 graph/ ──────────────┬───────────────┐
#         ├─ A 카메라 데이터셋 dataset/<log>/ ◄─┘(자세 보간)     │
#         ├─ S LiDAR 스캔 lidar_scans/ ◄────────────────────────┘(최종 프레임 보정)
#         └─ R 세션 RGB sem/rgb/ ─► B 세그 모델 sem/sem/ (GPU)
#
#   S ───────────────► M 메인 맵 5 cm 확률 복셀 main_map/main_map.pcd ─┬─► localization_map.pcd (0.25 m)        ┐
#   A + B + M ───────► C 점별 의미 라벨 main_map_semantic.pcd (+ voxsem 0.1 m)  │                                 │ ① 메인 맵
#   A ───────────────► D ConceptGraphs 객체 (GPU) ─► E 3D scene graph (노드·엣지) │                                 │
#   M ───────────────► N 통행성 그래프·토폴로지 nav3d/nav3d_graph.bin ◄─────────┘                                 ┘
#   S ───────────────► L 보행 맵 2.5 cm (5 m) ─► 계단 보완 locomotion/tiles/                                         ② 보행 맵
#   ①② ──────────────► G map_manifest.json ─► H Potree 뷰어·summary ─► V 영상
#
#   각 step_X 메서드 위 주석에 그 단계의 목적·스크립트·산출물을 적어 두었다.
# ────────────────────────────────────────────────────────────────────────────────────────────────────
#   단계 구현은 pipeline/ 의 블록별 모듈에 있다 (Miro 구조도와 같은 나눔):
#     preprocess_steps (Q P A S R B) · main_map_steps (M C) · semantic_steps (D E, 옛 O) · nav3d_steps (N) · locomotion_map_steps (L) · package_steps (G H V) · app_steps (옛 코스트맵) · core (공통·실행)
class Build(PreprocessSteps, MainMapSteps, SemanticSteps, LocomotionMapSteps, Nav3dSteps, PackageSteps, AppSteps, BuildCore):
    """세션 하나를 빌드한다. 단계 메서드는 각 블록 mixin 에, 공통 유틸·실행 순서는 BuildCore 에."""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("session_yaml"); ap.add_argument("--steps", default="Q,P,A,S,M,R,B,C,D,E,N,L,G,H,V")
    ap.add_argument("--force", default="", help="다시 만들 단계 (쉼표)"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.session_yaml))
    b = Build(cfg, dry=a.dry_run, force=[s for s in a.force.split(",") if s])
    b.go([s.strip() for s in a.steps.split(",") if s.strip()])
