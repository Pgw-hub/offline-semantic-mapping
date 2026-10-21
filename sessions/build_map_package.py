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
  P  posegraph  robot_map: preprocess/geometry/robot_graph.py --traj (로봇 키프레임 자세 → graph/) | 트랙1: deskew 키프레임 + 루프클로저 + GTSAM PGO (preprocess/geometry/build_graph.py)
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
                + 보행용 3D 확률 복셀 사전맵 (locomotion_map/build_voxel3d.py: OctoMap 식 log-odds + 복셀 평균 z, 로봇 VoxelMap 격자 0.025×0.025×0.1 m) → compiled/session_voxel3d.npz
                  + session_voxel3d_tiles/(10 m 타일 바이너리, 로봇 적재용) + voxel3d_view.rrd/voxel3d_bev.png (기본 0.025 정육면체, 2026-10-15~) + 선택 변형 z01(로봇 VoxelMap 격자 0.025×0.025×0.1, options.voxel3d.variants). options.voxel3d: false 면 생략
  G  package    map_manifest.json 작성 + 코스트맵(costmap_from_occupancy) 등록
  H  view       compiled/map_package.rrd — 기본 0.1m 점 표현 (options.view_cubes: true 면 view_coarse m 큐브)
                + compiled/potree/ Potree 웹 뷰어 (package/potree_export.py; options.potree: false 면 생략)
  Q  quality    mcap 데이터 품질(토픽별 dt/rate/size, 유실·커버리지 이벤트) + 재생 rerun → compiled/data_quality.{rrd,json}
  V  video      RGB 배속 영상 compiled/session_rgb_x<speed>.mp4 + 3분할(RGB|검출|세그) session_pipeline_x<speed>.mp4 (options.video_speed 0 이면 생략)
기하 전용 (options.geometry_only: true): 카메라 없이 LiDAR 만으로 보행 맵(voxel3d 타일)을 만든다. Q,P,S,F,G,H 만 돌고(A,R,B,C,D,E,V 생략),
                F 는 0.2 m 코스트맵(라벨 필요)을 건너뛰고 voxel3d 만 --voxsem 없이(클래스 unlabeled) 만든다. 의미는 나중에 별도 세션으로 입힌다.
레이아웃: <root>/{dataset,sem,voxel,compiled}/ ; 매니페스트 경로는 <root> 기준 상대경로 (load_map 규약).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # 저장소 루트 (패키지 import)
import argparse, glob, json, os, shutil, subprocess, sys, time
import yaml

from pipeline.env import REPO, PY, CGREPO, HF_HOME, CD            # 공통 상수 (pipeline/env.py)
from pipeline.core import BuildCore
from pipeline.preprocess_steps import PreprocessSteps
from pipeline.semantic_steps import SemanticSteps
from pipeline.locomotion_map_steps import LocomotionMapSteps
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
#   A + B (+S 기하) ─► C 0.1 m semantic 복셀 voxel/session_voxsem.npz          ← 지형(stuff) 레이어
#   A ───────────────► D ConceptGraphs 객체 (GPU) ─► E 세션 scene graph (노드·엣지) ← 객체(things) 레이어
#   S + C ───────────► F 확률 occupancy 0.2 m  +  F3d 보행용 voxel3d (0.025/0.1 m 타일)
#   C,E,F,F3d ───────► G map_manifest.json + 코스트맵   ─► H rerun·summary   ─► V 영상
#
#   각 step_X 메서드 위 주석에 그 단계의 목적·스크립트·산출물을 적어 두었다.
# ────────────────────────────────────────────────────────────────────────────────────────────────────
#   단계 구현은 pipeline/ 의 블록별 모듈에 있다 (Miro 구조도와 같은 나눔):
#     preprocess_steps (Q P A S R B) · semantic_steps (C D E F) · locomotion_map_steps (F3d) · package_steps (G H V) · app_steps (코스트맵) · core (공통·실행)
class Build(PreprocessSteps, SemanticSteps, LocomotionMapSteps, PackageSteps, AppSteps, BuildCore):
    """세션 하나를 빌드한다. 단계 메서드는 각 블록 mixin 에, 공통 유틸·실행 순서는 BuildCore 에."""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("session_yaml"); ap.add_argument("--steps", default="Q,P,A,S,R,B,C,D,E,F,G,H,V")
    ap.add_argument("--force", default="", help="다시 만들 단계 (쉼표)"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.session_yaml))
    b = Build(cfg, dry=a.dry_run, force=[s for s in a.force.split(",") if s])
    b.go([s.strip() for s in a.steps.split(",") if s.strip()])
