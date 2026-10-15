# Semantic Map 데이터 흐름

offline_semantic_mapping 의 모든 모듈과 데이터를 하나의 의존 그래프로 그린 것.
위에서 아래로 흐르고, 화살표는 "이 데이터를 저 모듈이 읽는다" 또는 "저 모듈이 이 데이터를 만든다".

- 마우스를 올리면 연결선만 남는 인터랙티브 버전: [data_flow.html](data_flow.html) (브라우저로 열기)
- 아래 Mermaid 그림은 VSCode(Markdown Preview Mermaid 확장) / GitHub 에서 렌더링됨

```mermaid
flowchart TD
  %% 외부 입력
  odom[/"mcap /Odometry/base · /path/pgo<br/>+ 로봇 저장 궤적 &lt;name&gt;.txt"/]:::in
  creg[/"mcap /cloud_registered<br/>SLAM 정합 점군"/]:::in
  calib[/"calibration/<br/>LiDAR·카메라 파라미터"/]:::in
  rgbd[/"mcap d455 rgb · depth"/]:::in
  cdict[/"classdicts/&lt;name&gt;.json<br/>클래스·role·cost"/]:::in
  models[/"모델 가중치<br/>SegFormer · YOLO/SAM/CLIP · Qwen"/]:::in

  %% 자세
  P["P · robot_graph.py<br/>로봇 PGO 자세 → graph/"]:::mod
  graph[("graph/graph.json<br/>T_odom ↔ T_corr, 키프레임 시각")]:::mid
  odom --> P --> graph

  %% 1차 추출
  S["S · export_registered_scans.py<br/>Δ(t) 로 최종 프레임 보정"]:::mod
  A["A · mcap_to_cg.py<br/>rgb/depth 짝짓기, 카메라 자세"]:::mod
  R["R · extract_rgb_session.py<br/>LiDAR 시각 10 Hz RGB"]:::mod
  graph -- "Δ(t) 보정" --> S
  creg --> S
  calib --> S
  graph -- "자세 보간" --> A
  calib --> A
  rgbd --> A
  rgbd --> R

  scans[("dataset/&lt;log&gt;/lidar_scans/<br/>스캔별 xyz(world) + 센서 원점")]:::mid
  dataset[("dataset/&lt;log&gt;/<br/>frame·depth·traj.txt·stamps")]:::mid
  srgb[("sem/rgb/ + rgb_index<br/>세션 RGB")]:::mid
  S --> scans
  A --> dataset
  R --> srgb

  %% GPU 인식
  D["D · ConceptGraphs<br/>YOLO·SAM·CLIP [GPU]"]:::mod
  B["B · precompute_semantic.py<br/>SegFormer-B4 [GPU]"]:::mod
  dataset --> D
  models -.-> D
  srgb --> B
  cdict --> B
  models -.-> B

  exps[("dataset/&lt;log&gt;/exps/<br/>로그별 객체 bbox·점군·CLIP")]:::mid
  sem[("sem/sem/ + classes.json<br/>픽셀별 N채널 확률")]:::mid
  D --> exps
  B --> sem

  %% 라벨 투영
  C["C · scans_to_pcd + label_map_points<br/>점에 라벨 투영, depth 채움"]:::mod
  graph -- "키프레임 목록" --> C
  scans -- "기하 (키프레임 스캔)" --> C
  dataset -- "traj · depth(z-buffer) · calib" --> C
  calib --> C
  sem -- "라벨" --> C

  V["V · make_*_video.py<br/>배속·3분할 영상"]:::mod
  dataset --> V
  sem --> V
  exps --> V
  mp4[["session_*_x15.mp4"]]:::out
  V --> mp4

  voxsem[["voxel/session_voxsem.npz<br/>0.1 m semantic 복셀 (지형)"]]:::out
  C --> voxsem

  %% 레이어 생성
  F3d["F3d · build_voxel3d.py<br/>확률 점유 + mean_z, 타일"]:::mod
  F["F · build_occupancy.py<br/>레이캐스팅 hit/miss log-odds"]:::mod
  E["E · map_compiler merge·edges / voxel_rerun_v2<br/>객체 병합, 엣지 (옵션 VLM)"]:::mod
  scans --> F3d
  scans -- "레이캐스팅" --> F
  voxsem -- "클래스·depth_xyz" --> F3d
  voxsem -- "동적 복셀 판정" --> F
  voxsem -- "객체-지형 엣지" --> E
  exps --> E
  models -. "Qwen (옵션)" .-> E

  voxel3d[["session_voxel3d.npz + _tiles/<br/>보행용 3D 사전맵"]]:::out
  occ[["session_occupancy.npz<br/>0.2 m 확률 occupancy"]]:::out
  objects[["session_objects.json · session_edges*.json<br/>+ voxel_session_relations.json (scene graph)"]]:::out
  F3d --> voxel3d
  F --> occ
  E --> objects

  %% 패키지
  G["G · step_G (매니페스트 작성)<br/>파일 목록·프레임·옵션 + sem/classes.json"]:::mod
  voxsem --> G
  voxel3d --> G
  occ --> G
  objects --> G
  manifest[["compiled/map_manifest.json<br/>맵 패키지 진입점"]]:::out
  G --> manifest

  costmap_m["G · costmap_from_occupancy.py<br/>지면 hits≥H, 밴드 장애물, role/cost"]:::mod
  H["H · load_map.py<br/>패키지 전체 로드 → 시각화·요약"]:::mod
  consumers(["소비자 (매니페스트만 읽음)<br/>map_compiler · localize(:9093) · verify_requests · export_pcd"]):::use
  manifest --> costmap_m
  manifest --> H
  manifest --> consumers

  costmap[["session_costmap_occ.npz<br/>nav2 등급 코스트맵"]]:::out
  rrd[["map_package.rrd · summary.json · bev.png"]]:::out
  costmap_m --> costmap
  H --> rrd
  costmap -. "costmap_2d 로 등록" .-> manifest

  portal(["tools/map_portal.py (웹 포털 :9092)<br/>카드·상세·빌드 큐·다운로드"]):::use
  rrd --> portal
  mp4 --> portal
  manifest --> portal

  classDef in fill:#e9eef7,stroke:#6b7fa6,color:#1f2320
  classDef mod fill:#e4efe6,stroke:#4f7f5a,color:#1f2320
  classDef mid fill:#f1efe8,stroke:#8a8577,stroke-dasharray:4 3,color:#1f2320
  classDef out fill:#fbead8,stroke:#b5622a,stroke-width:2px,color:#1f2320
  classDef use fill:#efe8f5,stroke:#7a5e9a,color:#1f2320
```

**범례**

| 모양·색 | 의미 |
|---|---|
| 평행사변형 (파랑) | 외부 입력 |
| 사각형 (초록) | 모듈 (단계 글자 · 스크립트) |
| 원통 (회색 점선) | 중간 데이터 (패키지에 안 들어감) |
| 이중 테두리 (주황) | 패키지 산출물 (매니페스트 등록) |
| 둥근 사각형 (보라) | 패키지를 읽는 쪽 |
| 점선 화살표 | 모델 가중치 / 되돌아 등록 |

Q(mcap 품질 점검, `tools/mcap_quality.py`)는 mcap 만 읽고 포털에만 표시되는 독립 가지라 생략했다.
트랙 1(로봇 저장 지도 없이 서버가 PGO 를 직접 하는 옛 방식)은 P·S·C 자리의 스크립트만 다르고 그래프 모양은 같다.

## 그림에서 보이는 재사용 지점

- **graph/ 는 세 번 쓰인다.** S 가 스캔을 최종 프레임으로 옮길 때, A 가 카메라 프레임 자세를 보간할 때, C 가 "키프레임 시각" 목록을 고를 때. 자세 하나를 모든 레이어가 공유하므로 지도 안에서 기하가 어긋나지 않는다.
- **lidar_scans/ 는 세 번 쓰인다.** C 의 기하 점(키프레임 스캔), F 의 레이캐스팅, F3d 의 3D 사전맵. 같은 점군을 해상도와 목적만 바꿔 세 레이어로 만든다.
- **session_voxsem.npz 는 출력이면서 네 모듈의 입력이다.** F·F3d 는 동적 복셀 제거와 클래스 부여에, E 는 객체 아래 지형을 보는 데, G 는 패키지에 넣는 데 쓴다.
- **dataset/ 은 네 모듈이 읽는다.** D(객체 검출)뿐 아니라 C(카메라 자세·depth 가림 판정), V(영상)도 같은 프레임 파일을 쓴다.
- **map_manifest.json 은 끝이 아니라 허브다.** 코스트맵·시각화·소비자·포털이 전부 이것만 읽고, 코스트맵은 만들어진 뒤 다시 매니페스트에 등록된다(되돌아가는 점선).
- **GPU 가 필요한 모듈은 B, D 와 옵션 E(VLM) 뿐**이다. 나머지는 CPU 기하 계산이라 산출물이 있으면 건너뛰고 재개할 수 있다.
