# 의미 코스트맵을 raisin 내비게이션에 넣기 — 작업 의뢰서

작성: 2026-10-03, 오프라인 의미 맵 서버 세션(blackwell 서버, `~/offline_semantic_mapping`)
대상: raisin / raisin_gui 가 빌드돼 있고 GUI 를 볼 수 있는 컨테이너에서 작업할 에이전트

## 1. 목표
로봇(GPU 없음)이 인도·횡단보도로만 다니고 차도는 피하게 한다. 서버가 카메라·LiDAR 녹화로 만든 의미 맵에서
**차도에 높은 코스트, 인도·횡단보도에 낮은 코스트**를 준 2D 격자(OccupancyGrid 규약)를 파일로 넘기고,
`raisin_navigation_plugin` 의 costmap 이 이 격자를 반영하게 한다. 로봇은 파일을 읽기만 하며 추론은 하지 않는다.

## 2. 넘기는 파일
위치: 서버 `/data2/semantic_nav_handoff/t3_sim_1002b_tex/` (컨테이너에 /data2 가 없으면 이 폴더를 통째로 복사)

| 파일 | 내용 |
|---|---|
| `semantic/semantic_mask.png` + `.yaml` | nav2 map_server 규약 이미지, `mode: raw`. 픽셀 값 = OccupancyGrid 값(0~100), 255 = -1(정보 없음). 0.2 m, origin `[-40.4, -20.2]`, 328×299 셀(65.6×59.8 m) |
| `semantic/semantic_mask.npy` | 같은 데이터를 OccupancyGrid `data` 순서 그대로 (int8, shape (H=y, W=x), 0행 = y 최소). C++ 에서 PNG 대신 쓰고 싶으면 참고 |
| `semantic/semantic_info.json` | 값 대응표, 원점, 셀 수, 출처 |
| `semantic/geofence_block_road.json` | 차도에서 횡단보도를 뺀 영역의 폴리곤 4개 (맵 좌표 m). 코드 수정 없이 시험하는 1단계용 |
| `semantic/topview.png` | 미리보기 (흰 인도, 노랑 횡단보도, 보라 차도, 초록 잔디, 회색 정보 없음) |
| `reference/robot_map_semantic.pcd` | 의미 라벨이 붙은 3D 점군 (시각화·좌표 정합용) |
| `reference/map_manifest.json`, `topview_compare.png` | 빌드 정보, GT 대비 정확도 그림 |

값 대응 (KeepoutFilter 가 `cost = round(v × 254 / 100)` 로 변환):

| 클래스 | 마스크 값 | nav2 코스트 |
|---|---|---|
| 인도 | 4 | 10 |
| 횡단보도 | 8 | 20 |
| 잔디 | 35 | 89 |
| 차도 | 90 | 229 |
| 그 외(라벨 없는 지면, 건물, 장애물, 미관측) | -1 | 필터가 건너뜀 → 실시간 레이어가 판단 |

정적 장애물은 일부러 넣지 않았다. 실시간 ObstacleLayer·HeightMapLayer 가 이미 처리한다.

정확도(시뮬 테스트베드 설계 기하를 GT 로 셀 단위 비교): 인도 IoU 0.92, 횡단보도 0.92, 차도 0.86, GT 인도 중 차도로 잘못 칠해진 비율 1.3%, 횡단보도 재현 99%.

## 2-1. 권장 방식: 서버 빌드 맵을 그대로 Load Map (2026-10-03 추가)
서버 포털 맵 상세 페이지의 **Download for raisin** 버튼이 `<맵이름>_raisin_r90.zip` 을 준다. 압축을 `<raisin 워크스페이스>/log/map/` 아래에 풀면:
```
log/map/<맵이름>/
  <맵이름>.pcd              x y z intensity (lidar_slam::Point 필드; intensity = 의미 라벨 id, semantic_pcd_legend.json)
  semantic/semantic_mask.png + .yaml   이 PCD 와 같은 좌표계의 의미 코스트 격자 (frame_id = <맵이름>)
  semantic/geofence_block_road.json    1단계 시험용 차도 폴리곤
  README_semantic.txt
```
- lidar_slam `loadMapFromDisk` 는 `<이름>.pcd` 만 필수이고 `_sc.bin`·`_georef.txt` 는 없으면 건너뛴다. 그래서 GUI Load Map 목록에 바로 뜬다.
- scan context 가 없으므로 전역 위치추정 대신 **초기 자세를 주고 재위치추정**한다. 맵 원점 = 녹화 시작 시 로봇 자세(sim spawn, yaw≈0)라서, sim 을 같은 spawn 에서 띄우면 초기 자세 (0, 0, 0)으로 맞는다.
- 이 방식이면 PCD 와 의미 격자가 같은 빌드에서 나와 **아래 3절의 좌표계 문제가 생기지 않는다.**

## 3. 좌표계 — 가장 먼저 확인할 것
- 격자 좌표는 **녹화 `railab_raibo_20261002_105020` 당시 lidar_slam 의 월드 프레임**이다(`/cloud_registered` 를 그대로 누적). 그 녹화에는 로봇이 저장한 맵(PCD)이 없었다.
- 로봇의 Load Map 은 lidar_slam 이 저장한 맵(`<ws>/log/map/<name>/<name>.pcd` + `_sc.bin` + `_georef.txt`)을 불러와 재위치추정한다. 재위치추정 중에는 `map_name → id_` TF 가 항등이라 costmap 프레임(`id_`) = 맵 프레임이 된다(아래 4절).
- 따라서 이 격자를 쓰려면 로봇이 Load Map 할 맵의 프레임과 격자 프레임이 같아야 한다. 둘 중 하나:
  1. **권장**: 시뮬에서 매핑한 뒤 그 세션에서 맵을 저장하고, 그 녹화(mcap)를 서버에 주면 서버가 같은 프레임으로 격자를 다시 만들어 준다. 서버 빌드는 약 5분.
  2. 이미 저장된 시뮬 맵이 있으면 `reference/robot_map_semantic.pcd` 를 그 맵 PCD 에 ICP 로 정합해 2D 변환(x, y, yaw)을 구하고, 격자에 그 변환을 적용한다. 테스트베드가 같으므로 정합은 잘 될 것이다.
- 시험 전에 RViz/GUI 에서 `topview.png` 가 로드한 맵의 도로와 겹치는지 눈으로 먼저 확인할 것.

## 4. raisin 내비게이션 현재 구조 (서버에서 소스를 읽어 확인한 내용)
경로는 `raisin_master/src/raisin_plugin/raisin_navigation_plugin/` 기준. 줄 번호는 서버에 있는 사본 기준이라 컨테이너 사본과 다를 수 있다.

- **costmap 레이어는 C++ 에 하드코딩**되어 있다(`src/navigation_node.cpp` 약 390–473행; YAML 의 `plugin:` 키는 사실상 설명용). 로컬·글로벌 둘 다 같은 순서다.
  1. ObstacleLayer (`/cloud_registered`)
  2. `raisin::plugin::HeightMapLayer`
  3. DenoiseLayer
  4. InflationLayer
  5. **KeepoutFilter** (`addFilter`)
- **StaticLayer 는 없고, `/map` 퍼블리셔·map_server 도 없다.** 글로벌 costmap 도 100×100 m rolling window 이고 `track_unknown_space: false` 다. 지도에서 오는 정보가 지금은 없다.
- **KeepoutFilter 동작**:
  - 마스크 0~100 을 코스트 0~254 로 선형 변환하고 -1 은 건너뛴다(`thirdParty/nav2_costmap_2d/plugins/costmap_filters/costmap_filter.cpp` `getMaskCost`). 그래서 중간 값이 그대로 소프트 코스트가 된다.
  - 기존 코스트보다 클 때만 덮어쓴다(`keepout_filter.cpp` 약 291–304행, max 병합). 실시간 장애물이 살아 있는 이유다.
  - 레이어가 다 돈 뒤에 적용되므로 마스크 코스트는 인플레이션되지 않는다.
  - 로봇이 마스크 셀 위에 있으면 `lethal_override_cost` 200 으로 쓴다.
- **마스크 공급**: NavigationNode 가 직접 만든다.
  - `CostmapFilterInfo` 를 `keepout_costmap_filter_info` 로, 마스크(OccupancyGrid)를 `keepout_filter_mask` 로 발행한다(navigation_node.cpp 약 300–355행).
  - 마스크 프레임은 `id_`, 크기는 파라미터 `NavigationPlugin.keepout_mask`(배포값 100×100 m, 0.1 m, `id_` 원점 중심)다.
- **지오펜스가 이 마스크를 쓰고 있다**:
  1. GUI 지오펜스 도구가 `planning/set_geofence` 를 호출한다(`SetGeofence{zones, frame, mode ALLOW/BLOCK, clear}`).
  2. `NavigationNode::onSetGeofence`(약 955–998행)가 영역을 `id_` 로 변환하고 `paintZone`(약 127–153행)으로 0/100 을 칠한다. 이때 **마스크 전체를 새로 만든다**.
  3. 맵이 바뀌면 지워진다(약 811–840행).
- **플래너·컨트롤러는 중간 코스트를 소프트 비용으로 쓴다.**
  - 배포값 `main_planner SmacLattice`, `cost_penalty 2.0`, `allow_unknown true`. 이동 비용 = 길이 × (1 + 2.0 × cost/252). 차도(229)는 인도(10)보다 1 m 당 약 2.6배 비싸다.
  - ThetaStar 도 252 이하를 소프트로 쓴다.
  - MPPI CostCritic 은 253 미만 코스트를 더한다.
- **GUI Load Map**: `raisin_gui_map_window_map_files.cpp` 에서 `get_point_cloud_map` 을 호출하고, 부가 파일은 `paths/`, `missions/` 만 받는다(`raisin_gui_map_window_sidecars.cpp`).

## 5. 작업 단계

### 1단계 — 코드 변경 없이 동작 확인 (지오펜스)
1. 3절의 좌표계를 맞춘다. 정합 변환이 있으면 `geofence_block_road.json` 폴리곤에 적용한다.
2. 로봇(시뮬)에 맵을 Load Map 하고 재위치추정한다.
3. `planning/set_geofence` 에 `mode: BLOCK`, `frame: <맵 이름>`, `zones: geofence_block_road.json 의 폴리곤 4개` 를 보낸다(GUI 지오펜스 도구로 그려도 된다).
4. 인도 위 출발점에서 길 건너 인도로 goal 을 주고 확인한다.
   - 경로가 횡단보도로만 건너는지
   - `navigation/global_costmap` 에 차도가 막혀 보이는지
- 한계: 값이 100(lethal)뿐이라 차도가 완전히 막힌다. 마스크가 100×100 m 로 제한되고, 맵을 바꾸면 사라진다.

### 2단계 — 정식 구현 (등급 코스트, 맵과 함께 저장)
1. **맵 부가 파일**: `<ws>/log/map/<name>/semantic/` 에 `semantic_mask.png` + `.yaml`(+ 선택적으로 `semantic_info.json`)을 둔다. GUI 부가 파일 동기화 목록에 `semantic/` 을 추가해 Load Map 할 때 함께 오가게 한다(`raisin_gui_map_window_sidecars.cpp`, 그리고 서버 쪽 sidecar 서비스가 있다면 거기도).
2. **NavigationNode**:
   - 맵 이름이 바뀔 때(현재 맵 변경 감지 지점) `log/map/<name>/semantic/semantic_mask.yaml` 이 있으면 읽어 **semantic 기본 마스크**로 보관한다. 없으면 지금처럼 동작한다.
   - 마스크를 발행할 때 셀 값 = `max(semantic, geofence)` 로 합친다. 지오펜스가 없으면 semantic 그대로다. 지오펜스 `clear` 도 semantic 은 남긴다.
   - 프레임: semantic 은 맵 프레임이다. 재위치추정 중에는 `id_` 와 항등이지만, 지오펜스처럼 TF 로 변환해 마스크 격자에 다시 샘플링하는 게 안전하다(`onSetGeofence` 의 변환 코드 재사용).
   - 마스크 크기: 지금 맵은 66×60 m 라 기본 100×100 m 에 들어간다. TIPS 의 1 km 도로 구간에 대비해 `keepout_mask` 크기를 맵에 맞춰 늘리거나, semantic 이 있으면 그 extent·해상도(0.2 m)를 그대로 쓰도록 한다. 1 km × 1 km 를 0.2 m 로 하면 2,500만 셀(int8 25 MB)이다.
3. **튜닝**: 차도를 인도로 2.6배 넘게 돌아가는 상황에서는 플래너가 아직 차도를 가로지를 수 있다. 더 엄격하게 하려면 둘 중 하나:
   - 서버에서 차도 값을 99(코스트 251)로 내보내고, `cost_penalty` 를 2→5 정도로 올린다(권장).
   - 차도를 100(lethal)으로 막는다. 이 경우 횡단보도 라벨에 빈틈이 있으면 경로가 끊긴다.
   - 서버 쪽 값 변경은 `tools/export_nav_semantic.py --road 99` 한 줄이다.

## 6. 완료 기준
- [ ] Load Map 후 `navigation/global_costmap` 시각화에 차도(229 부근)·인도(10)·횡단보도(20)가 보인다.
- [ ] 같은 인도 위 두 점 사이 goal: 경로가 인도에 머문다.
- [ ] 길 건너 goal: 경로가 횡단보도로 건넌다.
- [ ] 차도 위에 놓인 실시간 장애물(차량 등)이 여전히 lethal 로 반영된다(max 병합 확인).
- [ ] 지오펜스 BLOCK 을 추가로 그려도 semantic 이 사라지지 않는다(2단계).
- [ ] semantic 파일이 없는 맵은 지금과 똑같이 동작한다.

## 7. 서버에 다시 요청할 것
- 시뮬에서 **맵을 저장한 세션**의 녹화(mcap + config_snapshot)와 저장된 맵 이름. 받으면 같은 프레임으로 `semantic/` 을 다시 만들어 준다.
- 실험 결과(경로 스크린샷, 플래너 파라미터)를 알려주면 마스크 값을 맞춰 다시 내보낸다.
