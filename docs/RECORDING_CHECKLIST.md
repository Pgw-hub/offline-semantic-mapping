# 2차 데이터셋 녹화 체크리스트 (트랙 2: 로봇 SLAM 기하 위에 의미 입히기)

## ★ 20260911_103657 녹화 결과 점검 (2026-09-11)
- `/cloud_registered` **손상**: `pointcloud`(Livox CustomMsg) typekey 로 기록 → 실제 메시지는 sensor_msgs::PointCloud2. 메시지마다 2,308 B(필드 테이블 x,y,z,intensity,t,timestamp + 메모리 쓰레기), 점 데이터 없음. 복구 불가.
  → 레코더에 `pointcloud2` typekey 추가(docs/mcap_recorder_pointcloud2.patch.md) 후 `"/cloud_registered pointcloud2"` 로 다시 녹화해야 함.
- `/pgo/keyframe_cloud` 없음 (config 에 안 넣었거나 typekey 없어 무시).
- `/Odometry/base` 는 Mapping 모드에서 **PGO 보정이 실시간 반영된 자세** (getPublishWorldBase). 루프클로저마다 수 m 점프
  (248s 3.8m · 416s 4.2m · 584s 2.9m · 639.6s ~10m(96샘플 동안 두 프레임을 번갈아 발행) · 725.6s ~9.6m(101샘플)). 즉 실시간 오도메트리로 지도를 만들면 프레임이 섞임.
- `/path/pgo` 998 키프레임(824s, 1m 간격). pose 별 stamp 없음 → geometry/robot_graph.py 가 오도메트리 상대이동 매칭으로 시각 복원(636s 이전은 오차 ≤15cm, 이후 번갈아 발행 창 때문에 불확실).
  **로봇 저장 `<name>.txt`(TUM, timestamp 포함)를 받으면 `--traj` 로 정확히 대체.**
- raisin_config 미포함 → 20260827 것 복사해 사용(같은 로봇).


근거: refer/raisin_plugin/raisin_lidar_slam_plugin 소스 (2026-09-10 확인)

## mcap 에 녹화할 토픽
| 토픽 | 용도 | 비고 |
|---|---|---|
| `/livox/lidar` (raw) | 백업·deskew 재현·트랙 1 | 지금까지와 동일 |
| `/Odometry/base` | 프론트엔드 자세 (임의 시각 보간용) | 10 Hz, 필수 |
| `/cloud_registered` | deskew+정합된 스캔 (world) | `cloud_registered_voxel_size` 기본 0.1 m (mid360.yaml). **compressed 버전(1.0 m) 은 쓰지 말 것** |
| `/pgo/keyframe_cloud` (KeyframeCloud: idx, cloud, header.stamp=스캔 시각) | 키프레임 클라우드 (base 프레임) | ★실로봇은 `gui_publish_voxel_size: 0.5` 로 다운샘플해 발행 → **0.05~0.1 로 낮춰서 녹화** |
| `/path/pgo` | 보정 자세 (키프레임 순서 = idx 순) | 매번 전체 경로 재발행. **마지막 메시지**가 최종 보정값. 최적화 안 끝난 꼬리는 잘려 나감 |
| `d455_front/rgb`, `d455_front/depth`(aligned) | 의미 부여 | 지금까지와 동일. 헤더 시각 = 같은 시계 |
| (선택) `/imu`, GPS | 트랙 1 개선용 | 있으면 좋음 |

## mcap recorder 설정 (raisin_mcap_recorder_plugin/config/params.yaml `record_topics`)
- 추가(config 만으로 됨): `"/Odometry/base odom"`, `"/path/pgo path"` — 이전 데이터셋엔 /Odometry/base 만 있었음
- **레코더에 PointCloud2·KeyframeCloud 직렬화기가 없음** → `/cloud_registered`, `/pgo/keyframe_cloud` 는 소스 추가 + 재빌드 필요. 초안: docs/mcap_recorder_pointcloud2.patch.md. 빌드 전이면 두 줄은 무시됨.
- 이미지: `d455_front/rgb rgb_image`, `d455_front/depth depth_image` 의 `off` 를 지우거나 웹 UI 에서 켜기 (이전과 동일)
- 그래도 안 되면 대안: /cloud_registered 없이 `/livox/lidar` + `/Odometry/base` + 저장된 `<name>.txt`/`.pcd` 만으로도 트랙 2 진행 가능 (deskew 는 우리 쪽에서, 키프레임 클라우드는 raw 스캔에서 재구성)

## 종료 시 로봇에서 저장해 올 파일 (`save_point_cloud_map` 서비스, log/map/<name>/)
- `<name>.pcd` — 키프레임 클라우드를 보정 자세로 합친 전역 지도. ★실로봇은 `result.save_map_voxel_size: 0.5` 로 다운샘플 → **0.05~0.1 로 바꿔서 저장** (기본값이면 의미 입히기엔 너무 거침)
- `<name>.txt` — 키프레임 궤적, TUM 형식 `# timestamp tx ty tz qx qy qz qw` (초 단위 double, 키프레임 보정 자세). **이게 keyframe_cloud 의 자세 정답** (idx 순서·timestamp 로 매칭)
- `<name>_sc.bin` — Scan Context 디스크립터 (재지역화용, 우리는 안 씀)
- `<name>_auto_waypoints*` — 웨이포인트(SaveGraphFile 계열). 자세 그래프 아님
- 가능하면 `raisin_config/` (calibration + lidar_slam 파라미터) 도 함께

## 왜 이렇게 필요한가
- 키프레임 클라우드는 base 프레임이라 자세가 있어야 world 로 놓임 → `<name>.txt` 또는 `/path/pgo` 마지막 메시지.
- `/path/pgo` 는 시간이 지나며 과거 자세가 계속 바뀌므로 중간 메시지는 쓰지 않음.
- RGB/depth 프레임 → 자세: `T(t) = T_corr(k)·T_odom(k)⁻¹·T_odom(t)` (k = 직전 키프레임, T_odom 은 /Odometry/base 보간). 카메라 시각은 키프레임 사이에 있으므로 /Odometry/base 녹화가 꼭 필요.
- 키프레임 간격은 1 m / 45° (`keyframe_thresh_dist_`, `keyframe_thresh_rot_`) → 회전 구간에서 성기다. 의미 부여는 키프레임 클라우드가 아니라 `/cloud_registered` 전 스캔에 하고, 자세만 그래프에서 받는 방식이 더 촘촘함.

## 확인할 것 (녹화 전 1분 테스트)
1. `/pgo/keyframe_cloud` 가 mcap 에 들어오는지 (Mapping 모드에서만 발행됨).
2. keyframe_cloud 한 개의 점 수 (~1만 이상이면 다운샘플 완화 성공).
3. `/path/pgo` 마지막 메시지 poses 수 == 마지막 keyframe idx+1.
4. `<name>.txt` 줄 수 == 키프레임 수, timestamp 가 keyframe_cloud header.stamp 와 일치.

## 시뮬레이션 데이터셋 (트랙 2 첫 시도, 2026-09-11 계획)
- lidar_slam 은 `Raibo.real_robot: false` 면 sim 모드: keyframe_cloud 발행 다운샘플 없음(원본), save_point_cloud_map 복셀 0.1 m 고정, `sim_evaluation` 자동 켜져
  `<name>_gt.pcd` / `<name>_gt.txt`(정답 지도·궤적) 도 함께 저장 → 우리 지도 정합 정확도 평가에 그대로 사용.
- 저장 위치: `<workspace>/log/map/<name>/{<name>.pcd, <name>.txt, <name>_gt.pcd, <name>_gt.txt, <name>_sc.bin, waypoints}` 폴더째 가져올 것.
- 레코더: `/Odometry/truth odom`(기본 있음, sim 에서만 값이 옴) + `/Odometry/base odom` + `/path/pgo path` + d455 raw rgb/depth(off 제거). `/cloud_registered`·`/pgo/keyframe_cloud` 는 pointcloud2 패치 빌드에서만.
- 주의: 매핑은 `start_mapping` 으로 시작(Mapping 모드 아니면 keyframe/path 발행 없음). 종료 전 `save_point_cloud_map` 호출. raisin_config(calibration) 포함.
- 1분 테스트 후 확인: `/path/pgo` 마지막 poses 수 == `<name>.txt` 줄 수, `.txt` timestamp 가 `/Odometry/base` stamp 와 일치, mcap 메시지 크기가 점 수에 따라 변하는지(고정 크기면 오독).

## ★ 20260914_033204 (sim, railab_raibo, dungok_nav_testbed_2f) 점검 결과 (2026-09-14)
| 항목 | 상태 |
|---|---|
| `/cloud_registered` PointCloud2 | ✔ 새 `pointcloud2` typekey 로 정상 (스캔당 ~21.8k 점, 690~740 KB, 필드 x,y,z,intensity,t) — 파서 geometry/pc2_reader.py |
| `/livox/lidar` | ✔ sim 은 CustomMsg 가 아니라 PointCloud2(x,y,z,intensity,tag,line,t). 점별 t 가 전부 0 → sim 에선 deskew 불필요/불가. 우리 파이프라인 리더에 PointCloud2 분기 필요 |
| `/Odometry/base` | ✔ 10 Hz, cloud_registered 와 stamp 일치 |
| `d455_front/rgb`, `d455_front/depth` | ✘ config 에 켜져 있으나 메시지 0개 → sim 에서 카메라가 발행되지 않음. 의미 부여 불가 |
| `/path/pgo` | ✘ recorder config 에 없음 |
| `/pgo/keyframe_cloud` | ✘ config 에 없음 (typekey 도 없음) |
| `/Odometry/truth` | ✘ config 에 있으나 0개 (sim GT 발행 안 됨) |
| 저장 지도 `log/map/<name>/` (.pcd/.txt/_gt) | ✘ 없음. raisin_config/map 은 GIS 타일(korea.mbtiles) |
| raisin_config/calibration | ✔ (실로봇 시리얼의 d455 설정 그대로) |
| 길이 | sim 시각 기준 45 s (log_000 33.6 s + log_001 11.4 s; wall 80 s), 이동 범위 6×3 m |

## ★ 기본 파이프라인 = 트랙 2 keyframe_scans (2026-09-16 채택)
녹화 폴더에 반드시 함께 넣을 것: (1) mcap (토픽: /Odometry/base, /cloud_registered[pointcloud2], /path/pgo, d455_front/rgb·depth raw, /livox/lidar) (2) 로봇 저장 지도 폴더 `map_*/<name>.pcd + <name>.txt`
(save_point_cloud_map) (3) `raisin_config/` (calibration — GUI Calibration 창 Save). 포털은 데이터셋 폴더에서 .pcd+.txt 짝을 자동으로 찾아 robot_map 으로 넣는다. 지도 폴더가 없으면 geometry=raw_scans(기존 방식) 만 가능.
