# 2차 데이터셋 녹화 체크리스트 (트랙 2: 로봇 SLAM 기하 위에 의미 입히기)

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
