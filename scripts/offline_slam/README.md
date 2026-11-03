# 오프라인 lidar_slam 재생 (`scripts/offline_slam`)

녹화에는 센서만 받고(`/livox/lidar`, `/livox/imu`, 카메라, `tf_static`), SLAM 은 서버에서 다시 돌려 그 출력으로 지금과 같은 오프라인 맵(기하 + 시맨틱)을 만든다.
실시간 SLAM 결과(`/cloud_registered`, `/Odometry/base`, `/path/pgo`, `/pgo/keyframe_cloud`)가 녹화에 있어도 오프라인 결과로 바꿔 쓴다.

## 하는 일

```
녹화 mcap ──export_sensor_dump.py──▶ sensors.bin (/livox/imu·/livox/lidar, 레코더 수신 순서)
                                         │
                       offline_slam_host (raisin_novnc 컨테이너)
                       ├ libraisin_lidar_slam_plugin.so 를 그대로 dlopen·create()  ← 플러그인 코드 수정 없음
                       ├ raisin 재생 시계(simulatedClockMode): 주기 작업(PGO 루프 탐색 1 Hz, /path/pgo 0.5 Hz)이 녹화 시각으로 돈다
                       ├ 스캔마다 /Odometry/base 응답을 받은 뒤 다음 스캔 → 기계가 느려도 스캔을 버리지 않음
                       └ 출력 토픽을 slam_out.bin 으로, 끝나면 save_point_cloud_map (최종 PGO 궤적 offline.txt)
                                         │
녹화 mcap + slam_out.bin ──import_outputs_to_mcap.py──▶ <work>/mcap/log_XXX.mcap  (카메라 등은 그대로, SLAM 토픽만 교체)
                                         │
                       scripts/build_map_package.py (세션 mcap_dir = <work>/mcap, robot_map.traj = <work>/mcap/offline_slam_map/offline.txt)
```

## 실행

```bash
# 서버 호스트에서 (컨테이너 offline_3d_semantic_map, raisin_novnc 가 떠 있어야 함)
scripts/offline_slam/offline_slam.sh <녹화 mcap_dir> /data2/offline_slam/<이름> [lidar_slam 설정 폴더]
```

- lidar_slam 설정: 기본은 녹화의 `config_snapshot/base/raisin_lidar_slam_plugin/config/{params.yaml,mid360.yaml}`, 없으면 raisin 설치본.
- 결과: `<work>/mcap` (맵 파이프라인 입력), `<work>/host.log` (플러그인 로그), `<work>/compare.json` (녹화에 실시간 SLAM 출력이 있으면 비교).
- 호스트 빌드는 처음 실행 때 자동(`/tmp/osl_build`, raisin 과 같은 `-march=x86-64-v3` 필수 — 다르면 Eigen 정렬 차이로 tf 에서 segfault).

## 실시간 SLAM 과 다른 점

- **스캔 누락 없음**: 실시간은 기계가 밀리면 스캔을 버린다(KAIST 20261023: 분당 최대 84개, 555.9 s 에 3.8 s 멈춤 → 560 s 발산). 재생은 응답을 기다린다.
- **IMU 보정을 다시 한다**: 시작 후 `imu.calibration_time` 동안(약 1.8 s) 출력 없음. 출발 자세가 실시간과 0.5–0.9° 다를 수 있다(강체 오프셋, 맵 품질과 무관).
- **frame_id**: 출력 frame 은 `offline_slam-<숫자>` (파이프라인은 가장 많은 frame 을 world 로 씀 — 문제없음).
- **재현성**: GICP·PGO 가 멀티스레드라 같은 입력도 실행마다 cm 이하로 다름 (151520 두 번: 같은 시각 위치 차 중앙 0.85 cm, 최대 3.4 cm).
- **스레드**: 로봇 노드와 같이 `lidar`·`imu` 를 다른 스레드에 둔다. 같이 두면 deskew 가 기다리는 IMU 가 못 들어와 `Bad time sync` 가 매 스캔 난다.
- **TF**: 플러그인 `init()` 이 `<id>/livox_lidar_0:imu` 를 항등으로 한 번 보내므로, 녹화 `tf_static` 의 base→IMU 는 `init()` 뒤에 넣는다(로그에 `base->sensor TF unavailable` 가 없어야 정상).

## 검증 (2026-11-03)

| 데이터 | 실시간 vs 오프라인 |
|---|---|
| 20261016_151520 (핸드헬드, 166 s) | 키프레임 50 / 50, 경로 68.72 / 68.63 m, 같은 시각 위치 차(강체 정렬 후) 중앙 0.71 cm · 최대 3.8 cm, 28 s 에 처리 |
| 20261023 KAIST DTIPS (758 s) | 실시간이 정상이던 0–540 s: 경로 630.94 / 630.87 m, 정렬 후 중앙 3.5 cm · p90 5.4 cm. 오프라인은 758 s 끝까지 순간이동 0·출력 간격 최대 0.12 s (실시간은 560 s 부터 발산). 5.6 분에 처리 |

세부 수치: `/data2/offline_slam/*/compare*.json`.
