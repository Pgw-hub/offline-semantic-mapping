# preprocess — mcap → Map Builder 입력 (ROS 불필요, 컨테이너 offline_3d_semantic_map 에서 실행)

| 스크립트 | 입력 | 출력 |
|---|---|---|
| mcap_to_cg.py `<log.mcap> <calib_dir> <out_dir> --lidar` | mcap | `<out>/results/frame%06d.jpg, depth%06d.png`, `traj.txt`, `stamps.npz`, `lidar_world/` (ConceptGraphs·복셀 빌더 입력) |
| extract_rgb_session.py `<session_dir> <sem_dir> <a-b>` | 세션의 log_a..b mcap | `<sem_dir>/rgb/%06d.png` (LiDAR 스캔 시각 10Hz RGB), `rgb_index.npz` |
| precompute_semantic.py `<sem_dir>` | `<sem_dir>/rgb` | `<sem_dir>/sem/%06d.npz` (grass/path/tree/person 확률, SegFormer-B4 ADE20K, GPU) |
| build_voxel_sem_map.py `<log.mcap> <calib> <cg_out_dir> <sem_dir> <out.npz>` | 위 세 출력 | 로그별 0.1 m semantic 복셀 npz |
| export_lidar_scans.py `<log.mcap> <calib_dir> <out_dir>` | mcap | 스캔별 world 점 + 센서 원점 npz (occupancy 레이캐스팅 입력) |

calib_dir = `<session>/raisin_config/calibration` (livox_lidar_0_lidar.yaml, d455 yaml).
emc 컨테이너의 /opt/catkin_ws/tools 에서 옮겨온 것이며(2026-09-04), 이후 수정은 여기서만 한다.
