"""LiDAR 외부파라미터(base→lidar) 단일 출처.
lidar_slam 플러그인은 base→imu TF(calibration/livox_lidar_0_imu.yaml 과 동일) 에 소스 하드코딩 imu→lidar 오프셋 (-0.011, -0.02329, 0.04412) m 를 더해 LiDAR 위치를 잡는다
(refer/raisin_lidar_slam_plugin/src/lidar_slam/util/params.cpp loadExtrinsicsFromTf). /Odometry/base 는 그 외부파라미터로 만든 자세이므로 점을 얹을 때도 같은 값을 써야 일관된다.
calibration/livox_lidar_0_lidar.yaml 은 imu yaml 과 같은 값이라 SLAM 과 4.4 cm 차이 → 회전 시 최대 9 cm 이중 벽 (2026-09-14 sim A/B 로 확인).
모드: 환경변수 LIDAR_EXTRINSIC = calib(기본) | slam
★ 2026-09-14 판정: sim(20260914) 에서는 SLAM 값이 /cloud_registered 와 정확히 일치(4.4 cm 차이가 벽 이중화의 전부). 그러나 실로봇(172935 log_002, PGO 자세) 에서 δ 스윕을 하면
  calib(0,0,0) 이 최적이고 ±2 cm 만 움직여도 일관성이 나빠진다(4.01→4.11~4.15 cm). 실로봇 raw 점은 리더가 180° 뒤집어 쓰므로 SLAM 오프셋 부호 관계가 sim 과 다르고,
  103657 재빌드(slam)도 번짐 지표가 동일(두께 6.83→6.84 cm, 지도만 3.7 cm 강체 이동). → 실로봇 기본은 calib 유지, sim 은 slam 지정.
"""
import os
import numpy as np
import yaml
SLAM_IMU_TO_LIDAR = np.array([-0.011, -0.02329, 0.04412])

def _q2R(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def lidar_extrinsic(calib_dir, mode=None):
    """→ (t_bl (3,), R_bl (3,3), mode). calib_dir = raisin_config/calibration"""
    mode = (mode or os.environ.get("LIDAR_EXTRINSIC", "calib")).lower()
    if mode == "calib":
        tr = yaml.safe_load(open(os.path.join(calib_dir, "livox_lidar_0_lidar.yaml")))["transform"]
        t = np.array([tr["translation"][k] for k in "xyz"], float); R = _q2R(*[tr["rotation"][k] for k in "wxyz"])
        return t, R, "calib"
    tr = yaml.safe_load(open(os.path.join(calib_dir, "livox_lidar_0_imu.yaml")))["transform"]
    t_imu = np.array([tr["translation"][k] for k in "xyz"], float); R_imu = _q2R(*[tr["rotation"][k] for k in "wxyz"])
    return t_imu + R_imu @ SLAM_IMU_TO_LIDAR, R_imu, "slam"

def lidar_t_bl(calib_dir, mode=None):
    return lidar_extrinsic(calib_dir, mode)[0]
