#!/bin/bash
# 오프라인 lidar_slam 재생 한 번에 (서버 호스트에서 실행): 센서 추출 → 플러그인 재생 → 원래 모양의 mcap 데이터셋
#   scripts/offline_slam/offline_slam.sh <src mcap_dir> <work_dir under /data2> [slam_config_dir]
#   src mcap_dir   : 녹화 폴더 (log_XXX.mcap + config_snapshot). /livox/lidar·/livox/imu·tf_static 만 있으면 됨
#   work_dir       : 결과 폴더. <work_dir>/mcap 이 맵 파이프라인에 줄 mcap_dir (세션 robot_map.traj 는 <work_dir>/mcap/offline_slam_map/offline.txt)
#   slam_config_dir: lidar_slam params.yaml + mid360.yaml (기본: 녹화의 config_snapshot/base/raisin_lidar_slam_plugin/config,
#                    없으면 raisin 설치본 /home/gw.park/raisin_master/install/config/raisin_lidar_slam_plugin/config)
# 컨테이너: offline_3d_semantic_map (mcap 읽기·쓰기, /data2 rw), raisin_novnc (raisin 설치본, /data2·저장소 ro, raisin_master rw)
set -eu
SRC=$1; W=$2; CFG=${3:-}
REPO=/home/gw.park/offline_semantic_mapping
RM=/home/gw.park/raisin_master
NAME=$(basename "$W")
mkdir -p "$W/slam_config"
if [ -z "$CFG" ]; then
  CFG=$SRC/config_snapshot/base/raisin_lidar_slam_plugin/config
  [ -f "$CFG/params.yaml" ] || CFG=$RM/install/config/raisin_lidar_slam_plugin/config
fi
cp "$CFG/params.yaml" "$CFG/mid360.yaml" "$W/slam_config/"; echo "$CFG" > "$W/slam_config/SOURCE.txt"
echo "[1/4] 센서 추출"
docker exec offline_3d_semantic_map bash -c "cd $REPO && python scripts/offline_slam/export_sensor_dump.py $SRC $W/sensors" 2>&1 | grep -v Warn
echo "[2/4] lidar_slam 재생 (raisin_novnc)"
RUN=/home/raisin_master/log/offline_slam/$NAME
docker exec raisin_novnc bash -c "rm -rf $RUN && mkdir -p $RUN && cp $W/slam_config/* $RUN/ 2>/dev/null; /home/raion/offline_semantic_mapping/scripts/offline_slam/run.sh $W/sensors.bin $RUN $RUN && chown -R 1007:1007 /home/raisin_master/log/offline_slam"
cp "$RM/log/offline_slam/$NAME/slam_out.bin" "$RM/log/offline_slam/$NAME/host.log" "$W/"
rm -rf "$W/map"; cp -r "$RM/log/offline_slam/$NAME/log/map/offline" "$W/map"
echo "[3/4] mcap 데이터셋으로"
docker exec offline_3d_semantic_map bash -c "cd $REPO && python scripts/offline_slam/import_outputs_to_mcap.py $SRC $W/slam_out.bin $W/mcap --map-dir $W/map && chown -R 1007:1007 $W" 2>&1 | grep -v Warn
echo "[4/4] 실시간 SLAM 출력과 비교 (녹화에 있으면)"
docker exec offline_3d_semantic_map bash -c "cd $REPO && python scripts/offline_slam/compare_trajectories.py $SRC $W/mcap --out $W/compare.json" 2>&1 | grep -v Warn | grep -E '"(matched|live_keyframes|offline_keyframes|path_live_m|path_offline_m)"|median' || true
echo "완료: mcap_dir = $W/mcap · 궤적 = $W/mcap/offline_slam_map/offline.txt"
