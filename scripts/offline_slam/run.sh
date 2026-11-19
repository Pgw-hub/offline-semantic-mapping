#!/bin/bash
# Offline lidar_slam replay — run inside the raisin_novnc container (raisin install at /home/raisin_master).
#   run.sh <sensors.bin from export_sensor_dump.py> <out_dir> [config_dir] [map_name]
#   config_dir: folder holding the lidar_slam params.yaml + mid360.yaml to use (default: next to sensors.bin,
#               <dir>/slam_config; export step copies the recording's config_snapshot there)
# out_dir/slam_out.bin  host dump (see offline_slam_host.cpp) · host.log · profile/.raisin (HOME of the run)
set -eu
IN=$1; OUT=$2; CFG=${3:-$(dirname "$IN")/slam_config}; NAME=${4:-offline}
R=/home/raisin_master
HOSTBIN=${HOSTBIN:-/tmp/osl_build/offline_slam_host}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ ! -x "$HOSTBIN" ]; then
  cmake -S "$HERE/host" -B "$(dirname "$HOSTBIN")" -DRAISIN_ROOT=$R > /dev/null && cmake --build "$(dirname "$HOSTBIN")" -j 16 > /dev/null
fi
mkdir -p "$OUT/profile/.raisin/raisin_lidar_slam_plugin"
cp "$CFG"/params.yaml "$CFG"/mid360.yaml "$OUT/profile/.raisin/raisin_lidar_slam_plugin/"
J=${IN%.bin}.json
TF=$(python3 -c "
import json; d=json.load(open('$J'))['tf_static']
f=lambda k: ' '.join(str(v) for v in d[k]['t']+d[k]['q_xyzw'])
print(f('livox_lidar_0:imu'), f('livox_lidar_0:lidar'))")
export LD_LIBRARY_PATH=$R/install/lib
cd "$OUT"
HOME="$OUT/profile" stdbuf -oL -eL "$HOSTBIN" "$IN" "$OUT/slam_out.bin" $R/install/lib/libraisin_lidar_slam_plugin.so "$NAME" $TF > "$OUT/host.log" 2>&1
grep RESULT "$OUT/host.log"
