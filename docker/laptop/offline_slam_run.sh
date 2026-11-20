#!/usr/bin/env bash
# 노트북용 오프라인 lidar_slam 재생 한 단계 (scripts/offline_slam/run.sh 의 노트북판). 보통은 docker/laptop/offline_slam.sh 가 부른다.
#   docker/laptop/offline_slam_run.sh <sensors.bin> <out_dir> [config_dir] [map_name]
#   raisin 설치본: $RAISIN_ROOT (기본 ~/raisin_ws_handheld, install/lib/libraisin_lidar_slam_plugin.so 가 있어야 함)
#   호스트 바이너리: $OSL_BUILD/offline_slam_host (기본 ~/offline_slam_build). 없거나 소스·플러그인보다 오래되면
#   scripts/offline_slam/host 를 raisin 과 같은 -march(CMakeCache 의 RAISIN_MARCH, 없으면 x86-64-v3) 로 다시 빌드한다 (호스트에 cmake·g++ 필요).
# 결과: <out_dir>/slam_out.bin · host.log · log/map/<map_name>/ (최종 PGO 궤적 <map_name>.txt) · profile/ (재생 때 HOME)
set -euo pipefail
IN=$1; OUT=$2; CFG=${3:-$(dirname "$IN")/slam_config}; NAME=${4:-offline}
R=${RAISIN_ROOT:-$HOME/raisin_ws_handheld}
B=${OSL_BUILD:-$HOME/offline_slam_build}; HOSTBIN=$B/offline_slam_host
HERE=$(cd "$(dirname "$0")/../.." && pwd)
SO=$R/install/lib/libraisin_lidar_slam_plugin.so
[ -f "$SO" ] || { echo "lidar_slam 플러그인이 없습니다: $SO — RAISIN_ROOT 를 빌드된 raisin 워크스페이스로 지정하세요" >&2; exit 1; }
[ -f "$CFG/params.yaml" ] && [ -f "$CFG/mid360.yaml" ] || { echo "lidar_slam 설정(params.yaml, mid360.yaml)이 없습니다: $CFG" >&2; exit 1; }
if [ ! -x "$HOSTBIN" ] || [ "$HOSTBIN" -ot "$HERE/scripts/offline_slam/host/offline_slam_host.cpp" ] || [ "$HOSTBIN" -ot "$HERE/scripts/offline_slam/host/CMakeLists.txt" ] || [ "$HOSTBIN" -ot "$SO" ]; then
  command -v cmake >/dev/null && command -v g++ >/dev/null || { echo "호스트 빌드에 cmake·g++ 가 필요합니다 (sudo apt install cmake g++)" >&2; exit 1; }
  M=$(grep -oE '^RAISIN_MARCH:STRING=.*' "$R"/cmake-build-release/CMakeCache.txt 2>/dev/null | cut -d= -f2); M=${M:-x86-64-v3}
  echo "offline_slam_host 빌드 (RAISIN_ROOT=$R, -march=$M) → $HOSTBIN"
  cmake -S "$HERE/scripts/offline_slam/host" -B "$B" -DRAISIN_ROOT="$R" -DRAISIN_MARCH="$M" > "$B.cmake.log" 2>&1 && cmake --build "$B" -j "$(nproc)" >> "$B.cmake.log" 2>&1 || { echo "빌드 실패 — $B.cmake.log" >&2; exit 1; }
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
HOME="$OUT/profile" stdbuf -oL -eL "$HOSTBIN" "$IN" "$OUT/slam_out.bin" "$SO" "$NAME" $TF > "$OUT/host.log" 2>&1 || { echo "재생 실패 — $OUT/host.log 끝부분:" >&2; tail -5 "$OUT/host.log" >&2; exit 1; }
grep RESULT "$OUT/host.log" || { echo "재생이 RESULT 없이 끝났습니다 — $OUT/host.log" >&2; exit 1; }
[ -f "$OUT/log/map/$NAME/$NAME.txt" ] || { echo "저장 지도 궤적이 없습니다: $OUT/log/map/$NAME/$NAME.txt — $OUT/host.log" >&2; exit 1; }
