#!/usr/bin/env bash
# 센서(/livox/lidar · /livox/imu · 카메라 · tf_static)만 있는 녹화(매핑을 켜지 않아 SLAM 출력이 없는 것)로 맵을 만들기 위해
# 노트북에서 lidar_slam 을 다시 돌려 맵 파이프라인용 mcap_dir 를 만든다. 서버용 scripts/offline_slam/offline_slam.sh 의 노트북판.
# 녹화에 실시간 SLAM 출력이 있어도 쓸 수 있다 (오프라인 결과로 바꿔 쓰고, 실시간과 비교한 compare.json 을 남긴다).
#
#   docker/laptop/offline_slam.sh <녹화 mcap_dir> <work_dir> [--t-end-s N] [--logs 0 1 2] [--config DIR] [--resume]
#     work_dir   : 결과 폴더 (예: output/_offline_slam/<이름>). <work_dir>/mcap 이 세션 yaml 의 mcap_dir,
#                  <work_dir>/mcap/offline_slam_map/offline.txt 가 options.robot_map.traj. 녹화 폴더 안에는 둘 수 없다.
#     --t-end-s N: 녹화 시작 기준 N 초 이후는 재생하지 않는다 (끝부분이 맵을 뒤트는 경우). 다른 N 으로 다시 돌릴 때는 다른 work_dir 를 쓴다.
#     --logs     : 일부 log_XXX.mcap 만 (기본 전부)
#     --config   : lidar_slam params.yaml + mid360.yaml 폴더 (기본: 녹화의 config_snapshot/base/raisin_lidar_slam_plugin/config,
#                  없으면 $RAISIN_ROOT/install/config/raisin_lidar_slam_plugin/config)
#     --resume   : 이미 있는 단계 결과(sensors.bin, replay/, mcap/)는 건너뛴다
#   필요: 상주 컨테이너 offline-map (없으면 만든다), 호스트에 빌드된 raisin 워크스페이스 $RAISIN_ROOT (기본 ~/raisin_ws_handheld) 와 cmake·g++.
#   단계: [1/4] 센서 추출(컨테이너) → [2/4] lidar_slam 재생(호스트, docker/laptop/offline_slam_run.sh) → [3/4] mcap 변환(컨테이너) → [4/4] 실시간 SLAM 과 비교(있으면).
#   용량: work_dir 에 녹화 mcap 의 절반쯤이 더 생긴다 (sensors.bin + 재생 출력 + 다시 쓴 mcap; 시작 조건은 녹화 크기만큼 여유). 맵을 만든 뒤 sensors.bin 과 replay/slam_out.bin 은 지워도 된다.
#   로그: <work_dir>/offline_slam.log. 오래 걸리니(43 GB 녹화: 추출 2 분 · 재생 1 분 · 변환 4 분) 백그라운드로 돌리고 로그를 본다.
set -euo pipefail
NAME="${CONTAINER:-offline-map}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
R=${RAISIN_ROOT:-$HOME/raisin_ws_handheld}
[ $# -ge 2 ] || { sed -n '5,13p' "$0" >&2; exit 1; }
SRC="$(realpath "$1")"; W="$2"; shift 2
TEND=0; LOGS=(); CFG=""; RESUME=0
while [ $# -gt 0 ]; do
  case "$1" in
    --t-end-s) TEND="$2"; shift 2;;
    --logs) shift; while [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; do LOGS+=("$1"); shift; done;;
    --config) CFG="$(realpath "$2")"; shift 2;;
    --resume) RESUME=1; shift;;
    *) echo "모르는 옵션: $1" >&2; exit 1;;
  esac
done
[[ "$W" == /* ]] || W="$PWD/$W"
[ -d "$SRC" ] && ls "$SRC"/log_[0-9][0-9][0-9].mcap >/dev/null 2>&1 || { echo "녹화 폴더가 아닙니다 (log_XXX.mcap 없음): $SRC" >&2; exit 1; }
case "$W/" in "$SRC"/*) echo "work_dir 는 녹화 폴더 안에 둘 수 없습니다: $W" >&2; exit 1;; esac
[ -f "$R/install/lib/libraisin_lidar_slam_plugin.so" ] || { echo "lidar_slam 플러그인이 없습니다: $R/install/lib/libraisin_lidar_slam_plugin.so — RAISIN_ROOT 를 빌드된 raisin 워크스페이스로 지정하세요" >&2; exit 1; }
command -v cmake >/dev/null && command -v g++ >/dev/null || { echo "호스트 빌드에 cmake·g++ 가 필요합니다 (sudo apt install cmake g++)" >&2; exit 1; }
"$REPO/docker/laptop/container.sh" up
docker exec "$NAME" test -d "$SRC" || { echo "컨테이너 $NAME 안에서 녹화 폴더가 안 보입니다: $SRC — \$HOME 아래로 옮기거나 EXTRA_RO=<경로> container.sh rm && up" >&2; exit 1; }
mkdir -p "$W"
docker exec "$NAME" test -d "$W" || { echo "컨테이너 $NAME 안에서 work_dir 가 안 보입니다: $W — 저장소 아래(output/_offline_slam/<이름>)에 두세요" >&2; exit 1; }
NEED=$(du -sm "$SRC" | cut -f1); FREE=$(df -Pm "$W" | awk 'NR==2{print $4}')
[ "$FREE" -gt "$NEED" ] || { echo "디스크 여유가 부족합니다: $W 에 ${FREE} MB, 녹화 크기 ${NEED} MB 이상 필요" >&2; exit 1; }
exec > >(tee -a "$W/offline_slam.log") 2>&1
echo "===== $(date '+%F %T') offline_slam.sh $SRC → $W (t-end ${TEND}s, logs ${LOGS[*]:-all})"
U=(-u "$(id -u):$(id -g)" -w "$REPO" -e HOME=/tmp)
LOGARG=(); [ ${#LOGS[@]} -gt 0 ] && LOGARG=(--logs "${LOGS[@]}")
# lidar_slam 설정
mkdir -p "$W/slam_config"
if [ -z "$CFG" ]; then
  CFG=$SRC/config_snapshot/base/raisin_lidar_slam_plugin/config
  [ -f "$CFG/params.yaml" ] || CFG=$R/install/config/raisin_lidar_slam_plugin/config
fi
[ -f "$CFG/params.yaml" ] && [ -f "$CFG/mid360.yaml" ] || { echo "lidar_slam 설정(params.yaml, mid360.yaml)이 없습니다: $CFG" >&2; exit 1; }
cp "$CFG/params.yaml" "$CFG/mid360.yaml" "$W/slam_config/"; echo "$CFG" > "$W/slam_config/SOURCE.txt"
echo "lidar_slam 설정: $CFG"
echo "##### [1/4] 센서 추출"
if [ "$RESUME" = 1 ] && [ -s "$W/sensors.bin" ] && [ -f "$W/sensors.json" ]; then echo "(있음, 건너뜀)"; else
  rm -f "$W/sensors.bin" "$W/sensors.json"
  TARG=(); [ "$TEND" != 0 ] && TARG=(--t-end-s "$TEND")
  docker exec "${U[@]}" "$NAME" python scripts/offline_slam/export_sensor_dump.py "$SRC" "$W/sensors" "${TARG[@]}" "${LOGARG[@]}"
fi
echo "##### [2/4] lidar_slam 재생"
if [ "$RESUME" = 1 ] && [ -f "$W/replay/log/map/offline/offline.txt" ]; then echo "(있음, 건너뜀)"; else
  rm -rf "$W/replay"
  "$REPO/docker/laptop/offline_slam_run.sh" "$W/sensors.bin" "$W/replay" "$W/slam_config" offline
fi
echo "##### [3/4] mcap 변환"
if [ "$RESUME" = 1 ] && [ -f "$W/mcap/offline_slam_map/offline.txt" ] && [ -f "$W/mcap/offline_slam_import.json" ]; then echo "(있음, 건너뜀)"; else
  rm -rf "$W/mcap"
  docker exec "${U[@]}" "$NAME" python scripts/offline_slam/import_outputs_to_mcap.py "$SRC" "$W/replay/slam_out.bin" "$W/mcap" --map-dir "$W/replay/log/map/offline" "${LOGARG[@]}"
fi
[ -f "$W/mcap/offline_slam_map/offline.txt" ] || { echo "변환 결과에 궤적이 없습니다: $W/mcap/offline_slam_map/offline.txt" >&2; exit 1; }
echo "##### [4/4] 실시간 SLAM 출력과 비교 (녹화에 있으면)"
docker exec "${U[@]}" "$NAME" python scripts/offline_slam/compare_trajectories.py "$SRC" "$W/mcap" --out "$W/compare.json" "${LOGARG[@]}" 2>&1 | grep -E '"(matched|live_keyframes|offline_keyframes|path_live_m|path_offline_m)"|median' || echo "(실시간 SLAM 출력 없음 — 비교 생략)"
CALIB=$SRC/config_snapshot/raisin_root/calibration; [ -d "$CALIB" ] || CALIB="(없음 — 사용자에게 확인)"
cat <<MSG
완료. 세션 yaml 에 다음을 적고 docker/laptop/run.sh sessions/<세션>.yaml 로 빌드:
  mcap_dir: $W/mcap
  calib: $CALIB
  options:
    robot_map: {geometry: all_scans, traj: $W/mcap/offline_slam_map/offline.txt}
로그: $W/offline_slam.log · 플러그인 로그: $W/replay/host.log
MSG
