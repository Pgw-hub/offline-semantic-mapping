#!/usr/bin/env bash
# 매핑 커버리지 모니터 원탭 실행 (GNOME 독 아이콘용. 센서 수집 아이콘 raisin_sensor_collect.sh 와 같은 방식)
#   coverage_monitor_launch.sh          : 레코더가 녹화 중이면 그 세션을 --live 로 모니터, 아니면 녹화 시작을 기다렸다가 시작.
#                                         다시 누르면 종료 (coverage_summary.json · png 저장)
#   coverage_monitor_launch.sh stop     : 종료만
#   coverage_monitor_launch.sh replay   : 가장 최근 세션 재생 (녹화 끝난 뒤 다시 보기)
#   coverage_monitor_launch.sh results  : 결과 폴더(output/_coverage) 열기
# 세션 이름은 레코더 제어 소켓(127.0.0.1:$REC_CTRL_PORT, 기본 8089) 의 status 로 받고, 폴더는
# $REC_OUTPUT_DIR (센서 수집 아이콘은 ~/logger_data 로 줌) 밑 <날짜>/<세션> 에서 찾는다.

REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
MONITOR=$REPO/scripts/recording/coverage_monitor.py
VENV=$HOME/.venvs/coverage
LOGGER_ROOT=${REC_OUTPUT_DIR:-$HOME/logger_data}
LOGGER_ROOTS=("$LOGGER_ROOT" "$HOME/logger_data" /home/raibo2/logger_data)   # 세션을 찾을 곳 (coverage_monitor.py 의 logger_roots 와 같음; 레코더 기본값은 /home/raibo2/logger_data)
OUT_ROOT=$REPO/output/_coverage
STATE_DIR=$HOME/.local/state/raisin_coverage_monitor
LOG_DIR=$LOGGER_ROOT/launcher_logs
CTRL_PORT=${REC_CTRL_PORT:-8089}
MONITOR_ARGS=(${COVERAGE_MONITOR_ARGS:-})   # 예: COVERAGE_MONITOR_ARGS="--mask-bottom 60"
TITLE="매핑 커버리지 모니터"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$OUT_ROOT"
trap "" PIPE

alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
read_pid() { cat "$STATE_DIR/$1.pid" 2>/dev/null; }
info()  { zenity --info     --title "$TITLE" --width 480 --text "$1" "${@:2}" 2>/dev/null; }
error() { zenity --error    --title "$TITLE" --width 480 --text "$1" 2>/dev/null; }
ask()   { zenity --question --title "$TITLE" --width 480 --text "$1" --ok-label "${2:-계속}" --cancel-label "${3:-취소}" 2>/dev/null; }
rec_cmd() {
  timeout "${2:-2}" bash -c "exec 3<>/dev/tcp/127.0.0.1/$CTRL_PORT && echo '$1' >&3 && head -n1 <&3" 2>/dev/null | tr -d '\r'
}
field() { sed -n "s/.*\\b$1=\\([^ ]*\\).*/\\1/p" <<< "$2"; }
session_dir() {   # 레코더 status 의 session= 은 "<날짜>/<이름>" (예전 판은 이름만). 둘 다 받는다
  local r n; n=${1##*/}
  for r in "${LOGGER_ROOTS[@]}"; do
    [ -d "$r/$1" ] && { echo "$r/$1"; return; }
    ls -d "$r"/*/"$n" 2>/dev/null | tail -1 | grep . && return
  done; return 1
}
latest_session() {   # mcap 이 있는 가장 최근 세션 폴더 (모든 root, 수정 시각순)
  local d r; for d in $(ls -td $(for r in "${LOGGER_ROOTS[@]}"; do [ -d "$r" ] && echo "$r"/*/*/; done) 2>/dev/null); do
    [ -n "$(ls "$d"/log_*.mcap 2>/dev/null)" ] && { echo "${d%/}"; return; }
  done
}
last_line() { grep -a "t=" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

# ---------------------------------------------------------------- 실행 중인 인스턴스로의 요청
case $1 in
  results) mkdir -p "$OUT_ROOT"; xdg-open "$OUT_ROOT" 2>/dev/null & exit 0 ;;
esac
MAIN_PID=$(read_pid main)
if alive "$MAIN_PID"; then touch "$STATE_DIR/stop_request"; exit 0; fi   # 둘째 탭 = 종료
[ "$1" = stop ] && exit 0

# ---------------------------------------------------------------- 준비 확인
if [ ! -x "$VENV/bin/python" ]; then
  error "모니터 venv 가 없습니다: $VENV\n\n터미널에서 한 번 설치하세요:\ncd $REPO\npython3 -m venv $VENV\n$VENV/bin/pip install -r requirements-recording.txt"
  exit 1
fi
[ -f "$MONITOR" ] || { error "모니터 스크립트가 없습니다:\n$MONITOR"; exit 1; }

echo $$ > "$STATE_DIR/main.pid"
rm -f "$STATE_DIR/stop_request"
trap 'rm -f "$STATE_DIR/main.pid" "$STATE_DIR/stop_request" "$STATE_DIR/monitor.pid"' EXIT

# ---------------------------------------------------------------- 세션 고르기
MODE=live; SESSION=""; DIR=""
if [ "$1" = replay ]; then
  MODE=replay; DIR=$(latest_session)
  [ -n "$DIR" ] || { error "재생할 세션이 없습니다.\n$LOGGER_ROOT 에 log_*.mcap 이 있는 세션 폴더가 없습니다."; exit 1; }
  SESSION=$(basename "$DIR")
else
  # 녹화 시작을 기다린다 (취소 가능). 레코더 소켓이 없으면 노드·GUI 가 아직 안 뜬 것.
  fifo=$STATE_DIR/wait.fifo; rm -f "$fifo"; mkfifo "$fifo"
  zenity --progress --pulsate --title "$TITLE" --width 480 --text "녹화 시작 기다리는 중…" --cancel-label "취소" < "$fifo" 2>/dev/null &
  wait_pid=$!; exec 4> "$fifo"
  while :; do
    alive "$wait_pid" || { exec 4>&-; rm -f "$fifo"; exit 0; }
    [ -f "$STATE_DIR/stop_request" ] && { exec 4>&-; kill "$wait_pid" 2>/dev/null; rm -f "$fifo"; exit 0; }   # 기다리는 중 둘째 탭 = 취소
    st=$(rec_cmd status 2)
    if [ "$(field recording "$st")" = 1 ]; then
      SESSION=$(field session "$st"); DIR=$(session_dir "$SESSION"); SESSION=${SESSION##*/}
      [ -n "$DIR" ] && break
      printf '# 세션 폴더 기다리는 중: %s\\n%s\n' "$SESSION" "${LOGGER_ROOTS[*]}" >&4 2>/dev/null
    elif [ -z "$st" ]; then
      printf '# 녹화기 대기 중: 센서 수집 아이콘 → GUI 접속 → Turn on\\n(레코더 소켓 127.0.0.1:%s 없음)\n' "$CTRL_PORT" >&4 2>/dev/null
    else
      printf '# 녹화 시작 기다리는 중: GUI tools > mcap recorder > start\\n녹화가 시작되면 자동으로 모니터를 띄웁니다\n' >&4 2>/dev/null
    fi
    sleep 1
  done
  exec 4>&-; kill "$wait_pid" 2>/dev/null; rm -f "$fifo"
fi
OUT=$OUT_ROOT/$SESSION
STAMP=$(date +%Y%m%d_%H%M%S); LOG=$LOG_DIR/coverage_$STAMP.log

# ---------------------------------------------------------------- 모니터 실행
# env --default-signal: 백그라운드 자식은 SIGINT 무시 상태로 시작하므로 되돌린다 (모니터는 INT/TERM 에 결과를 저장하고 끝남).
# rerun 뷰어는 python 의 자식이라 python 만 끝내면 창은 남는다 (결과를 계속 볼 수 있음).
args=("$DIR" --out "$OUT" "${MONITOR_ARGS[@]}")
[ $MODE = live ] && args+=(--live --done-idle 15)
(cd "$REPO" && exec env --default-signal=INT,TERM PATH="$VENV/bin:$PATH" "$VENV/bin/python" "$MONITOR" "${args[@]}") > "$LOG" 2>&1 &
MON_PID=$!; echo $MON_PID > "$STATE_DIR/monitor.pid"
sleep 3
if ! alive $MON_PID; then
  error "모니터가 바로 종료되었습니다.\n\n로그: $LOG\n\n$(tail -n 8 "$LOG")"; exit 1
fi
notify-send -i "$HOME/.local/share/icons/hicolor/scalable/apps/raisin-coverage-monitor.svg" "$TITLE" \
  "$([ $MODE = live ] && echo "녹화 중 세션" || echo "재생"): $SESSION  (다시 누르면 종료)" 2>/dev/null

# ---------------------------------------------------------------- 상태 창 (버튼 또는 둘째 탭 = 종료)
rm -f "$STATE_DIR/stop_request"
fifo=$STATE_DIR/status.fifo; rm -f "$fifo"; mkfifo "$fifo"
zenity --progress --pulsate --title "$TITLE" --width 560 --text "모니터 시작 중…" --cancel-label "■  모니터 종료" < "$fifo" 2>/dev/null &
status_pid=$!; exec 4> "$fifo"
start=$(date +%s); reason=""; rec_off=0
while :; do
  [ -f "$STATE_DIR/stop_request" ] && { reason=dock; break; }
  alive "$status_pid" || { reason=button; break; }
  alive $MON_PID || { reason=exited; break; }
  if [ $MODE = live ]; then
    st=$(rec_cmd status 2)
    if [ "$(field recording "$st")" = 1 ]; then rec_off=0; else rec_off=$((rec_off + 1)); fi
    [ $rec_off -ge 20 ] && { reason=rec_done; break; }   # 녹화가 끝나고 20 s: 모니터가 스스로 안 끝나면 종료시킴
  fi
  el=$(( $(date +%s) - start ))
  line=$(last_line "$LOG"); [ -n "$line" ] || line="데이터 기다리는 중…"
  warn=$(grep -ac "⚠" "$LOG" 2>/dev/null)
  if [ $MODE = live ]; then head="● 녹화 모니터 중  $((el / 60))분  $SESSION"; else head="▶ 재생 중  $((el / 60))분  $SESSION"; fi
  [ $rec_off -gt 0 ] && [ $MODE = live ] && head="녹화 끝 — 마무리 중…  $SESSION"
  printf '# %s\\n%s\\n경고 %s건 · 결과: %s\n' "$head" "$line" "${warn:-0}" "$OUT" >&4 2>/dev/null
  sleep 1
done
exec 4>&-; kill "$status_pid" 2>/dev/null; rm -f "$fifo"

# ---------------------------------------------------------------- 종료 (결과 저장 기다림)
if alive $MON_PID; then
  ( echo "# 결과 저장 중… (coverage_summary.json · png)"
    kill -TERM $MON_PID 2>/dev/null
    for _ in $(seq 300); do alive $MON_PID || break; sleep 0.1; done
    alive $MON_PID && kill -KILL $MON_PID 2>/dev/null ) | \
    zenity --progress --pulsate --title "$TITLE" --width 480 --auto-close --no-cancel --text "종료 중…" 2>/dev/null
fi

summary=$("$VENV/bin/python" - "$OUT/coverage_summary.json" <<'PY' 2>/dev/null
import json, sys
try: s = json.load(open(sys.argv[1]))
except Exception as e: print(f"요약 없음 ({type(e).__name__})"); sys.exit()
a = s.get("area_m2", {}); lat = s.get("slam_latency_ms") or {}
print(f"녹화 {s.get('duration_s', 0):.0f} s · 이동 {s.get('path_m', 0):.0f} m · 스캔 {s.get('scans', 0)} · depth {s.get('depth_frames', 0)}")
print(f"보행용 맵 {a.get('near_ok', 0):.0f} m² (약함 {a.get('near_weak', 0):.0f}) · 카메라 {a.get('camera', 0):.0f} m² · LiDAR만 {a.get('near_without_camera', 0):.0f} m²")
print(f"SLAM 지연 중앙값 {lat.get('median', '-')} ms · 레코더 유실 {s.get('recorder', {}).get('dropped', '-')} · 경고 {len(s.get('warnings', []))}건")
PY
)
case $reason in
  exited)   [ $MODE = live ] && note="모니터가 스스로 끝났습니다 (녹화 종료 또는 오류, 로그: $LOG)" || note="재생 끝" ;;
  rec_done) note="녹화가 끝나 모니터를 종료했습니다" ;;
  *)        note="종료 완료" ;;
esac
if ask "$note\n\n$SESSION\n$summary\n\n$OUT" "결과 폴더 열기" "닫기"; then xdg-open "$OUT" 2>/dev/null & fi
