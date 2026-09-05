#!/usr/bin/env bash
# Map Portal(9092) + rerun 웹뷰어(9091) 를 호스트에서 띄운다. 이미 떠 있으면 건너뜀. 로그: /tmp/map_portal.log, /tmp/rerun_viewer.log
REPO="$(cd "$(dirname "$0")/.." && pwd)"; PY="$HOME/.venvs/rerun/bin/python"; RR="$HOME/.venvs/rerun/bin/rerun"
if pgrep -u "$USER" -f 'tools/map_porta[l].py' >/dev/null; then echo "portal already running"; else
  (cd "$REPO" && setsid nohup "$PY" tools/map_portal.py >/tmp/map_portal.log 2>&1 < /dev/null &) ; sleep 1; echo "portal started: http://localhost:9092/"; fi
if pgrep -u "$USER" -x rerun >/dev/null; then echo "rerun viewer already running"; else
  (setsid nohup "$RR" --serve --web-viewer-port 9091 --ws-server-port 9877 --port 9876 >/tmp/rerun_viewer.log 2>&1 < /dev/null &) ; sleep 1; echo "rerun viewer started: http://localhost:9091"; fi
