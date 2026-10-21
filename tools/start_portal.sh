#!/usr/bin/env bash
# Map Portal(9092) + rerun 웹뷰어(9091) 를 호스트에서 띄운다. 이미 떠 있으면 건너뜀. 로그: /tmp/map_portal.log, /tmp/rerun_viewer.log
REPO="$(cd "$(dirname "$0")/.." && pwd)"; PY="$HOME/.venvs/rerun/bin/python"; RR="$HOME/.venvs/rerun/bin/rerun"
# venv 가 없으면(노트북 컨테이너 offline-map 안 등) PATH 의 python · rerun 을 쓴다 (이미지에 rerun-sdk 0.16.1 포함)
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"; [ -x "$RR" ] || RR="$(command -v rerun)"
USER="${USER:-$(id -un)}"
# pgrep 이 없는 환경(slim 이미지)용 대체: /proc 의 cmdline 에서 찾는다
command -v pgrep >/dev/null || pgrep() { local pat="${@: -1}"; for f in /proc/[0-9]*/cmdline; do tr '\0' ' ' <"$f" 2>/dev/null | grep -qE -- "$pat" && return 0; done; return 1; }
if pgrep -u "$USER" -f 'tools/map_porta[l].py' >/dev/null; then echo "portal already running"; else
  (cd "$REPO" && setsid nohup "$PY" tools/map_portal.py >/tmp/map_portal.log 2>&1 < /dev/null &) ; sleep 1; echo "portal started: http://localhost:9092/"; fi
if pgrep -u "$USER" -f 'reru[n] --serve' >/dev/null; then echo "rerun viewer already running"; else
  (setsid nohup "$RR" --serve --web-viewer-port 9091 --ws-server-port 9877 --port 9876 >/tmp/rerun_viewer.log 2>&1 < /dev/null &) ; sleep 1; echo "rerun viewer started: http://localhost:9091"; fi
