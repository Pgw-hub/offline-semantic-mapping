#!/usr/bin/env bash
# 노트북 빌드용 상주 컨테이너 관리. 빌드마다 컨테이너를 만들고 지우지 않고, 하나를 계속 띄워 두고 그 안에서 작업한다.
# 사용: docker/laptop/container.sh up      # 없으면 만들고, 멈춰 있으면 시작 (여러 번 실행해도 됨)
#       docker/laptop/container.sh shell   # 컨테이너 안 bash (저장소 루트에서 시작)
#       docker/laptop/container.sh stop    # 멈춤 (컨테이너는 남는다)
#       docker/laptop/container.sh rm      # 지움 (이미지는 남는다)
#       docker/laptop/container.sh status
# 마운트 (만들 때 고정): $HOME 전체 읽기 전용(mcap 녹화가 홈 아래 어디 있든 보인다), 저장소 읽기·쓰기, 모델 캐시 $HF_CACHE → /models/hf_cache.
#   홈 밖의 데이터를 쓰려면 EXTRA_RO="/data2:/mnt/usb" 처럼 콜론으로 나열하고 rm 후 up (마운트는 바꿀 수 없다).
# 네트워크는 기본 host (NETWORK 로 변경): 컨테이너 안에서 띄운 포털(:9092)·rerun 뷰어(:9091)가 호스트 localhost 로 바로 보인다.
# 포털용 rerun 은 SDK 9886 · WS 9887 로 띄운다 (host 네트워크라 녹화 커버리지 모니터의 rerun spawn(9876)과 겹치면 모니터 창이 안 뜬다).
# 컨테이너가 시작될 때 tools/start_portal.sh 를 실행해 포털(:9092)·rerun 뷰어(:9091)를 같이 띄운다 (재부팅 후 자동 재시작 포함). 포트를 다른 프로그램(VS Code 포트 포워딩 등)이 잡고 있으면 그 둘만 실패하고 컨테이너는 정상.
# 컨테이너는 root 로 돈다. 빌드 산출물(output/)은 파이프라인이 끝날 때 호스트 사용자 소유로 돌린다.
set -euo pipefail
IMAGE="${IMAGE:-offline-map-laptop}"
NAME="${CONTAINER:-offline-map}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/offline_map_hf}"
cmd="${1:-up}"

exists()  { docker container inspect "$NAME" >/dev/null 2>&1; }
running() { [ "$(docker container inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]; }

case "$cmd" in
  up)
    if ! exists; then
      mkdir -p "$HF_CACHE"
      VOLS=(-v "$HOME:$HOME:ro" -v "$REPO:$REPO" -v "$HF_CACHE:/models/hf_cache")
      IFS=: read -ra EXTRA <<< "${EXTRA_RO:-}"
      for d in "${EXTRA[@]}"; do [ -n "$d" ] && VOLS+=(-v "$d:$d:ro"); done
      docker create --name "$NAME" --gpus all --restart unless-stopped --network "${NETWORK:-host}" \
        -e PYTHONDONTWRITEBYTECODE=1 -e RERUN_SDK_PORT=9886 -e RERUN_WS_PORT=9887 "${VOLS[@]}" -w "$REPO" "$IMAGE" \
        bash -c 'bash tools/start_portal.sh >/tmp/start_portal.log 2>&1 || true; exec sleep infinity' >/dev/null   # 컨테이너가 뜰 때(재부팅 포함) 포털·rerun 뷰어도 같이
      echo "컨테이너 생성: $NAME (이미지 $IMAGE)"
    fi
    running || { docker start "$NAME" >/dev/null; echo "컨테이너 시작: $NAME"; }
    ;;
  shell)  "$0" up; exec docker exec -it -w "$REPO" "$NAME" bash ;;
  stop)   exists && docker stop "$NAME" >/dev/null && echo "멈춤: $NAME" ;;
  rm)     exists && docker rm -f "$NAME" >/dev/null && echo "삭제: $NAME (이미지 $IMAGE 는 남음)" ;;
  status) docker ps -a --filter "name=^/$NAME$" ;;
  *) echo "사용: $0 {up|shell|stop|rm|status}" >&2; exit 1 ;;
esac
