#!/usr/bin/env bash
# 노트북 이미지로 세션을 빌드한다. 상주 컨테이너(docker/laptop/container.sh, 기본 이름 offline-map) 안에서 실행하므로
# 빌드가 끝나도 컨테이너는 남는다 (없으면 자동으로 만든다). 저장소·홈·출력 폴더가 호스트와 같은 경로라 세션 yaml 을 그대로 쓴다.
# 사용: docker/laptop/run.sh <session.yaml> [--steps ... --force ... --dry-run]
#   세션 yaml 에 options.objects: false (scene graph 없이) 또는 options.geometry_only: true (카메라 없이 기하만: 메인 맵 기하 + 보행 맵) 를 넣을 것.
#   mcap_dir 는 컨테이너에 마운트된 경로($HOME 아래, 또는 EXTRA_RO 로 추가한 경로)여야 한다.
#   모델 캐시: $HF_CACHE (기본 ~/.cache/offline_map_hf) — 처음 실행 때 SegFormer 를 내려받는다.
set -euo pipefail
NAME="${CONTAINER:-offline-map}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
[ $# -ge 1 ] || { echo "사용: $0 <session.yaml> [build_map_package 옵션]" >&2; exit 1; }
YAML="$(realpath "$1")"; shift
val() { grep -E "^$1:" "$YAML" | head -1 | sed -E "s/^$1:[[:space:]]*//; s/[\"']//g; s/[[:space:]]+#.*$//"; }
MCAP="$(val mcap_dir)"
[ -d "$MCAP" ] || { echo "mcap_dir 가 없습니다: $MCAP" >&2; exit 1; }
"$REPO/docker/laptop/container.sh" up
docker exec "$NAME" test -d "$MCAP" || { echo "컨테이너 $NAME 안에서 mcap_dir 가 안 보입니다: $MCAP — \$HOME 아래로 옮기거나 EXTRA_RO=<경로> container.sh rm && up 으로 다시 만드세요." >&2; exit 1; }
docker exec "$NAME" test -r "$YAML" || { echo "컨테이너 $NAME 안에서 세션 yaml 이 안 보입니다: $YAML" >&2; exit 1; }
# SegFormer 가 캐시에 있으면 Hugging Face 에 접속하지 않는다 (인터넷이 없을 때 HEAD 요청 재시도로 B 단계가 10 분 지연되는 것 방지)
HF_CACHE="${HF_CACHE:-$HOME/.cache/offline_map_hf}"; OFFLINE=()
[ -d "$HF_CACHE/hub/models--nvidia--segformer-b4-finetuned-ade-512-512/snapshots" ] && OFFLINE=(-e HF_HUB_OFFLINE=1)
docker exec -w "$REPO/sessions" "${OFFLINE[@]}" "$NAME" python ../scripts/build_map_package.py "$YAML" "$@"
# (2026-11-24~) 보행 맵은 파이프라인 L 단계가 LiDAR 만 + 계단 보완까지 한 번에 만든다 (compiled/locomotion/tiles) — 예전 빌드 뒤 변형(lidaronly·stair_lidaronly) 단계는 없앴다.
