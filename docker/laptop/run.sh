#!/usr/bin/env bash
# 노트북 이미지로 세션을 빌드한다. 저장소 · mcap · 출력 폴더를 호스트와 같은 경로로 마운트하므로 세션 yaml 을 그대로 쓴다.
# 사용: docker/laptop/run.sh <session.yaml> [--steps ... --force ... --dry-run]
#   세션 yaml 에 options.objects: false (semantic 맵까지) 또는 options.geometry_only: true (보행용 맵만) 를 넣을 것.
#   모델 캐시: $HF_CACHE (기본 ~/.cache/offline_map_hf) — 처음 실행 때 SegFormer 를 내려받는다.
set -euo pipefail
IMAGE="${IMAGE:-offline-map-laptop}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
[ $# -ge 1 ] || { echo "사용: $0 <session.yaml> [build_map_package 옵션]" >&2; exit 1; }
YAML="$(realpath "$1")"; shift
val() { grep -E "^$1:" "$YAML" | head -1 | sed -E "s/^$1:[[:space:]]*//; s/[\"']//g; s/[[:space:]]+#.*$//"; }
MCAP="$(val mcap_dir)"; ROOT="$(val root || true)"; CALIB="$(val calib || true)"
[ -d "$MCAP" ] || { echo "mcap_dir 가 없습니다: $MCAP" >&2; exit 1; }
HF_CACHE="${HF_CACHE:-$HOME/.cache/offline_map_hf}"; mkdir -p "$HF_CACHE"
VOLS=(-v "$REPO:$REPO" -v "$MCAP:$MCAP:ro" -v "$HF_CACHE:/models/hf_cache" -v "$(dirname "$YAML"):$(dirname "$YAML")")
if [[ -n "$CALIB" && "$CALIB" != "$MCAP"* ]]; then VOLS+=(-v "$CALIB:$CALIB:ro"); fi
if [[ "$ROOT" == /* ]]; then mkdir -p "$(dirname "$ROOT")"; VOLS+=(-v "$(dirname "$ROOT"):$(dirname "$ROOT")"); fi
exec docker run --rm --gpus all "${VOLS[@]}" -w "$REPO/sessions" "$IMAGE" python build_map_package.py "$YAML" "$@"
