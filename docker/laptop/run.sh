#!/usr/bin/env bash
# 노트북 이미지로 세션을 빌드한다. 상주 컨테이너(docker/laptop/container.sh, 기본 이름 offline-map) 안에서 실행하므로
# 빌드가 끝나도 컨테이너는 남는다 (없으면 자동으로 만든다). 저장소·홈·출력 폴더가 호스트와 같은 경로라 세션 yaml 을 그대로 쓴다.
# 사용: docker/laptop/run.sh <session.yaml> [--steps ... --force ... --dry-run]
#   세션 yaml 에 options.objects: false (semantic 맵까지) 또는 options.geometry_only: true (보행용 맵만) 를 넣을 것.
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
# 빌드 뒤: LiDAR 전용 보행 맵 변형(depth·평탄 채움 제외) — 서버에서 수동으로 만들던 session_voxel3d_lidaronly (포털 다운로드 'LiDAR 만'). dry-run 이나 이미 있으면 건너뜀
case " $* " in *" --dry-run "*) exit 0;; esac
ROOT="$(val root)"; [[ "$ROOT" == /* ]] || ROOT="$REPO/output/$ROOT"
if [ -f "$ROOT/compiled/session_voxel3d.npz" ] && [ ! -f "$ROOT/compiled/session_voxel3d_lidaronly.npz" ]; then
  docker exec -w "$REPO" "$NAME" python scripts/locomotion_map/voxel3d_drop_fill.py "$ROOT/compiled" --drop-source 2 3 --suffix lidaronly
  docker exec "$NAME" chown -R "$(id -u):$(id -g)" "$ROOT/compiled"
fi
# 계단 채움을 LiDAR 전용 타일에도 (session_voxel3d_stair_lidaronly_tiles, 대전 11/03 실기에 쓴 조합) — 파이프라인 계단 채움은 depth 채움이 섞인 기본 타일 위라서.
# 옵션(risers·clean·roi)은 세션 yaml 의 options.stair_complete 를 파이프라인과 같은 기본값으로 읽는다. 로봇 맵 패키지(robot_map_package.py --tiles auto)가 이것을 먼저 고른다.
if [ -f "$ROOT/compiled/session_voxel3d_stair_tiles/index.json" ] && [ -f "$ROOT/compiled/session_voxel3d_lidaronly_tiles/index.json" ] && [ ! -f "$ROOT/compiled/session_voxel3d_stair_lidaronly_tiles/index.json" ]; then
  docker exec -i -w "$REPO" -e CUDA_VISIBLE_DEVICES= "$NAME" python - "$YAML" "$ROOT/compiled" <<'PY' || echo "경고: LiDAR 전용 계단 채움 실패 — 로봇 맵 패키지는 다음 변형을 쓴다" >&2
import subprocess, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])); comp = sys.argv[2]
sc = (cfg.get("options") or {}).get("stair_complete") or True
sc = dict({"risers": True, "roi": None, "clean": True}, **(sc if isinstance(sc, dict) else {}))
cmd = [sys.executable, "scripts/locomotion_map/voxel3d_stair_complete.py", f"{comp}/session_voxel3d_lidaronly_tiles", f"{comp}/session_voxel3d_stair_lidaronly_tiles"]
cmd += (["--risers"] if sc["risers"] else []) + (["--clean"] if sc["clean"] else []) + (["--roi", *map(str, sc["roi"])] if sc["roi"] else [])
subprocess.run(cmd, check=True)
PY
  docker exec "$NAME" chown -R "$(id -u):$(id -g)" "$ROOT/compiled"
fi
# 계단 채움(options.stair_complete) 결과는 타일로만 나오므로 npz 로도 만들어 포털 변형 'stair' · Potree 레이어 'voxel3d_stair' 로 보이게 한다
if [ -f "$ROOT/compiled/session_voxel3d_stair_tiles/index.json" ] && [ ! -f "$ROOT/compiled/session_voxel3d_stair.npz" ]; then
  docker exec -w "$REPO" "$NAME" python scripts/locomotion_map/voxel3d_tiles_to_npz.py "$ROOT/compiled/session_voxel3d_stair_tiles" "$ROOT/compiled/session_voxel3d_stair.npz"
  docker exec -w "$REPO" "$NAME" python scripts/package/potree_export.py "$ROOT" --force
  docker exec "$NAME" chown -R "$(id -u):$(id -g)" "$ROOT/compiled"
fi
