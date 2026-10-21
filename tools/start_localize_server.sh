#!/usr/bin/env bash
# 대화형 초기 위치 서버를 컨테이너 안에서 띄운다. 사용: tools/start_localize_server.sh <manifest> [logs=6,7,8,9,10]
M="${1:-output/web_172935_l05_cdv2_fix/compiled/map_manifest.json}"; LOGS="${2:-6,7,8,9,10}"
docker exec offline_3d_semantic_map pkill -f "scripts/apps/localize_server.py" 2>/dev/null; sleep 1
docker exec -d offline_3d_semantic_map bash -c "cd /home/gw.park/offline_semantic_mapping && CUDA_VISIBLE_DEVICES= python scripts/apps/localize_server.py $M --logs $LOGS > /tmp/localize_server.log 2>&1"
echo "starting localize server on :9093 (map $M) — log: docker exec offline_3d_semantic_map tail /tmp/localize_server.log"
