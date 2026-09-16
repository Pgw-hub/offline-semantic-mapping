#!/bin/bash
# 실로봇 트랙 2 (20260916) 6개 빌드 순차 실행 — 컨테이너 offline_3d_semantic_map 안에서 실행.
# 데이터셋별로 pcd 변형을 전체(P..V) 먼저 돌려 dataset/sem/D 결과를 만들고, keyframe_scans / all_scans 변형은 심볼릭링크로 공유하며 P,S,C,E,F,G,H,V 만 다시.
cd /home/gw.park/offline_semantic_mapping/sessions || exit 1
for DS in t2_b1f2 t2_nearblock2; do
  echo "##### $DS pcd  $(date +%H:%M)"; python build_map_package.py session_${DS}_pcd.yaml --steps P,A,S,R,B,C,D,E,F,G,H,V > ${DS}_pcd_build.out 2>&1 || echo "!!! ${DS}_pcd FAILED"
  for G in keyframe_scans all_scans; do
    echo "##### ${DS}_$G  $(date +%H:%M)"; python build_map_package.py session_${DS}_$G.yaml --steps P,S,C,E,F,G,H,V > ${DS}_${G}_build.out 2>&1 || echo "!!! ${DS}_$G FAILED"
  done
done
echo "##### ALL DONE $(date +%H:%M)"
