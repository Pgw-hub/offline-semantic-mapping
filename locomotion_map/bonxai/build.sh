#!/bin/bash
# bonxai_occupancy 빌드 (컨테이너 offline_3d_semantic_map 안에서; Eigen 3 헤더 필요: apt-get install libeigen3-dev)
# build_voxel3d.py --occupancy bonxai 가 실행 파일이 없거나 소스보다 오래되면 이 스크립트를 부른다.
set -e
D=$(cd "$(dirname "$0")" && pwd)
B=$D/../../third_party/bonxai
[ -d /usr/include/eigen3 ] || { echo "Eigen 3 헤더 없음: apt-get install -y libeigen3-dev" >&2; exit 1; }
g++ -O3 -std=c++17 -DNDEBUG -I"$B/bonxai_core/include" -I"$B/bonxai_map/include" -I/usr/include/eigen3 \
  "$D/bonxai_occupancy.cpp" "$B/bonxai_map/src/probabilistic_map.cpp" -o "$D/bonxai_occupancy"
echo "built $D/bonxai_occupancy"
