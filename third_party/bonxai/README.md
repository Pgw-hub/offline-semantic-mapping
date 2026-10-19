# Bonxai (vendored)

Upstream: https://github.com/facontidavide/Bonxai, commit 8ba04b2 ("perf(core): CoordMap root map …"). License: MPL-2.0 (see LICENSE).
Only the parts the map builder needs are copied, unmodified: `bonxai_core/include/` (header-only sparse voxel grid) and
`bonxai_map` ProbabilisticMap (`probabilistic_map.hpp/.cpp`, OctoMap-style log-odds occupancy with ray casting).
Used by `tools/bonxai/bonxai_occupancy.cpp`. Build needs Eigen 3 headers (`libeigen3-dev`, installed in the
`offline_3d_semantic_map` container on 2026-10-15).
