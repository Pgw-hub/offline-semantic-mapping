// Bonxai ProbabilisticMap 으로 스캔 묶음의 점유를 판정한다 (build_voxel3d.py --occupancy bonxai 가 호출).
// 사용: bonxai_occupancy <scans.bin> <res> <max_range> <free_range> <prob_hit> <prob_miss> <out.bin>
// 입력 scans.bin: 스캔마다 uint32 n, float32 origin[3], float32 xyz[n*3] (world 좌표, 원점 = 그 시각 LiDAR 위치)
// 범위 규칙 (build_voxel3d 자체 판정과 같게):
//   거리 ≤ free_range       : 적중 + 원점→점 레이로 빈 공간 (Bonxai addHitPoint)
//   free_range < 거리 ≤ max : 원점→free_range 지점까지만 빈 공간 (addMissPoint), 끝점 셀은 레이 없이 적중만 반영
//   거리 > max_range, < 5 cm: 버림
// 출력 out.bin: uint64 n, 그 뒤 n × {int32 ix, iy, iz; float32 logodds}  (점유 셀만, logodds > 0)
#include <bonxai_map/probabilistic_map.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unordered_set>
#include <vector>

namespace {
uint64_t keyOf(const Bonxai::CoordT& c) {
  return (uint64_t(uint32_t(c.x) & 0x1FFFFF) << 42) | (uint64_t(uint32_t(c.y) & 0x1FFFFF) << 21) |
         uint64_t(uint32_t(c.z) & 0x1FFFFF);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: %s scans.bin res max_range free_range prob_hit prob_miss out.bin\n", argv[0]);
    return 1;
  }
  const double res = std::atof(argv[2]), maxRange = std::atof(argv[3]), freeRange = std::atof(argv[4]);
  using Map = Bonxai::ProbabilisticMap;
  Map map(res);
  Map::Options opt;
  opt.prob_hit_log = Map::logods(float(std::atof(argv[5])));
  opt.prob_miss_log = Map::logods(float(std::atof(argv[6])));
  map.setOptions(opt);
  auto accessor = map.grid().createAccessor();

  std::ifstream in(argv[1], std::ios::binary);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> buf;
  size_t nScan = 0, nPts = 0, nFar = 0;
  std::unordered_set<uint64_t> nearHits, farHits;
  std::vector<Bonxai::CoordT> farCoords;
  while (true) {
    uint32_t n = 0;
    float o[3];
    if (!in.read(reinterpret_cast<char*>(&n), 4)) break;
    in.read(reinterpret_cast<char*>(o), 12);
    buf.resize(size_t(n) * 3);
    in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size() * 4));
    const Eigen::Vector3d origin(o[0], o[1], o[2]);
    nearHits.clear(); farHits.clear(); farCoords.clear();
    for (uint32_t i = 0; i < n; ++i) {
      const Eigen::Vector3d p(buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]);
      const Eigen::Vector3d d = p - origin;
      const double r = d.norm();
      if (r < 0.05 || r > maxRange) continue;
      ++nPts;
      if (r <= freeRange) {
        map.addHitPoint(p);
        nearHits.insert(keyOf(map.grid().posToCoord(p.x(), p.y(), p.z())));
      } else {
        map.addMissPoint(origin + d * (freeRange / r));
        const auto c = map.grid().posToCoord(p.x(), p.y(), p.z());
        if (farHits.insert(keyOf(c)).second) farCoords.push_back(c);
      }
    }
    map.updateFreeCells(origin);
    // far hits: one hit update per cell and scan, no ray (free space is only carved up to free_range)
    for (const auto& c : farCoords) {
      if (nearHits.count(keyOf(c))) continue;
      auto* cell = accessor.value(c, true);
      cell->probability_log = std::min(cell->probability_log + opt.prob_hit_log, opt.clamp_max_log);
      ++nFar;
    }
    if (++nScan % 500 == 0) std::fprintf(stderr, "  bonxai %zu scans\n", nScan);
  }

  std::vector<Bonxai::CoordT> occ;
  std::vector<float> lo;
  size_t nCells = 0, nFree = 0;
  map.grid().forEachCell([&](Map::CellT& cell, const Bonxai::CoordT& c) {
    ++nCells;
    if (cell.probability_log > opt.occupancy_threshold_log) {
      occ.push_back(c);
      lo.push_back(float(cell.probability_log) * 1e-6f);
    } else if (cell.probability_log < opt.occupancy_threshold_log) {
      ++nFree;
    }
  });
  std::ofstream out(argv[7], std::ios::binary);
  const uint64_t m = occ.size();
  out.write(reinterpret_cast<const char*>(&m), 8);
  for (size_t i = 0; i < occ.size(); ++i) {
    const int32_t ijk[3] = {occ[i].x, occ[i].y, occ[i].z};
    out.write(reinterpret_cast<const char*>(ijk), 12);
    out.write(reinterpret_cast<const char*>(&lo[i]), 4);
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("{\"scans\": %zu, \"points\": %zu, \"far_hit_updates\": %zu, \"cells\": %zu, \"occupied\": %zu, \"free\": %zu, \"seconds\": %.1f}\n",
    nScan, nPts, nFar, nCells, occ.size(), nFree, sec);
  return 0;
}
