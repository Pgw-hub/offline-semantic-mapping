# mcap_recorder: PointCloud2 / KeyframeCloud 기록 추가 (초안, 미빌드)

★ 20260911_103657 녹화에서 `/cloud_registered` 를 `pointcloud` typekey 로 기록한 결과: 매 메시지 2.3KB 헤더 조각만 남고 점 데이터 손실 (CustomMsg 구조체로 PointCloud2 를 재해석). 아래 패치 없이는 어떤 config 로도 기록 불가.

대상: `raisin_mcap_recorder_plugin/src/mcap_recorder_plugin.cpp`
현재 레코더의 typekey 11종에는 `sensor_msgs::msg::PointCloud2` 와 `raisin_interfaces::msg::KeyframeCloud` 가 없어
`/cloud_registered`, `/pgo/keyframe_cloud` 는 config 에 적어도 "unknown typekey" 로 무시된다. 아래를 추가하고 재빌드.

## 1. include / using (파일 상단, 다른 msg include 옆)
```cpp
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "raisin_interfaces/msg/keyframe_cloud.hpp"
using PC2 = raisin::sensor_msgs::msg::PointCloud2;              // lidar_slam /cloud_registered
using KFC = raisin::raisin_interfaces::msg::KeyframeCloud;      // lidar_slam /pgo/keyframe_cloud (idx + cloud)
```

## 2. 직렬화기 (serializePath 아래)
PointCloud2 를 필드 레이아웃째 그대로 저장한다 (PointXYZI 가정 없음 → 나중에 필드가 바뀌어도 리더가 해석).
```cpp
// PointCloud2: 필드 테이블 + raw data 그대로. lidar_slam 은 pcl::PointXYZI (x,y,z,intensity; point_step 16 또는 32).
static void serializePointCloud2(std::vector<char>& b, const PC2& m) {
  put<uint32_t>(b, m.height);
  put<uint32_t>(b, m.width);
  put<uint32_t>(b, static_cast<uint32_t>(m.fields.size()));
  for (const auto& f : m.fields) {
    put<uint8_t>(b, static_cast<uint8_t>(f.name.size()));
    b.insert(b.end(), f.name.begin(), f.name.end());
    put<uint32_t>(b, f.offset);
    put<uint8_t>(b, f.datatype);
    put<uint32_t>(b, f.count);
  }
  put<uint8_t>(b, m.is_bigendian ? 1 : 0);
  put<uint32_t>(b, m.point_step);
  put<uint32_t>(b, m.row_step);
  put<uint8_t>(b, m.is_dense ? 1 : 0);
  put<uint32_t>(b, static_cast<uint32_t>(m.data.size()));
  b.insert(b.end(), m.data.begin(), m.data.end());
}
static void serializeKeyframeCloud(std::vector<char>& b, const KFC& m) {
  put<uint64_t>(b, m.idx);
  serializePointCloud2(b, m.cloud);
}
```
`/cloud_registered` 는 0.1 m voxel 후 스캔당 ~수천 점 × 16 B ≈ 100 KB, 10 Hz → 1 MB/s 수준. 필요하면 depth_image 처럼 zlib 을 씌울 수 있다.

## 3. 등록 (reg_by_type 블록)
KeyframeCloud 는 `header` 가 최상위가 아니라 `cloud.header` 에 있어 `regHdr` 를 그대로 못 쓴다 → regJS 처럼 전용 registrar.
```cpp
reg_by_type["pointcloud2"] = {"sensor_msgs::msg::PointCloud2",
  "u32 h,w; u32 F; F*{u8 nlen; name; u32 offset; u8 datatype; u32 count}; u8 bigendian; "
  "u32 point_step,row_step; u8 is_dense; u32 len; data  # raw PointCloud2 (lidar_slam PointXYZI)",
  [&](const std::string& t, Chan* ch) { regHdr(t, ch, serializePointCloud2); }};
reg_by_type["keyframe_cloud"] = {"raisin_interfaces::msg::KeyframeCloud",
  "u64 idx; <pointcloud2>  # stamp = cloud.header.stamp (키프레임 스캔 시각), frame = base",
  [&](const std::string& t, Chan* ch) { regKFC(t, ch); }};
```
`regKFC` 는 `regJS` 를 복사해 `createSubscriber<KFC>` 로 바꾸고, Job 의 sec/nsec 에 `c->cloud.header.stamp.sec / .nanosec` 를 넣고, 람다에서 `serializeKeyframeCloud(b, *c)` 를 호출하면 된다.

## 4. CMakeLists
raisin_interfaces 가 이미 링크돼 있으면(RobotState/Command 를 쓰므로) 추가 의존성 없음. sensor_msgs 도 Imu/Image 로 이미 포함.

## 5. 오프라인 리더 (우리 쪽)
`geometry/deskew.py::read_mcap` 에 위 레이아웃 파서를 추가한다: 필드 테이블에서 x,y,z(,intensity) offset 을 읽어 `np.frombuffer(data, dtype=[...])`. `/pgo/keyframe_cloud` 는 (idx, stamp, xyz) 로, `/path/pgo` 는 마지막 메시지의 poses 배열을 idx 순서로 keyframe 에 매칭.

## 6. 녹화 전 검증
1분 녹화 후 mcap 요약에서 `/cloud_registered` ≈ 10 Hz × 시간, `/pgo/keyframe_cloud` ≈ 이동거리(m) 개, `/path/pgo` ≈ 0.5 Hz × 시간이 들어왔는지 확인.
