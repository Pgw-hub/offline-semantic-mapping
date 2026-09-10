# Keyframe Graph 형식 (두 트랙 공용 계약)

지도의 기하 뼈대. **로봇 SLAM(raisin_lidar_slam_plugin)이 만든 것**(트랙 2) 과 **서버 오프라인 PGO 가 만든 것**(트랙 1) 을 같은 형식으로 저장해, 이후 의미 부여 파이프라인이 출처를 몰라도 되게 한다.

```
<root>/graph/
├── graph.json          메타 + 노드 목록 + 엣지 목록
└── clouds/%06d.npz     노드별 deskew 된 LiDAR 클라우드 (base 프레임, float32 xyz [+ intensity])
```

## graph.json
```json
{
  "format_version": 1,
  "source": "robot_slam | server_pgo",          // 누가 만들었나
  "frame": "map",                                // 노드 자세의 기준 프레임 (로봇 SLAM 의 map 프레임 = /Odometry/base 의 world 와 동일 원점)
  "time_base": "header_stamp_ns",                // 모든 stamp 는 메시지 헤더 시각(ns, 로봇 시계). mcap log_time 아님(40~100ms 지연)
  "extrinsics": {"T_base_lidar": [[...4x4...]]}, // 클라우드가 놓인 base 프레임과 LiDAR 의 관계 (기록용)
  "nodes": [
    {"id": 0, "stamp_ns": 1787819495542313930, "log": "log_000",
     "T_odom": [[...4x4...]],                     // 그 시각의 /Odometry/base (보정 전)
     "T_corr": [[...4x4...]],                     // PGO 보정 자세 (world←base). 클라우드를 이걸로 옮기면 world
     "cloud": "clouds/000000.npz", "n_pts": 41230}
  ],
  "edges": [
    {"from": 0, "to": 1, "type": "odom | loop | gps | prior", "T_rel": [[...4x4...]], "info": [6개 대각 정보량], "fitness": 0.97}
  ],
  "loops": 12, "optimized_at": "2026-09-11 10:20", "residual_rms_m": 0.03
}
```

## 규약
- 노드 시각 = 그 keyframe 의 **LiDAR 스캔 헤더 시각**. 플러그인의 KeyframeCloud.header.stamp 도 같은 값(`raisin::toTimestamp(scan_stamp_)`).
- RGB/depth 프레임의 헤더 시각도 같은 시계(확인: 2026-09-11, log_002 에서 네 토픽 헤더가 같은 epoch·ns, log_time 은 40~100ms 뒤).
- 임의 시각 t 의 보정 자세: `T(t) = T_corr(k) · T_odom(k)⁻¹ · T_odom(t)`, k = t 를 덮는(직전) 노드. 카메라 프레임을 노드에 붙일 때 이 식을 쓴다.
- 클라우드는 노드 base 프레임. world 로 옮길 때 `T_corr(k)`.

## 로봇 SLAM → 이 형식 (트랙 2 변환기 요구사항)
- `/pgo/keyframe_cloud` (KeyframeCloud: id, cloud(base 프레임), header.stamp) → nodes[].cloud, stamp
- 매핑 종료 시 최종 보정 자세 목록(`/path/pgo` 마지막 발행 또는 별도 저장) → nodes[].T_corr
- ★플러그인의 `SaveGraphFile`(NODES id x y z / EDGES from to cost) 은 **웨이포인트 그래프**라 자세가 없음. 보정 자세는 /path/pgo 나 새 내보내기가 필요.
- 실로봇은 keyframe_cloud 를 voxel-down 해 발행(`keyframe_publish_voxel_filter_`) → 해상도 파라미터를 0.1m 이하로 요청.
