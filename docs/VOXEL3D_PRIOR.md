# voxel3d — 보행용 3D 확률 복셀 사전맵 (규격 · 로봇 적재 설계)

작성 2026-09-19. 산출물: `output/<map>/compiled/session_voxel3d.npz` + `session_voxel3d_tiles/` (빌더 `build_voxel3d.py`, 파이프라인 F 단계, 매니페스트 키 `voxel3d_prior` / `voxel3d_tiles`).

## 무엇인가
- **원천**: 키프레임 시각의 `/cloud_registered` (로봇 lidar_slam 이 기하 정합·보정한 점군, 최종 PGO 지도 프레임). 로봇 저장 지도와 2 cm 안에서 일치하는 것을 0916 데이터로 확인한 기하.
- **확률 격자 (0.1 m 정육면체, OctoMap 식)**: 스캔마다 LiDAR 원점→점 레이캐스팅. 끝점 셀 hit(+log 0.85/0.15), 통과 셀 miss(+log 0.45/0.55), 스캔당 셀 1회, 배치(≈4 s)마다 순차 갱신·클램프 [-2, 3.5]. 점유 = log-odds > 0. 한두 번 스친 사람·튀는 점·로봇 자기 몸은 miss 가 쌓여 지워진다.
- **세부 격자 (0.025 × 0.025 × 0.1 m = 로봇 `voxel_mapping` VoxelMap 격자)**: hit 점마다 count · 평균 z(voxblox 의 sub-voxel 표면에 해당) · z 표준편차 · 관측 스캔 수. 점유 판정은 부모 확률 셀이 한다(얇은 표면이 세부 격자에서 침식되지 않게).
- **depth 채움 (source 2)**: `label_map_points.py` 가 남긴 depth 카메라 점(LiDAR 0.15 m 밖) 중 LiDAR 가 hit 한 적 없고 miss 도 ≤ 3 인 부모 셀에만.
- **평탄 채움 (source 3)**: 같은 0.1 m 셀 안 LiDAR 세부 셀들의 z 퍼짐 ≤ 4 cm(국소 평면)이면 4×4 xy 발자국의 빈 세부 셀을 평균 z 로 채움. LiDAR 밀도가 2.5 cm 보다 성겨서 생기는 바닥 빈칸(getHeight NaN)을 없앤다. 벽·모서리는 채우지 않는다.
- **클래스**: 세부 셀 중심이 속한 0.1 m semantic 복셀의 argmax (254 = 확신 낮음, 255 = 라벨 없음). 동적 클래스(person) 셀 제거.

## 파일 규격
### `session_voxel3d.npz` (분석·시각화용)
| 키 | 형 | 뜻 |
|---|---|---|
| `ijk` | int32 (N,3) | 세부 격자 인덱스. x=(i+0.5)·res_xy, y=(j+0.5)·res_xy, 셀 중심 z=(k+0.5)·res_z |
| `mean_z` | f32 | 셀 안 점들의 평균 z (절대, 지도 프레임). getHeight 가 돌려줄 값 |
| `std_z`, `count`, `n_scans` | f16, u16, u16 | z 표준편차, 점 수, 관측 스캔 수 (채움 셀은 count=n_scans=0) |
| `logodds` | f16 | 부모 확률 셀 log-odds (depth 셀은 관측 수 기반 의사값) |
| `cls`, `source` | u8 | 클래스 id / 1 LiDAR · 2 depth · 3 평탄 채움 |
| `tile` | int32 | 타일 번호 (index.json 순서) |
| `coarse_ijk`, `coarse_logodds`, `coarse_hits`, `coarse_misses` | | 확률 격자 전체 (free 공간 포함) — 유령 검수·free 질의용 |
| `removed_ijk`, `removed_why` | | 제거된 세부 셀 (1 확률 free, 2 동적 클래스) — 검수용 |
| `params` | json | 해상도·확률·프레임·클래스 목록 |

### `session_voxel3d_tiles/` (로봇 적재용)
- `index.json`: `res_xy, res_z, tile_m, record(필드·dtype 순서), record_bytes=24, tiles[{tx, ty, x0, y0, n, bytes, z_min, z_max, file}]`, 프레임 설명, 클래스 목록.
- `t_<tx>_<ty>.bin`: little-endian struct 배열, 셀 (ix, iy, iz) 정렬.

```c
#pragma pack(push, 1)
struct PriorCell {            // 24 B
  int32_t  ix, iy;            // x=(ix+0.5)*res_xy, y=(iy+0.5)*res_xy   (지도 프레임, 전역 인덱스)
  int16_t  iz;                // 셀 중심 z=(iz+0.5)*res_z
  uint8_t  cls;               // 클래스 id (254 other, 255 unlabeled)
  uint8_t  src;               // 1 lidar, 2 depth, 3 flat-fill
  float    mean_z;            // 표면 높이 (절대 z, 셀 안 점 평균)
  float    max_z;             // 셀 안 최대 z — std 가 큰 셀(계단 코처럼 두 표면이 섞임)에서는 이 값을 쓰는 보수 규칙 권장
  uint16_t logodds_f16;       // IEEE half. 신뢰도 (LiDAR: OctoMap log-odds)
  uint16_t count;             // 점 수
};
#pragma pack(pop)
// 타일: tx = floor(x / tile_m), ty = floor(y / tile_m). 로봇 위치 (x,y) 에서 창 10 m 를 덮으려면 (tx-1..tx+1, ty-1..ty+1) 9개.
```

## 크기 (2026-09-16 데이터)
| 지도 | 세부 셀 | 전체 | 타일 | 최대 타일 | 비고 |
|---|---|---|---|---|---|
| b1f_2 (실내 112 s) | 1.42 M | 34 MB | 8 | 12.6 MB | 통째로 올려도 됨 |
| nearblock_2 (실외 300 m) | 19.1 M | ≈ 460 MB | 162 | ≈ 14 MB | 타일 필수 — 3×3 타일 ≤ ~130 MB 상한 |

RAM 상한 = 9 타일 × 최대 타일 크기. 타일 안에서 창 밖 셀을 버리면 더 줄어든다(10 m 창 = 400×400×40 셀; 그 안의 점유 셀은 보통 수십만 개 = 수 MB).

## 로봇 쪽 적재 설계 (권장: 2-계층 조회, 라이브 지도는 손대지 않음)
1. **프레임 정렬**: 사전맵은 SLAM 지도 프레임, 컨트롤러는 IEKF(`/Odometry/state_estimator`, odom) 프레임. `raisin_navigation_plugin` 이 하는 SLAM↔estimator 정렬(T_est←map)을 그대로 쓴다. 정렬이 없거나 SLAM fitness 가 낮으면 사전맵을 끈다(게이팅).
2. **prior VoxelMap**: `voxel_mapping` 의 `VoxelMap` 과 같은 기하(0.025/0.1, 10 m, z ±2 m)로 인스턴스를 하나 더 두고 timeout 없이 유지. `move()` 가 창을 옮길 때마다 새로 드러난 셀 범위를 타일에서 채운다: `occupied=true, count=1(또는 prior count), sumZ = mean_z − voxelCenterZ`.
3. **조회**: `getHeight(x,y,maxZ)` 를 라이브 → NaN 이면 prior 순으로. 라이브 관측이 있으면 항상 라이브가 이긴다(바뀐 환경·사람은 라이브가 본다).
4. **선택**: `cls` 로 계단·문턱 등에서 정책/속도 제한, `src==3`(채움)·`logodds` 낮은 셀은 라이브가 없을 때만 쓰기 등 신뢰도 정책.

## 검증 (hold-out: 스캔 10개 중 1개를 빌드에서 빼고 '나중 관측' 으로 사용)
`tools/voxel3d_eval.py` — getHeight 와 같은 규칙으로 질의해 |오차|·미응답(NaN)·유령(사전맵에 있으나 held-out 이 통과한 셀) 비율을 잰다. 결과는 `compiled/voxel3d_eval.json` 과 HANDOFF 참조.

## 재생성
```
python build_voxel3d.py --scans output/<map>/dataset/log_*/lidar_scans --voxsem output/<map>/voxel/session_voxsem.npz --out output/<map>/compiled/session_voxel3d
python voxel3d_view.py output/<map>/compiled/session_voxel3d.npz output/<map>/compiled/voxel3d_view.rrd --voxsem ... --bev ...
```
(컨테이너 `offline_3d_semantic_map`. 파이프라인은 `options.voxel3d` 로 켜고 끔.)

## 기하 전용 빌드와 Bonxai 점유 판정 (2026-10-15)
- **기하 전용** (`options.geometry_only: true`, 템플릿 `sessions/session_TEMPLATE_geometry_only.yaml`): 카메라 없이 Q·P·S·F·G·H 만 돈다.
  `build_voxel3d.py` 는 `--voxsem` 없이 실행되어 클래스가 전부 255(unlabeled), depth 채움·동적 클래스 제거가 없다. 0.2 m 코스트맵(라벨 필요)은 만들지 않는다.
  검증: t2_sim_1009_110835 를 기하 전용으로 다시 빌드 → 기존(카메라) 빌드와 셀 1,361,949 개·mean_z·max_z·logodds 가 정확히 같고 depth 채움 498 셀만 빠짐.
- **Bonxai 점유 판정** (`build_voxel3d.py --occupancy bonxai`, 세션 옵션 `voxel3d.variants: [bonxai]` → `session_voxel3d_bonxai_tiles/`):
  확률 격자(prob_res)의 점유 판정만 Bonxai ProbabilisticMap(`third_party/bonxai`, 커밋 8ba04b2, MPL-2.0)이 하고, 세부 격자 높이 통계·평탄 채움·클래스·타일은 같은 코드.
  범위 규칙은 자체 판정과 같다: 적중 ≤ max_range(30 m), 빈 공간 레이캐스팅 ≤ free_range(8 m) — 8 m 넘는 점은 8 m 지점까지만 레이(addMissPoint), 끝점 셀은 레이 없이 적중만.
  실행 파일 `tools/bonxai/bonxai_occupancy` 는 처음 쓸 때 `tools/bonxai/build.sh` 로 자동 빌드 (Eigen 3 헤더 필요: 컨테이너에 `libeigen3-dev`).
  비교 (t2_sim_1009_110835, 990 스캔): 보행 영역 2.5 cm 칸 커버리지 자체 90.3 % / Bonxai 90.1 %, 두 맵 모두 있는 칸의 바닥 높이 차 중앙 0.00 cm(3 cm 초과 0.3 %),
  점유 판정 시간 자체 약 200 s / Bonxai 13 s. 옛 벤치(nearblock)의 지면 침식은 빈 공간 레이를 30 m 전체로 그은 탓이었다.
  주의: Bonxai 는 점유 셀만 넘기므로(hit/miss 횟수 없음) depth 채움의 "LiDAR 가 여러 번 비었다고 본 칸은 채우지 않음(depth_max_miss)" 조건이 bonxai 변형에서는 적용되지 않는다.
  카메라 세션에서는 bonxai 변형의 depth 채움 셀이 자체 판정보다 많을 수 있다 (기하 전용 빌드에는 depth 채움이 없어 무관).
