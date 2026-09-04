# 세션 인계 — raibo offline semantic map + Map Compiler (호스트 세션)

새 Claude Code 세션을 `/home/gw.park` 에서 시작하고 이 문서를 붙여넣으면 이어서 작업 가능.
(시간순 상세 기록은 `HANDOFF_기록_20260902.md`, emc 쪽 상세는 `emc:/opt/catkin_ws/HANDOFF.md`)

## 최종 목표 (사용자 확정)
서버: `.mcap` → **offline semantic map** (dense 지형 + 3D scene graph) → **Map Compiler** 가
요청("이 구역 semantic layer 줘", "전자레인지 옆에 가서 사진 찍어와")을 받아 **CPU 로봇용 산출물**
(goal pose, costmap, action 시퀀스)로 컴파일 → 로봇(nav2 + behavior tree)이 수행.
- 역할 분담 확정: 지형(stuff) = dense 레이어, 객체(things) = ConceptGraphs. 통행성 맵은 로봇 nav2 도 자체 보유.

## 환경
- 이 세션 = blackwell 서버 호스트 (docker 가능, GPU RTX PRO 6000 97GB — 타 학습작업과 공유)
- 컨테이너 `emc`: ROS Noetic / elevation_mapping_cupy / rviz(noVNC :6081). Py3.8, sm_120 torch 불가.
  ★emc 안에 별도 Claude 세션이 있을 수 있음 — **ROS replay 는 한 세션만** (rosnode kill 충돌 이력)
- 컨테이너 `cg`: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-**devel** (nvcc), /data2 마운트, 상주형.
  ConceptGraphs ali-dev + 패치. 버전 고정: supervision 0.21.0, rerun-sdk 0.16.1, opencv 4.10, numpy 1.26.
- 데이터: /data2/20260827/20260827/KR-RRT-2604-008_20260827_172935 (야외 강변 log_000~010, 11분)
  - log_000 실내 사무실, 001 전환, 002~010 야외. **011~014 SLAM 발산 → 사용 금지**
  - 자세 = mcap `/Odometry/base` (LiDAR SLAM). depth = `d455_front/depth` (color 정렬 640x360)

## 완성된 파이프라인 (전부 검증됨)
### A. Dense 지형 레이어 (emc, 이전 단계 완료분)
- 전역 semantic 맵: /data2/20260827/_peek/global_session_full.npz (grass/path/tree/person)
- 전역 DINO 16D 맵: /data2/20260827/_peek/global_session_feat.npz (167×140m, res 0.10, feat셀 622k)
  - 고정 PCA /data2/20260827/_dino/dino_pca.npz — **재학습 절대 금지** (기존 맵 전부 무효화됨)
  - 교차로그 게이트 통과: 선형 0.841/kNN15 0.844 (기준선 0.416)
- ★전역 npz grid 규약: **row=x, col=y**

### B. ConceptGraphs 객체 레이어 (cg)
- 변환기: emc:/opt/catkin_ws/tools/mcap_to_cg.py — mcap→Replica 레이아웃
  (rgb 640x480 + depth 워프 + slerp 보간 c2w traj.txt). `--lidar` = ±0.4s LiDAR 누적→z-buffer→D455 빈픽셀 채움
- 실행: `docker exec cg bash -c "cd /data2/cg/concept-graphs/conceptgraph && export YOLO_CONFIG_DIR=/root/.yolo CG_MASK_BOTTOM_PX=60 CG_LOG_SCENE_CLOUD=4 [CG_LOG_LIDAR=1] [CG_DROP_CLASSES=person] CG_HEADLESS_RRD=<out.rrd> && python slam/rerun_realtime_mapping.py"`
  - scene 은 hydra_configs/raibo.yaml 의 scene_id, stride/suffix 는 rerun_realtime_mapping.yaml
  - ★객체 bbox_center 는 **이미 world 좌표** (스크립트가 dataset.poses raw 사용)
  - ★코드 수정 후엔 해당 scene 의 exps 삭제 (불완전 detections 재사용 사고 이력)
- 세션 전체 결과: 11로그 매핑 완료, 통합 338노드(session_objects.json) + 휴리스틱 엣지 671(session_edges.json)
- LiDAR 융합 결론(log_007): 원거리 기하 살아남(9→96객체), 동적 유령은 CG_DROP_CLASSES=person
  + obj_min_detections=2 로 억제 → **정적 6객체가 현재 기준선**. stride↓는 벤치 재현율 해결 못함(RGB 검출 한계).

### C. Map Compiler (cg:/data2/cg/map_compiler.py) — 산출물은 /data2/cg/compiled/
| 요청 | 출력 |
|---|---|
| text "질의" | CLIP 유사도 객체 순위 + world 좌표 JSON |
| terrain grass\|path\|tree | 전역 feat 코사인 활성도 grid (622k셀 68ms) |
| region x0 y0 x1 y1 | 지형 비율 + 영역 내 객체 |
| costmap | ROS uint8 costmap (path10/grass90/tree253+경사>0.35 치명) |
| merge / edges | 세션 통합 노드 / 기하 휴리스틱 엣지 |
| prompt "질문" [--planner] | LLM reasoning 프롬프트 (planner=원논문 JSON 응답 형식+엣지 포함) |
- 로봇 CPU 실측: CLIP 텍스트 인코딩 121ms(8스레드), 338객체 매칭 0.01ms

### D. 시각화
- rerun 웹: 포트 **9091(뷰어)+9092(rrd HTTP)** 포워딩 →
  `http://localhost:9091?url=http://localhost:9092/<rrd 상대경로(/data2/cg 기준)>`
  (ws 9877 은 VSCode 포워딩 불가. 서버: `~/.venvs/rerun/bin/rerun --serve --web-viewer-port 9091 --port 9877 <rrd>` + 호스트 rrd_http.py :9092)
- rrd 트리: world/objects/{rgb_pcd,seg_pcd,inst_pcd,bbox,bbox_w_labels}, world/scene(배경), world/lidar(회색 무라벨)
  — seg_pcd 만 켜면 논문의 "Incrementally Built 3D Map" 뷰. timeless 아님 → 타임라인 따라 누적.
- DINO 전역맵은 emc rviz(noVNC :6081, publish_global_feat.py + global_feat_view.rviz)

## 함정 (반복 사고 이력)
- `docker exec … python - <<EOF` 는 **-i 필수** (없으면 무음 no-op)
- pkill 패턴이 자기 명령줄과 매칭되면 셸 자살 → `pkill -u gw.park -x rerun`
- bg_classes 는 skip_bg=False 면 맵에서 안 걸러짐 → 동적 클래스 제외는 CG_DROP_CLASSES(자체 패치)
- ultralytics = AGPL-3.0 → 제품 배포 시 GroundingDINO+SAM(Apache) 교체 경로. Replica 데이터는 연구용 전용.
- `run_in_background`+`&` 동시 사용 금지(런처만 종료됨). 완료 확인은 로그 문자열/파일 수로.

## 열려 있는 결정 / 다음 후보
1. **검출 어휘 도메인 정리** — 야외 벤치 재현율의 진짜 해법 (classes 목록 교체 + conf, 재실행 수반)
2. 관측 위치 분산 기반 동적 객체 필터 (움직이는 자전거 억제)
3. GPT-4o 엣지·캡션 — /data2/cg/.openai_key 생성 대기 (로그당 $2~5)
4. 복셀맵(3D)·층 분리, Map Compiler 데몬화, task 요청 타입(goal+action 시퀀스)
5. ConceptGraphs 근본 한계 인지: stuff(지형) 미표현 → dense 레이어 병행이 우리 설계

## 참고
- 아티팩트: "Scene Graph 데이터 흐름", "파이프라인 속 CLIP과 LLM" (claude.ai/code/artifacts)
- 메모리: conceptgraphs-cg-setup, dino-feat-map-status, emc-ros-session-conflict

## 스냅샷·유스케이스 (2026-09-02 밤)
- 형상 저장: cg → 이미지 `cg-conceptgraphs:20260902` (docker commit). emc 는 베이스 레이어 손상으로
  commit 불가 → git 커밋(f3311a5, /opt/catkin_ws) + `docker export` → /data2/backups/emc_20260902.tar.gz (11.8GB).
  concept-graphs 패치는 레포 git 커밋(8ebc68a). map_compiler/cg_query 사본 → /data2/backups/code/.
- CPU 로봇 유스케이스 검증 완료: /data2/cg/robot_usecase_demo.py → /data2/cg/compiled/usecase/
  A(잔디 goal: 기하 불가→semantic 72m 도달) · B(경로준수: on-path 21%→79%) ·
  C(텍스트 "bench"→CLIP 124ms→goal→106m 경로). 리포트 아티팩트 "CPU 로봇 Semantic Navigation 검증".
  ★베이스라인은 보고된 실패모드의 모델링(융합 elevation 은 평활화되어 잔디 거칠기 재현 불가).
  ★semantic 전역맵은 0.05m/4800² (feat 맵과 해상도 다름). NaN 구멍→경사 계산 시 정규화 컨볼루션 필수.

## Semantic 3D 복셀 맵 (2026-09-03) — heightmap 대체 마스터 표현 프로토타입
- emc:/opt/catkin_ws/tools/build_voxel_sem_map.py <mcap> <calib> <cg_dataset(D455 depth)> <_sem/session> <out.npz>
  LiDAR(360°·40m)+D455 depth 점 → 0.1m 복셀 누적, semantic 은 카메라 z-buffer 로 보이는 점만 수령
  (_sem/session 은 LiDAR 스캔별 RGB 라 10Hz 로 자연 정렬). DINO 미포함(사용자 결정).
- log_007 결과: /data2/cg/voxel/log007_voxsem.npz — 복셀 1.03M, 라벨 32%/unknown 68%, LiDAR 점 라벨율 16.5%
- rerun: cg:/data2/cg/voxel_rerun.py → /data2/cg/compiled/voxel_log007.rrd (9.7MB)
  트리: world/voxels/{semantic,rgb,unknown}/f*, world/objects(bbox+"X on 지형"), world/robot. 프레임 따라 누적.
- 객체-지형 관계는 bbox 발자국 아래 복셀 지배 클래스로 자동 생성 (voxel_log007_relations.json) — 휴리스틱 엣지 대체 후보
- 세션 전체 복셀 맵: /data2/cg/voxel/session_voxsem.npz (merge_voxels.py) — 복셀 7.55M, 라벨 44%.
  rerun: /data2/cg/compiled/voxel_session.rrd (73MB, unknown 1/2 서브샘플), 관계 297개(voxel_session_relations.json)
  ★실내 log_000 구간의 "door on grass"류 관계는 semantic 모델이 야외 4클래스뿐이라 생긴 아티팩트 — 실내 어휘 없음.
  ★자기매칭 함정 재발: 대기 루프에서 `pgrep -f <스크립트명>` 은 루프 자신을 잡는다 → 파일 존재/로그 문자열로 판정할 것.
- ★복셀 표시 버그 수정: semantic 4클래스 확률이 전부 낮은 복셀(벽·차·바닥 등 야외 어휘 밖)이 argmax=0(grass)로
  연두색 표시되던 문제 → voxel_rerun_v2.py 에 "other"(max<0.3, 청회색) 추가, 관계 투표에서 제외.
  근본 해법은 _sem 사전계산의 채널 확장(ADE wall/floor/car/building… 추가) = 어휘 정리 항목.
- 관계 투표 최종 규칙: other·person 제외 → 세션 객체-지형 관계 99개 (bicycle on path 10, bench on path/grass 10, trash bin on path 5 …). 실내 아티팩트 소거.
- 복셀 후처리: cg:/data2/cg/voxel_postprocess.py <voxel.npz> <out_prefix> [--res 0.1 --clearance 1.2 --min-pts 6 --rrd]
  (a) occupancy 복셀(점유=관측 복셀, free/unknown 미구분 — ray-cast 정보 없음) 높이색 → world/occupancy_height
  (b) n×n costmap: 컬럼 지면(하위10% z, 3×3 min 평활) 위 [0.15, clearance] 에 non-grass 복셀 → 254 lethal,
      지면 semantic → path10/grass90/other150/기하만60, 희소컬럼(n_pts<6)·미관측 → 255. ROS 규약 npz+png, rerun world/costmap2d
  log_007: 1377×696, 관측 134k셀, 장애물 99k, path 7.1k, grass 16.7k. 코드 사본 /data2/backups/code/
- costmap 규칙 최종(등급형): 지면 위 [0.15, clearance] 구간에서 grass 라벨 → 통과(무시) / tree·other 라벨 또는
  라벨 없는 0.5m 초과 물체 → 254 lethal / 라벨 없는 ≤0.5m 물체(LiDAR 만 봄) → 200 고비용(불확실).
  log_007: lethal 44k · uncertain 48k · path 13k · grass 17k (이전 lethal 99k 에서 과판정 해소). 색: 주황=200.
- 세션 등급형 costmap: 2152×1856 @0.1m, lethal 355k · uncertain 161k (이전 573k lethal). rerun: compiled/post_session.rrd(39MB)
  world/occupancy_height(높이색 점유 복셀) + world/costmap2d(지면 위 비용 색). log_007: compiled/post_log007.rrd
- 무료 엣지 VLM: vlm_edges.py --backend local (기본) — Qwen2.5-VL 로컬(HF_HOME=/data2/cg/hf_cache). 7B 결과: 12쌍 정확도 25%
  (bbox 중첩을 'inside'로, 위아래를 'on top of'로 과판정 — 수치 메타 제거·박스 주석 무시 프롬프트로도 불충분).
  정답지 compiled/vlm_edges_log_000/ground_truth.json (Claude 육안), 채점 score_edges.py. 32B 다운로드 중 → 비교 예정.
- ★무료 VLM 엣지 채택: Qwen2.5-VL-32B-Instruct (Apache-2.0, bf16 ~66GB, 로드 41s, 쌍당 7.5s) — 12쌍 정확도 8/12=67%
  (오답 4개 중 3개는 검출 라벨 오류/애매 쌍). 7B 는 25% 로 부적합. --local-model 로 선택. 결과 compiled/vlm_edges_log_000_32b/
- 32B 세션 규모 시험: log_000 후보 80쌍(≤1.5m) → next to 36 · none 25 · on top of 16 · under 3, 쌍당 7.2s, 총 ~10분.
  결과 compiled/vlm_edges_log_000_full/edges_claude.json, rerun compiled/vlm_edges_log000_32b_80.rrd (edges_rerun.py).
  전 로그(수백 쌍) 확장은 같은 명령을 --scene 별로 반복 후 병합.
- 프롬프트 v2(지지 접촉 필수·A/B 색 재확인): 80쌍 → next to 37 · none 30 · on top of 10(전부 cup/laptop on desk, 오탐 제거) · under 3.
  12쌍 정확도 8/12 유지. 이 설정 확정 → 전 로그 실행.
- 세션 VLM 엣지 완료(Qwen2.5-VL-32B, 프롬프트 v2): 55개 (next to 42 · on top of 10 · under 3), 참여 객체 50.
  실내 log_000 이 50개, 야외는 정적 객체가 적고 2m 내 쌍이 드물어 5개(log_004 1, log_010 4).
  → 야외 관계는 객체-지형 측정 엣지(99)가 주력, 객체-객체 VLM 엣지는 실내 주력. 산출물 compiled/session_edges_vlm.json,
  rerun compiled/vlm_edges_session.rrd (edges_session_rerun.py). 엣지 3원: 휴리스틱 671 / 객체-지형 99 / VLM 55.
- 맵 패키지 매니페스트: compiled/map_manifest.json (8파일, 합계 ~100MB). 콜드 로드 0.52s(CPU), 재계산 불필요.
- 맵 로더: cg:/data2/cg/load_map.py <manifest> [--rrd] — MapPackage 클래스(복셀·객체·CLIP·엣지3종·costmap 일괄 로드 0.65s,
  summary(), labels(), to_rerun()). rerun compiled/map_package.rrd (map/voxels/{semantic/{grass,path,tree,person},rgb,other,unknown} · costmap2d · objects · edges/{heuristic,object_terrain,vlm}).
- rerun 기본 가시성: rerun 0.16 은 엔티티별 on/off 저장 불가 → Blueprint 의 뷰 contents 포함/제외로 대신함
  (load_map.py default_blueprint(): 탭 4개 — Map(기본: unknown·rgb·other·heuristic·costmap 제외) / Costmap / RGB / All layers,
  rr.send_blueprint(make_default=True) 로 rrd 에 저장). 제외된 엔티티는 뷰어 Blueprint 패널 "+" 로 다시 추가 가능.
