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

## 코드 저장소 (2026-09-04 커밋 완료)
| 위치 | 내용 | 최신 커밋 |
|---|---|---|
| emc:/opt/catkin_ws (git) | ROS 도구, mcap_to_cg.py, build_voxel_sem_map.py | 99e5be5 |
| cg:/data2/cg/concept-graphs (git, ali-dev 로컬 브랜치) | ConceptGraphs 호환 패치·raibo 설정 | fabcd43 |
| /data2/cg (git, main — 신설) | map_compiler·복셀 도구·vlm_edges·load_map·뷰어·docs/HANDOFF.md | b4811bb |
- 커밋은 컨테이너 안에서: `docker exec cg bash -c "cd /data2/cg && git add -A && git commit -m ..."` (git identity 는 -c 로 지정 또는 git config 설정)
- 데이터·모델·산출물(data/ voxel/ compiled/ hf_cache/ *.rrd *.npz)은 .gitignore — 백업은 docker 이미지 cg-conceptgraphs:20260902 + /data2/backups
- 원격(remote) 미설정 — GitHub/GitLab 에 올리려면 `git remote add origin <url> && git push -u origin main`

## 확률적 occupancy (OctoMap) 실험 (2026-09-04)
- 도구: emc tools/export_lidar_scans.py (스캔별 world 점+센서 원점) → cg build_octomap.py (octomap-python, 레이캐스팅 log-odds,
  tree.search 로 우리 복셀과 조인) → compiled/octo_log007_{occupancy.npz,octomap.bt,rrd}
- 결과(0.2m, 기본 hit0.7/miss0.4, 599스캔 23s): occupied 114k · free 2.48M 리프. 우리 0.1m 복셀의 65.8% 가 OctoMap free —
  동적 제거가 아니라 **지면 침식(path 86% free)·식생 통과** 아티팩트. person 은 94% free → 동적 지우기는 작동.
  ★우리 semantic 복셀 맵 = 표면 복셀(관측 점 누적), free/unknown 없음. 진짜 occupancy 는 별도 층으로 병행이 맞음.
- OctoMap 3설정 비교(우리 표면 복셀 중 free 판정 = 침식률): 0.2m 기본 65.8% / 0.2m hit.85·miss.45 **31.2%** / 0.1m hit.85·miss.45 39.5%.
  → miss 약화가 침식을 절반으로, 해상도↑는 오히려 악화(셀당 hit 희소). person 제거는 86~94% 유지.
  권장 하이브리드: occupied = 표면 복셀 ∪ OctoMap occupied / free = OctoMap free ∧ 표면 복셀 없음 / 그 외 unknown,
  "표면 복셀이 free 안" 은 person 등 동적 클래스에만 삭제 근거로 사용. rrd: compiled/octo_log007.rrd(0.2 기본), octo_log007_tuned.rrd(0.1 튠)
- rerun 정육면체 렌더: 0.16 Boxes3D 는 와이어프레임만 → 복셀 중심에서 8정점·12삼각 Mesh3D 를 클래스별 1개로 생성(cubes_mesh, 크기 0.96×res).
  build_octomap.py 기본 적용(occ/occupied/<class>), load_map.py --cubes (--sub 4~8 권장, 수백만 큐브는 브라우저 부담).
  rrd: compiled/octo_log007_tuned02.rrd (0.2m 튠, 권장 설정) · compiled/map_package_cubes.rrd
  · 검은 모서리: cube_mesh.py (면별 24정점 음영 메시 + <entity>/wire 에 검은 Boxes3D 와이어). load_map --cubes 기본 켜짐(--no-edges 로 끔), OctoMap rrd 는 "Occupied (no edges)" 탭으로 비교.
  · 해상도: load_map --coarse <m> 는 서브샘플이 아니라 병합(semantic n_sem 가중, rgb n_pts 가중). 세션 라벨 큐브 수: 0.2m 85만 · 0.3m 41만 · 0.4m 23만 · 0.5m 15만.
    rrd: compiled/map_package_cubes03.rrd (0.3m, 105MB) · map_package_cubes05.rrd (0.5m, 40MB). OctoMap rrd 는 --objects <session_objects.json> 으로 씬그래프 박스 겹침(occ/objects/*).
  · 투명도: --alpha (RGBA 굽힘). 0.16 뷰어는 런타임 슬라이더 없음, 메시 알파 지원 불확실 → compiled/octo_log007_tuned02_a40.rrd 로 확인.

## 확률적 occupancy 레이어 — 맵 패키지 내장 (2026-09-04)
- 외부 OctoMap 단계 제거: cg:/data2/cg/build_occupancy.py (numpy 레이캐스팅). 입력 = emc export_lidar_scans.py 의 스캔별 world 점+원점
  (/data2/cg/data/raibo/log_XXX/lidar_scans, 11개 로그 전부 export 완료) + session_voxsem.npz.
- 규칙: 스캔당 셀 1회 hit/miss, log-odds hit .85/miss .45 클램프[-2,3.5]; 하이브리드 = occupied ∪ 정적 표면 복셀, free = free ∧ 표면 없음,
  동적(person) 표면 복셀이 free 안이면 제거(surf_removed). 0.2m, 11로그 6526스캔 266s.
- 세션 결과: 셀 863만 (occupied 214만 · free 649만). occupied 출처: 레이만 292 · 표면복원 86만 · 둘 다 128만.
  표면 복셀 755만 중 레이 free 234만(31%) — 지면 침식/식생 통과라 복원됨. person 14,383 중 11,610 제거.
- 매니페스트 files.occupancy_probabilistic = compiled/session_occupancy.npz (41MB; 패키지 총 124MB). 로드 1.0s.
  MapPackage.occ / .occupancy_state(xyz) / .surf_removed. load_map --occ-view 0.4 (표시 병합).
- rrd: compiled/map_package_occ.rrd (0.3m 큐브 + occ 0.4m, 230MB) · map_package_occ_lite.rrd (0.5m/0.6m).
- 미완: 코스트맵(voxel_postprocess) 이 아직 이 레이어를 안 씀 — free 셀을 '관측된 자유공간' 으로 넣으면 255(미관측) 줄일 수 있음.

## 요청 검증 + 코스트맵 v2 (2026-09-04)
- cg:/data2/cg/verify_requests.py <manifest> --n 20 --exclude-indoor --costmap-key costmap_2d_occ
  랜덤 시작 → 문장 파싱(사전 KO2EN) → CLIP 객체 → "옆" 목표(0.5~1.5m 링, 같은 연결요소, 3D occupancy 몸통 검사) → 가중 A*.
  + 지형 코스트맵 요청 grass_only/avoid_grass/grass_layer/graded, n×n 크롭, nav2 map_server pgm+yaml. 출력 compiled/verify_occ/.
- ★발견: v1 costmap_2d(voxel_postprocess) 는 통행면 3,253m² / lethal 3,553m², 연결요소 14,961(최대 557m²) → 4/20.
  원인 = 표면 복셀 1개(hit 1회)도 장애물 + 열 최저점 지면. occupancy 셀 34% 가 hit 1회.
- costmap_from_occupancy.py → compiled/session_costmap_occ.npz, 매니페스트 costmap_2d_occ (0.2m). 지면 hits≥2 최저 셀, 장애물 hits≥3,
  grass/person 제외, tree 또는 >0.5m lethal. 통행 9,483m² / lethal 719m², 최대 요소 8,907m² → 20/20 (평균 62m, 우회비 1.56).
  MapPackage(manifest, costmap_key="costmap_2d_occ"). inflation 붙인 뒤 v1 제거 권장. voxel_postprocess 의 rrd/뷰는 아직 v1.
- 리포트 아티팩트: "벤치 옆으로 가줘 검증".
- 세션 RGB 배속 영상: compiled/session_rgb_x15.mp4 (43.6s, 57MB) · session_rgb_x10.mp4 (65s, 81MB). 11로그 30fps 프레임을 stride 15/10 으로 뽑고(cg cv2, 960px, 상단 오버레이) emc ffmpeg libx264 로 인코딩.
  브라우저 재생: http://localhost:9092/compiled/session_rgb_x15.mp4 (rrd_http.py 서버). 프레임 임시폴더 video_frames*/ 는 삭제.
- 기하 vs semantic 정적 코스트맵 비교 (compare_costmaps.py, 아티팩트 "기하 vs Semantic 정적 코스트맵"): 세션 잔디 1,142m² 중 lethal 비율
  A 원시점 기하(0.1m, ≥0.15m) 87.6% · B occupancy 기하(hits≥1) 6.5% · C semantic 5.2%. 최대 통행영역 60 / 4,221 / 8,293 m².
  ★핵심: 잔디 통과의 대부분은 다중스캔 occupancy 에서 이미 나옴; semantic 은 등급(잔디 90, 산책로 선호)·낮은 미라벨 장애물 200 (연결성)·tree/other 보수 처리를 더함.
  B 도 hits≥3 이면 한 덩이(8,384m²). 파일 compiled/cmp_costmap_*_h1.npz, compiled/compare_costmap/.
- 산책로 준수 정책 검증 (policy_adherence.py, 아티팩트 "산책로 준수 정책 검증"): 산책로 셀 50쌍(20~80m), GEO/SEM/STRICT 세 맵 모두 도달.
  산책로 위 비율 42.2% → 74.6% → 78.9%, 잔디 침범 합계 176.5m → 47.5m → 0, 길이 +4.6% / +10.0%. 잔디 밟는 쌍 36 → 19 → 0.
  ★compare_costmaps/policy_adherence 의 경로 오버레이는 격자에 그린 뒤 전치+반전해야 함(전치 후 그리면 대각 반전 버그, 수정 완료·재발행).

## 컨테이너 통합 (2026-09-04) — offline_3d_semantic_map
- 새 컨테이너 `offline_3d_semantic_map` (이미지 offline-3d-semantic-map:20260904 = cg 스냅샷 + mcap 1.4.0 + ffmpeg).
  run: docker run -d --name offline_3d_semantic_map --gpus all --shm-size 16g -v /data2:/data2 -w /data2/cg --restart unless-stopped offline-3d-semantic-map:20260904 sleep infinity
  버전 고정 유지: numpy 1.26.4 · cv2 4.10 · torch 2.7.1+cu128 · transformers 5.16.1 · rerun-sdk 0.16.1 · supervision 0.21.0.
- 전처리 5종을 emc 에서 /data2/cg/tools/preprocess/ 로 이전 (ROS 불필요): mcap_to_cg, extract_rgb_session, precompute_semantic(=구 precompute_semantic_outdoor, 인자화),
  build_voxel_sem_map, export_lidar_scans. log_007 검증: 프레임·depth·traj·lidar_world 바이트 동일, SegFormer 확률 동일, 복셀 키 집합 동일.
  precompute_semantic 은 이제 이 GPU 에서 돈다(40프레임 11s; 세션 6600프레임 ≈ 30분).
- ★기존 참조 복셀(voxel/log007_voxsem.npz, session)은 LiDAR 융합 없는 data/raibo/log_007 depth 로 만든 것. --lidar 데이터셋(log_007_lidar)으로 만들면 복셀 +0.14%.
- 이후 모든 작업은 offline_3d_semantic_map 에서. cg 는 참고용으로 남김(중지 여부는 사용자 결정), emc 는 옛 히트맵 파이프라인·rviz 전용.
- 다음: 한 번에 도는 빌드 오케스트레이터(build_map_package.py + 세션 YAML) — 단일 컨테이너라 docker exec 없이 파이썬 한 프로세스로 가능.

## 단일 스크립트 빌드 (2026-09-04) — sessions/build_map_package.py
- 컨테이너 offline_3d_semantic_map 에서: `cd /data2/cg/sessions && python build_map_package.py <session.yaml> [--steps A,S,R,B,C,D,E,F,G,H] [--force X] [--dry-run]`
  YAML: root, mcap_dir, logs(연속 범위), calib(기본 mcap_dir/raisin_config/calibration), options{lidar_fusion, mask_bottom_px, drop_classes, vlm_edges, vlm_max_pairs, occupancy_res, prob_hit/miss, costmap{...}, view_coarse, map_id}
  레이아웃 <root>/{dataset/log_XXX, sem, voxel, compiled, logs}. 산출물 있으면 건너뜀. 단계별 로그 <root>/logs/*.log, 전체 로그 sessions/<name>_build.out.
- 검증: sessions/test_2logs (log_007+008) 전체 13분 → 매니페스트 9 파일 25.7MB, 복셀 166만, occupancy 296만 셀, 객체 12, 코스트맵 813×320@0.2m, map_package.rrd 62MB.
  단계 시간: A 데이터셋 81s/로그 · B SegFormer 268s(1,200프레임) · D ConceptGraphs ≈110s/로그(scene cloud 로깅 끔) · E 115s · F 30s.
- 고친 것: merge_voxels 파일명 log_XXX 허용, MapPackage 코스트맵 없이 로드 가능, map_compiler 경로 env(CG_RAIBO_ROOT/CG_OUT_DIR).
- ★야외 로그는 객체가 적어(로그당 6개) VLM 후보 쌍이 0 일 수 있음 → edges_vlm 빈 목록도 정상. 전체 세션 재현용 YAML: sessions/session_riverside_20260827.yaml (미실행, 예상 2시간).
- 기존 세션(data/raibo, compiled/)은 손대지 않음. 새 세션은 모두 sessions/<name>/ 아래.

## 저장소 이전 + session_3 (2026-09-04)
- ★코드·세션 설정·결과의 정본은 /home/gw.park/offline_semantic_mapping (git). /data2/cg 는 mcap 원본·ConceptGraphs 설치본·hf_cache·옛 산출물만.
  컨테이너 offline_3d_semantic_map 는 이 폴더를 같은 경로로 마운트(-w 도 여기). YAML root 에 이름만 쓰면 output/<이름>/.
  빌드 끝나면 output 소유권을 gw.park 으로 chown. 9092 파일 서버는 tools/rrd_http.py [dir] (기본 = 저장소) → 링크 http://localhost:9092/output/<이름>/compiled/map_package.rrd,
  옛 산출물은 심볼릭링크 data2_cg/ 경유 (http://localhost:9092/data2_cg/compiled/...).
- session_3 (log_000~002, 실내→야외) 35분: A 253s(84s/로그) · R+B 370s(1,800프레임) · C 35s · D 1,066s(≈6분/로그, 실내 객체 多) · E 298s(VLM 포함) · F 88s · G+H 6s.
  결과: 복셀 225만 · occupancy 316만 셀 · 객체 76 · 엣지 휴리스틱 76/지형 29/VLM 13 · 코스트맵 460×598@0.2m · 패키지 9파일 31.7MB · rrd 67MB.
- 단계 V 추가: options.video_speed (기본 15, 0=생략) → compiled/session_rgb_x15.mp4 (tools/make_rgb_video.py, 30fps 원본 stride=배속, 960px, H.264). session_3 3로그 → 12s 14.6MB, 5초 소요.
- Map Portal (2026-09-04): tools/map_portal.py 가 9092 를 맡음(rrd_http.py 대체, 파일 서빙 겸함). http://localhost:9092/ 에 output/*/compiled/map_manifest.json 스캔 카드
  (데이터셋·로그·통계 chips·클래스 분포·rerun 열기·매니페스트·빌드 로그·영상 자동재생). step H 가 compiled/summary.json 작성(포털이 읽음). 시작: tools/start_portal.sh.
  test_2logs 도 output/ 아래로 이동(구 /data2/cg/sessions 비움).
- 단계 Q 데이터 품질 (2026-09-05): tools/mcap_quality.py → compiled/data_quality.{rrd,json}. 모든 토픽의 dt/rate/size 시계열(고주파 IMU 는 초 단위 집계), gap(dt>2.5×중앙값)·로그별 커버리지(1s 이상 늦은 시작/이른 끝) 이벤트 TextLog,
  재생 탭(RGB stride15 jpeg, depth, LiDAR 1Hz 1/4점, odom 궤적). 3로그 6초, 87MB. 포털 카드에 품질 표(토픽별 Hz·개수·유실≈·gap·dt max·MB·늦은 시작) + "데이터 품질·재생" 버튼.
  ★발견: log_000 /Odometry/base 가 로그 시작 후 6.3s 부터 나옴(SLAM 초기화) → 그 구간 프레임은 자세 없음(pose스킵). IMU gap 은 1~2 msg 수준(base_imu 120건).
- 콘솔 상시 구동 (2026-09-05): systemd --user 서비스 map-portal.service(:9092) · rerun-viewer.service(:9091, ws 9877, sdk 9876) 등록·enable, linger on → 재부팅/로그아웃 후에도 유지.
  파일 사본 tools/*.service. 모두 0.0.0.0 바인딩이라 사내망에서 http://192.168.100.161:9092/ 로 접근 가능(방화벽은 sudo 없어 미확인). 인증 없음 → 외부 공개는 Tailscale/Cloudflare Access 등 앞단 필요.
  ★rerun --serve 는 --port(SDK)와 ws 포트를 다르게 줘야 함(같으면 즉시 종료). 콘솔 상세 화면의 RGB 단독 탭 제거(3분할에 포함).

## 콘솔에서 빌드 (2026-09-09)
- Datasets 탭: DATASET_ROOTS(기본 /data2) 아래 깊이≤4 에서 log_*.mcap 폴더 스캔(60s 캐시, Rescan 버튼). 로그 수·범위·연속성·크기·캘리브 유무·이미 빌드된 맵 표시.
- Build map: 웹 폼(이름·로그 범위·고급 옵션=OPTION_SPEC 범위 검증) → sessions/session_<name>.yaml 생성 → 큐(한 번에 1개) → `docker exec offline_3d_semantic_map python build_map_package.py …`
  (포털은 호스트 venv 에서 돌고 빌드는 컨테이너). 로그 sessions/<name>_build.out, 큐 상태 sessions/jobs.json. Processing queue 탭: 단계 진행바·Cancel·Log.
- 포털 재시작 시 진행 중이던 작업은 status=unknown 으로 표시(빌드 자체는 컨테이너에서 계속 돎). 쓰기 동작 보호: env PORTAL_TOKEN 설정 시 X-Portal-Token 필요(기본 미설정 — 사내망 한정 전제).
- 첫 웹 빌드: web_172218_l01 (172218 세션 log_000~001, VLM off).

## 클래스 사전 (2026-09-09)
- classdict.py + classdicts/<name>.json: SegFormer ADE20K 150 라벨 → 클래스(ade 부분문자열 묶음) + role(ground_preferred/ground_soft/obstacle/dynamic/context) + cost + color.
  default 는 내장(초기 4클래스, ids 동일: grass[9] path[6,11,13,29,52,54,91] tree[4,17,72,87] person[12]) → 기존 sem 과 수치 동일 검증(24프레임 max diff 0).
  ★'tree' 부분문자열이 'streetlight' 도 잡음(원래부터). 새 사전에서 tree 를 정확히 하려면 ade 에 "tree" 대신 라벨 전체("tree", "plant", "palm") 를 두고 streetlight 를 별도 클래스로.
- 파이프라인: precompute_semantic --classdict → sem/classes.json + N채널; build_voxel_sem_map/merge_voxels N채널 + npz classdict; build_occupancy dynamic=role; costmap_from_occupancy 역할 규칙
  (ground_*: 지면값=cost, soft·dynamic 은 밴드 통과, obstacle 은 밴드 lethal, context 지면 150); load_map/voxel_rerun 색은 사전에서. 구 npz(classdict 없음)는 이름으로 legacy 복원.
- 오케스트레이터 options.classdict (기본 default). sem/ 이 다른 사전으로 만들어져 있으면 중단(--force B 또는 새 root). 매니페스트에 classdict 전체 기록.
- 포털 Class dictionary 탭: 목록/편집(ADE 라벨 검색 추가, 역할, 코스트, 색)/Duplicate/Save/Delete, 빌드 폼에 사전 선택. classdicts/ade20k_labels.json 은 컨테이너에서 모델 config 로 덤프.
- 예시 사전 outdoor_v2 (car=obstacle, building/wall/fence=context) → 웹 빌드 web_172218_cdv2 로 검증 중.
- 라벨 오염 수정 (2026-09-10): build_voxel_sem_map --edge-px 4 (클래스 경계 4px 안 라벨 보류) + --require-depth 1 (depth 없는 픽셀엔 LiDAR 라벨 안 줌). 옵션 label_edge_px/require_depth (기본 4/true).
  172935 log0-5 outdoor_v2 비교: 차량 위 0~1.2m 오라벨 19,973→10,344 (-48%), 라벨 복셀 -9%(경계·원거리 배경이 unknown 으로), car -13%.
  남은 오라벨의 원인: 차 지붕·유리의 반사(SegFormer 가 tree/building 으로 판정) → 세그 정확도 몫. 후처리(차량 덩어리 위 0.6m 관측≤2 재배정)는 미적용.
  비교 rrd: tools/compare_maps_rerun.py A B out.rrd (before/·after/ 나란히) → output/web_172935_l05_cdv2_fix/compiled/compare_before_after.rrd.
  ★비교 빌드 요령: 새 root 에 dataset/·sem/ 심볼릭링크 → C 부터만 재실행 (4분).

## 초기 위치 잡기 (2026-09-10)
- localize.py: 정합용 지도 = map_points(source: voxels | voxels_occ(표면복셀 ∩ occupancy≠free, 동적 제거) | occupancy), 제외 클래스 기본 person,car.
  다중 스케일 point-to-plane ICP(0.4/2.5m → 0.2/1.0 → 0.1/0.4). --sweep 는 보류 로그 스캔(0.5s 누적)에 (r, dyaw) 오차를 주고 수렴 성공률 표. 정합 1회 ≈ 2s(전역 340만 점).
- tools/localize_server.py (:9093, 컨테이너, 포트 공개를 위해 컨테이너 재생성 -p 9093:9093): 지도 메모리 상주, 스캔 후보(보류 로그 6~10, 커버리지≥0.4) 186개,
  요청마다 초기 자세 주변 40m crop → ICP ≈1.1s. 결과: 단계별 자세·최종·fitness/rmse·정답 대비 오차. rrd 는 RecordingStream 을 만들고 disconnect 로 flush(전역 rr.save 는 서버에서 안 씀).
  시작: tools/start_localize_server.sh <manifest> [logs]. 포털 Localization 탭이 host:9093 을 직접 호출(VSCode 면 9093 도 포워딩).
- 첫 결과 (지도 web_172935_l05_cdv2_fix, 스캔 log_006#0): 초기 오차 1.5m/20° → 0.017m/0.04° (1.1s); 4.2m/60° 는 yaw 다중 시작으로 → 0.018m/0.04° (4.2s).
- 배치 스윕(sweep)은 사용자 요청으로 중단 → 대화형 도구로 대체. 필요하면 localize.py --sweep 그대로 사용 가능.
- 주차장 held-out 초기화 검증 (2026-09-10): 지도 web_172935_l02_cdv2_fix(로그 0~2, dataset/sem 심볼릭링크 재사용) + log_003 스캔(지도 미포함, overlap 0.91~0.99).
  결과 3스캔: 초기 1.5m/20° → 0.04m/0.16°, 0.11m/3.84°(기준 3° 살짝 초과), 0.03m/2.5°. yaw 다중시작(4m/60°)도 같은 값으로 수렴 → ICP 자체는 수렴, 잔여 yaw 2~4° 는
  log_002 통과 시점과 log_003 시점 사이 SLAM 오도메트리 드리프트(지도 내부 일관성) 로 추정. in-map(log_002) 스캔은 0.04m/0.04°.
  Localization 탭: Map/Candidate logs 선택 후 "지도·로그 적용" (POST /api/loc/map). 스캔 목록에 in-map/held-out 표시. 서버 현재 상태: l02 지도, 로그 3,2.
  ★주의: 서버의 map 전환 실패 시 상태가 반쯤 바뀌던 버그 수정(원자적). 상대 경로는 저장소 루트 기준.
- PCD 내보내기 (2026-09-10): export_pcd.py <manifest> out.pcd --source voxels|voxels_occ|occupancy --exclude unknown,person,… --min-nsem N --occ-filter 1/0.
  PCD v0.7 binary, 필드 x y z rgb(packed float, PCL 관례) label(uint32; 사전 순서, other=N, unknown=N+1) + <out>_legend.json. 포털 상세 "Download PCD…" 대화상자(클래스 체크, 기본 unknown·person 제외).
  캐시: output/<맵>/compiled/export/<맵>_<src>_no-…_<hash>.pcd. Open3D 는 xyz+rgb 만 읽음(label 은 numpy 로).

## 기하 재구성 투트랙 (2026-09-10) — geom_rebuild / paint_slam
- 배경: 현재 맵은 raw(비-deskew) 스캔을 프론트엔드 `/Odometry/base`(드리프트, 루프클로저 없음) 에 그대로 얹음 → 주차장 held-out yaw 2~4°, 지도 내부 불일치.
  로봇 SLAM(refer/raisin_plugin/raisin_lidar_slam_plugin, DLIO+GTSAM iSAM2, 반경 루프클로저+GICP) 은 키프레임 그래프를 갖지만 우리 mcap 에 `/pgo/keyframe_cloud`·`/path/pgo` 가 없음.
- 트랙 1 `geom_rebuild` 컨테이너(같은 이미지 + gtsam pip): 서버가 기하를 다시 만든다 — deskew → 키프레임 → 오프라인 PGO → graph/graph.json.
- 트랙 2 `paint_slam` 컨테이너: 로봇 SLAM 키프레임 그래프 위에 의미만 입힌다. 로봇 데이터가 없어 트랙 1 그래프를 대용 입력으로 사용. 공유 규격 geometry/GRAPH_FORMAT.md.
- 시각 정합 확인: 모든 토픽 header.stamp 는 같은 시계(ns epoch), mcap log_time 은 header 보다 40~100ms 늦음. 키프레임 stamp = LiDAR 스캔 header stamp. Livox CustomMsg 점별 `ot` = 0~100ms 오프셋(99% 단조).
- deskew 검증 (geometry/deskew.py, deskew_eval.py; 172935/log_002, 이동 구간 81쌍 연속 스캔 NN 중앙값):
  raw 7.93cm → deskew 6.93cm (−12.6%), "헤더=스캔 끝" 가설은 8.85cm 로 악화 ⇒ `/Odometry/base` 와 스캔 header 는 스캔 시작 시각. 지면 두께 지표는 skew 에 무감(8.0→8.3cm) 이라 폐기.
- refer/ (497MB, 로봇 플러그인 복사본) 은 .gitignore.
- 오프라인 PGO 구현 (geometry/build_graph.py, 2026-09-10): 로그별 스트리밍 → deskew 키프레임(1m/10°, 앞뒤 ±2 스캔 누적, 5cm) → 오도메트리 엣지 →
  루프 후보(반경 8m, 인덱스 차 ≥40, i 당 최근접 1개) GICP(0.1m, 1.5→0.6m corr) + point-to-plane 마무리, 수락 fitness≥0.55·rmse≤0.10 → GTSAM LM(Huber) → 2라운드.
  오도메트리 점프>5m(SLAM 붕괴) 가드. 산출 graph/graph.json + clouds + odom_log_XXX.npz + graph_view.rrd(before/after). 실행: geom_rebuild 또는 offline_3d_semantic_map(gtsam pip 설치함) 컨테이너.
  172935 log0-10 (895 kf, 654s): 루프 55, 자세 변화 평균 0.16m 최대 0.32m, 루프 잔차 rms 0.9cm, 101s. held-out 재방문 쌍 60개 GICP 일관성(geometry/graph_eval.py):
  odom 초기화 5.4cm(p90 9.9) → PGO 2.5cm(p90 4.6) 중앙값 — 재방문 불일치 −54%. 회전은 0.15°→0.14°(이미 작음). output/geom_172935_l00_10/graph/{graph_view,eval}.rrd
  log0-2 만(231 kf): 루프 3, 자세 변화 ≤7cm → 짧은 구간은 PGO 효과 미미, 이득은 주로 deskew.
- 파이프라인 연결: geometry/pose_provider.py(PoseProvider.at, Δ 보간) · tools/preprocess/{mcap_to_cg,build_voxel_sem_map,export_lidar_scans}.py --pose-graph (export 는 점별 deskew) ·
  오케스트레이터 단계 P + options.pose_graph(기본 false) · 포털 빌드 폼 체크박스 "PGO 기하 재구성", 상세 Stats "Geometry" 행(kf/loops/shift/revisit, before/after rerun 링크).
  매니페스트 frame "SLAM world, PGO-corrected", pose_graph: graph/graph.json.
- 검증 빌드 sessions/session_web_172935_l02_pgo.yaml (root web_172935_l02_pgo, l02_cdv2_fix 와 동일 옵션 + pose_graph, sem 심볼릭링크 재사용) → 주차장 held-out(log_003) 초기화 재시험 예정.
- 두 인스턴스 동시 작업 주의: 다른 Claude 인스턴스(gw-park-35)가 build_graph.py 를 덮어쓴 일이 있었음 → 현재 파일이 정식(그쪽 버전 + SLAM 붕괴 가드). 동시에 같은 저장소를 만지지 말 것.
- 검증 결과 (2026-09-10, geometry/loc_bench.py, held-out log_003 25스캔×2 초기오차 1.5m/20°, 누적 5스캔, 지도 voxels_occ):
    old 지도 + raw 질의     : 위치 8.7/12.3cm(중앙/p90)  yaw 1.02°/2.47° (최대 3.85°)  성공 96%
    old 지도 + deskew 질의  : 위치 7.1/10.8cm            yaw 0.07°/0.18° (최대 0.23°)  성공 100%
    PGO 지도 + raw 질의     : 위치 7.1/ 9.3cm            yaw 0.99°/2.85°               성공 92%
    PGO 지도 + deskew 질의  : 위치 3.7/ 8.1cm            yaw 0.08°/0.18°               성공 98% (1건 국소해: 스캔 180, 2.9m/16.8°, fitness 0.57)
  ⇒ 이전 held-out yaw 잔차 2.5~3.8° 의 원인은 SLAM 드리프트가 아니라 회전 중 질의 스캔 skew (log0-10 PGO 상 log_003 구간 오도메트리 yaw 보정 ≤0.1°).
    질의 deskew 로 yaw 해결, 지도 기하(PGO+deskew) 로 위치 오차 절반. 새 지도 occupancy 셀 3.16M→2.95M(−6.7%, 번짐 감소), 객체 92→74.
  ★로봇 온라인 초기화도 질의 스캔을 deskew 해서 넣어야 함(/cloud_registered 사용 또는 IMU 보정). localize 서버는 지금 web_172935_l02_pgo + 로그 3,2(deskew 스캔) 로 전환됨.
  결과 파일: output/web_172935_l02_pgo/{loc_bench.json, loc_bench_oldmap.json}. held-out 질의 스캔은 export_lidar_scans.py --deskew (그래프 없이 원 오도메트리) 로 생성.

## 2차 데이터셋 20260911_103657 (2026-09-11)
- /home/gw.park/raisin_logs/20260911/... → /data2/20260911/KR-RRT-2604-008_20260911_103657 로 복사(컨테이너 마운트). 14 로그 824s 19GB, raisin_config 는 20260827 복사.
- 점검 결과는 docs/RECORDING_CHECKLIST.md 상단. 요약: /cloud_registered 손상(typekey 오류), /Odometry/base 는 PGO 실시간 보정 포함(루프클로저 점프 ≤10m), /path/pgo 998kf(stamp 없음).
- geometry/robot_graph.py: 로봇 최종 PGO(/path/pgo 마지막 메시지 또는 --traj <name>.txt) → graph/ (PoseProvider 규약). /path/pgo 는 점프 제거 오도메트리와 상대이동 매칭으로 시각 복원,
  --tmax/--max-err 로 신뢰 노드만. 노드 jump_ns → pose_provider.delta 가 그 구간은 보간 대신 점프 시각에서 계단 전환.
- geometry/build_graph.py: 점프 가드가 두 번째 로그에서 logs.index 로 죽던 버그 수정. 이 데이터셋은 실시간 오도메트리 점프 때문에 자체 PGO(odom 엣지) 부적합 → 로봇 PGO 자세 사용.
- 빌드: sessions/session_web_103657_l00_09_robotpgo.yaml (log0-9, 0~600s, 자세=로봇 PGO 546 노드). 전 구간(l00_13)은 <name>.txt 확보 후 --traj 로 graph 재생성 → 재빌드 (sem 은 심볼릭링크 재사용).
- geometry/robot_pgo_path.py 는 초기 진단용(robot_graph.py 로 대체).
- ★20260911 데이터 기하 판정 (2026-09-11 오후): 실시간 /Odometry/base 의 점프 3건(248/416/584s, 3~4m)과 로봇 최종 /path/pgo 의 같은 자리 3.5m 계단이 일치.
  점프 제거(dejump) 오도메트리로 자체 PGO(build_graph --dejump --loop-radius 12 --max-shift 10, output/web_103657_l00_09/graph)를 돌리면 log_009↔log_003 재방문에서 GICP 5~7m 이동(fit 0.9)
  루프 5개 수락, 최적화 후 루프 잔차 rms 0.57m(1차 데이터셋 0.9cm) → 루프끼리 불일치. 점프 제거 후 드리프트가 7m 라는 뜻이면 점프=PGO 보정이 맞지만, 주차장 반복 구조의
  거짓 정합 가능성도 있어 이 그래프는 지도 빌드에 쓰지 않음(rrd 만 남김). 639.6s(96샘플)·725.6s(101샘플) 창은 자세가 10초간 20m/s 로 미끄러짐 = SLAM 붕괴 또는 거짓 루프클로저, 최종 /path/pgo 에도 9.6m 계단과 2m+ 간격 113개.
  ⇒ 이 데이터셋에서 신뢰할 수 있는 구간은 0~600s(log0-9). 지도는 로봇 최종 PGO 자세(robot_graph.py, 시각 복원 ±0.1s)로 빌드 중(web_103657_l00_09_robotpgo). 로봇 저장 지도(.pcd/.txt)로 교차 확인 예정.
