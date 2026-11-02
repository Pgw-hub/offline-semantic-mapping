# 노트북 맵 빌드 작업 안내 (Claude Code 세션용)

이 문서는 **노트북에서 실행 중인 Claude Code 세션**에게 주는 작업 안내다.
사용자가 "이 문서 읽고 진행해줘" 라고 하면 아래 순서대로 진행한다.

## 0. 배경 — 왜 노트북에서 빌드하나

- 이 저장소는 로봇 `.mcap` 녹화로 오프라인 맵 패키지를 만든다. 구조는 [README.md](../README.md) 에 있다.
- 원래는 GPU 서버(`blackwell-01`)의 컨테이너에서 빌드한다. 그런데 mcap(세션당 수 GB~수십 GB)을 네트워크로 서버에 옮기는 데 시간이 너무 오래 걸린다.
  그래서 **데이터를 받은 노트북에서 바로 빌드**하고, 결과 폴더(`compiled/`, 1 GB 미만)만 서버로 보낸다.
- 노트북 사양: Ubuntu 24.04, RTX 5070 Laptop **8 GB**, NVIDIA 드라이버 595 (CUDA 13.2).
- 노트북에서 만드는 것은 **두 종류뿐이다** (사용자 결정):

  | 종류 | 세션 옵션 | 산출물 |
  |---|---|---|
  | 기하 전용 (보행용 맵만) | `geometry_only: true` | locomotion map (`session_voxel3d_tiles/`) |
  | semantic 맵까지 | `objects: false` | 위 + semantic 복셀 PCD · 0.2 m 점유 · 코스트맵 |

  **3D scene graph(ConceptGraphs, 단계 D·E)는 GPU 메모리가 약 10 GB 필요해서 노트북에서 돌리지 않는다.** 세션 yaml 에 `objects: false` 나 `geometry_only: true` 가 없으면 D·E 를 돌리려 하니 반드시 둘 중 하나를 넣는다.

## 1. 지켜야 할 규칙

- 사용자에게는 **한국어로** 답한다.
- **보행용 맵 기본값을 바꾸지 않는다.** 확률 격자 0.025 m, LiDAR 점과 빈 공간 레이 모두 센서 5 m 이내(`max_range 5`, `free_range 5`)는 사용자가 정한 값이고, 코드 기본값이다.
  세션 yaml 에 `voxel3d:` 를 적을 때도 `prob_res: 0.025, max_range: 5.0, free_range: 5.0` 을 유지한다.
- 코드(`scripts/`)는 고치지 않는다. 빌드가 코드 문제로 실패하면 고치기 전에 원인과 로그를 사용자에게 보고하고 지시를 받는다. 서버와 같은 코드여야 결과가 같다.
- mcap 원본 폴더에는 아무것도 쓰지 않는다 (`run.sh` 도 읽기 전용으로 마운트한다).
- 오래 걸리는 빌드는 백그라운드로 돌리고, 진행 상황은 `<root>/logs/build.log` 로 확인한다.

## 2. 준비 (처음 한 번)

### 2.1 확인

```bash
nvidia-smi                                   # GPU · 드라이버 570 이상
docker --version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi   # 컨테이너에서 GPU 보이는지
```

마지막 명령이 실패하면 NVIDIA Container Toolkit 이 없는 것이다. 설치는 sudo 가 필요하니 사용자에게 진행 여부를 묻는다
(공식 안내: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html, 설치 후 `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
`docker` 가 sudo 없이 안 되면 사용자를 `docker` 그룹에 넣어야 한다 (역시 사용자에게 묻는다).

### 2.2 저장소와 이미지

```bash
git clone https://github.com/Pgw-hub/offline-semantic-mapping.git   # 이미 있으면 git pull
cd offline-semantic-mapping
docker build -f docker/laptop/Dockerfile -t offline-map-laptop .     # 약 15 GB, 10–20 분
docker run --rm --gpus all offline-map-laptop python -c "import torch, open3d; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), open3d.__version__)"
# 기대값: True NVIDIA GeForce RTX 5070 Laptop GPU 0.19.0
docker run --rm offline-map-laptop bash -c 'ls -l $POTREE_CONVERTER'      # Potree 변환기 (이미지 안에서 빌드됨)
```

이미지는 서버에서 받는 것이 아니라 여기서 직접 빌드한다 (레지스트리에 없다). Potree 변환기(PotreeConverter 2.1.1)도 이미지 빌드 때 같이 컴파일된다.

### 2.3 결과를 노트북에서 보기 위한 준비 (포털 · Potree 뷰어)

```bash
tools/potree/fetch_viewer.sh                                            # Potree 1.8.2 웹 뷰어 파일 (65 MB, git 에 없음)
docker/laptop/container.sh up                                           # 상주 컨테이너 offline-map (4 참고). 이미지를 다시 빌드했으면 rm 후 up
```

포털 · rerun 뷰어는 컨테이너 안에서 띄우면 venv 가 필요 없다 (이미지에 rerun-sdk 0.16.1 포함, 6.1 참고).
호스트에서 띄우려면 `python3 -m venv ~/.venvs/rerun && ~/.venvs/rerun/bin/pip install -r requirements-geometry.txt` (`tools/start_portal.sh` 가 이 경로를 먼저 찾고, 없으면 PATH 의 python · rerun 을 쓴다).

## 3. 세션 yaml 만들기

빌드 설정은 세션 yaml 하나다. 사용자에게 다음을 확인한다: **mcap 폴더 경로**, **쓸 로그 번호**, **결과 이름(root)**, **빌드 종류**(기하 전용 / semantic 맵까지).

mcap 폴더는 레코더 v2 형식이어야 한다:

```
<mcap_dir>/
├── log_000.mcap, log_001.mcap, ...      ← logs: [0, 1, ...] 가 이 번호
├── config_snapshot/raisin_root/calibration/   ← LiDAR · 카메라 외부 파라미터 (자동으로 찾음)
└── log_manifest.json ...
```

`ls <mcap_dir>` 로 로그 파일과 `config_snapshot/raisin_root/calibration` 이 있는지 먼저 본다. calibration 이 없으면 빌드할 수 없으니 사용자에게 알린다.

### 3.1 기하 전용 (보행용 맵만)

`sessions/session_TEMPLATE_geometry_only.yaml` 을 복사해서 `root`, `mcap_dir`, `logs`, `map_id` 를 채운다. 나머지는 그대로 둔다.

```yaml
root: <결과 이름>                      # 이름만 쓰면 output/<결과 이름>/ 에 생긴다
mcap_dir: /절대/경로/<mcap 폴더>
logs: [0, 1, 2]
options:
  map_id: "<설명> · geometry only"
  geometry_only: true
  robot_map:
    geometry: all_scans
  lidar_extrinsic: calib
  livox_flip: false                    # 아래 3.3 참고
  aux_lidar: "off"
  prob_hit: 0.85
  prob_miss: 0.45
  voxel3d: {res_xy: 0.025, res_z: 0.025, prob_res: 0.025, max_range: 5.0, free_range: 5.0, tile_m: 10.0, fill_flat: 0.0}
  stair_complete: true                 # 계단 디딤판 채움 + 앞면 판. 아래 3.4 참고
```

### 3.2 semantic 맵까지

서버에서 검증한 핸드헬드 세션(`sessions/session_t3_handheld_1016_beolzip_grating_stair.yaml`)을 바탕으로 한다. `objects: false` 가 핵심이다.

```yaml
root: <결과 이름>
mcap_dir: /절대/경로/<mcap 폴더>
logs: [0, 1, 2]
options:
  map_id: "<설명> · SegFormer indoor_v1 · semantic (no scene graph)"
  objects: false                       # 3D scene graph(D·E) 생략 — 노트북 필수
  robot_map:
    geometry: all_scans
  lidar_extrinsic: calib
  livox_flip: false
  lidar_fusion: false
  mask_bottom_px: 0                    # 핸드헬드 리그는 0, 로봇 몸체가 화면 아래에 보이면 60
  drop_classes: person
  vlm_edges: false
  occupancy_res: 0.2
  prob_hit: 0.85
  prob_miss: 0.45
  costmap: {h_ground: 2, h_obs: 3, clearance: 1.4, band_min: 0.3}
  view_cubes: false
  view_coarse: 0.3
  video_speed: 15
  video_pipeline: false                # 3분할 영상은 scene graph 검출 화면이 필요해서 objects: false 면 어차피 생략
  classdict: indoor_v1                 # 실내. 실외 도로는 사용자에게 물을 것 (classdicts/)
  label_edge_px: 4
  require_depth: 1
  pose_graph: true
  depth_fill_range: 4.0
  depth_fill_gap: 0.15
  aux_lidar: "off"
  stair_complete: true                 # 계단 녹화면 (3.4)
  # seg_batch: 2                       # GPU 메모리 부족(OOM) 때만. 아래 5 참고
```

### 3.3 확인할 옵션

- `livox_flip`: 핸드헬드 리그와 레코더 v2 실기체 세션은 `false` 로 빌드해 왔다. 템플릿 기본값은 `true`(옛 실기체 규약)이니 어떤 리그인지 모르면 사용자에게 묻는다.
- `classdict`: 세그멘테이션 클래스 사전. 실내는 `indoor_v1`. 다른 값을 쓰려면 `classdicts/` 에 있는 이름이어야 한다.
- 녹화 앞부분의 센서 초기화 구간, 저장 지도 위치추정으로 좌표계가 바뀌는 구간은 스캔 내보내기 단계가 자동으로 뺀다.

### 3.4 계단 디딤판 채움 (`stair_complete`)

그레이팅 계단은 LiDAR 빔이 망 사이로 빠져 디딤판이 듬성듬성 잡힌다. 그대로 쓰면 prior map controller 가 디딤판 뒤쪽 구멍으로 계단 밑바닥을 읽어 발이 단차 틈에 빠진다.
`stair_complete: true` 면 F 단계 끝에서 `scripts/locomotion_map/voxel3d_stair_complete.py` 가 본 보행 맵에서 계단을 스스로 찾아

- 계단 전체에 높이 · 앞모서리 직선을 맞추고(전역 직선 모델), 각 디딤판 사각형 안의 빈 칸만 그 디딤판 높이로 채우고,
- 디딤판 앞모서리 아래에 앞면 판(수직 셀, `risers`)을 넣어 뚫린 단차를 막힌 계단처럼 만든다.

산출물은 본 맵과 **별도 폴더** `compiled/session_voxel3d_stair_tiles/` 이고(본 맵 `session_voxel3d_tiles/` 는 그대로), 계단 모델은 `session_voxel3d_stair_tiles_stair.json`, 확인 그림은 `voxel3d_stair_check.png`.

- 맵 전체에서 가장 긴 계단 하나만 잡는다. ROI 없이 둔곡 그레이팅 계단 핸드헬드 맵 두 개에서 맞게 찾았다.
  계단이 여럿이거나 엉뚱한 것을 잡으면 `stair_complete: {roi: [x0, x1, y0, y1]}` (맵 좌표 m) 로 좁힌다. 계단 결과만 지우고(`rm -r compiled/session_voxel3d_stair_tiles compiled/session_voxel3d_stair_tiles_stair.json compiled/voxel3d_stair_check.png`) `--steps F,G` 로 돌리면 계단 단계만 다시 돈다 (`--force F` 는 보행 맵 전체를 다시 만든다).
- 계단을 못 찾으면 build.log 에 `계단 채움 실패(빌드는 계속…)` 한 줄이 남고 나머지 빌드는 그대로 진행된다. 이때는 `logs/F_voxel3d_stair.log` 를 사용자에게 보여 준다.
- 둔곡 그레이팅 계단 기준값(10/16 녹화): 디딤판 9–10 개, 단 높이 약 20–20.5 cm, 단 너비 약 24 cm, 폭 약 85 cm. 크게 다르면 사용자에게 보고한다.

## 4. 빌드 실행

```bash
cd offline-semantic-mapping
docker/laptop/run.sh sessions/<세션>.yaml --dry-run      # 먼저 실행할 명령만 확인
docker/laptop/run.sh sessions/<세션>.yaml                # 실제 빌드
```

- `run.sh` 는 상주 컨테이너 `offline-map`(`docker/laptop/container.sh up` 으로 만든다, 없으면 run.sh 가 자동으로 만든다) 안에서 `docker exec` 로 빌드한다. 빌드가 끝나도 컨테이너는 지워지지 않는다 (사용자 결정, 2026-10-21).
  컨테이너는 저장소(읽기·쓰기) · `$HOME` 전체(읽기 전용) · 모델 캐시를 호스트와 **같은 경로**로 마운트한다. 그래서 yaml 에는 노트북의 실제 절대 경로를 쓰고, `mcap_dir` 는 `$HOME` 아래에 있어야 한다 (홈 밖이면 `EXTRA_RO=<경로> container.sh rm && up`).
  컨테이너 안에서 직접 작업하려면 `docker/laptop/container.sh shell`.
- 처음 semantic 빌드 때 SegFormer 모델(약 0.5 GB)을 `~/.cache/offline_map_hf` 로 내려받는다 (인터넷 필요, 이후 재사용).
- 일부 단계만 다시: `--steps F,G` / 지우고 다시: `--force F`. 이미 있는 산출물은 건너뛴다.
- 진행 확인: `tail -f output/<결과 이름>/logs/build.log`, 단계별 로그는 같은 폴더의 `<단계>*.log`.

서버 기준 시간 (beolzip 168 s 녹화, 로그 3개): 기하 전용 약 2 분, semantic 맵까지 약 8 분. 노트북은 이보다 느릴 수 있다.
단계 순서: Q 품질 → P 자세 → (A 데이터셋 → S 스캔 → R RGB → B 세그 → C semantic 복셀) → F 점유 + 보행용 맵 → G 패키지 → H 뷰어 → V 영상. 기하 전용은 Q P S F G H 만 돈다.

## 5. 문제 해결

| 증상 | 원인 · 조치 |
|---|---|
| 단계 B 에서 `CUDA out of memory` | 세그멘테이션 batch 8 은 GPU 약 7 GB 를 쓴다. yaml `options` 에 `seg_batch: 4` (또는 2, 약 2.4 GB) 를 넣고 `--steps B --force B` 후 이어서 빌드. 결과는 batch 8 과 세그 픽셀 약 1 % 다르고 맵 차이는 0.01 % 수준이라 써도 된다. 바꿨으면 사용자에게 보고한다. |
| 단계 H 로그에 `Potree 생성 실패(빌드는 계속)` · `PotreeConverter 없음` | 이미지가 변환기를 넣기 전(2026-10-21 이전) 버전. `git pull` 후 2.2 의 `docker build` 를 다시 하고, `docker/laptop/container.sh rm && docker/laptop/container.sh up` 으로 컨테이너를 새 이미지로 다시 만든 뒤 `--steps H --force H` 로 Potree 만 다시 만든다. |
| 포털의 Potree 화면이 비어 있음 | 뷰어 파일이 없음. `tools/potree/fetch_viewer.sh` 실행 (2.3). |
| `mcap_dir 가 없습니다` | yaml 의 `mcap_dir` 가 노트북 경로가 아님. 서버 경로(`/data2/...`)를 그대로 복사하지 않았는지 확인. |
| calibration 관련 파일 없음 | `<mcap_dir>/config_snapshot/raisin_root/calibration` 이 없는 녹화. 사용자에게 알린다 (yaml 에 `calib:` 로 다른 경로를 줄 수는 있다). |
| `docker: permission denied` | 사용자가 docker 그룹에 없음. 사용자에게 묻는다. |
| `컨테이너 offline-map 안에서 mcap_dir 가 안 보입니다` | mcap 폴더가 `$HOME` 밖에 있어 컨테이너에 마운트되지 않음. `EXTRA_RO=<폴더> docker/laptop/container.sh rm && EXTRA_RO=<폴더> docker/laptop/container.sh up` 으로 다시 만든다. |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit 없음 (2.1). |
| git push/pull 이 `vscode-git-*.sock ECONNREFUSED` 로 실패 | VS Code 로그인 연결이 끊긴 것. `GIT_ASKPASS= VSCODE_GIT_IPC_HANDLE= git ...` 로 실행하거나 VS Code 창을 새로 고친다. |

그 밖의 실패는 `output/<결과 이름>/logs/` 의 해당 단계 로그 끝부분을 사용자에게 보여주고 판단을 받는다.

## 6. 끝난 뒤 — 확인하고 서버로 보내기

### 6.1 결과 확인

```bash
ls output/<결과 이름>/compiled/
cat output/<결과 이름>/compiled/session_voxel3d_summary.json   # 셀 수 · 타일 수
```

- 기하 전용: `map_manifest.json`, `session_voxel3d.npz`, `session_voxel3d_tiles/`, `voxel3d_bev.png`, `voxel3d_view.rrd`, `data_quality.json`, `potree/`
- semantic 맵까지: 위 + `robot_map_semantic.pcd`, `session_occupancy.npz`, `session_costmap_occ.npz`, `bev.png`, `map_package.rrd`
- `stair_complete: true` 면 + `session_voxel3d_stair_tiles/`, `session_voxel3d_stair_tiles_stair.json`, `voxel3d_stair_check.png`.
  build.log 의 `계단: 디딤판 N 개 …` 줄과 `voxel3d_stair_check.png` 를 사용자에게 보여 준다. 그림 위 두 칸은 채움 전·후 측면도(초록 = 채운 셀, 점선 = 디딤판 높이), 아래는 디딤판 칸 윗면도(파랑 = 디딤판 높이 셀, 빨강 = 디딤판 사이 높이의 가짜 윗면)다. 디딤판 수가 실제 계단과 다르거나 점선이 계단 단과 어긋나면 보고한다.
- `voxel3d_bev.png`, `bev.png` 를 열어 사용자에게 보여주고(Read 로 이미지 확인 가능), 맵이 비었거나 벽이 두 겹이면 보고한다.

사용자에게 요약해 줄 것: 빌드 종류, 걸린 시간, 스캔 수, voxel3d 셀 수와 타일 수, semantic 이면 라벨 비율, `seg_batch` 를 바꿨는지, 이상한 점.

노트북에서 직접 보기 (2.3 준비가 되어 있어야 함):

```bash
docker exec -w /home/raion-seoul/offline-semantic-mapping offline-map bash tools/start_portal.sh   # 포털 http://localhost:9092 + rerun 웹 뷰어 :9091 (컨테이너 안, --network host)
# 또는 호스트 venv 가 있으면: tools/start_portal.sh
```

포털 맵 카드에서 **Potree**(보행용 맵 · 원본 스캔 점, 확대 · 단면 · 측정), **3D prior (locomotion)** · **Map in rerun** 을 연다.
포털의 빌드 버튼은 서버 컨테이너용이라 노트북에서는 쓰지 않는다 (빌드는 항상 `docker/laptop/run.sh`).

### 6.2 로봇에 넣기 (prior map controller)

포털 맵 화면의 **Download stair tiles** 버튼(계단 채움 맵, `stair_complete` 일 때만 보임) 또는 **Download PCD…** 의 `3D prior 타일 zip` 항목(본 맵)으로 zip 을 받는다.
zip 안은 `voxel3d_tiles/` 폴더 하나다(타일 + `index.json` + 요약, 계단 맵이면 `stair.json` · `stair_check.png` 도). 로봇의 raisin 워크스페이스에서

```bash
unzip <받은 zip> -d <raisin 워크스페이스>/log/map/<SLAM 맵 이름>/     # → log/map/<SLAM 맵 이름>/voxel3d_tiles/index.json
```

`<SLAM 맵 이름>` 은 로봇이 위치추정에 쓰는 lidar_slam 저장 지도 이름(= SLAM frame_id)이다. 컨트롤러는 `prior_map_dir` 가 비어 있으면 이 자리를 읽는다.
이미 `voxel3d_tiles/` 가 있으면 덮기 전에 옛 폴더 이름을 바꿔 둔다. 보행 맵과 SLAM 저장 지도가 **같은 좌표계**여야 하므로, 녹화 때 그 저장 지도로 위치추정했는지(또는 3 의 `robot_map.traj`) 사용자에게 확인한다.

### 6.3 서버로 보내기

서버로는 **`compiled/` 폴더만** 보내면 된다 (`dataset/`, `sem/` 은 수 GB 라 보내지 않는다).
받는 곳은 서버의 `/home/gw.park/offline_semantic_mapping/output/<결과 이름>/compiled/` 이고, 그러면 서버 맵 포털 목록에 나타난다.
전송 방법(scp/rsync, 외장 디스크 등)은 사용자에게 묻는다. 예:

```bash
rsync -av output/<결과 이름>/compiled/ <사용자>@<서버>:/home/gw.park/offline_semantic_mapping/output/<결과 이름>/compiled/
```

`compiled/potree/` 도 같이 가므로 서버 포털에서도 Potree 를 바로 열 수 있다.

## 7. 더 볼 것

- [README.md](../README.md) — 코드 구조와 파이프라인 그림
- [docker/laptop/README.md](../docker/laptop/README.md) — 노트북 이미지 요약과 서버 대비 검증 기록
- [docs/HANDOFF.md](HANDOFF.md) — 전체 이력 · 결정 · 함정 (길다, 필요한 날짜 항목만)
- [docs/VOXEL3D_PRIOR.md](VOXEL3D_PRIOR.md) — 보행용 맵 타일 형식
