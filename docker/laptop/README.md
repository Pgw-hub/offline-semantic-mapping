# 노트북에서 맵 빌드하기

mcap 을 서버로 옮기지 않고, 데이터를 받은 노트북에서 바로 맵을 만든다.
노트북 이미지는 **기하 전용 보행용 맵**과 **semantic 맵**까지 만든다.
3D scene graph(ConceptGraphs 객체 검출)는 GPU 메모리 ~10 GB 가 필요해 서버에서만 돌린다.
노트북의 Claude Code 세션에 작업을 맡길 때는 [docs/LAPTOP_CLAUDE_GUIDE.md](../../docs/LAPTOP_CLAUDE_GUIDE.md) 를 읽게 한다.

| 빌드 종류 | 세션 옵션 | GPU | 산출물 |
|---|---|---|---|
| 보행용 맵만 | `geometry_only: true` | 필요 없음 | locomotion map (voxel3d 타일), Potree |
| semantic 맵까지 | `objects: false` | NVIDIA, 8 GB 이상 | 위 + semantic 복셀 · 0.2 m 점유 맵 · 코스트맵 |
| 전체 | (기본) | 서버 | 위 + 3D scene graph (객체 · 엣지) |

## 준비 (한 번)

1. Ubuntu + NVIDIA 드라이버 570 이상, Docker, NVIDIA Container Toolkit.
   확인: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`
2. 저장소 clone 후 이미지 빌드 (약 15 GB, 처음 한 번 torch 를 내려받는다):

   ```
   git clone <repo> offline_semantic_mapping && cd offline_semantic_mapping
   docker build -f docker/laptop/Dockerfile -t offline-map-laptop .
   ```
3. 상주 컨테이너를 만든다 (이름 `offline-map`, 빌드가 끝나도 지워지지 않고 재부팅 후에도 다시 뜬다):

   ```
   docker/laptop/container.sh up
   ```

   마운트는 만들 때 고정된다: `$HOME` 전체 읽기 전용(mcap 녹화가 홈 아래 어디 있든 보인다), 저장소 읽기·쓰기, 모델 캐시(`$HF_CACHE`, 기본 `~/.cache/offline_map_hf`) → `/models/hf_cache`.
   이미지를 다시 빌드했으면 `container.sh rm && container.sh up` 으로 컨테이너도 다시 만든다 (기존 컨테이너는 옛 이미지를 계속 쓴다).
   홈 밖(`/data2`, 외장 디스크 등)의 녹화를 쓰려면 `EXTRA_RO="/data2:/mnt/usb" docker/laptop/container.sh up` 처럼 만든다 (이미 있으면 `container.sh rm` 후).

## 빌드

1. 세션 yaml 을 만든다. `sessions/session_TEMPLATE_geometry_only.yaml` 을 복사해 `mcap_dir`, `logs`, `root` 를 고친다.
   semantic 맵까지 만들려면 `geometry_only` 를 지우고 `options` 에 `objects: false` 와 `classdict` 를 넣는다.
2. 실행:

   ```
   docker/laptop/run.sh sessions/session_내세션.yaml
   ```

   상주 컨테이너 `offline-map` 안에서 `docker exec` 로 돈다 (없으면 자동으로 만든다). 저장소 · 홈 · 출력 폴더가 호스트와 같은 경로라 yaml 의 경로를 그대로 쓴다.
   `mcap_dir` 는 컨테이너에 마운트된 경로여야 한다 (`$HOME` 아래, 또는 `EXTRA_RO` 로 추가한 경로). 아니면 run.sh 가 바로 멈추고 알려 준다.
   처음 semantic 빌드 때 SegFormer(약 0.5 GB)를 `~/.cache/offline_map_hf` 로 내려받는다 (`HF_CACHE` 로 변경 가능).
   컨테이너는 root 로 돌지만 파이프라인이 끝날 때 `output/` 산출물을 호스트 사용자 소유로 돌린다 (`PYTHONDONTWRITEBYTECODE=1` 이라 `scripts/` 에 root 소유 `__pycache__` 도 안 생긴다).
3. 결과는 `<root>/compiled/` (기본 `output/<root>/compiled/`). 서버 포털로 보내려면 이 폴더째 복사한다.

## 컨테이너 안에서 작업

```
docker/laptop/container.sh shell    # 컨테이너 안 bash, 저장소 루트에서 시작 (GPU · torch · open3d 사용 가능)
docker/laptop/container.sh status   # 떠 있는지
docker/laptop/container.sh stop     # 멈춤 (컨테이너는 남는다)
docker/laptop/container.sh rm       # 지움 (이미지 offline-map-laptop 은 남는다). 마운트를 바꿀 때 rm 후 up
```

안에서 직접 빌드하려면 `cd sessions && python ../scripts/build_map_package.py <session.yaml>` (run.sh 가 하는 일과 같다).
컨테이너 안에서 `pip install` 등으로 바꾼 것은 컨테이너에 남지만 이미지에는 남지 않는다. 이미지에 넣으려면 Dockerfile 을 고쳐 다시 빌드한다.
`docker` 를 sudo 없이 쓰려면 사용자가 `docker` 그룹에 있어야 한다 (추가 직후에는 새 터미널 또는 `newgrp docker`).

## 맵 보기 (포털 :9092 + rerun 웹뷰어 :9091 + Potree)

이미지에는 PotreeConverter 2.1.1(서버와 같은 커밋 d9387d5)이 들어 있어 노트북 빌드도 `compiled/potree/` 를 만든다. Potree 웹 뷰어 정적 파일은 git 에 없으니 한 번만 받는다:

```
tools/potree/fetch_viewer.sh        # Potree 1.8.2 웹 뷰어 (65 MB) → tools/potree/viewer/
```

포털은 서버와 같은 `tools/start_portal.sh` 로 띄운다. 두 방법 중 하나:

```
# (a) 컨테이너 안에서 — 이미지에 rerun-sdk 0.16.1 뷰어가 있어 venv 가 필요 없다 (권장)
docker exec -w /home/raion-seoul/offline-semantic-mapping offline-map bash tools/start_portal.sh   # 또는 container.sh shell 안에서 tools/start_portal.sh
# (b) 호스트에서 — ~/.venvs/rerun 에 venv 를 만들어 둔다
python3 -m venv ~/.venvs/rerun && ~/.venvs/rerun/bin/pip install -r requirements-geometry.txt
tools/start_portal.sh
```

- 브라우저: http://localhost:9092 (포털) · http://localhost:9091 (rerun). (a) 는 컨테이너가 `--network host` 로 만들어졌을 때(기본) localhost 로 보인다.
  bridge 네트워크로 만든 옛 컨테이너면 `docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' offline-map` 의 IP 로 열거나 `container.sh rm && container.sh up` 으로 다시 만든다.
- 포털은 `output/*/compiled/map_manifest.json` 을 스캔해 세션 카드를 보여 준다. 카드에서 **Potree**(보행용 맵 · 원본 스캔 점), **3D prior (locomotion)** · **Map in rerun** 을 연다. rerun 링크는 같은 포털에서 `.rrd` 를 받아 연다.
- 포털의 "빌드" 버튼은 서버 컨테이너(`offline_3d_semantic_map`)를 부르므로 노트북에서는 쓰지 않는다. 빌드는 `docker/laptop/run.sh` 로 한다.
- 로그: `/tmp/map_portal.log`, `/tmp/rerun_viewer.log` (컨테이너 안에서 띄웠으면 컨테이너 안). 컨테이너를 재시작하면 포털도 내려가니 다시 `start_portal.sh` 를 실행한다.
- 이미지 안 PotreeConverter 로 만든 Potree 레이어: 점 수 · 범위가 서버 변환기 결과와 동일 (voxel3d 670,177 · raw 15,641,885).

## 이미지 없이 (보행용 맵만)

GPU 가 없거나 Docker 를 쓰지 않을 때:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-geometry.txt
.venv/bin/python scripts/build_map_package.py sessions/session_내세션.yaml   # geometry_only: true
```

Potree 는 `tools/potree/PotreeConverter/build/PotreeConverter` (2.1.1) 또는 환경변수 `POTREE_CONVERTER` 의 변환기가 있어야 만들어진다. 없으면 그 단계만 건너뛴다.

## 검증 기록 (2026-10-21, beolzip_grating_stair)

- 깨끗한 clone + venv 로 기하 전용 빌드 → 서버 컨테이너 결과와 1,623 파일 동일.
- 이 이미지로 `objects: false` 빌드 → 서버 컨테이너 결과와 동일 (SegFormer 출력 3,338 파일 포함, Potree 는 옥트리 순서만 다름). 470 s.
- 이미지 안 PotreeConverter 로 만든 Potree 레이어: 점 수 · 범위가 서버 변환기 결과와 동일 (voxel3d 670,177 · raw 15,641,885).
- GPU 메모리: SegFormer 단계 batch 8 약 7.0 GB, batch 2 약 2.4 GB. 노트북에서 메모리가 부족하면 `options` 에 `seg_batch: 2` (또는 4).
  batch 를 바꾸면 fp16 연산 경로가 달라져 세그 픽셀 약 1.4 % 가 달라지고, 보행용 맵 셀 0.009 % · 코스트맵 0.006 % 가 달라진다 (서버와 완전히 같게 하려면 기본 8).
