# 노트북에서 맵 빌드하기

mcap 을 서버로 옮기지 않고, 데이터를 받은 노트북에서 바로 맵을 만든다.
노트북 이미지는 **기하 전용 보행용 맵**과 **semantic 맵**까지 만든다.
3D scene graph(ConceptGraphs 객체 검출)는 GPU 메모리 ~10 GB 가 필요해 서버에서만 돌린다.

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

## 빌드

1. 세션 yaml 을 만든다. `sessions/session_TEMPLATE_geometry_only.yaml` 을 복사해 `mcap_dir`, `logs`, `root` 를 고친다.
   semantic 맵까지 만들려면 `geometry_only` 를 지우고 `options` 에 `objects: false` 와 `classdict` 를 넣는다.
2. 실행:

   ```
   docker/laptop/run.sh sessions/session_내세션.yaml
   ```

   저장소 · mcap · 출력 폴더를 호스트와 같은 경로로 마운트하므로 yaml 의 경로를 그대로 쓴다.
   처음 semantic 빌드 때 SegFormer(약 0.5 GB)를 `~/.cache/offline_map_hf` 로 내려받는다 (`HF_CACHE` 로 변경 가능).
3. 결과는 `<root>/compiled/` (기본 `output/<root>/compiled/`). 서버 포털로 보내려면 이 폴더째 복사한다.

## 이미지 없이 (보행용 맵만)

GPU 가 없거나 Docker 를 쓰지 않을 때:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-geometry.txt
.venv/bin/python scripts/build_map_package.py sessions/session_내세션.yaml   # geometry_only: true
```

Potree 뷰어는 `tools/potree/PotreeConverter/build/PotreeConverter` (2.1.1) 가 있어야 만들어진다. 없으면 그 단계만 건너뛴다.

## 검증 기록 (2026-10-21, beolzip_grating_stair)

- 깨끗한 clone + venv 로 기하 전용 빌드 → 서버 컨테이너 결과와 1,623 파일 동일.
- 이 이미지로 `objects: false` 빌드 → 서버 컨테이너 결과와 동일 (SegFormer 출력 3,338 파일 포함, Potree 는 옥트리 순서만 다름). 470 s.
- GPU 메모리: SegFormer 단계 batch 8 약 7.0 GB, batch 2 약 2.4 GB. 노트북에서 메모리가 부족하면 `options` 에 `seg_batch: 2` (또는 4).
  batch 를 바꾸면 fp16 연산 경로가 달라져 세그 픽셀 약 1.4 % 가 달라지고, 보행용 맵 셀 0.009 % · 코스트맵 0.006 % 가 달라진다 (서버와 완전히 같게 하려면 기본 8).
