# 노트북용 빌드 이미지: 기하 전용 보행용 맵 + semantic 맵 (3D scene graph 제외).
# 서버 이미지(offline-3d-semantic-map, 43 GB)에서 ConceptGraphs · YOLO · SAM · CLIP · VLM 을 뺀 구성.
# GPU: NVIDIA 드라이버 570 이상 + NVIDIA Container Toolkit (torch 2.7.1 / CUDA 12.8, Blackwell RTX 50 지원).
#
# 빌드 (저장소 루트에서):   docker build -f docker/laptop/Dockerfile -t offline-map-laptop .
# 실행 예:                  docker/laptop/run.sh <session.yaml>   (사용법은 docker/laptop/README.md)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
COPY requirements-geometry.txt requirements-semantic.txt /tmp/req/
RUN pip install --no-cache-dir -r /tmp/req/requirements-semantic.txt
# Open3D 헤드리스(EGL) 런타임 + PotreeConverter 런타임(TBB)
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libx11-6 libxext6 libxrender1 libsm6 libtbb12 \
    && rm -rf /var/lib/apt/lists/*

# 모델 캐시는 이미지에 넣지 않고 볼륨으로 (처음 실행 때 SegFormer 를 내려받아 재사용)
ENV HF_HOME=/models/hf_cache \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
WORKDIR /repo
