"""파이프라인 공통 상수: 코드(scripts/)·저장소 경로 · 파이썬 · 외부 의존(ConceptGraphs · 모델 캐시) · 클래스 사전."""
import os, sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # scripts/ (우리 코드)
REPO = os.path.dirname(CODE)                                                         # 저장소 루트 (output/ · sessions/ · tools/)
if CODE not in sys.path: sys.path.insert(0, CODE)
from common import classdict as CD                                                    # 사전·세그 모델 레지스트리 (step B 가드)
CGREPO = os.environ.get("CG_REPO", "/data2/cg/concept-graphs") + "/conceptgraph"   # ConceptGraphs 설치 위치 (컨테이너에 editable 설치됨)
HF_HOME = os.environ.get("HF_HOME", "/data2/cg/hf_cache")                        # 모델 캐시 (수십 GB, 데이터 디스크에 둠)
PY = sys.executable
