#!/usr/bin/env python3
"""클래스 사전 — 세그멘테이션 모델의 라벨을 어떤 클래스로 묶고, 각 클래스가 주행에서 어떤 역할인지 정의.

사전 JSON: {"name", "description", "model": MODELS 키, "classes": [{"name", "ade": [모델 라벨 부분문자열…], "color": "#rrggbb", "role", "cost"}]}
  model: 사전이 어떤 세그 모델(= 라벨 공간)에 대한 것인지. 사전에 묶여 있어 모델·라벨 조합이 깨지지 않는다. 없으면 segformer_b4_ade (구 사전 호환).
  "ade" 키 이름은 역사적 이유로 유지 — 실제로는 model 의 라벨 공간(ADE20K 150 / Mapillary Vistas 65)에서 부분문자열 매칭한다.
역할(role):
  ground_preferred  선호 지면 (산책로·보도)          코스트맵 지면값 = cost (기본 10)
  ground_soft       통행 가능하지만 비싼 지면 (잔디)  지면값 = cost (기본 90); 지면 위 밴드에서 장애물로 세지 않음
  obstacle          정적 장애물 (나무·건물·차)        밴드에 있으면 높이 무관 lethal; 지면이면 150
  dynamic           동적 (사람)                       occupancy free 안이면 제거, 장애물로 세지 않음
  context           문맥용 (벽·바닥·하늘 등)          지면 150, 밴드는 높이 규칙(>0.5m lethal, 이하 200)
파일 위치: <repo>/classdicts/<name>.json. 'default' 는 이 모듈에 내장(초기 4클래스와 수치 동일).
"""
import json, os

REPO = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(REPO, "classdicts")
ROLES = ["ground_preferred", "ground_soft", "obstacle", "dynamic", "context"]
ROLE_GROUND_COST = {"ground_preferred": 10, "ground_soft": 90, "obstacle": 150, "dynamic": 150, "context": 150}
# 세그 모델 레지스트리: 사전의 "model" 키 → precompute_semantic 이 올릴 모델. labels = classdicts/<labels>_labels.json (편집기 검색용 라벨 목록)
MODELS = {
    "segformer_b4_ade": {"title": "SegFormer-B4 · ADE20K (2021, 150 라벨)", "hub_id": "nvidia/segformer-b4-finetuned-ade-512-512", "backend": "segformer", "labels": "ade20k",
                         "note": "실내·실외 일반 장면. road/sidewalk 는 있으나 횡단보도 없음. 가볍고 빠름(GPU 0.03 s/프레임)."},
    "m2f_swinl_mapillary": {"title": "Mask2Former Swin-L · Mapillary Vistas (2021, 65 라벨)", "hub_id": "facebook/mask2former-swin-large-mapillary-vistas-semantic", "backend": "mask2former", "labels": "mapillary_vistas",
                            "note": "도로 장면 전용. Sidewalk / Crosswalk / Curb / Road / Bike Lane 분리. 비상업 라이선스(제품화 전 확인)."},
    "m2f_swinl_mapillary_pan": {"title": "Mask2Former Swin-L · Mapillary Vistas panoptic (2021, 65 라벨)", "hub_id": "facebook/mask2former-swin-large-mapillary-vistas-panoptic", "backend": "mask2former", "labels": "mapillary_vistas",
                                "note": "semantic 체크포인트와 같은 라벨, panoptic 학습본을 semantic 후처리로 사용 (비교용)."},
    "segformer_sim_ft": {"title": "SegFormer-B2 · 시뮬 GT 미세조정 (2026-10, 4 라벨)", "hub_id": os.path.join(REPO, "models", "segformer_sim_ft"), "backend": "segformer", "labels": "sim_gt4",
                         "note": "gt/robotplus_outdoor_testbed/train_segformer.py 로 시뮬 2D GT 학습. 로컬 체크포인트 models/segformer_sim_ft (git 제외)."},
    "eomt_l_ade": {"title": "EoMT-L · ADE20K (2025, 150 라벨)", "hub_id": "tue-mps/ade20k_semantic_eomt_large_512", "backend": "eomt", "labels": "ade20k",
                   "note": "SegFormer 와 같은 라벨, 경계가 깔끔하고 아스팔트 포장 인도 오판이 적음. 3배 느림(GPU 0.09 s/프레임)."},
}
DEFAULT_MODEL = "segformer_b4_ade"
DEFAULT = {"name": "default", "description": "초기 4클래스: grass / path / tree / person", "model": DEFAULT_MODEL,
           "classes": [{"name": "grass", "ade": ["grass"], "color": "#5faa5f", "role": "ground_soft", "cost": 90},
                       {"name": "path", "ade": ["path", "sidewalk", "road", "earth", "dirt", "field", "runway"], "color": "#c8a578", "role": "ground_preferred", "cost": 10},
                       {"name": "tree", "ade": ["tree", "plant", "palm"], "color": "#286e3c", "role": "obstacle", "cost": 254},
                       {"name": "person", "ade": ["person"], "color": "#dc3c3c", "role": "dynamic", "cost": 254}]}
OTHER_COLOR = "#7887a0"; UNKNOWN_COLOR = "#6e6e6e"


def validate(cd):
    if not isinstance(cd, dict) or not cd.get("classes"): raise ValueError("classes 가 비어 있음")
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}", str(cd.get("name", ""))): raise ValueError("사전 이름은 영문/숫자/_/- 1~41자")
    cd["model"] = cd.get("model") or DEFAULT_MODEL                      # 구 사전(model 없음) = SegFormer ADE
    if cd["model"] not in MODELS: raise ValueError(f"model 은 {list(MODELS)} 중 하나 (받은 값: {cd['model']})")
    names = set()
    for c in cd["classes"]:
        n = str(c.get("name", "")).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", n): raise ValueError(f"클래스 이름 '{n}' 은 소문자/숫자/_ 로 (예: grass)")
        if n in names or n in ("other", "unknown"): raise ValueError(f"클래스 이름 중복/예약어: {n}")
        names.add(n)
        if c.get("role") not in ROLES: raise ValueError(f"{n}: role 은 {ROLES} 중 하나")
        if not c.get("ade"): raise ValueError(f"{n}: ADE 라벨이 하나도 없음")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(c.get("color", ""))): raise ValueError(f"{n}: color 는 #rrggbb")
        c["cost"] = int(c.get("cost", ROLE_GROUND_COST[c["role"]]))
        if c.get("overrides") is not None and not isinstance(c["overrides"], list): raise ValueError(f"{n}: overrides 는 클래스 이름 목록")
        if c.get("overrides"): c["override_min"] = float(c.get("override_min", 0.3))   # 이 확률 이상이면 덮음
    for c in cd["classes"]:                                              # overrides: 이 클래스 확률 ≥0.3 인 픽셀에서 지정 클래스 확률을 0 으로 (예: crosswalk 가 road 를 덮음 — 노면 표시는 바탕면 위에 있다)
        for o in c.get("overrides") or []:
            if o not in names or o == c["name"]: raise ValueError(f"{c['name']}: overrides 에 모르는 클래스 {o}")
    if len(cd["classes"]) > 14: raise ValueError("클래스는 14개 이하 (코스트맵 4비트 라벨 패킹: 클래스 + other + unknown ≤ 16)")
    return cd


def model_info(cd):
    """사전의 모델 레지스트리 항목 (hub_id, backend, labels …)"""
    return MODELS[cd.get("model") or DEFAULT_MODEL]


def labels_for(space):
    """라벨 공간 이름('ade20k', 'mapillary_vistas') → [{"id", "label"}] (classdicts/<space>_labels.json, 없으면 [])"""
    p = os.path.join(DIR, f"{space}_labels.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []


def load(name_or_path=None):
    """이름('default', 'outdoor_v2') 또는 경로 → 사전 dict"""
    if not name_or_path or name_or_path == "default": return json.loads(json.dumps(DEFAULT))
    p = name_or_path if os.path.sep in name_or_path or name_or_path.endswith(".json") else os.path.join(DIR, f"{name_or_path}.json")
    return validate(json.load(open(p, encoding="utf-8")))


def list_all():
    out = [json.loads(json.dumps(DEFAULT))]
    if os.path.isdir(DIR):
        for f in sorted(os.listdir(DIR)):
            if f.endswith(".json") and not f.endswith("_labels.json"):
                try: out.append(load(os.path.join(DIR, f)))
                except Exception as e: out.append({"name": f[:-5], "error": str(e), "classes": []})
    return out


def save(cd):
    cd = validate(cd)
    if cd["name"] == "default": raise ValueError("default 는 수정할 수 없음 (복제해서 새 이름으로)")
    os.makedirs(DIR, exist_ok=True); p = os.path.join(DIR, f"{cd['name']}.json")
    json.dump(cd, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1); return p


def from_voxsem(npz):
    """복셀 npz 에서 사전 복원: classdict 키가 있으면 그것, 없으면(구 형식) classes 이름으로 default 에서 찾아 구성"""
    if "classdict" in npz.files:
        return json.loads(str(npz["classdict"]))
    names = [str(c) for c in npz["classes"]]; by = {c["name"]: c for c in DEFAULT["classes"]}
    return {"name": "legacy", "description": "npz classes 이름에서 복원", "model": DEFAULT_MODEL, "classes": [by.get(n, {"name": n, "ade": [n], "color": "#9aa0a6", "role": "context", "cost": 150}) for n in names]}


def names(cd): return [c["name"] for c in cd["classes"]]
def hex2rgb(h): h = h.lstrip("#"); return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
def colors_rgb(cd): return [hex2rgb(c["color"]) for c in cd["classes"]]
def ids_with_role(cd, *roles): return [i for i, c in enumerate(cd["classes"]) if c["role"] in roles]
def ade_ids(cd, id2label):
    """각 클래스의 모델 라벨 id 목록 (부분문자열 매칭, 소문자) — 이름은 역사적, 모델의 id2label 이면 무엇이든 됨"""
    out = []
    for c in cd["classes"]:
        subs = [s.lower() for s in c["ade"]]
        out.append([int(i) for i, l in id2label.items() if label_match(l, subs)])
    return out


def label_match(label, subs):
    """라벨 패턴: 'curb' = 부분문자열 / '=curb' = 정확히 일치 (Mapillary 'Curb' vs 'Curb Cut' 분리용)"""
    l = label.lower().strip()
    return any((l == s[1:].strip()) if s.startswith("=") else (s in l) for s in subs)
