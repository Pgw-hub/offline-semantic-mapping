#!/usr/bin/env python3
"""클래스 사전 — SegFormer(ADE20K 150 라벨)를 어떤 클래스로 묶고, 각 클래스가 주행에서 어떤 역할인지 정의.

사전 JSON: {"name", "description", "classes": [{"name", "ade": [ADE 라벨 부분문자열…], "color": "#rrggbb", "role", "cost"}]}
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
DEFAULT = {"name": "default", "description": "초기 4클래스: grass / path / tree / person",
           "classes": [{"name": "grass", "ade": ["grass"], "color": "#5faa5f", "role": "ground_soft", "cost": 90},
                       {"name": "path", "ade": ["path", "sidewalk", "road", "earth", "dirt", "field", "runway"], "color": "#c8a578", "role": "ground_preferred", "cost": 10},
                       {"name": "tree", "ade": ["tree", "plant", "palm"], "color": "#286e3c", "role": "obstacle", "cost": 254},
                       {"name": "person", "ade": ["person"], "color": "#dc3c3c", "role": "dynamic", "cost": 254}]}
OTHER_COLOR = "#7887a0"; UNKNOWN_COLOR = "#6e6e6e"


def validate(cd):
    if not isinstance(cd, dict) or not cd.get("classes"): raise ValueError("classes 가 비어 있음")
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}", str(cd.get("name", ""))): raise ValueError("사전 이름은 영문/숫자/_/- 1~41자")
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
    if len(cd["classes"]) > 32: raise ValueError("클래스는 32개 이하")
    return cd


def load(name_or_path=None):
    """이름('default', 'outdoor_v2') 또는 경로 → 사전 dict"""
    if not name_or_path or name_or_path == "default": return json.loads(json.dumps(DEFAULT))
    p = name_or_path if os.path.sep in name_or_path or name_or_path.endswith(".json") else os.path.join(DIR, f"{name_or_path}.json")
    return validate(json.load(open(p, encoding="utf-8")))


def list_all():
    out = [json.loads(json.dumps(DEFAULT))]
    if os.path.isdir(DIR):
        for f in sorted(os.listdir(DIR)):
            if f.endswith(".json") and f != "ade20k_labels.json":
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
    return {"name": "legacy", "description": "npz classes 이름에서 복원", "classes": [by.get(n, {"name": n, "ade": [n], "color": "#9aa0a6", "role": "context", "cost": 150}) for n in names]}


def names(cd): return [c["name"] for c in cd["classes"]]
def hex2rgb(h): h = h.lstrip("#"); return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
def colors_rgb(cd): return [hex2rgb(c["color"]) for c in cd["classes"]]
def ids_with_role(cd, *roles): return [i for i, c in enumerate(cd["classes"]) if c["role"] in roles]
def ade_ids(cd, id2label):
    """각 클래스의 ADE 라벨 id 목록 (부분문자열 매칭, 소문자)"""
    out = []
    for c in cd["classes"]:
        subs = [s.lower() for s in c["ade"]]
        out.append([int(i) for i, l in id2label.items() if any(s in l.lower() for s in subs)])
    return out
