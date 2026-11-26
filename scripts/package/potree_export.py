#!/usr/bin/env python3
"""맵 빌드 결과 → Potree 웹 뷰어 (포털의 기본 보기). 컨테이너 offline_3d_semantic_map 안에서 실행.

사용: python scripts/package/potree_export.py <map_root 또는 output 아래 이름> [--layers main,loco,nav,loc] [--raw-max 40000000] [--force]
새 빌드 (compiled/main_map/ 있음, 2026-11-24~) — 점 레이어 (PotreeConverter):
  main   메인 맵 5 cm (main_map_semantic.npz, 기하 전용이면 main_map.pcd). classification = 의미 클래스(id+1, 30 other, 31 없음),
         RGB = 카메라 색, → 패널에서 의미 색 / 카메라 색 / 높이 색, 클래스별 켜고 끄기
  loc    위치추정 지도 0.25 m (localization_map.pcd)
  loco   보행 맵 2.5 cm (locomotion/locomotion.npz). RGB = 출처 (LiDAR · 계단 채움)
  nav    통행성 그래프 노드 (nav3d/nav3d_nodes.csv). RGB = 장소(방마다 색, 계단 빨강, 장소 없음 회색), intensity = 벽까지 여유(cm)
  raw    (선택) 정합된 원시 LiDAR 점
  겹쳐 그리기(three.js 선·Potree 주석, 페이지에 JSON 으로 내장): 토폴로지 장소 라벨·포털 연결선, 3D scene graph 객체 상자·라벨·관계 엣지.
옛 빌드 (session_voxel3d*.npz) — 예전처럼 voxel3d 변형 + raw 레이어, 옛 페이지.
출력: compiled/potree/<레이어>/(metadata.json, octree.bin, hierarchy.bin) + compiled/potree/index.html
뷰어 정적 파일은 tools/potree/viewer/{build,libs} (Potree 1.8.2, 없으면 tools/potree/fetch_viewer.sh), 변환기는 $POTREE_CONVERTER 또는 tools/potree/PotreeConverter/build/PotreeConverter (2.1.1; 노트북 이미지는 /opt/potree/PotreeConverter).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # scripts/ (패키지 import)
import argparse, glob, json, os, shutil, struct, subprocess, sys, tempfile, time
import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/
REPO = os.path.dirname(CODE)                                         # 저장소 루트 (tools/ · output/)
sys.path.insert(0, CODE)
from common import classdict as CD  # noqa: E402

CONVERTER = os.environ.get("POTREE_CONVERTER") or os.path.join(REPO, "tools", "potree", "PotreeConverter", "build", "PotreeConverter")   # 노트북 이미지는 환경변수로
SRC_NAMES = {1: "LiDAR", 2: "depth fill", 3: "flat fill", 5: "stair fill"}                       # 5 = 계단 디딤판·앞면 채움 (voxel3d_stair_complete.py)
SRC_COLORS = {1: [0.35, 0.75, 0.45, 1.0], 2: [0.30, 0.55, 0.95, 1.0], 3: [0.62, 0.62, 0.62, 1.0], 5: [0.95, 0.55, 0.15, 1.0]}

PT2 = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("intensity", "<u2"), ("ret", "u1"), ("cls", "u1"),
                ("angle", "i1"), ("user", "u1"), ("psid", "<u2"), ("r", "<u2"), ("g", "<u2"), ("b", "<u2")])   # LAS 1.2 point format 2, 26 B


def write_las(path, xyz, intensity=None, classification=None, rgb=None, scale=0.001):
    """LAS 1.2 / point format 2 (laspy 없이). rgb 는 0..255 uint8 (N,3)."""
    xyz = np.round(np.asarray(xyz, np.float64) / scale) * scale     # 헤더 경계 상자가 양자화된 점을 감싸게 (반올림으로 최댓값을 넘으면 변환기가 거부)
    n = len(xyz); lo = xyz.min(0); hi = xyz.max(0); off = np.floor(lo)
    rec = np.zeros(n, PT2)
    q = np.round((xyz - off) / scale).astype(np.int64)
    rec["x"], rec["y"], rec["z"] = q[:, 0], q[:, 1], q[:, 2]
    rec["ret"] = 0b00001001                                               # return 1 of 1
    if intensity is not None: rec["intensity"] = intensity
    if classification is not None: rec["cls"] = classification
    if rgb is not None:
        rgb16 = rgb.astype(np.uint16) * 257
        rec["r"], rec["g"], rec["b"] = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    hdr = struct.pack("<4sHH16sBB32s32sHHHIIBHI5I3d3d6d", b"LASF", 0, 0, b"\0" * 16, 1, 2, b"offline_semantic_mapping".ljust(32, b"\0"),
                      b"potree_export.py".ljust(32, b"\0"), int(time.strftime("%j")), int(time.strftime("%Y")), 227, 227, 0, 2, PT2.itemsize,
                      n, n, 0, 0, 0, 0, scale, scale, scale, *off, *(v for a in range(3) for v in (hi[a] + scale / 2, lo[a] - scale / 2)))   # 경계 상자에 반 칸 여유 (복원 좌표의 부동소수 오차)
    assert len(hdr) == 227, len(hdr)
    with open(path, "wb") as f:
        f.write(hdr); rec.tofile(f)


def class_colors(root, classes):
    """클래스 id → RGB(0..255). 세션 사전(voxsem classdict)이 있으면 그 색, 없으면 회색. 254 other / 255 unlabeled."""
    pal = {}
    vs = os.path.join(root, "voxel", "session_voxsem.npz")
    if os.path.exists(vs):
        cd = CD.from_voxsem(np.load(vs, allow_pickle=True))
        pal = {i: CD.hex2rgb(c["color"]) for i, c in enumerate(cd["classes"])}
    pal[254] = CD.hex2rgb(CD.OTHER_COLOR); pal[255] = CD.hex2rgb(CD.UNKNOWN_COLOR)
    lut = np.full((256, 3), 160, np.uint8)
    for k, v in pal.items(): lut[k] = v
    return lut, [str(c) for c in classes]


def convert(las, out_dir, force):
    if os.path.exists(os.path.join(out_dir, "metadata.json")) and not force:
        print(f"  건너뜀 (있음): {out_dir}"); return
    if os.path.isdir(out_dir): shutil.rmtree(out_dir)
    subprocess.run([CONVERTER, las, "-o", out_dir], check=True, stdout=subprocess.DEVNULL)


def layer_voxel3d(root, comp, tmp, force):
    layers = []
    for npz in sorted(glob.glob(os.path.join(comp, "session_voxel3d*.npz"))):
        base = os.path.basename(npz)[:-4]
        if base.endswith("_summary") or base.startswith("eval_"): continue
        name = "voxel3d" if base == "session_voxel3d" else "voxel3d_" + base[len("session_voxel3d_"):]
        out = os.path.join(comp, "potree", name)
        d = np.load(npz, allow_pickle=True)
        I = d["ijk"].astype(np.float64); rx = float(d["res_xy"])
        xyz = np.stack([(I[:, 0] + 0.5) * rx, (I[:, 1] + 0.5) * rx, d["mean_z"].astype(np.float64)], -1)
        cls = d["cls"].astype(np.uint8); lut, classes = class_colors(root, d["classes"])
        print(f"  {name}: {len(xyz):,} 셀 (res_xy {rx}, res_z {float(d['res_z'])})")
        if force or not os.path.exists(os.path.join(out, "metadata.json")):
            las = os.path.join(tmp, name + ".las")
            write_las(las, xyz, intensity=cls.astype(np.uint16), classification=d["source"].astype(np.uint8), rgb=lut[cls])
            convert(las, out, True)
        layers.append({"name": name, "path": f"{name}/metadata.json", "points": int(len(xyz)), "kind": "voxel3d",
                       "res": [rx, float(d["res_z"])], "classes": classes})
    return layers


def layer_raw(comp, root, tmp, force, raw_max):
    files = sorted(glob.glob(os.path.join(root, "dataset", "*", "lidar_scans", "*.npz")))
    if not files: return []
    out = os.path.join(comp, "potree", "raw")
    if not force and os.path.exists(os.path.join(out, "metadata.json")):
        meta = json.load(open(os.path.join(out, "metadata.json"))); n = int(meta.get("points", 0))
        print(f"  raw: 건너뜀 (있음, {n:,} 점)")
        return [{"name": "raw", "path": "raw/metadata.json", "points": n, "kind": "raw"}]
    pts = [np.load(f)["xyz"].astype(np.float32) for f in files]
    total = sum(len(p) for p in pts); keep = min(1.0, raw_max / max(total, 1))
    if keep < 1.0:
        rng = np.random.default_rng(0); pts = [p[rng.random(len(p)) < keep] for p in pts]
    xyz = np.concatenate(pts).astype(np.float64)
    print(f"  raw: 스캔 {len(files)}개 · {total:,} 점" + (f" → {len(xyz):,} 점 추출" if keep < 1.0 else ""))
    las = os.path.join(tmp, "raw.las"); write_las(las, xyz, rgb=np.full((len(xyz), 3), 200, np.uint8)); convert(las, out, True)
    return [{"name": "raw", "path": "raw/metadata.json", "points": int(len(xyz)), "kind": "raw", "scans": len(files)}]


PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__ · Potree</title>
<link rel="stylesheet" href="/tools/potree/viewer/build/potree/potree.css">
<link rel="stylesheet" href="/tools/potree/viewer/libs/jquery-ui/jquery-ui.min.css">
<link rel="stylesheet" href="/tools/potree/viewer/libs/openlayers3/ol.css">
<link rel="stylesheet" href="/tools/potree/viewer/libs/spectrum/spectrum.css">
<link rel="stylesheet" href="/tools/potree/viewer/libs/jstree/themes/mixed/style.css">
<style>
#help{position:absolute;right:12px;top:12px;z-index:10;max-width:340px;background:rgba(20,24,28,.86);color:#e6eaee;font:13px/1.55 system-ui,sans-serif;padding:10px 12px;border-radius:8px}
#help b{color:#9fd3dc}#help .x{float:right;cursor:pointer;color:#9aa5ae}
</style>
</head>
<body>
<script src="/tools/potree/viewer/libs/jquery/jquery-3.1.1.min.js"></script>
<script src="/tools/potree/viewer/libs/spectrum/spectrum.js"></script>
<script src="/tools/potree/viewer/libs/jquery-ui/jquery-ui.min.js"></script>
<script src="/tools/potree/viewer/libs/other/BinaryHeap.js"></script>
<script src="/tools/potree/viewer/libs/tween/tween.min.js"></script>
<script src="/tools/potree/viewer/libs/d3/d3.js"></script>
<script src="/tools/potree/viewer/libs/proj4/proj4.js"></script>
<script src="/tools/potree/viewer/libs/openlayers3/ol.js"></script>
<script src="/tools/potree/viewer/libs/i18next/i18next.js"></script>
<script src="/tools/potree/viewer/libs/jstree/jstree.js"></script>
<script src="/tools/potree/viewer/build/potree/potree.js"></script>
<div class="potree_container" style="position:absolute;width:100%;height:100%;left:0;top:0">
  <div id="potree_render_area"></div>
  <div id="potree_sidebar_container"></div>
</div>
<div id="help"><span class="x" onclick="this.parentNode.remove()">닫기 ✕</span>
<b>__TITLE__</b><br>
더블클릭: 그 점으로 다가가며 회전 중심 지정 · 휠: 확대/축소 · 좌드래그: 회전 · 우드래그: 이동<br>
<b>자르기</b>: 왼쪽 Tools → Clipping 의 상자 아이콘, 상자를 드래그로 이동·크기 조절. Clip task 를 "Show inside" 로<br>
<b>단면</b>: Tools → Measurement 의 Profile(높이 프로파일) 로 계단을 가로질러 두 점 클릭 → 아래 창에 2D 단면<br>
<b>측정</b>: Distance / Height 도구로 단 높이·디딤 깊이<br>
<b>레이어</b>: Scene 목록에서 켜고 끄기 (raw = 정합된 원시 LiDAR 점)<br>
<b>색</b>: 레이어 선택 → Attribute: elevation(높이) · classification(출처: LiDAR/depth/평탄 채움) · rgba(클래스 색)
</div>
<script type="module">
const LAYERS = __LAYERS__;
window.viewer = new Potree.Viewer(document.getElementById("potree_render_area"));
viewer.setEDLEnabled(true);
viewer.setFOV(60);
viewer.setPointBudget(6_000_000);
viewer.setBackground("gradient");
viewer.setClassifications(__CLASSES__);
viewer.loadGUI(() => {
  viewer.setLanguage("en");
  $("#menu_tools").next().show();
  $("#menu_clipping").next().show();
  viewer.toggleSidebar();
});
let first = true;
for (const L of LAYERS) {
  Potree.loadPointCloud(L.path, L.name, e => {
    const pc = e.pointcloud, m = pc.material;
    viewer.scene.addPointCloud(pc);
    m.activeAttributeName = "elevation";
    m.pointSizeType = Potree.PointSizeType.FIXED;
    m.size = L.kind === "raw" ? 1.0 : 2.0;
    m.shape = Potree.PointShape.SQUARE;
    pc.visible = (L.name === "voxel3d");
    if (first) { viewer.fitToScreen(); first = false; }
  });
}
</script>
</body>
</html>
"""


# ─────────────────────────── 새 빌드 (메인 맵 + 보행 맵) ───────────────────────────
OTHER_CODE, NONE_CODE = 30, 31                                   # 의미 클래스 classification: id+1 · other 30 · 라벨 없음 31 (LAS 5 bit 안)
PLACE_PAL = np.array([[31, 119, 180], [174, 199, 232], [44, 160, 44], [152, 223, 138], [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
                      [227, 119, 194], [247, 182, 210], [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]], np.uint8)   # 빨강·주황 없음: 계단(빨강)·계단참(주황) 전용


def sem_class_code(lab):
    lab = lab.astype(np.int64); c = lab + 1
    c[lab == 254] = OTHER_CODE; c[lab == 255] = NONE_CODE
    return c.astype(np.uint8)


def layer_main(root, comp, tmp):
    sem = os.path.join(comp, "main_map", "main_map_semantic.npz"); pcd = os.path.join(comp, "main_map", "main_map.pcd")
    from package.robot_map_package import read_pcd_xyz
    if os.path.exists(sem):
        z = np.load(sem, allow_pickle=True); xyz = z["xyz"].astype(np.float64); cls = sem_class_code(z["label"]); rgb = z["cam_rgb"]
        classes = [str(c) for c in z["classes"]]; cd = json.loads(str(z["classdict"])) if "classdict" in z.files else None
    elif os.path.exists(pcd):
        xyz = read_pcd_xyz(pcd); cls = np.full(len(xyz), NONE_CODE, np.uint8); rgb = np.full((len(xyz), 3), 190, np.uint8); classes = []; cd = None
    else:
        return [], None
    las = os.path.join(tmp, "main.las"); write_las(las, xyz, classification=cls, rgb=rgb); convert(las, os.path.join(comp, "potree", "main"), True)
    u, n = np.unique(cls, return_counts=True)
    print(f"  main: {len(xyz):,} 점 (의미 {'있음' if classes else '없음'})")
    return [{"name": "main", "path": "main/metadata.json", "points": int(len(xyz)), "semantic": bool(classes), "class_counts": {int(k): int(c) for k, c in zip(u, n)}}], (classes, cd)


def layer_loc(comp, tmp):
    p = os.path.join(comp, "main_map", "localization_map.pcd")
    if not os.path.exists(p): return []
    from package.robot_map_package import read_pcd_xyz
    xyz = read_pcd_xyz(p); las = os.path.join(tmp, "loc.las")
    write_las(las, xyz, rgb=np.tile(np.array([[90, 170, 255]], np.uint8), (len(xyz), 1))); convert(las, os.path.join(comp, "potree", "loc"), True)
    print(f"  loc: {len(xyz):,} 점"); return [{"name": "loc", "path": "loc/metadata.json", "points": int(len(xyz))}]


LOCO_SRC = {1: ("LiDAR", [80, 190, 110]), 2: ("depth fill", [80, 140, 240]), 3: ("flat fill", [160, 160, 160]), 4: ("pit fill", [200, 80, 200]), 5: ("stair fill", [245, 140, 40])}


def layer_loco(comp, tmp):
    p = os.path.join(comp, "locomotion", "locomotion.npz")
    if not os.path.exists(p): return []
    d = np.load(p, allow_pickle=True); I = d["ijk"].astype(np.float64); rx = float(d["res_xy"])
    xyz = np.stack([(I[:, 0] + 0.5) * rx, (I[:, 1] + 0.5) * rx, d["mean_z"].astype(np.float64)], -1); src = d["source"].astype(np.uint8)
    lut = np.full((256, 3), 200, np.uint8)
    for k, (_, c) in LOCO_SRC.items(): lut[k] = c
    las = os.path.join(tmp, "loco.las"); write_las(las, xyz, intensity=src.astype(np.uint16), rgb=lut[src]); convert(las, os.path.join(comp, "potree", "loco"), True)
    u, n = np.unique(src, return_counts=True)
    print(f"  loco: {len(xyz):,} 셀 (출처 {dict(zip(u.tolist(), n.tolist()))})")
    return [{"name": "loco", "path": "loco/metadata.json", "points": int(len(xyz)), "res": rx, "sources": {LOCO_SRC.get(int(k), (str(k),))[0]: int(c) for k, c in zip(u, n)},
             "source_colors": {LOCO_SRC[k][0]: LOCO_SRC[k][1] for k in LOCO_SRC if k in u}}]


def layer_nav(comp, tmp):
    p = os.path.join(comp, "nav3d", "nav3d_nodes.csv"); sp = os.path.join(comp, "nav3d", "nav3d_graph_summary.json")
    if not os.path.exists(p): return []
    A = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    if not len(A): return []
    places = {pl["id"]: pl for pl in (json.load(open(sp)).get("places", []) if os.path.exists(sp) else [])}
    pid = A[:, 4].astype(np.int64); rgb = np.full((len(A), 3), 150, np.uint8)
    for i, pl in places.items():
        m = pid == i; rgb[m] = [230, 40, 40] if pl["type"] == "stair" else ([255, 158, 61] if pl["type"] == "landing" else PLACE_PAL[i % len(PLACE_PAL)])
    las = os.path.join(tmp, "nav.las"); write_las(las, A[:, :3], intensity=np.clip(A[:, 3] * 100, 0, 65535).astype(np.uint16), rgb=rgb)
    convert(las, os.path.join(comp, "potree", "nav"), True)
    print(f"  nav: {len(A):,} 노드, 장소 {len(places)}")
    return [{"name": "nav", "path": "nav/metadata.json", "points": int(len(A)), "clearance_max_cm": float(A[:, 3].max() * 100)}]


def overlays(comp):
    """페이지에 내장할 겹쳐 그리기 자료: 토폴로지(장소·포털) + 3D scene graph(객체·엣지)."""
    out = {"places": [], "portals": [], "objects": [], "edges": [], "terrain": []}
    sp = os.path.join(comp, "nav3d", "nav3d_graph_summary.json")
    if os.path.exists(sp):
        S = json.load(open(sp))
        for pl in S.get("places", []):
            col = [230, 40, 40] if pl["type"] == "stair" else ([255, 158, 61] if pl["type"] == "landing" else PLACE_PAL[pl["id"] % len(PLACE_PAL)].tolist())
            out["places"].append({"id": pl["id"], "type": pl["type"], "c": [round(v, 3) for v in pl["centroid"]], "nodes": pl.get("nodes"), "color": col})
        for po in S.get("portals", []):
            out["portals"].append({"id": po["id"], "type": po["type"], "c": [round(v, 3) for v in po["centroid"]], "places": po.get("places", []), "width": round(po.get("width", 0), 2)})
    op = os.path.join(comp, "session_objects.json")
    if os.path.exists(op):
        for i, o in enumerate(json.load(open(op))):
            out["objects"].append({"id": i, "tag": o.get("tag", "?"), "c": o["center_w"], "e": o["extent"], "n": o.get("num_det")})
    for f, src in (("session_edges.json", "heuristic"), ("session_edges_vlm.json", "vlm")):
        fp = os.path.join(comp, f)
        if os.path.exists(fp):
            for e in json.load(open(fp)) or []:
                if "object_1_center" in e and "object_2_center" in e:
                    out["edges"].append({"a": e["object_1_center"], "b": e["object_2_center"], "d": e.get("edge_description", ""), "src": src})
    rp = os.path.join(comp, "voxel_session_relations.json")
    if os.path.exists(rp):
        out["terrain"] = [{"c": r["center"], "obj": r["object"], "on": r["on"]} for r in json.load(open(rp)) or []]
    return out


PAGE2 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "potree_page.html"), encoding="utf-8").read() if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "potree_page.html")) else ""


def export_new(root, comp, want, raw_max, title):
    tmp = tempfile.mkdtemp(prefix="potree_"); layers = []; sem = None
    try:
        if "main" in want:
            L, sem = layer_main(root, comp, tmp); layers += L
        if "loc" in want: layers += layer_loc(comp, tmp)
        if "loco" in want: layers += layer_loco(comp, tmp)
        if "nav" in want: layers += layer_nav(comp, tmp)
        if "raw" in want: layers += layer_raw(comp, root, tmp, True, raw_max)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not layers: sys.exit("레이어가 없습니다 (compiled/main_map 확인)")
    classes, cd = sem if sem else ([], None)
    cls = {"0": {"visible": True, "name": "(의미 없는 레이어)", "color": [0.8, 0.8, 0.8, 1.0]}}
    legend = []
    for i, name in enumerate(classes):
        col = CD.hex2rgb(cd["classes"][i]["color"]) if cd and i < len(cd["classes"]) else (160, 160, 160)
        cls[str(i + 1)] = {"visible": True, "name": name, "color": [c / 255 for c in col] + [1.0]}; legend.append({"code": i + 1, "name": name, "color": list(col)})
    for code, name, hexc in ((OTHER_CODE, "other", CD.OTHER_COLOR), (NONE_CODE, "unlabeled", CD.UNKNOWN_COLOR)):
        col = CD.hex2rgb(hexc); cls[str(code)] = {"visible": True, "name": name, "color": [c / 255 for c in col] + [1.0]}; legend.append({"code": code, "name": name, "color": list(col)})
    main_l = next((l for l in layers if l["name"] == "main"), None)
    if main_l:
        for g in legend: g["points"] = main_l["class_counts"].get(g["code"], 0)
    ov = overlays(comp)
    page = PAGE2.replace("__TITLE__", title).replace("__LAYERS__", json.dumps(layers)).replace("__CLASSES__", json.dumps(cls)) \
                .replace("__LEGEND__", json.dumps(legend)).replace("__OVERLAYS__", json.dumps(ov, separators=(",", ":")))
    open(os.path.join(comp, "potree", "index.html"), "w", encoding="utf-8").write(page)
    json.dump({"layers": layers, "overlays": {k: len(v) for k, v in ov.items()}, "built": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(comp, "potree", "layers.json"), "w"), indent=1)
    print(f"뷰어: output/{title}/compiled/potree/index.html ({len(layers)} 레이어, 겹쳐 그리기 {', '.join(f'{k} {len(v)}' for k, v in ov.items())})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map"); ap.add_argument("--layers", default=None, help="새 빌드 기본 main,loco,nav,loc (raw 추가 가능) · 옛 빌드 기본 voxel3d,raw"); ap.add_argument("--raw-max", type=float, default=40e6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = a.map if os.path.isabs(a.map) else os.path.join(REPO, "output", a.map)
    comp = os.path.join(root, "compiled"); os.makedirs(os.path.join(comp, "potree"), exist_ok=True)
    if not os.path.exists(CONVERTER): sys.exit(f"PotreeConverter 없음: {CONVERTER} (POTREE_CONVERTER 환경변수로 지정 가능)")
    title = os.path.basename(os.path.normpath(root))
    if os.path.isdir(os.path.join(comp, "main_map")):
        want = [s.strip() for s in (a.layers or "main,loco,nav,loc").split(",") if s.strip()]
        for d in ("voxel3d", "voxel3d_lidaronly", "voxel3d_stair"):                     # 옛 레이어 폴더가 남아 있으면 정리 (같은 root 를 새 구조로 다시 빌드한 경우)
            if os.path.isdir(os.path.join(comp, "potree", d)): shutil.rmtree(os.path.join(comp, "potree", d))
        return export_new(root, comp, want, a.raw_max, title)
    want = [s.strip() for s in (a.layers or "voxel3d,raw").split(",") if s.strip()]
    tmp = tempfile.mkdtemp(prefix="potree_")
    try:
        layers = []
        if "voxel3d" in want: layers += layer_voxel3d(root, comp, tmp, a.force)
        if "raw" in want: layers += layer_raw(comp, root, tmp, a.force, a.raw_max)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not layers: sys.exit("레이어가 없습니다 (voxel3d npz / lidar_scans 확인)")
    classes = {str(k): {"visible": True, "name": v, "color": SRC_COLORS[k]} for k, v in SRC_NAMES.items()}
    classes["0"] = {"visible": True, "name": "raw (unclassified)", "color": [0.8, 0.8, 0.8, 1.0]}
    page = PAGE.replace("__TITLE__", title).replace("__LAYERS__", json.dumps(layers)).replace("__CLASSES__", json.dumps(classes))
    open(os.path.join(comp, "potree", "index.html"), "w", encoding="utf-8").write(page)
    json.dump({"layers": layers, "built": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(comp, "potree", "layers.json"), "w"), indent=1)
    print(f"뷰어: output/{title}/compiled/potree/index.html ({len(layers)} 레이어)")


if __name__ == "__main__":
    main()
