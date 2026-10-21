#!/usr/bin/env python3
"""맵 빌드 결과 → Potree 웹 뷰어 (포털에서 열기). 컨테이너 offline_3d_semantic_map 안에서 실행.

사용: python tools/potree/potree_export.py <map_root 또는 output 아래 이름> [--layers voxel3d,raw] [--raw-max 40000000] [--force]
레이어
  voxel3d       compiled/session_voxel3d.npz — 보행 맵 셀 중심 (x, y) 와 컨트롤러가 읽는 높이 mean_z.
                변형(session_voxel3d_<이름>.npz: bonxai, lidaronly …)도 각각 레이어로.
                classification = 출처(1 LiDAR · 2 depth 채움 · 3 평탄 채움), intensity = 클래스 id, RGB = 클래스 색.
  raw           dataset/<log>/lidar_scans/*.npz — S 단계의 정합된 스캔 점 전부(최종 PGO 프레임). 너무 많으면 --raw-max 로 균등 추출.
출력: compiled/potree/<레이어>/(metadata.json, octree.bin, hierarchy.bin) + compiled/potree/index.html
뷰어 정적 파일은 tools/potree/viewer/{build,libs} (Potree 1.8.2), 변환기는 tools/potree/PotreeConverter/build/PotreeConverter (2.1.1).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # 저장소 루트 (패키지 import)
import argparse, glob, json, os, shutil, struct, subprocess, sys, tempfile, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from common import classdict as CD  # noqa: E402

CONVERTER = os.path.join(REPO, "tools", "potree", "PotreeConverter", "build", "PotreeConverter")
SRC_NAMES = {1: "LiDAR", 2: "depth fill", 3: "flat fill"}
SRC_COLORS = {1: [0.35, 0.75, 0.45, 1.0], 2: [0.30, 0.55, 0.95, 1.0], 3: [0.62, 0.62, 0.62, 1.0]}

PT2 = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("intensity", "<u2"), ("ret", "u1"), ("cls", "u1"),
                ("angle", "i1"), ("user", "u1"), ("psid", "<u2"), ("r", "<u2"), ("g", "<u2"), ("b", "<u2")])   # LAS 1.2 point format 2, 26 B


def write_las(path, xyz, intensity=None, classification=None, rgb=None, scale=0.001):
    """LAS 1.2 / point format 2 (laspy 없이). rgb 는 0..255 uint8 (N,3)."""
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
                      n, n, 0, 0, 0, 0, scale, scale, scale, *off, hi[0], lo[0], hi[1], lo[1], hi[2], lo[2])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map"); ap.add_argument("--layers", default="voxel3d,raw"); ap.add_argument("--raw-max", type=float, default=40e6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = a.map if os.path.isabs(a.map) else os.path.join(REPO, "output", a.map)
    comp = os.path.join(root, "compiled"); os.makedirs(os.path.join(comp, "potree"), exist_ok=True)
    if not os.path.exists(CONVERTER): sys.exit(f"PotreeConverter 없음: {CONVERTER}")
    want = [s.strip() for s in a.layers.split(",") if s.strip()]
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
    title = os.path.basename(os.path.normpath(root))
    page = PAGE.replace("__TITLE__", title).replace("__LAYERS__", json.dumps(layers)).replace("__CLASSES__", json.dumps(classes))
    open(os.path.join(comp, "potree", "index.html"), "w", encoding="utf-8").write(page)
    json.dump({"layers": layers, "built": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(comp, "potree", "layers.json"), "w"), indent=1)
    print(f"뷰어: output/{title}/compiled/potree/index.html ({len(layers)} 레이어)")


if __name__ == "__main__":
    main()
