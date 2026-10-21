#!/usr/bin/env python3
"""semantic 복셀 맵 + ConceptGraphs 객체 + 객체-지형 관계(별도 엔티티) -> rerun.

사용: python voxel_rerun_v2.py <voxel.npz> <objects.json> <traj_dir_or_file> <out.rrd> [--unknown-sub N]
  objects.json : session_objects.json (merge 산출물) 또는 obj_json_*.json
  traj         : raibo/ 디렉토리(모든 log_*/traj.txt 이어붙임) 또는 단일 traj.txt
엔티티:
  world/voxels/{semantic,rgb,unknown}/f*   복셀 (최초 관측 프레임에 등장)
  world/objects/<tag>_<i>                  bbox, 라벨 = 클래스 이름만
  world/relations/<i>                      객체 중심 → 지형 접점 선 + "on <지형>" 라벨 (별도 토글)
  world/robot                              궤적
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))  # 저장소 루트 (패키지 import)
import argparse, glob, json, os
import numpy as np
import rerun as rr

ap = argparse.ArgumentParser()
ap.add_argument("vox"); ap.add_argument("objects"); ap.add_argument("traj"); ap.add_argument("out")
ap.add_argument("--unknown-sub", type=int, default=1, help="unknown 복셀 표시 서브샘플")
ap.add_argument("--drop-tags", default="person", help="제외 태그 (쉼표)")
a = ap.parse_args()

d = np.load(a.vox)
ijk = d["ijk"]; V = float(d["vox"]); classes = [str(c) for c in d["classes"]]
n_sem = d["n_sem"]; sem = d["sem_mean"].astype(np.float32); rgb = d["rgb"]; first = d["first_frame"]
ctr = (ijk + 0.5) * V
lab = sem.argmax(1); labeled = n_sem > 0
# 4클래스(grass/path/tree/person) 어디에도 해당 안 되는 복셀(벽·차·바닥 등) = "other"
# (argmax 가 0=grass 로 떨어지는 버그 방지). 관측은 됐으나 어휘에 없는 것.
OTHER = len(classes)
other = labeled & (sem.max(1) < 0.3)
lab = np.where(other, OTHER, lab)
from common import classdict as CD
cd = CD.from_voxsem(d); palette = CD.colors_rgb(cd) + [CD.hex2rgb(CD.OTHER_COLOR)]   # other = 청회색 (unknown 회색과 구분)
NON_TERRAIN = set(CD.ids_with_role(cd, "dynamic", "context"))                          # 지형 판정에서 제외 (obstacle 은 'on tree' 허용)
semcol = np.array([palette[i] for i in lab], np.uint8)
print(f"other(4클래스 해당없음) 복셀: {other.sum():,} / 라벨 {labeled.sum():,}")
nF = int(d["n_frames"])
print(f"복셀 {len(ijk):,} (라벨 {labeled.sum():,}), 프레임 {nF}")

rr.init("voxel_semantic_session")
rr.save(a.out)
CH = 20
for c0 in range(0, nF + CH, CH):
    m = (first >= c0) & (first < c0 + CH)
    if not m.any():
        continue
    rr.set_time_sequence("frame", c0)
    ml = m & labeled; mu = m & ~labeled
    if ml.any():
        rr.log(f"world/voxels/semantic/f{c0}", rr.Points3D(ctr[ml], colors=semcol[ml], radii=V * 0.45))
        rr.log(f"world/voxels/rgb/f{c0}", rr.Points3D(ctr[ml], colors=rgb[ml], radii=V * 0.45))
    if mu.any():
        sub = np.flatnonzero(mu)[:: a.unknown_sub]
        rr.log(f"world/voxels/unknown/f{c0}", rr.Points3D(ctr[sub], colors=[[110, 110, 110]], radii=V * 0.35))

# 궤적: 디렉토리면 log 순서로 이어붙임 (30Hz → 3프레임마다 ≈ 10Hz semantic 프레임)
trajs = []
if os.path.isdir(a.traj):
    for p in sorted(glob.glob(os.path.join(a.traj, "log_???", "traj.txt"))):
        li = int(os.path.basename(os.path.dirname(p))[4:])
        T = np.loadtxt(p).reshape(-1, 4, 4)
        trajs.append((li, T))
else:
    trajs.append((0, np.loadtxt(a.traj).reshape(-1, 4, 4)))
for li, T in trajs:
    for k, fi in enumerate(range(0, len(T), 3)):
        rr.set_time_sequence("frame", li * 600 + k)
        rr.log("world/robot", rr.Points3D([T[fi][:3, 3]], colors=[[255, 255, 255]], radii=0.25))

# 객체 + 관계
rr.set_time_sequence("frame", nF)
raw = json.load(open(a.objects))
objs = list(raw.values()) if isinstance(raw, dict) else raw
drop = set(t.strip() for t in a.drop_tags.split(",") if t.strip())
rels = []
for i, o in enumerate(objs):
    tag = o.get("object_tag") or o.get("tag")
    if tag in drop:
        continue
    c = np.array(o.get("bbox_center") or o["center_w"], float); e = np.array(o.get("bbox_extent") or o.get("extent") or [0.5, 0.5, 0.5], float)
    rr.log(f"world/objects/{tag.replace(' ', '_')}_{i}",
           rr.Boxes3D(centers=[c], half_sizes=[e / 2], labels=[tag], colors=[[255, 200, 40]]))
    foot = labeled & (np.abs(ctr[:, 0] - c[0]) < e[0] / 2 + 0.3) & (np.abs(ctr[:, 1] - c[1]) < e[1] / 2 + 0.3) \
        & (ctr[:, 2] < c[2] - e[2] / 2 + 0.3) & (ctr[:, 2] > c[2] - e[2] / 2 - 0.6)
    foot = foot & (lab < OTHER) & ~np.isin(lab, list(NON_TERRAIN))   # other·동적·문맥 클래스는 지형이 아님
    if foot.any():
        terrain = classes[int(np.bincount(lab[foot], minlength=len(classes)).argmax())]
        gz = float(ctr[foot, 2].mean())
        rr.log(f"world/relations/{i}", rr.LineStrips3D([[c, [c[0], c[1], gz]]], colors=[[80, 200, 255]], radii=0.03,
                                                       labels=[f"on {terrain}"]))
        rels.append({"object": tag, "center": c.round(2).tolist(), "on": terrain, "n_voxels": int(foot.sum())})
json.dump(rels, open(a.out.replace(".rrd", "_relations.json"), "w"), ensure_ascii=False, indent=1)
import collections
print("객체", len(objs), "→ 관계", len(rels), collections.Counter((r["object"], r["on"]) for r in rels).most_common(8))
print("저장:", a.out)
