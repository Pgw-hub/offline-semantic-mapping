#!/usr/bin/env python3
"""semantic 복셀 맵 + ConceptGraphs 객체를 같은 world 프레임으로 rerun 에 기록.

엔티티:
  world/voxels/semantic/f*  클래스 색 (grass 초록 / path 황갈 / tree 짙초록 / person 빨강)
  world/voxels/rgb/f*       실제 색
  world/voxels/unknown/f*   카메라가 못 본 복셀 (회색, 기하만)
  world/objects/*           ConceptGraphs bbox + 라벨 (+ 객체-지형 관계)
  world/robot               궤적
타임라인 'frame' = semantic 프레임(10Hz). 복셀은 최초 관측 프레임에 등장.
"""
import gzip, json, pickle, sys
import numpy as np
import rerun as rr

vox_path, obj_dir, traj_path, out = sys.argv[1:5]
d = np.load(vox_path)
ijk = d["ijk"]; V = float(d["vox"]); classes = [str(c) for c in d["classes"]]
n_sem = d["n_sem"]; sem = d["sem_mean"].astype(np.float32); rgb = d["rgb"]; first = d["first_frame"]
ctr = (ijk + 0.5) * V
lab = sem.argmax(1); conf = sem.max(1)
labeled = n_sem > 0
CCOL = {"grass": (95, 170, 95), "path": (200, 165, 120), "tree": (40, 110, 60), "person": (220, 60, 60)}
semcol = np.array([CCOL[classes[i]] for i in lab], np.uint8)
semcol[~labeled] = 110
print(f"복셀 {len(ijk):,} (라벨 {labeled.sum():,})")

rr.init("voxel_semantic_map")
rr.save(out)
CH = 10   # 프레임 10개씩 묶어 로깅
nF = int(d["n_frames"])
traj = np.loadtxt(traj_path).reshape(-1, 4, 4)
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
        rr.log(f"world/voxels/unknown/f{c0}", rr.Points3D(ctr[mu], colors=[[110, 110, 110]], radii=V * 0.35))
# 로봇 궤적 (cg 프레임 30Hz → 3프레임마다 ≈ semantic 10Hz)
for k, fi in enumerate(range(0, len(traj), 3)):
    rr.set_time_sequence("frame", k)
    rr.log("world/robot", rr.Points3D([traj[fi][:3, 3]], colors=[[255, 255, 255]], radii=0.25))

# ConceptGraphs 객체 + 객체-지형 관계
rr.set_time_sequence("frame", nF)
objs = json.load(open([p for p in __import__("glob").glob(obj_dir + "/obj_json_*.json")][0]))
rels = []
for k, o in objs.items():
    c = np.array(o["bbox_center"]); e = np.array(o["bbox_extent"])
    # bbox 발자국 아래 0.6m 이내 라벨 복셀의 지배 클래스 = 놓인 지형
    foot = labeled & (np.abs(ctr[:, 0] - c[0]) < e[0] / 2 + 0.3) & (np.abs(ctr[:, 1] - c[1]) < e[1] / 2 + 0.3) \
        & (ctr[:, 2] < c[2] - e[2] / 2 + 0.3) & (ctr[:, 2] > c[2] - e[2] / 2 - 0.6)
    terrain = classes[int(np.bincount(lab[foot], minlength=4).argmax())] if foot.any() else "unknown"
    rels.append({"object": o["object_tag"], "center": o["bbox_center"], "on": terrain, "n_voxels": int(foot.sum())})
    rr.log(f"world/objects/{k}_{o['object_tag'].replace(' ', '_')}",
           rr.Boxes3D(centers=[c], half_sizes=[e / 2], labels=[f"{o['object_tag']} on {terrain}"], colors=[[255, 200, 40]]))
json.dump(rels, open(out.replace(".rrd", "_relations.json"), "w"), ensure_ascii=False, indent=1)
print("객체-지형 관계:", [(r["object"], r["on"], r["n_voxels"]) for r in rels])
print("저장:", out)
