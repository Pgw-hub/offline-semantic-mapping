#!/usr/bin/env python3
"""전 로그의 VLM 엣지(vlm_edges_<log>_v2/edges_claude.json)를 하나의 rerun + 통합 json 으로.
사용: python edges_session_rerun.py <raibo_root> <compiled_root> <out.rrd> [--min-conf 0.6]
"""
import argparse, collections, glob, json, os
import numpy as np
import rerun as rr

ap = argparse.ArgumentParser()
ap.add_argument("raibo"); ap.add_argument("compiled"); ap.add_argument("out")
ap.add_argument("--min-conf", type=float, default=0.6)
a = ap.parse_args()
COL = {"on top of": (255, 170, 40), "under": (120, 170, 255), "next to": (90, 220, 130),
       "leaning on": (230, 120, 200), "inside": (200, 200, 80), "attached to": (255, 110, 110)}
rr.init("vlm_edges_session"); rr.save(a.out)
allE = []; kept = collections.Counter(); n_obj = 0
for ed_path in sorted(glob.glob(os.path.join(a.compiled, "vlm_edges_log_*_v2", "edges_claude.json"))):
    log = os.path.basename(os.path.dirname(ed_path)).replace("vlm_edges_", "").replace("_v2", "")
    oj = glob.glob(os.path.join(a.raibo, log, "exps", "r_mapping_stride10", "obj_json_*.json"))
    if not oj:
        continue
    objs = {o["id"]: o for o in json.load(open(oj[0])).values()}
    edges = json.load(open(ed_path))
    used = set()
    for ed in edges:
        rel = ed.get("relation"); conf = ed.get("confidence") or 0
        if not rel or rel == "none" or conf < a.min_conf:
            continue
        A = objs.get(ed["object_1_id"]); B = objs.get(ed["object_2_id"])
        if A is None or B is None:
            continue
        ca, cb = np.array(A["bbox_center"]), np.array(B["bbox_center"])
        rr.log(f"world/edges/{rel.replace(' ', '_')}/{log}_{ed['pair_id']}",
               rr.LineStrips3D([[ca, cb]], colors=[COL.get(rel, (255, 255, 255))], radii=0.04,
                               labels=[f"{A['object_tag']} {rel} {B['object_tag']}"]))
        kept[rel] += 1; used.update([A["id"], B["id"]])
        allE.append({"log": log, "object_1": {"id": A["id"], "tag": A["object_tag"], "center": A["bbox_center"]},
                     "relation": rel, "confidence": conf,
                     "object_2": {"id": B["id"], "tag": B["object_tag"], "center": B["bbox_center"]},
                     "evidence": ed.get("evidence"), "source": ed.get("source")})
    for oid in used:   # 엣지에 참여한 객체만 박스로
        o = objs[oid]; c = np.array(o["bbox_center"]); e = np.array(o["bbox_extent"])
        rr.log(f"world/objects/{log}/{o['object_tag'].replace(' ', '_')}_{oid}",
               rr.Boxes3D(centers=[c], half_sizes=[e / 2], labels=[o["object_tag"]], colors=[[200, 200, 200]]))
        n_obj += 1
json.dump(allE, open(os.path.join(a.compiled, "session_edges_vlm.json"), "w"), ensure_ascii=False, indent=1)
print(f"세션 VLM 엣지 {len(allE)}개 {dict(kept)} · 참여 객체 {n_obj} → {a.out}")
