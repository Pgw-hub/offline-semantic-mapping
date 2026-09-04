#!/usr/bin/env python3
"""VLM 판정 엣지(객체↔객체)를 rerun 에 선+라벨로 기록.
사용: python edges_rerun.py <obj_json> <edges_json> <out.rrd> [--min-conf 0.6]
  obj_json : ConceptGraphs obj_json_*.json (id = curr_obj_num)
  edges_json: vlm_edges.py 출력 (object_1_id/object_2_id = curr_obj_num, relation, confidence)
"""
import argparse, json, collections
import numpy as np
import rerun as rr

ap = argparse.ArgumentParser()
ap.add_argument("objs"); ap.add_argument("edges"); ap.add_argument("out")
ap.add_argument("--min-conf", type=float, default=0.6)
a = ap.parse_args()

objs = {o["id"]: o for o in json.load(open(a.objs)).values()}
edges = json.load(open(a.edges))
COL = {"on top of": (255, 170, 40), "under": (120, 170, 255), "next to": (90, 220, 130),
       "leaning on": (230, 120, 200), "inside": (200, 200, 80), "attached to": (255, 110, 110)}
rr.init("vlm_edges"); rr.save(a.out)
kept = collections.Counter()
for o in objs.values():
    c = np.array(o["bbox_center"]); e = np.array(o["bbox_extent"])
    rr.log(f"world/objects/{o['object_tag'].replace(' ', '_')}_{o['id']}",
           rr.Boxes3D(centers=[c], half_sizes=[e / 2], labels=[o["object_tag"]], colors=[[200, 200, 200]]))
for ed in edges:
    rel = ed.get("relation"); conf = ed.get("confidence") or 0
    if not rel or rel == "none" or conf < a.min_conf:
        continue
    A = objs.get(ed["object_1_id"]); B = objs.get(ed["object_2_id"])
    if A is None or B is None:
        continue
    ca, cb = np.array(A["bbox_center"]), np.array(B["bbox_center"])
    rr.log(f"world/edges/{rel.replace(' ', '_')}/{ed['pair_id']}",
           rr.LineStrips3D([[ca, cb]], colors=[COL.get(rel, (255, 255, 255))], radii=0.03,
                           labels=[f"{A['object_tag']} {rel} {B['object_tag']} ({conf:.1f})"]))
    kept[rel] += 1
print("표시 엣지:", dict(kept), "→", a.out)
