#!/usr/bin/env python3
"""저장된 ConceptGraphs 맵(pkl.gz)에 CLIP 텍스트 질의 (headless CLI).

사용: python cg_query.py <pcd_pkl_gz> "질의문" [질의문2 ...]
"""
import gzip, pickle, sys

import numpy as np
import torch
import open_clip

path = sys.argv[1]
queries = sys.argv[2:] or ["something to sit on", "a place to put a cup", "something soft"]

with gzip.open(path, "rb") as f:
    res = pickle.load(f)
objs = res["objects"]
print(f"객체 {len(objs)}개 로드: {path}")

fts, meta = [], []
for o in objs:
    ft = o.get("clip_ft")
    if ft is None:
        continue
    ft = ft.cpu().numpy() if torch.is_tensor(ft) else np.asarray(ft)
    fts.append(ft.ravel())
    bbox = o.get("bbox")
    center = np.asarray(bbox.center).round(2).tolist() if bbox is not None else None
    meta.append((o.get("curr_obj_num", "?"), o.get("class_name", "?"), center,
                 int(o.get("num_detections", 0))))
F = np.stack(fts)
F = F / np.linalg.norm(F, axis=1, keepdims=True)
print(f"CLIP feature {F.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
model = model.to(device).eval()
tok = open_clip.get_tokenizer("ViT-H-14")

for q in queries:
    with torch.no_grad():
        t = model.encode_text(tok([q]).to(device)).cpu().numpy().ravel()
    t = t / np.linalg.norm(t)
    sim = F @ t
    order = np.argsort(-sim)[:5]
    print(f"\n질의: {q!r}")
    for r, i in enumerate(order, 1):
        oid, cls, center, ndet = meta[i]
        print(f"  {r}. sim={sim[i]:.3f}  [{cls}] id={oid} center={center} det={ndet}")
