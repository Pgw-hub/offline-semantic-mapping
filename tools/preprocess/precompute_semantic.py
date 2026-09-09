#!/usr/bin/env python3
"""SegFormer-B4 (ADE20K) → 야외 4채널 semantic 확률 (grass / path / tree / person), 프레임별 npz.

입력: <sem_dir>/rgb/*.png (extract_rgb_session.py 출력, 640x480)   출력: <sem_dir>/sem/<같은이름>.npz {sem: (480,640,4) uint8 = 확률×255}
채널 정의(ADE 라벨 부분문자열 합):
  grass  = grass
  path   = path, sidewalk, road, earth, dirt, field, runway
  tree   = tree, plant, palm
  person = person
사용: python precompute_semantic.py <sem_dir> [--batch 8] [--skip-existing]
(구 precompute_semantic_outdoor.py 와 수치 동일. SEMDIR 환경변수 대신 인자.)
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # <repo>
import classdict as CD
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ap = argparse.ArgumentParser()
ap.add_argument("sem_dir"); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--model", default="nvidia/segformer-b4-finetuned-ade-512-512")
ap.add_argument("--classdict", default="default", help="클래스 사전 이름(classdicts/<name>.json) 또는 경로. 기본 default = grass/path/tree/person")
a = ap.parse_args(); D = a.sem_dir; os.makedirs(D + "/sem", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = SegformerImageProcessor.from_pretrained(a.model)
model = SegformerForSemanticSegmentation.from_pretrained(a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
cd = CD.load(a.classdict); id2l = {int(k): v for k, v in model.config.id2label.items()}
groups = list(zip(CD.names(cd), CD.ade_ids(cd, id2l)))
for n, g in groups: print(n, "ids:", g, [id2l[i] for i in g])
json.dump(cd, open(D + "/classes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)     # 복셀 빌더가 읽음
files = sorted(glob.glob(D + "/rgb/*.png"))
if a.skip_existing:
    files = [f for f in files if not os.path.exists(D + "/sem/%s.npz" % os.path.basename(f)[:-4])]
print("프레임:", len(files), "device:", dev); t0 = time.time()
with torch.no_grad():
    for s0 in range(0, len(files), a.batch):
        batch = files[s0:s0 + a.batch]
        ims = [Image.open(p).convert("RGB") for p in batch]
        inp = proc(images=ims, return_tensors="pt").to(dev)
        pv = inp["pixel_values"].half() if dev == "cuda" else inp["pixel_values"]
        logits = model(pixel_values=pv).logits
        up = torch.nn.functional.interpolate(logits.float(), size=(480, 640), mode="bilinear", align_corners=False)
        probs = up.softmax(1)
        chans = [probs[:, g].sum(1) if g else torch.zeros_like(probs[:, 0]) for _, g in groups]
        arr = (torch.stack(chans, dim=-1).clamp(0, 1) * 255).byte().cpu().numpy()      # (B,H,W,N)
        for j, p in enumerate(batch):
            np.savez_compressed(D + "/sem/%s.npz" % os.path.basename(p)[:-4], sem=arr[j])
        if (s0 // a.batch) % 25 == 0: print(f"  {s0}/{len(files)}  {time.time()-t0:.0f}s")
print("완료:", len(glob.glob(D + "/sem/*.npz")), f"[{time.time()-t0:.0f}s]")
