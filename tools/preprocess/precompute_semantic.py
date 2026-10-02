#!/usr/bin/env python3
"""세그멘테이션 모델 → 클래스 사전 N채널 semantic 확률, 프레임별 npz. 모델은 사전의 "model" 키가 정한다 (classdict.MODELS).

입력: <sem_dir>/rgb/*.png (extract_rgb_session.py 출력, 640x480)   출력: <sem_dir>/sem/<같은이름>.npz {sem: (480,640,N) uint8 = 확률×255}
                                                                       + <sem_dir>/classes.json (사전 전체, model 포함 — 복셀 빌더가 읽음)
백엔드:
  segformer   픽셀별 softmax 확률 → 클래스별 라벨 합 (초기 4클래스 default 사전과 수치 동일)
  mask2former / eomt   마스크 모델: processor.post_process_semantic_segmentation(return_segmentation_scores) 의 클래스 점수
              (= Σ_query softmax(class)·sigmoid(mask), 0~1) → 클래스별 라벨 합, clamp(0,1)
사용: python precompute_semantic.py <sem_dir> [--classdict default|<name>|<path>] [--model <레지스트리 키>] [--batch 8] [--skip-existing]
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # <repo>
import classdict as CD
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("sem_dir"); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--classdict", default="default", help="클래스 사전 이름(classdicts/<name>.json) 또는 경로. 기본 default = grass/path/tree/person")
ap.add_argument("--model", default=None, help=f"사전의 model 을 덮어씀 (실험용). 키: {list(CD.MODELS)}")
a = ap.parse_args(); D = a.sem_dir; os.makedirs(D + "/sem", exist_ok=True)
H, W = 480, 640
cd = CD.load(a.classdict)
if a.model: cd["model"] = a.model; CD.validate(cd)
mi = CD.model_info(cd); backend, hub = mi["backend"], mi["hub_id"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("모델:", cd["model"], "=", hub, f"[{backend}]")

if backend == "segformer":
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(hub)
    model = SegformerForSemanticSegmentation.from_pretrained(hub, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    def scores(ims):                                                   # → (B, C, H, W) 클래스 확률 (softmax)
        inp = proc(images=ims, return_tensors="pt").to(dev)
        pv = inp["pixel_values"].half() if dev == "cuda" else inp["pixel_values"]
        logits = model(pixel_values=pv).logits
        return torch.nn.functional.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False).softmax(1)
else:
    from transformers import AutoImageProcessor
    if backend == "mask2former": from transformers import Mask2FormerForUniversalSegmentation as MaskModel
    elif backend == "eomt": from transformers import EomtForUniversalSegmentation as MaskModel
    else: raise SystemExit(f"알 수 없는 backend: {backend}")
    proc = AutoImageProcessor.from_pretrained(hub); model = MaskModel.from_pretrained(hub).to(dev).eval()   # fp32 (마스크 모델 fp16 은 미검증)
    def scores(ims):                                                   # → (B, C, H, W) 클래스 점수 0~1
        out = model(**proc(images=ims, return_tensors="pt").to(dev))
        res = proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)] * len(ims), return_segmentation_scores=True)
        return torch.stack([r.segmentation_scores for r in res]).float().clamp(0, 1)

id2l = {int(k): v for k, v in model.config.id2label.items()}
groups = list(zip(CD.names(cd), CD.ade_ids(cd, id2l)))
for n, g in groups: print(n, "ids:", g, [id2l[i] for i in g])
names_ = CD.names(cd); overrides = [(names_.index(c["name"]), [names_.index(o) for o in c["overrides"]], float(c.get("override_min", 0.3))) for c in cd["classes"] if c.get("overrides")]
for i, os_, th in overrides: print(f"override: {names_[i]} ≥{th} → {[names_[o] for o in os_]} = 0")
if any(not g for _, g in groups): print("경고: 라벨이 하나도 매칭되지 않은 클래스가 있음 (채널은 0 으로 채움)")
json.dump(cd, open(D + "/classes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)     # 복셀 빌더가 읽음
files = sorted(glob.glob(D + "/rgb/*.png"))
if a.skip_existing:
    files = [f for f in files if not os.path.exists(D + "/sem/%s.npz" % os.path.basename(f)[:-4])]
print("프레임:", len(files), "device:", dev); t0 = time.time()
with torch.no_grad():
    for s0 in range(0, len(files), a.batch):
        batch = files[s0:s0 + a.batch]
        ims = [Image.open(p).convert("RGB") for p in batch]
        probs = scores(ims)
        chans = [probs[:, g].sum(1) if g else torch.zeros_like(probs[:, 0]) for _, g in groups]
        for i, os_, th in overrides:                                   # 노면 표시 우선: 예) crosswalk ≥override_min 인 픽셀은 road 표를 받지 않음
            m = chans[i] >= th
            for o in os_: chans[o] = torch.where(m, torch.zeros_like(chans[o]), chans[o])
        arr = (torch.stack(chans, dim=-1).clamp(0, 1) * 255).byte().cpu().numpy()      # (B,H,W,N)
        for j, p in enumerate(batch):
            np.savez_compressed(D + "/sem/%s.npz" % os.path.basename(p)[:-4], sem=arr[j])
        if (s0 // a.batch) % 25 == 0: print(f"  {s0}/{len(files)}  {time.time()-t0:.0f}s")
print("완료:", len(glob.glob(D + "/sem/*.npz")), f"[{time.time()-t0:.0f}s]")
