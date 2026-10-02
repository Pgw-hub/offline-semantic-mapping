#!/usr/bin/env python3
"""SegFormer 미세조정 — 시뮬 2D GT(stage1 --write-labels: 0 sidewalk 1 road 2 crosswalk 3 grass, 255 무시) 로 학습.
사용: python train_segformer.py --train <rgb_dir>:<label_dir> [--train ...] [--val <rgb_dir>:<label_dir>] --out <ckpt_dir> [--base nvidia/segformer-b2-finetuned-ade-512-512]
      [--steps 3000 --bs 8 --lr 6e-5 --size 512 --holdout-every 10]
산출: <out>/ (HF 형식: config.json + model.safetensors + preprocessor) — classdict.MODELS 에 hub_id 로 등록해 파이프라인에서 사용. id2label = sidewalk/road/crosswalk/grass."""
import argparse, glob, json, os, random, time
import numpy as np, torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
NAMES = ["sidewalk", "road", "crosswalk", "grass"]
ap = argparse.ArgumentParser(); ap.add_argument("--train", action="append", required=True); ap.add_argument("--val", action="append", default=[]); ap.add_argument("--out", required=True)
ap.add_argument("--base", default="nvidia/segformer-b2-finetuned-ade-512-512"); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=6e-5)
ap.add_argument("--size", type=int, default=512, help="학습 입력 긴 변 (4:3)"); ap.add_argument("--holdout-every", type=int, default=0, help="train 쌍에서 N 프레임마다 1개를 val 로 떼어냄 (val 미지정 시)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); dev = "cuda"
def pairs(spec):
    rgb_dir, lab_dir = spec.split(":"); out = []
    for lp in sorted(glob.glob(os.path.join(lab_dir, "*.png"))):
        rp = os.path.join(rgb_dir, os.path.basename(lp))
        if os.path.exists(rp): out.append((rp, lp))
    return out
train = sum((pairs(s) for s in a.train), []); val = sum((pairs(s) for s in a.val), [])
if not val and a.holdout_every: val = train[::a.holdout_every]; train = [p for i, p in enumerate(train) if i % a.holdout_every]
print(f"train {len(train)} · val {len(val)} · base {a.base}")
proc = SegformerImageProcessor.from_pretrained(a.base); proc.do_reduce_labels = False
model = SegformerForSemanticSegmentation.from_pretrained(a.base, num_labels=len(NAMES), id2label={i: n for i, n in enumerate(NAMES)}, label2id={n: i for i, n in enumerate(NAMES)}, ignore_mismatched_sizes=True).to(dev)
W_, H_ = a.size, int(a.size * 3 / 4); mean = torch.tensor(proc.image_mean, device=dev).view(1, 3, 1, 1); std = torch.tensor(proc.image_std, device=dev).view(1, 3, 1, 1)
def load(batch, aug=True):
    xs, ys = [], []
    for rp, lp in batch:
        im = Image.open(rp).convert("RGB").resize((W_, H_), Image.BILINEAR); lb = Image.open(lp).resize((W_, H_), Image.NEAREST)
        x = np.asarray(im, np.float32) / 255.0; y = np.asarray(lb, np.int64).copy()
        if aug and random.random() < 0.5: x = x[:, ::-1]; y = y[:, ::-1]
        if aug:   # 밝기·대비 지터 (sim 조명 단조로움 보완)
            x = np.clip((x - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.08, 0.08), 0, 1)
        xs.append(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)); ys.append(torch.from_numpy(np.ascontiguousarray(y)))
    return (torch.stack(xs).to(dev) - mean) / std, torch.stack(ys).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
scaler = torch.amp.GradScaler("cuda")
@torch.no_grad()
def evaluate(pairs_, n=200):
    model.eval(); inter = np.zeros(len(NAMES)); uni = np.zeros(len(NAMES)); sel = pairs_[:: max(1, len(pairs_) // n)]
    for i in range(0, len(sel), 8):
        x, y = load(sel[i:i + 8], aug=False); lg = model(pixel_values=x).logits; lg = torch.nn.functional.interpolate(lg, size=y.shape[-2:], mode="bilinear", align_corners=False); p = lg.argmax(1)
        for c in range(len(NAMES)):
            m = y != 255; inter[c] += ((p == c) & (y == c) & m).sum().item(); uni[c] += (((p == c) | (y == c)) & m).sum().item()
    model.train(); return {n_: round(float(inter[c] / max(uni[c], 1)), 3) for c, n_ in enumerate(NAMES)}
model.train(); t0 = time.time(); best = -1
for step in range(1, a.steps + 1):
    batch = random.sample(train, a.bs); x, y = load(batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(pixel_values=x, labels=y)      # SegFormer 는 라벨 255 무시(ignore_index) + 1/4 해상도 로짓을 upsample 해 CE
    opt.zero_grad(set_to_none=True); out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 100 == 0: print(f"step {step} loss {out.loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
    if step % 500 == 0 or step == a.steps:
        ev = evaluate(val) if val else {}; mean_iou = float(np.mean(list(ev.values()))) if ev else 0.0; print(f"  val IoU {ev} mean {mean_iou:.3f}", flush=True)
        if mean_iou >= best:
            best = mean_iou; os.makedirs(a.out, exist_ok=True); model.save_pretrained(a.out); proc.save_pretrained(a.out)
            json.dump({"base": a.base, "train": a.train, "val": a.val or f"holdout_every={a.holdout_every}", "steps": step, "val_iou": ev, "size": a.size}, open(os.path.join(a.out, "train_info.json"), "w"), indent=1)
print("saved best (val mean IoU %.3f) →" % best, a.out)
