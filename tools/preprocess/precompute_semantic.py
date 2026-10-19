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
ap.add_argument("--hybrid-suppress", action="store_true", help="hybrid: secondary 비지면 확신만큼 primary 지면 채널 억제 (실험 결과 지면 IoU 손해 → 기본 off)")
ap.add_argument("--hub-id", default=None, help="모델 가중치 경로/허브 id 덮어쓰기 (레지스트리 backend 유지; 미세조정 체크포인트 비교용)")
ap.add_argument("--size", type=int, default=0, help="모델 입력 긴 변(px) 덮어쓰기 (0=processor 기본; Mapillary M2F 기본 384²). 4:3 로 맞춤")
ap.add_argument("--tta-flip", action="store_true", help="좌우 플립 TTA (확률 평균)")
ap.add_argument("--sky-black", type=int, default=0, help="RGB 최대값 < 이 값인 픽셀(검은 하늘)은 모든 채널 0 (sim 전용)")
ap.add_argument("--ema", type=float, default=0.0, help="시간 평활: p = ema·p_prev + (1−ema)·p_cur (프레임 순서대로)")
a = ap.parse_args(); D = a.sem_dir; os.makedirs(D + "/sem", exist_ok=True)
H, W = 480, 640
cd = CD.load(a.classdict)
if a.model: cd["model"] = a.model; CD.validate(cd)
mi = CD.model_info(cd); backend, hub = mi["backend"], (a.hub_id or mi["hub_id"])
if not a.size and mi.get("size"): a.size = int(mi["size"])                  # 레지스트리 기본 입력 크기
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("모델:", cd["model"], "=", hub, f"[{backend}]")
if backend == "hybrid":                                                 # 두 모델을 올리고 클래스별로 채널 출처를 나눔
    import importlib; HYB = mi
    def _load(key, hub_override=None):
        m_ = CD.MODELS[key]; h_ = hub_override or m_["hub_id"]
        if m_["backend"] == "segformer":
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            p_ = SegformerImageProcessor.from_pretrained(h_); md = SegformerForSemanticSegmentation.from_pretrained(h_, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
            def sc(ims):
                inp = p_(images=ims, return_tensors="pt").to(dev); pv = inp["pixel_values"].half() if dev == "cuda" else inp["pixel_values"]
                return torch.nn.functional.interpolate(md(pixel_values=pv).logits.float(), size=(H, W), mode="bilinear", align_corners=False).softmax(1)
        else:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            p_ = AutoImageProcessor.from_pretrained(h_); md = Mask2FormerForUniversalSegmentation.from_pretrained(h_).to(dev).eval()
            SZ = {"height": int(a.size * 3 / 4), "width": a.size} if a.size else None
            def sc(ims):
                kw = {"size": SZ} if SZ else {}; out = md(**p_(images=ims, return_tensors="pt", **kw).to(dev))
                res = p_.post_process_semantic_segmentation(out, target_sizes=[(H, W)] * len(ims), return_segmentation_scores=True)
                return torch.stack([r.segmentation_scores for r in res]).float().clamp(0, 1)
        return {int(k): v for k, v in md.config.id2label.items()}, sc
    id2l_p, sc_p = _load(HYB["primary"], a.hub_id); id2l_s, sc_s = _load(HYB["secondary"])
    pri = {l.lower() for l in id2l_p.values()}
    use_primary = [all(x.lstrip("=") in pri for x in c["ade"]) for c in cd["classes"]]          # 사전 패턴이 전부 primary 라벨이면 primary 채널
    gp = CD.ade_ids(cd, id2l_p); gs = CD.ade_ids(cd, id2l_s)
    print("hybrid: primary 클래스", [c["name"] for c, u in zip(cd["classes"], use_primary) if u], "| secondary", [c["name"] for c, u in zip(cd["classes"], use_primary) if not u])
    class _M: pass
    model = _M(); model.config = _M(); model.config.id2label = {**{k: v for k, v in id2l_s.items()}}
    def scores(ims): return (sc_p(ims), sc_s(ims))


if backend == "hybrid": pass
elif backend == "segformer":
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
    SIZE = {"height": int(a.size * 3 / 4), "width": a.size} if a.size else None
    def scores_once(ims):
        kw = {"size": SIZE} if SIZE else {}
        out = model(**proc(images=ims, return_tensors="pt", **kw).to(dev))
        res = proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)] * len(ims), return_segmentation_scores=True)
        return torch.stack([r.segmentation_scores for r in res]).float().clamp(0, 1)
    def scores(ims):                                                   # → (B, C, H, W) 클래스 점수 0~1 (+ 플립 TTA)
        s_ = scores_once(ims)
        if a.tta_flip: s_ = 0.5 * (s_ + scores_once([im.transpose(Image.FLIP_LEFT_RIGHT) for im in ims]).flip(-1))
        return s_
    def _unused(ims):
        out = model(**proc(images=ims, return_tensors="pt").to(dev))
        res = proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)] * len(ims), return_segmentation_scores=True)
        return torch.stack([r.segmentation_scores for r in res]).float().clamp(0, 1)

id2l = {int(k): v for k, v in model.config.id2label.items()}
groups = list(zip(CD.names(cd), CD.ade_ids(cd, id2l)))
if backend == "hybrid": groups = [(n, (gp[i] if use_primary[i] else gs[i])) for i, n in enumerate(CD.names(cd))]
for n, g in groups: print(n, "ids:", g, [id2l[i] for i in g])
names_ = CD.names(cd); overrides = [(names_.index(c["name"]), [names_.index(o) for o in c["overrides"]], float(c.get("override_min", 0.3))) for c in cd["classes"] if c.get("overrides")]
for i, os_, th in overrides: print(f"override: {names_[i]} ≥{th} → {[names_[o] for o in os_]} = 0")
if any(not g for _, g in groups): print("경고: 라벨이 하나도 매칭되지 않은 클래스가 있음 (채널은 0 으로 채움)")
json.dump(cd, open(D + "/classes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)     # 복셀 빌더가 읽음
files = sorted(glob.glob(D + "/rgb/*.png"))
if files:                                                              # 출력 해상도 = 긴 변 640, 입력 RGB 비율 유지 (4:3 → 640x480, 16:9 → 640x360)
    from PIL import Image as _PI; _w, _h = _PI.open(files[0]).size; W = 640; H = int(round(W * _h / _w)); print(f"입력 {_w}x{_h} → 출력 {W}x{H}")
if a.skip_existing:
    files = [f for f in files if not os.path.exists(D + "/sem/%s.npz" % os.path.basename(f)[:-4])]
print("프레임:", len(files), "device:", dev); t0 = time.time(); ema_prev = None
with torch.no_grad():
    for s0 in range(0, len(files), a.batch):
        batch = files[s0:s0 + a.batch]
        ims = [Image.open(p).convert("RGB") for p in batch]
        probs = scores(ims)
        if backend == "hybrid":
            pp, ps = probs; chans = [(pp[:, g].sum(1) if use_primary[i] else ps[:, g].sum(1)) if g else torch.zeros_like(ps[:, 0]) for i, (_, g) in enumerate(groups)]
            sec_ng = torch.stack([c for c, u in zip(chans, use_primary) if not u]).amax(0).clamp(0, 1)   # secondary 가 보는 비지면(장애물·문맥) 확신
            for i, u in enumerate(use_primary):                                                           # primary 는 4클래스 softmax 라 기둥·나무에도 지면 확률을 줌 → (옵션) 비지면 확신만큼 억제
                if u and a.hybrid_suppress: chans[i] = chans[i] * (1 - sec_ng)
            probs = None
        if probs is not None and a.sky_black > 0:                                               # 검은 하늘 픽셀은 투표 제외 (sim: 하늘=building 번짐 차단)
            dark = torch.stack([torch.from_numpy(np.asarray(im)).amax(-1) < a.sky_black for im in ims]).to(dev)
            probs = probs * (~dark)[:, None].float()
        if probs is not None: chans = [probs[:, g].sum(1) if g else torch.zeros_like(probs[:, 0]) for _, g in groups]
        for i, os_, th in overrides:                                   # 노면 표시 우선: 예) crosswalk ≥override_min 인 픽셀은 road 표를 받지 않음
            m = chans[i] >= th
            for o in os_: chans[o] = torch.where(m, torch.zeros_like(chans[o]), chans[o])
        st = torch.stack(chans, dim=-1).clamp(0, 1)
        if a.ema > 0:                                                     # 시간 평활 (배치 안에서도 순서대로)
            for b in range(st.shape[0]):
                st[b] = a.ema * ema_prev + (1 - a.ema) * st[b] if ema_prev is not None else st[b]; ema_prev = st[b].clone()
        arr = (st * 255).byte().cpu().numpy()      # (B,H,W,N)
        for j, p in enumerate(batch):
            np.savez_compressed(D + "/sem/%s.npz" % os.path.basename(p)[:-4], sem=arr[j])
        if (s0 // a.batch) % 25 == 0: print(f"  {s0}/{len(files)}  {time.time()-t0:.0f}s")
print("완료:", len(glob.glob(D + "/sem/*.npz")), f"[{time.time()-t0:.0f}s]")
