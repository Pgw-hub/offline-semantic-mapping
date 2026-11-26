#!/usr/bin/env python3
"""E ① 객체 융합 — 로그별 ConceptGraphs 객체 → 세션 3D scene graph 노드 (compiled/session_objects.json).

옛 map_compiler.py merge 를 대신한다. 옛 방식의 문제 (서울 cut715 에서 사진으로 확인, analysis/objects/):
  · 로그 간 병합이 '중심 거리 1.5 m + CLIP 0.75' 라 서로 다른 물체를 묶고 (싱크+쓰레기통, 마우스+상자 13로그),
    같은 물체가 겹쳐 여러 개로 남는다 (같은 의자·화분·티슈 상자가 2–3개)
  · 라벨 = 첫 검출의 클래스 (ConceptGraphs 는 병합해도 class_name 을 안 바꾼다) → 상자를 storage bin, 사무용 의자를 folded chair,
    조명을 monitor 로. 바닥·계단·유리벽도 객체로 남는다
새 방식:
  1. 점구름 겹침 + 의미로 전체 군집화 (MaskClustering 처럼 순차 병합 대신 한꺼번에): 5 cm 칸, 1칸 팽창 겹침 oa·ob 에서
     min ≥ t_min 이고 (CLIP ≥ c1 또는 검출 라벨 분포 겹침 ≥ lab) → 같은 물체, max ≥ t_max 이고 (CLIP ≥ c2 또는 라벨 ≥ lab) → 같은 물체
     (선반 위 상자·책상 위 책처럼 포함된 다른 물체는 CLIP·라벨이 달라 남는다)
  2. 라벨: 서로 다른 프레임의 잘 보인 장면 4개를 2x2 로 붙여 Qwen2.5-VL 에 보여 주고 후보 목록(검출 투표 상위 5 + 실내 공통 어휘
     + 구조물 단어)에서 고르게 한다. 구조물(floor·wall·stairs·railing·glass·several objects)을 고르면 객체 아님 → rejected 로.
     VLM 이 없으면(--vlm-model none) 신뢰도 가중 다수결
  3. 2차 병합: 겹치는 두 군집이 라벨이 같으면 합침 (의자–좌석, 화분–화분통처럼 일부만 보인 같은 물체)
서울 cut715 (점검 표본, 기준값 정할 때 안 쓴 48개): 라벨 정확도 34 % → 83 %, 객체 아닌 것 제거 0/9 → 8/9.
출력: session_objects.json(남긴 객체, 옛 형식 + label_source·vote_tag·views), session_objects_clip.npz(같은 순서),
      session_objects_rejected.json(객체 아님), session_objects_fusion.json(요약·파라미터)
사용: object_fusion.py --root <dataset> --out <compiled> [--vlm-model Qwen/Qwen2.5-VL-32B-Instruct | none]
"""
import argparse, collections, glob, gzip, json, os, pickle, sys, time
import numpy as np

COMMON = ["chair", "office chair", "armchair", "table", "desk", "box", "storage bin", "basket", "shelf", "bookshelf", "cabinet", "file cabinet", "trash can",
          "potted plant", "light", "lamp", "sign", "monitor", "computer tower", "keyboard", "mouse", "laptop", "printer", "telephone", "bag", "backpack",
          "paper bag", "bottle", "cup", "fire extinguisher", "power outlet", "power strip", "light switch", "whiteboard", "picture", "window", "door",
          "sink", "refrigerator", "microwave", "water cooler", "cart", "ladder", "divider", "pillow", "couch", "book", "toolbox", "fan", "clock", "mirror",
          "radiator", "speaker", "container", "tissue box", "suitcase", "shoe", "jacket", "broom", "bucket", "stool", "bench", "coat rack", "tv", "projector", "tape"]
STRUCTURE = ["floor", "wall", "ceiling", "stairs", "railing", "glass wall", "floor mat", "doorway", "several objects"]


# ── 1. 로그별 객체 ─────────────────────────────────────────────────────────────────────────────
def load_objects(root):
    objs = []
    for exp in sorted(glob.glob(os.path.join(root, "log_*/exps/r_mapping_stride10"))):
        log = exp.split("/")[-3]; p = glob.glob(os.path.join(exp, "pcd_*.pkl.gz"))
        if not p: continue
        r = pickle.load(gzip.open(p[0], "rb")); names = r["class_names"]
        for i, o in enumerate(r["objects"]):
            ft = o.get("clip_ft")
            if ft is None: continue
            ft = np.asarray(ft.cpu().numpy() if hasattr(ft, "cpu") else ft, np.float32).ravel()
            P = np.asarray(o["pcd_np"], np.float32)
            if not len(P): continue
            keep = np.unique(np.floor(P / 0.05).astype(np.int64), axis=0, return_index=True)[1]
            dets = [(str(cp), [float(v) for v in xy], names[c] if 0 <= c < len(names) else "?", float(cf))
                    for cp, xy, c, cf in zip(o["color_path"], o["xyxy"], o["class_id"], o["conf"])]
            objs.append(dict(log=log, idx=i, tag=o.get("class_name", "?"), dets=dets, pcd=P[keep], clip=ft / (np.linalg.norm(ft) + 1e-9)))
    return objs


def votes_of(dets):
    v = collections.Counter()
    for _, _, c, cf in dets: v[c] += cf
    t = sum(v.values()) or 1.0
    return {k: x / t for k, x in v.items()}


def pair_features(objs, max_dist=4.0):
    V = [set(map(tuple, np.floor(o["pcd"] / 0.05).astype(np.int64))) for o in objs]
    D = [{(x + a, y + b, z + c) for (x, y, z) in v for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)} for v in V]
    C = np.array([o["pcd"].mean(0) for o in objs]); F = np.stack([o["clip"] for o in objs]); W = [votes_of(o["dets"]) for o in objs]
    P = {}
    for a in range(len(objs)):
        for b in range(a + 1, len(objs)):
            if np.linalg.norm(C[a] - C[b]) > max_dist: continue
            oa = len(V[a] & D[b]) / len(V[a]); ob = len(V[b] & D[a]) / len(V[b])
            if max(oa, ob) < 0.05: continue
            P[(a, b)] = dict(mn=min(oa, ob), mx=max(oa, ob), cos=float(F[a] @ F[b]), lab=sum(min(W[a].get(k, 0), W[b].get(k, 0)) for k in W[a]))
    return P


def union_find(n, links):
    parent = list(range(n))
    def f(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in links:
        ra, rb = f(a), f(b)
        if ra != rb: parent[rb] = ra
    return [f(i) for i in range(n)]


# ── 2. 라벨 (VLM) ─────────────────────────────────────────────────────────────────────────────
IMG_WH = {}


def image_size(path):
    d = os.path.dirname(path)
    if d not in IMG_WH:
        import cv2
        img = cv2.imread(path); IMG_WH[d] = (img.shape[1], img.shape[0]) if img is not None else (640, 360)
    return IMG_WH[d]


def best_views(dets, k=4):
    def score(d):
        img_wh = image_size(d[0])
        x0, y0, x1, y1 = d[1]; edge = x0 < 3 or y0 < 3 or x1 > img_wh[0] - 3 or y1 > img_wh[1] - 3
        return d[3] * min(1.0, np.sqrt(max(1.0, (x1 - x0) * (y1 - y0))) / 80) * (0.6 if edge else 1.0)
    out, seen = [], []
    for d in sorted(dets, key=lambda d: -score(d)):
        try: fr = int(d[0].split("frame")[-1].split(".")[0])
        except ValueError: fr = len(seen) * 100
        if any(abs(fr - s) < 20 for s in seen): continue
        seen.append(fr); out.append(d)
        if len(out) == k: break
    return out


def view_sheet(dets, size=320):
    import cv2
    tiles = []
    for d in best_views(dets):
        img = cv2.imread(d[0])
        if img is None: continue
        h, w = img.shape[:2]; x0, y0, x1, y1 = d[1]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(x1 - x0, y1 - y0) * 0.8 + 40
        a, b, c, e = int(max(0, cx - s)), int(max(0, cy - s)), int(min(w, cx + s)), int(min(h, cy + s)); im = img[b:e, a:c].copy()
        cv2.rectangle(im, (int(x0 - a), int(y0 - b)), (int(x1 - a), int(y1 - b)), (0, 255, 0), 3)
        f = size / max(im.shape[:2]); im = cv2.resize(im, (max(1, int(im.shape[1] * f)), max(1, int(im.shape[0] * f))))
        pad = np.full((size, size, 3), 40, np.uint8); oy, ox = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
        pad[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im; tiles.append(pad)
    tiles += [np.full((size, size, 3), 40, np.uint8)] * (4 - len(tiles))
    return cv2.cvtColor(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), cv2.COLOR_BGR2RGB)


class VlmLabeler:
    def __init__(self, model_id):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch; self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")

    def label(self, dets, cands):
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        prompt = ("The image shows up to four views of ONE physical thing, outlined by the green box in each view, seen by a robot in an office/workshop building. "
                  "What is inside the green box? Choose the single best label from this list: " + ", ".join(cands + STRUCTURE) + ". "
                  "Use a structure label (" + ", ".join(STRUCTURE) + ") when the box mainly covers that rather than one object. Reply with only the label.")
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(view_sheet(dets))}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True); imgs, vids = process_vision_info(msgs)
        inp = self.proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
        with self.torch.no_grad(): out = self.model.generate(**inp, max_new_tokens=12, do_sample=False)
        return self.proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip().strip(".").lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("CG_RAIBO_ROOT")); ap.add_argument("--out", default=os.environ.get("CG_OUT_DIR"))
    ap.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-32B-Instruct", help="none = 검출 투표 라벨")
    ap.add_argument("--t-min", type=float, default=0.3); ap.add_argument("--c1", type=float, default=0.8)
    ap.add_argument("--t-max", type=float, default=0.5); ap.add_argument("--c2", type=float, default=0.85); ap.add_argument("--lab", type=float, default=0.3)
    ap.add_argument("--t2-min", type=float, default=0.2); ap.add_argument("--t2-max", type=float, default=0.5)
    a = ap.parse_args(); t0 = time.time()
    objs = load_objects(a.root); n = len(objs)
    if not n:
        json.dump([], open(os.path.join(a.out, "session_objects.json"), "w")); print("객체 없음"); return
    P = pair_features(objs)
    # 1차: 겹침 + 의미
    links = [(i, j) for (i, j), p in P.items()
             if (p["mn"] >= a.t_min and (p["cos"] >= a.c1 or p["lab"] >= a.lab)) or (p["mx"] >= a.t_max and (p["cos"] >= a.c2 or p["lab"] >= a.lab))]
    L1 = union_find(n, links); groups = collections.defaultdict(list)
    for i, c in enumerate(L1): groups[c].append(i)
    # 라벨
    labeler = None
    if a.vlm_model and a.vlm_model.lower() != "none":
        try: labeler = VlmLabeler(a.vlm_model)
        except Exception as e: print(f"VLM 사용 불가 ({e}) → 검출 투표 라벨", flush=True)
    lab, raw, vote = {}, {}, {}
    for c, mem in groups.items():
        dets = [d for i in mem for d in objs[i]["dets"]]; v = votes_of(dets); top = sorted(v, key=lambda k: -v[k])[:5]
        vote[c] = top[0] if top else "?"
        if labeler:
            ans = labeler.label(dets, top + [x for x in COMMON if x not in top]); raw[c] = ans; lab[c] = "none" if ans in STRUCTURE else ans
        else:
            lab[c] = vote[c]; raw[c] = vote[c]
    # 2차: 같은 라벨 + 겹침
    idx = {c: k for k, c in enumerate(groups)}; links2 = []
    for (i, j), p in P.items():
        ci, cj = L1[i], L1[j]
        if ci != cj and lab[ci] != "none" and lab[ci] == lab[cj] and (p["mn"] >= a.t2_min or p["mx"] >= a.t2_max): links2.append((idx[ci], idx[cj]))
    L2 = union_find(len(groups), links2); final = collections.defaultdict(list)
    for c, k in idx.items(): final[L2[k]].append(c)
    kept, rejected, clips = [], [], []
    for subs in final.values():
        mem = [i for c in subs for i in groups[c]]; dets = [d for i in mem for d in objs[i]["dets"]]
        main_c = max(subs, key=lambda c: sum(len(objs[i]["dets"]) for i in groups[c]))
        pts = np.concatenate([objs[i]["pcd"] for i in mem]); lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
        w = np.array([max(1, len(objs[i]["dets"])) for i in mem], np.float32); f = (np.stack([objs[i]["clip"] for i in mem]) * w[:, None]).sum(0); f /= np.linalg.norm(f) + 1e-9
        rec = dict(tag=lab[main_c], center_w=((lo + hi) / 2).round(3).tolist(), extent=(hi - lo).round(3).tolist(), num_det=len(dets),
                   logs=sorted({objs[i]["log"] for i in mem}), n_merged=len(mem), label_source="vlm" if labeler else "vote", vote_tag=vote[main_c],
                   vlm_answer=raw[main_c], members=[[objs[i]["log"], objs[i]["idx"]] for i in mem], views=[[d[0], [round(x, 1) for x in d[1]]] for d in best_views(dets, 3)])
        if rec["tag"] == "none": rejected.append(rec)
        else: kept.append(rec); clips.append(f)
    order = np.argsort([-r["num_det"] for r in kept]); kept = [kept[k] for k in order]; clips = [clips[k] for k in order]
    os.makedirs(a.out, exist_ok=True)
    json.dump(kept, open(os.path.join(a.out, "session_objects.json"), "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(a.out, "session_objects_clip.npz"), clip=np.stack(clips).astype(np.float16) if clips else np.zeros((0, 1024), np.float16))
    json.dump(rejected, open(os.path.join(a.out, "session_objects_rejected.json"), "w"), ensure_ascii=False, indent=1)
    summ = dict(per_log_objects=n, first_pass=len(groups), objects=len(kept), rejected=len(rejected), label_source="vlm" if labeler else "vote",
                vlm_model=a.vlm_model if labeler else None, seconds=round(time.time() - t0), params={k: getattr(a, k) for k in ("t_min", "c1", "t_max", "c2", "lab", "t2_min", "t2_max")})
    json.dump(summ, open(os.path.join(a.out, "session_objects_fusion.json"), "w"), indent=1)
    print(f"로그별 객체 {n} → 1차 {len(groups)} → 객체 {len(kept)} (+ 객체 아님 {len(rejected)}), 라벨 {summ['label_source']}, {summ['seconds']} s")


if __name__ == "__main__":
    main()
