#!/usr/bin/env python3
"""ConceptGraphs 객체 쌍의 관계(엣지)를 Claude 비전으로 판정 — 맵 후처리 방식.

흐름:  객체 pkl → 후보 쌍(기하) → 두 객체가 함께 보인 프레임 크롭(증거 이미지)
       → Claude (이미지 + 기하 메타) → 구조화 JSON {relation, confidence, evidence}
       → <out>/edges_claude.json (+ evidence/*.jpg, raw/*.json)

사용 (cg 컨테이너):
  export ANTHROPIC_API_KEY=sk-ant-...
  python vlm_edges.py --scene log_000 --max-pairs 30 --dry-run   # 후보·증거 이미지만 (API 호출 없음)
  python vlm_edges.py --scene log_000 --max-pairs 30             # 실제 판정
"""
import argparse, base64, glob, gzip, io, json, os, pickle, time
import numpy as np
import cv2

RELATIONS = ["on top of", "under", "next to", "leaning on", "inside", "attached to", "none"]
SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": RELATIONS},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["relation", "confidence", "evidence"],
    "additionalProperties": False,
}
SYSTEM = (
    "You judge PHYSICAL spatial relations between two objects in a robot's camera images for a 3D scene graph. "
    "Object A is outlined in RED, object B in BLUE. Several views of the same pair may be given. "
    "Answer only about the relation of A relative to B, using exactly one of: "
    + ", ".join(RELATIONS) + ". "
    "'on top of' = A rests on B's upper surface; 'under' = A is beneath B; 'next to' = side by side within reach, "
    "not touching in a supporting way; 'leaning on' / 'attached to' / 'inside' as commonly understood. "
    "Use 'none' when the objects are merely in the same view without a meaningful physical relation, or when "
    "the boxes look like a mis-detection. Be conservative: confidence < 0.5 when unsure. Keep evidence to one sentence."
)


def load_objects(pkl):
    with gzip.open(pkl, "rb") as f:
        return pickle.load(f)["objects"]


def bbox_of(o):
    p = np.asarray(o["pcd_np"])
    lo, hi = p.min(0), p.max(0)
    return (lo + hi) / 2, hi - lo


def evidence_image(o1, o2, views, out_dir, pair_id):
    """두 객체가 함께 관측된 프레임을 골라 bbox 표시 후 크롭. (경로, base64) 리스트 반환."""
    f1 = {str(p): i for i, p in enumerate(o1["color_path"])}
    common = [(str(p), f1[str(p)], j) for j, p in enumerate(o2["color_path"]) if str(p) in f1]
    if not common:
        return []
    # 시간적으로 퍼진 프레임을 고르기 위해 균등 샘플
    pick = [common[k] for k in np.linspace(0, len(common) - 1, min(views, len(common))).astype(int)]
    outs = []
    for n, (path, i1, i2) in enumerate(pick):
        img = cv2.imread(path)
        if img is None:
            continue
        b1 = np.asarray(o1["xyxy"][i1]).astype(int); b2 = np.asarray(o2["xyxy"][i2]).astype(int)
        cv2.rectangle(img, (b1[0], b1[1]), (b1[2], b1[3]), (60, 60, 230), 2)
        cv2.rectangle(img, (b2[0], b2[1]), (b2[2], b2[3]), (230, 120, 40), 2)
        cv2.putText(img, f"A:{o1['class_name']}", (b1[0], max(b1[1] - 6, 12)), 0, 0.55, (60, 60, 230), 2)
        cv2.putText(img, f"B:{o2['class_name']}", (b2[0], max(b2[1] - 6, 12)), 0, 0.55, (230, 120, 40), 2)
        x0, y0 = max(min(b1[0], b2[0]) - 40, 0), max(min(b1[1], b2[1]) - 40, 0)
        x1, y1 = min(max(b1[2], b2[2]) + 40, img.shape[1]), min(max(b1[3], b2[3]) + 40, img.shape[0])
        crop = img[y0:y1, x0:x1]
        if crop.shape[0] < 96 or crop.shape[1] < 96:
            crop = img
        p = os.path.join(out_dir, f"pair{pair_id:03d}_v{n}.jpg")
        cv2.imwrite(p, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        outs.append((p, base64.standard_b64encode(buf.tobytes()).decode()))
    return outs


def ask_claude(client, model, images_b64, meta_text):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}} for b in images_b64]
    content.append({"type": "text", "text": meta_text})
    kwargs = dict(model=model, max_tokens=1024, system=SYSTEM,
                  messages=[{"role": "user", "content": content}],
                  output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    try:
        # 안전 분류기 거절 시 서버가 대체 모델로 재시도 (opus-5 권장 기본)
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    except Exception as e:   # 베타 파라미터 미지원 등 -> 일반 엔드포인트
        print("   (beta fallback 사용 불가, 일반 호출:", type(e).__name__, ")")
        resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        return {"relation": "none", "confidence": 0.0, "evidence": "refused"}, resp
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp


class LocalVLM:
    """무료 로컬 비전 LLM (기본 Qwen2.5-VL-7B-Instruct, Apache-2.0). 서버 GPU 에서 실행, API 비용 0."""

    def __init__(self, model_id):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model_id = model_id

    def ask(self, image_paths, tag_a, tag_b):
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in image_paths]
        # 수치 메타는 빼고 이미지 위주로 묻는다 (bbox 중심 거리·높이차는 7B 모델을 오도함)
        content.append({"type": "text", "text":
            f"Object A is inside the RED box and is labeled '{tag_a}'. Object B is inside the BLUE box and is labeled '{tag_b}'. "
            "The colored boxes are only annotations drawn by a detector — ignore how the boxes overlap. "
            "Judge the REAL objects: is A resting on B's top surface with physical support contact (on top of), beneath B (under), beside B (next to), "
            "leaning on B, physically contained within B like an item in a bin (inside — NOT merely drawn within B's box), "
            "attached to B, or not physically related (none)? "
            "Important: being higher in the image than B is NOT 'on top of' unless B actually supports A — "
            "e.g. a laptop on a desk above a chair is 'none' relative to the chair; blinds above a TV are 'none'. "
            "Double-check which object is RED (A) and which is BLUE (B) before answering; the relation must be stated as A relative to B. "
            "First write one short sentence describing what you see, then output a JSON object on its own line: "
            '{"relation": <one of ' + str(RELATIONS) + '>, "confidence": <number 0..1>, "evidence": "<that sentence>"}'})
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = self.processor(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=220, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        # JSON 추출 (코드펜스/잡문 방어)
        s, e = reply.find("{"), reply.rfind("}")
        try:
            ans = json.loads(reply[s:e + 1])
            if ans.get("relation") not in RELATIONS:
                ans["relation"] = "none"
            ans["confidence"] = float(ans.get("confidence", 0.0))
            ans["evidence"] = str(ans.get("evidence", ""))[:300]
        except Exception:
            ans = {"relation": "none", "confidence": 0.0, "evidence": "unparseable: " + reply[:120]}
        return ans, reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="log_000")
    ap.add_argument("--exp", default="r_mapping_stride10")
    ap.add_argument("--dataset-root", default="/data2/cg/data/raibo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-dist", type=float, default=1.5, help="후보 쌍 중심 거리 상한(m)")
    ap.add_argument("--max-pairs", type=int, default=30)
    ap.add_argument("--min-det", type=int, default=2)
    ap.add_argument("--drop", default="person")
    ap.add_argument("--views", type=int, default=2, help="쌍당 증거 이미지 수")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--backend", choices=["claude", "local"], default="local",
                    help="local = 무료 로컬 VLM (GPU), claude = Anthropic API")
    ap.add_argument("--local-model", default="Qwen/Qwen2.5-VL-32B-Instruct",
                    help="7B 는 정확도 25% 로 부적합, 32B 67% (12쌍 정답지 기준)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    exp = os.path.join(a.dataset_root, a.scene, "exps", a.exp)
    pkl = glob.glob(os.path.join(exp, "pcd_*.pkl.gz"))[0]
    out = a.out or f"/data2/cg/compiled/vlm_edges_{a.scene}"
    os.makedirs(os.path.join(out, "evidence"), exist_ok=True); os.makedirs(os.path.join(out, "raw"), exist_ok=True)

    drop = set(t.strip() for t in a.drop.split(",") if t.strip())
    objs = [o for o in load_objects(pkl) if o["class_name"] not in drop and o["num_detections"] >= a.min_det]
    geo = [bbox_of(o) for o in objs]
    print(f"객체 {len(objs)}개 (필터 후)")

    # 후보 쌍: 중심 거리 + 함께 관측된 프레임 존재
    cands = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            d = float(np.linalg.norm(geo[i][0] - geo[j][0]))
            if d > a.max_dist:
                continue
            shared = len(set(map(str, objs[i]["color_path"])) & set(map(str, objs[j]["color_path"])))
            if shared == 0:
                continue
            # A = 더 높은 쪽 (on top of 판정 자연스럽게)
            hi, lo = (i, j) if geo[i][0][2] >= geo[j][0][2] else (j, i)
            cands.append((d, hi, lo, shared))
    cands.sort(key=lambda c: c[0])
    cands = cands[: a.max_pairs]
    print(f"후보 쌍 {len(cands)}개 (거리 ≤ {a.max_dist}m, 공동 관측 프레임 있음)")

    client = local = None
    if not a.dry_run:
        if a.backend == "claude":
            import anthropic
            client = anthropic.Anthropic()
        else:
            t0 = time.time(); local = LocalVLM(a.local_model)
            print(f"로컬 VLM 로드: {a.local_model} ({time.time()-t0:.0f}s)")

    edges, cost_in = [], 0
    for k, (d, i, j, shared) in enumerate(cands):
        A, B = objs[i], objs[j]
        (ca, ea), (cb, eb) = geo[i], geo[j]
        ev = evidence_image(A, B, a.views, os.path.join(out, "evidence"), k)
        meta = (f"A = '{A['class_name']}' (size {ea[0]:.2f}x{ea[1]:.2f}x{ea[2]:.2f} m). "
                f"B = '{B['class_name']}' (size {eb[0]:.2f}x{eb[1]:.2f}x{eb[2]:.2f} m). "
                f"Center distance {d:.2f} m; A is {ca[2]-cb[2]:+.2f} m higher than B; "
                f"horizontal offset {np.linalg.norm((ca-cb)[:2]):.2f} m. Seen together in {shared} frames. "
                f"What is the physical relation of A to B?")
        print(f"[{k:02d}] {A['class_name']} ↔ {B['class_name']}  d={d:.2f}m dz={ca[2]-cb[2]:+.2f}  증거 {len(ev)}장")
        src = f"claude:{a.model}" if a.backend == "claude" else f"local:{a.local_model}"
        rec = {"pair_id": k, "object_1_id": A["curr_obj_num"], "object_1_tag": A["class_name"],
               "object_2_id": B["curr_obj_num"], "object_2_tag": B["class_name"],
               "distance_m": round(d, 2), "dz_m": round(float(ca[2] - cb[2]), 2),
               "evidence_images": [p for p, _ in ev], "source": src}
        if a.dry_run or not ev:
            rec.update(relation=None, confidence=None, evidence=None)
        elif a.backend == "claude":
            t0 = time.time()
            ans, resp = ask_claude(client, a.model, [b for _, b in ev], meta)
            rec.update(relation=ans["relation"], confidence=ans["confidence"], evidence=ans["evidence"],
                       served_by=resp.model, in_tokens=resp.usage.input_tokens, out_tokens=resp.usage.output_tokens,
                       seconds=round(time.time() - t0, 1))
            cost_in += resp.usage.input_tokens
            json.dump(resp.model_dump(), open(os.path.join(out, "raw", f"pair{k:03d}.json"), "w"), default=str)
            print(f"     → {ans['relation']}  ({ans['confidence']:.2f})  {ans['evidence']}")
        else:
            t0 = time.time()
            ans, raw = local.ask([p for p, _ in ev], A["class_name"], B["class_name"])
            rec.update(relation=ans["relation"], confidence=ans["confidence"], evidence=ans["evidence"],
                       served_by=a.local_model, seconds=round(time.time() - t0, 1))
            open(os.path.join(out, "raw", f"pair{k:03d}.txt"), "w").write(raw)
            print(f"     → {ans['relation']}  ({ans['confidence']:.2f})  {ans['evidence']}  [{rec['seconds']}s]")
        edges.append(rec)
    json.dump(edges, open(os.path.join(out, "edges_claude.json"), "w"), ensure_ascii=False, indent=1)
    if not a.dry_run:
        import collections
        print("\n관계 분포:", collections.Counter(e["relation"] for e in edges).most_common())
        if a.backend == "claude":
            print(f"입력 토큰 합계 {cost_in:,} (≈ ${cost_in/1e6*5:.2f} @ opus-5 입력 $5/1M)")
        else:
            print(f"로컬 VLM — API 비용 $0, 쌍당 평균 {np.mean([e.get('seconds', 0) for e in edges if e.get('seconds')]):.1f}s")
    print("저장:", os.path.join(out, "edges_claude.json"))


if __name__ == "__main__":
    main()
