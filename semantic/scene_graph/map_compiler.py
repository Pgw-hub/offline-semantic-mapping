#!/usr/bin/env python3
"""Map Compiler 프로토타입 (서버측).

입력(고표현력 맵):
  - 전역 DINO feat 맵: /data2/20260827/_peek/global_session_feat.npz (16D/셀, world 좌표)
  - 지형 프로토타입:   /data2/20260827/_dino/proto_session.npz
  - ConceptGraphs 객체 맵: /data2/cg/data/raibo/<log>/exps/r_mapping_stride10/
      (pkl.gz 의 clip_ft + obj json 의 bbox — GradSLAM relative_pose=True 라서
       traj.txt 첫 줄 T_w_c0 로 world 변환해서 통합)

요청 3종 → 로봇(CPU)용 압축 산출물(npz/json + 미리보기 png):
  text_query   "벤치"  → CLIP 유사도 순위 객체 리스트(월드 좌표) [+ 활성 셀 마킹]
  terrain      grass   → 전역 feat 맵 코사인 활성도 grid (float16, bbox crop)
  region       x0 y0 x1 y1 → 영역 안 객체 + 지형 프로토타입 비율 요약

사용:
  python map_compiler.py text "a bench to sit on" [--topk 10]
  python map_compiler.py terrain grass|path|tree
  python map_compiler.py region <x0> <y0> <x1> <y1>
"""
import argparse, glob, gzip, json, os, pickle, time

import numpy as np

FEAT_NPZ = "/data2/20260827/_peek/global_session_feat.npz"
PROTO_NPZ = "/data2/20260827/_dino/proto_session.npz"
RAIBO_ROOT = os.environ.get("CG_RAIBO_ROOT", "/data2/cg/data/raibo")
OUT_DIR = os.environ.get("CG_OUT_DIR", "/data2/cg/compiled")


# ---------- 고표현력 맵 로드 ----------

def load_feat_map():
    d = np.load(FEAT_NPZ)
    feat = d["feat"].astype(np.float32)              # (N,N,16)
    cnt = d["feat_cnt"]
    gx0, gy0, res = float(d["gx0"]), float(d["gy0"]), float(d["gres"])
    n = np.linalg.norm(feat, axis=2, keepdims=True)
    valid = (cnt > 0) & (n[..., 0] > 1e-6)
    feat = np.where(n > 1e-6, feat / n, 0.0)
    return feat, valid, gx0, gy0, res, d["bbox"]


def load_objects():
    """모든 log 의 ConceptGraphs 객체를 world 좌표로 통합."""
    objs = []
    for exp in sorted(glob.glob(os.path.join(RAIBO_ROOT, "log_*/exps/r_mapping_stride10"))):
        log = exp.split("/")[-3]
        pkls = glob.glob(os.path.join(exp, "pcd_*.pkl.gz"))
        jp = glob.glob(os.path.join(exp, "obj_json_*.json"))
        if not pkls or not jp:
            continue   # 매핑 진행 중이거나 실패한 로그는 건너뜀
        with gzip.open(pkls[0], "rb") as f:
            res = pickle.load(f)
        jpath = jp[0]
        jinfo = json.load(open(jpath))
        jlist = list(jinfo.values())
        for i, o in enumerate(res["objects"]):
            ft = o.get("clip_ft")
            if ft is None:
                continue
            ft = np.asarray(ft.cpu().numpy() if hasattr(ft, "cpu") else ft, np.float32).ravel()
            ji = jlist[i] if i < len(jlist) else {}
            # rerun_realtime_mapping 은 dataset.poses(raw world c2w)를 사용 -> 이미 world 좌표
            c_w = np.array(ji.get("bbox_center", [0, 0, 0]))
            objs.append({
                "log": log,
                "tag": ji.get("object_tag", o.get("class_name", "?")),
                "center_w": c_w.round(2).tolist(),
                "extent": ji.get("bbox_extent"),
                "num_det": int(o.get("num_detections", 0)),
                "clip_ft": ft / (np.linalg.norm(ft) + 1e-9),
            })
    return objs


# ---------- 요청 처리 ----------

def req_text(args):
    import torch, open_clip
    objs = load_objects()
    F = np.stack([o["clip_ft"] for o in objs])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-H-14")
    with torch.no_grad():
        t = model.encode_text(tok([args.query]).to(dev)).cpu().numpy().ravel()
    t /= np.linalg.norm(t)
    sim = F @ t
    order = np.argsort(-sim)[: args.topk]
    out = [{k: v for k, v in objs[i].items() if k != "clip_ft"} | {"sim": round(float(sim[i]), 3)}
           for i in order]
    path = save_json("text_query", {"query": args.query, "results": out})
    for r in out:
        print(f"  sim={r['sim']:.3f} [{r['tag']}] {r['log']} @ world {r['center_w']} det={r['num_det']}")
    print("저장:", path)


def req_terrain(args):
    feat, valid, gx0, gy0, res, bbox = load_feat_map()
    p = np.load(PROTO_NPZ)
    names = [str(x) for x in p["names"]] if "names" in p else ["grass", "path", "tree"]
    q = p["proto"][names.index(args.proto)].astype(np.float32)
    q /= np.linalg.norm(q)
    t0 = time.time()
    sim = feat @ q
    sim[~valid] = np.nan
    dt = (time.time() - t0) * 1000
    r0, r1, c0, c1 = bbox
    crop = sim[r0:r1 + 1, c0:c1 + 1].astype(np.float16)
    path = os.path.join(OUT_DIR, f"terrain_{args.proto}.npz")
    np.savez_compressed(path, sim=crop, x0=gx0 + r0 * res, y0=gy0 + c0 * res, res=res,
                        axes="row=x,col=y")
    png = save_grid_png(crop.astype(np.float32), f"terrain_{args.proto}")
    v = crop[~np.isnan(crop)]
    print(f"'{args.proto}' 활성도 grid {crop.shape} (유효 {v.size}셀, 계산 {dt:.0f}ms)")
    print(f"  sim 분포: p10={np.percentile(v,10):.2f} p50={np.percentile(v,50):.2f} p90={np.percentile(v,90):.2f}")
    print("저장:", path, "+", png)


def req_region(args):
    feat, valid, gx0, gy0, res, _ = load_feat_map()
    p = np.load(PROTO_NPZ)
    names = [str(x) for x in p["names"]]
    P = p["proto"].astype(np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    # 누적 grid 는 row=x, col=y
    r0 = int((args.x0 - gx0) / res); r1 = int((args.x1 - gx0) / res)
    c0 = int((args.y0 - gy0) / res); c1 = int((args.y1 - gy0) / res)
    sub = feat[r0:r1 + 1, c0:c1 + 1]; vm = valid[r0:r1 + 1, c0:c1 + 1]
    lab = (sub @ P.T).argmax(-1)
    frac = {names[i]: round(float(((lab == i) & vm).sum() / max(vm.sum(), 1)), 3)
            for i in range(len(names))}
    objs = [o for o in load_objects()
            if args.x0 <= o["center_w"][0] <= args.x1 and args.y0 <= o["center_w"][1] <= args.y1]
    out = {"region": [args.x0, args.y0, args.x1, args.y1],
           "terrain_frac": frac, "n_valid_cells": int(vm.sum()),
           "objects": [{k: v for k, v in o.items() if k != "clip_ft"} for o in objs]}
    path = save_json("region", out)
    print("지형 비율:", frac, f"(유효 {int(vm.sum())}셀)")
    print(f"객체 {len(objs)}개:", sorted({o['tag'] for o in objs}))
    print("저장:", path)


def req_costmap(args):
    """전역 DINO/elevation 맵 -> 로봇 planner 용 traversability costmap.

    출력: uint8 grid (0=자유, 254=치명, 255=미관측), ROS costmap 규약.
    비용 = 지형 클래스 기본비용 + 경사 페널티.
    """
    d = np.load(FEAT_NPZ)
    feat, valid, gx0, gy0, res, bbox = load_feat_map()
    pz = np.load(PROTO_NPZ)
    names = [str(x) for x in pz["names"]]
    P = pz["proto"].astype(np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    lab = (feat @ P.T).argmax(-1)

    base = {"path": 10, "grass": 90, "tree": 253, "person": 200}
    cost = np.full(lab.shape, 255, np.uint8)          # 미관측
    for i, n in enumerate(names):
        cost[valid & (lab == i)] = base.get(n, 120)

    # 경사 페널티: elevation 기울기 (관측 셀만)
    el = d["elevation"] / np.maximum(d["el_cnt"], 1)
    el_ok = d["el_cnt"] > 0
    gy, gx = np.gradient(np.where(el_ok, el, np.nan), res)
    slope = np.hypot(gx, gy)
    steep = el_ok & valid & (slope > args.max_slope)
    cost[steep] = np.maximum(cost[steep], 253)

    r0, r1, c0, c1 = bbox
    crop = cost[r0:r1 + 1, c0:c1 + 1]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "costmap.npz")
    np.savez_compressed(path, cost=crop, x0=gx0 + r0 * res, y0=gy0 + c0 * res,
                        res=res, axes="row=x,col=y",
                        legend="0-252 cost, 253+ lethal, 255 unknown")
    import cv2
    vis = np.zeros((*crop.shape, 3), np.uint8)
    vis[crop == 255] = (40, 40, 40)
    ok = crop < 255
    g = (255 - crop[ok].astype(np.int32) * 255 // 254).astype(np.uint8)
    vis[ok] = np.stack([np.zeros_like(g), g, 255 - g], -1)   # 초록=자유, 빨강=치명
    png = os.path.join(OUT_DIR, "costmap.png")
    cv2.imwrite(png, vis.transpose(1, 0, 2)[::-1])
    n_ok = int(ok.sum())
    print(f"costmap {crop.shape} 저장: {path} + {png}")
    print(f"  관측 {n_ok}셀 | 자유(<50) {(crop<50).sum()} | 치명(>=253) {int(((crop>=253)&ok).sum())}")


def req_edges(args):
    """기하 휴리스틱 관계 엣지 (GPT 없이): on top of / under / next to.

    입력: merge 가 만든 session_objects.json (없으면 먼저 merge 실행 필요)
    """
    jp = os.path.join(OUT_DIR, "session_objects.json")
    objs = json.load(open(jp))
    edges = []
    cand_next = []
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i == j:
                continue
            a, b = objs[i], objs[j]
            ca, cb = np.array(a["center_w"]), np.array(b["center_w"])
            ea = np.array(a.get("extent") or [0.5] * 3); eb = np.array(b.get("extent") or [0.5] * 3)
            dxy = np.linalg.norm((ca - cb)[:2])
            dz = ca[2] - cb[2]
            # extent 는 정렬축 기준 대략치 -> 반경으로 환산
            ra = max(ea[:2]) / 2; rb = max(eb[:2]) / 2
            if dxy < max(ra, rb) and dz > 0.15 and dz < (ea[2] + eb[2]):
                edges.append((i, "on top of", j))
            elif i < j and dxy < args.next_to and abs(dz) < 0.8 and dxy > max(ra, rb) * 0.5:
                cand_next.append((dxy, i, j))
    # next-to 는 객체당 최근접 k개만 (밀집 구간 폭발 방지)
    per = {}
    for dxy, i, j in sorted(cand_next):
        if per.get(i, 0) < args.max_next and per.get(j, 0) < args.max_next:
            edges.append((i, "next to", j))
            per[i] = per.get(i, 0) + 1; per[j] = per.get(j, 0) + 1
    out = [{"edge_id": k,
            "edge_description": f"{objs[i]['tag']} {rel} {objs[j]['tag']}",
            "object_1_id": i, "object_1_tag": objs[i]["tag"], "object_1_center": objs[i]["center_w"],
            "object_2_id": j, "object_2_tag": objs[j]["tag"], "object_2_center": objs[j]["center_w"],
            "source": "geometric_heuristic"}
           for k, (i, rel, j) in enumerate(edges)]
    op = os.path.join(OUT_DIR, "session_edges.json")
    json.dump(out, open(op, "w"), ensure_ascii=False, indent=1)
    import collections
    kinds = collections.Counter(e[1] for e in edges)
    print(f"엣지 {len(out)}개 ({dict(kinds)}) 저장: {op}")
    for e in out[:6]:
        print("  -", e["edge_description"])


def req_merge(args):
    """로그 간 중복 객체 병합 -> 세션 통합 객체 맵 (union-find)."""
    objs = load_objects()
    n = len(objs)
    C = np.array([o["center_w"] for o in objs])
    F = np.stack([o["clip_ft"] for o in objs])
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    n_link = 0
    for i in range(n):
        for j in range(i + 1, n):
            if objs[i]["log"] == objs[j]["log"]:
                continue   # 동일 로그 내 병합은 ConceptGraphs 가 이미 수행 — 로그 경계만 봉합
            if np.linalg.norm(C[i] - C[j]) < args.dist and float(F[i] @ F[j]) > args.cos:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri; n_link += 1
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for g in groups.values():
        mem = [objs[i] for i in g]
        w = np.array([max(m["num_det"], 1) for m in mem], np.float32)
        c = (C[g] * w[:, None]).sum(0) / w.sum()
        f = (F[g] * w[:, None]).sum(0); f /= np.linalg.norm(f) + 1e-9
        best = mem[int(np.argmax(w))]
        merged.append({"tag": best["tag"], "center_w": c.round(2).tolist(),
                       "extent": best.get("extent"),
                       "num_det": int(sum(m["num_det"] for m in mem)),
                       "logs": sorted({m["log"] for m in mem}),
                       "n_merged": len(mem), "clip_ft": f})
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "session_objects.json")
    json.dump([{k: v for k, v in m.items() if k != "clip_ft"} for m in merged],
              open(out_json, "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(OUT_DIR, "session_objects_clip.npz"),
                        clip=np.stack([m["clip_ft"] for m in merged]).astype(np.float16))
    multi = sum(1 for m in merged if m["n_merged"] > 1)
    print(f"객체 {n} -> 병합 후 {len(merged)} (연결 {n_link}쌍, 다중관측 객체 {multi}개)")
    print("저장:", out_json)


def req_prompt(args):
    """맵 전체를 LLM 에 붙여넣을 수 있는 reasoning 프롬프트로 변환."""
    objs = load_objects()
    objs = [o for o in objs if o["num_det"] >= args.min_det]
    lines = []
    for i, o in enumerate(objs):
        x, y, z = o["center_w"]
        ext = o.get("extent") or [0, 0, 0]
        lines.append(f"- id{i} [{o['tag']}] 위치=({x:.1f}, {y:.1f}, {z:.1f})m"
                     f" 크기={ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f}m 관측={o['num_det']}회 ({o['log']})")
    if args.planner:
        # 원논문 planner 형식: scene JSON + 구조화된 JSON 응답 강제
        # merge 산출물이 있으면 그것(통합 객체+엣지 id 일치)을 사용
        import json as _json
        mp = os.path.join(OUT_DIR, "session_objects.json")
        ep = os.path.join(OUT_DIR, "session_edges.json")
        edges_txt = ""
        if os.path.exists(mp):
            mobjs = _json.load(open(mp))
            scene = [{"id": i, "object_tag": o["tag"], "bbox_center": o["center_w"],
                      "bbox_extent": o.get("extent") or [0, 0, 0],
                      "num_detections": o["num_det"], "logs": o["logs"]} for i, o in enumerate(mobjs)]
            if os.path.exists(ep):
                ee = _json.load(open(ep))
                lines = [f'({e["object_1_id"]}) {e["edge_description"]} ({e["object_2_id"]})' for e in ee]
                edges_txt = ("\n\nObject relations (geometric heuristic; ids refer to the scene JSON):\n"
                             + "\n".join(lines))
        else:
            scene = [{"id": i, "object_tag": o["tag"],
                      "bbox_center": o["center_w"],
                      "bbox_extent": o.get("extent") or [0, 0, 0],
                      "num_detections": o["num_det"], "log": o["log"]} for i, o in enumerate(objs)]
        prompt = f"""The input is a 3D scene described in a JSON format. Each entry describes one object:
"id", "object_tag", "bbox_center" (world meters), "bbox_extent", "num_detections", "log".
log_000 is an indoor office; log_002~010 are an outdoor riverside walkway.

{_json.dumps(scene, ensure_ascii=False)}{edges_txt}

For each user question, respond with a JSON dictionary with fields:
1. "inferred_query": your interpretation of the user query in a succinct form
2. "relevant_objects": list of relevant object ids (if applicable)
3. "query_achievable": whether the query is achievable with the objects in this scene
4. "final_relevant_objects": final relevant object ids, most relevant first
5. "explanation": brief explanation of the most relevant object(s), and how they achieve the task
6. "goal_xyz": if the robot should move, the target world coordinate

질문: {args.question}
"""
        out = os.path.join(OUT_DIR, "reason_prompt_planner.txt")
        os.makedirs(OUT_DIR, exist_ok=True)
        open(out, "w").write(prompt)
        print(prompt if args.echo else f"planner 프롬프트 저장: {out} (객체 {len(scene)}개, {len(prompt)}자)")
        return

    prompt = f"""당신은 사족보행 로봇의 공간 추론 어시스턴트입니다.
아래는 로봇이 탐사하며 만든 3D semantic map 의 객체 목록입니다.
좌표는 미터 단위 world 좌표 (x, y, z)이고, 로봇 시작점이 원점 부근입니다.
log_000 은 실내 사무실, log_002~010 은 야외 강변 산책로 구간입니다.

[객체 목록: {len(lines)}개]
{chr(10).join(lines)}

위 맵을 근거로 다음 질문에 답하세요. 반드시 객체 id 와 world 좌표를 근거로 들고,
로봇이 이동해야 한다면 목표 좌표를 제시하세요.

질문: {args.question}
"""
    out = os.path.join(OUT_DIR, "reason_prompt.txt")
    os.makedirs(OUT_DIR, exist_ok=True)
    open(out, "w").write(prompt)
    print(prompt if args.echo else f"프롬프트 저장: {out} (객체 {len(lines)}개, {len(prompt)}자)")


# ---------- 출력 ----------

def save_json(kind, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{kind}_{int(time.time())}.json")
    json.dump(obj, open(path, "w"), ensure_ascii=False, indent=1)
    return path


def save_grid_png(g, name):
    import cv2
    v = np.nan_to_num(g, nan=-1)
    lo, hi = np.nanpercentile(g, 5), np.nanpercentile(g, 99)
    img = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS))
    img[np.isnan(g)] = 32
    path = os.path.join(OUT_DIR, f"{name}.png")
    cv2.imwrite(path, img.transpose(1, 0, 2)[::-1])   # 가로=world x, 세로=world y(위로+)
    return path


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text"); t.add_argument("query"); t.add_argument("--topk", type=int, default=10)
    tr = sub.add_parser("terrain"); tr.add_argument("proto")
    eg = sub.add_parser("edges"); eg.add_argument("--next-to", type=float, default=1.5); eg.add_argument("--max-next", type=int, default=3)
    mg = sub.add_parser("merge"); mg.add_argument("--dist", type=float, default=1.5); mg.add_argument("--cos", type=float, default=0.75)
    cm = sub.add_parser("costmap"); cm.add_argument("--max-slope", type=float, default=0.35)
    pr = sub.add_parser("prompt"); pr.add_argument("question"); pr.add_argument("--min-det", type=int, default=2); pr.add_argument("--echo", action="store_true"); pr.add_argument("--planner", action="store_true")
    rg = sub.add_parser("region")
    for k in ("x0", "y0", "x1", "y1"):
        rg.add_argument(k, type=float)
    args = ap.parse_args()
    {"text": req_text, "terrain": req_terrain, "region": req_region, "prompt": req_prompt, "costmap": req_costmap, "merge": req_merge, "edges": req_edges}[args.cmd](args)


if __name__ == "__main__":
    main()
