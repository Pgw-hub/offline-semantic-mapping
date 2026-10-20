#!/usr/bin/env python3
"""Map Portal — 빌드된 Map Package 들을 한 페이지에서 보고 관리하는 로컬 웹 서버 (9092).
- /                         포털 페이지 (세션 카드: 데이터셋·로그·통계·rerun 링크·영상 자동재생·빌드 로그)
- /api/maps                 output/*/compiled/map_manifest.json 스캔 → JSON
- /api/log/<name>           빌드 로그 tail
- 그 외 경로                저장소 파일 서빙 (CORS) — rerun 뷰어가 .rrd 를 여기서 받는다 (rrd_http.py 대체)
사용: python tools/map_portal.py [저장소 루트=이 파일의 상위] [포트=9092]
"""
import http.server, functools, glob, json, os, re, subprocess, sys, threading, time, urllib.parse

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9092
OUT = os.path.join(ROOT, "output"); SESS = os.path.join(ROOT, "sessions")
DATASET_ROOTS = [p for p in os.environ.get("DATASET_ROOTS", "/data2").split(":") if p]
CONTAINER = os.environ.get("BUILD_CONTAINER", "offline_3d_semantic_map")
TOKEN = os.environ.get("PORTAL_TOKEN", "")            # 설정되면 빌드/취소 POST 에 X-Portal-Token 헤더 필요
JOBS_FILE = os.path.join(SESS, "jobs.json")
STEPS = "Q,A,S,R,B,C,D,E,F,G,H,V".split(",")
OPTION_SPEC = {   # 웹 폼에서 받는 옵션: (타입, 기본값, 범위)
    "lidar_fusion": ("bool", True, None), "mask_bottom_px": ("int", 60, (0, 200)), "drop_classes": ("str", "person", None),
    "vlm_edges": ("bool", True, None), "vlm_max_pairs": ("int", 30, (0, 200)),
    "occupancy_res": ("float", 0.2, (0.1, 0.5)), "prob_hit": ("float", 0.85, (0.5, 0.99)), "prob_miss": ("float", 0.45, (0.05, 0.5)),
    "h_ground": ("int", 2, (1, 10)), "h_obs": ("int", 3, (1, 10)), "clearance": ("float", 1.4, (0.5, 3.0)), "band_min": ("float", 0.3, (0.1, 1.0)),
    "view_cubes": ("bool", False, None), "view_coarse": ("float", 0.3, (0.1, 1.0)), "video_speed": ("int", 15, (0, 60)), "video_pipeline": ("bool", True, None),
    "classdict": ("str", "default", None), "label_edge_px": ("int", 4, (0, 12)), "require_depth": ("str", "true", None),
    "geometry": ("str", "keyframe_scans", None),   # keyframe_scans(기본, 트랙 2) | all_scans | pcd(로봇 저장 지도 그대로) | raw_scans(기존 트랙 1)
    "pose_graph": ("bool", False, None), "lidar_extrinsic": ("str", "calib", None), "livox_flip": ("bool", True, None),
    "depth_fill_range": ("float", 4.0, (0.0, 10.0)), "depth_fill_gap": ("float", 0.15, (0.05, 0.5)),
    "voxel3d": ("bool", True, None)}   # 보행용 3D 확률 복셀 사전맵(build_voxel3d.py) 생성 여부
GEOMETRIES = ("keyframe_scans", "all_scans", "pcd", "raw_scans")
def find_robot_map(dp):
    """데이터셋 폴더 안 로봇 저장 지도 (map_*/ 또는 하위 폴더의 <name>.pcd + <name>.txt 짝) → {"pcd","traj","name"} 또는 None"""
    cands = []
    for sub in [dp] + [os.path.join(dp, d) for d in sorted(os.listdir(dp)) if os.path.isdir(os.path.join(dp, d)) and d not in ("raisin_config",)]:
        for f in os.listdir(sub):
            if f.endswith(".pcd") and not f.endswith("_gt.pcd") and os.path.exists(os.path.join(sub, f[:-4] + ".txt")): cands.append(os.path.join(sub, f))
    if not cands:                                             # 지도 폴더가 데이터셋 옆(형제 폴더)에 있는 경우 (예: /data2/20260915/260915_test)
        par = os.path.dirname(dp)
        for d in sorted(os.listdir(par)) if os.path.isdir(par) else []:
            sub = os.path.join(par, d)
            if sub == dp or not os.path.isdir(sub) or any(re.fullmatch(r"log_\d{3}\.mcap", f) for f in os.listdir(sub)): continue
            for f in os.listdir(sub):
                if f.endswith(".pcd") and not f.endswith("_gt.pcd") and os.path.exists(os.path.join(sub, f[:-4] + ".txt")): cands.append(os.path.join(sub, f))
    if not cands: return None
    p = sorted(cands, key=os.path.getmtime)[-1]; return {"pcd": p, "traj": p[:-4] + ".txt", "name": os.path.basename(p)[:-4], "sibling": os.path.dirname(p).startswith(dp) is False}
HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_portal.html"), encoding="utf-8").read()
_pub = {"t": 0.0, "v": ""}
def public_url():
    """외부 접속 주소: PORTAL_PUBLIC_URL 환경변수, 없으면 cloudflared quick tunnel 로그에서 최신 주소 (프로세스가 살아 있을 때만)."""
    if os.environ.get("PORTAL_PUBLIC_URL"): return os.environ["PORTAL_PUBLIC_URL"]
    if time.time() - _pub["t"] < 30: return _pub["v"]
    url = ""
    try:
        alive = subprocess.run(["pgrep", "-f", f"cloudflared tunnel .*{PORT}"], capture_output=True).returncode == 0
        log = os.environ.get("PORTAL_TUNNEL_LOG", f"/tmp/cf_quick_{PORT}.log")
        if alive and os.path.exists(log):
            m = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log, errors="replace").read())
            if m: url = m[-1]
    except Exception: pass
    _pub.update(t=time.time(), v=url); return url
sys.path.insert(0, ROOT); import classdict as CD
LABELS = {space: CD.labels_for(space) for space in {m["labels"] for m in CD.MODELS.values()}}   # 라벨 공간별 목록 (편집기 검색용): classdicts/<space>_labels.json
ADE_LABELS = LABELS.get("ade20k", [])

_ds_cache = {"t": 0, "v": []}
def scan_datasets():
    """DATASET_ROOTS 아래(깊이 ≤4)에서 log_*.mcap 이 있는 폴더 = 데이터셋"""
    if time.time() - _ds_cache["t"] < 60: return _ds_cache["v"]
    out = []
    for root in DATASET_ROOTS:
        if not os.path.isdir(root): continue
        base_depth = root.rstrip("/").count("/")
        for dp, dn, fn in os.walk(root):
            if dp.count("/") - base_depth >= 4 or os.path.basename(dp).startswith(("_", ".")) or "/output/" in dp + "/" or dp.startswith(OUT):
                dn[:] = []; continue
            logs = sorted(f for f in fn if re.fullmatch(r"log_\d{3}\.mcap", f))
            if not logs: continue
            sizes = [os.path.getsize(os.path.join(dp, f)) for f in logs]; nums = [int(f[4:7]) for f in logs]
            mt = max(os.path.getmtime(os.path.join(dp, f)) for f in logs)
            out.append({"path": dp, "name": os.path.basename(dp), "parent": os.path.basename(os.path.dirname(dp)), "n_logs": len(logs), "log_min": nums[0], "log_max": nums[-1],
                        "contiguous": nums == list(range(nums[0], nums[-1] + 1)), "size_gb": round(sum(sizes) / 1e9, 2), "recorded": time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)),
                        "has_calib": os.path.isdir(os.path.join(dp, "raisin_config", "calibration")), "robot_map": find_robot_map(dp)})
            dn[:] = []
    out.sort(key=lambda d: d["recorded"], reverse=True); _ds_cache.update(t=time.time(), v=out); return out

class Jobs:
    """빌드 작업 큐: 한 번에 하나만 (GPU 공유). jobs.json 에 영속."""
    def __init__(self):
        self.lock = threading.Lock(); self.jobs = []; self.load()
        threading.Thread(target=self.worker, daemon=True).start()
    def load(self):
        try: self.jobs = json.load(open(JOBS_FILE))
        except Exception: self.jobs = []
        for j in self.jobs:            # 포털 재시작으로 추적을 잃은 작업 정리
            if j["status"] in ("running", "queued"): j["status"] = "unknown"; j["note"] = "포털 재시작으로 추적 끊김 — 로그 확인"
        self.save()
    def save(self):
        os.makedirs(SESS, exist_ok=True); json.dump(self.jobs, open(JOBS_FILE, "w"), ensure_ascii=False, indent=1)
    def add(self, name, yaml_path, spec):
        with self.lock:
            if any(j["name"] == name and j["status"] in ("queued", "running") for j in self.jobs): raise ValueError("같은 이름의 작업이 이미 진행 중")
            self.jobs.append({"name": name, "yaml": yaml_path, "spec": spec, "status": "queued", "queued_at": time.time(), "started_at": None, "ended_at": None, "rc": None, "pid": None}); self.save()
    def cancel(self, name):
        with self.lock:
            for j in self.jobs:
                if j["name"] == name and j["status"] == "queued": j["status"] = "cancelled"; j["ended_at"] = time.time(); self.save(); return True
                if j["name"] == name and j["status"] == "running" and j.get("pid"):
                    subprocess.run(["docker", "exec", CONTAINER, "pkill", "-f", f"build_map_package.py session_{name}.yaml"]); j["status"] = "cancelled"; j["ended_at"] = time.time(); self.save(); return True
        return False
    def worker(self):
        while True:
            job = None
            with self.lock:
                if not any(j["status"] == "running" for j in self.jobs):
                    job = next((j for j in self.jobs if j["status"] == "queued"), None)
                    if job: job["status"] = "running"; job["started_at"] = time.time(); self.save()
            if not job: time.sleep(2); continue
            log = os.path.join(SESS, f"{job['name']}_build.out")
            cmd = ["docker", "exec", CONTAINER, "bash", "-lc", f"cd {SESS} && python build_map_package.py {os.path.basename(job['yaml'])}"]
            with open(log, "a") as lf:
                lf.write(f"===== WEB BUILD {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"); lf.flush()
                p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
                with self.lock: job["pid"] = p.pid; self.save()
                rc = p.wait()
            with self.lock:
                if job["status"] == "running": job["status"] = "done" if rc == 0 else "failed"
                job["rc"] = rc; job["ended_at"] = time.time(); self.save()
    def view(self):
        out = []
        for j in self.jobs[::-1]:
            v = dict(j); log = os.path.join(SESS, f"{j['name']}_build.out"); step = None
            if os.path.exists(log):
                try:
                    txt = open(log, errors="replace").read()[-20000:]
                    m = re.findall(r"===== 단계 (\w) =====", txt); step = m[-1] if m else None
                except Exception: pass
            v["step"] = step; v["step_index"] = STEPS.index(step) + 1 if step in STEPS else 0; v["n_steps"] = len(STEPS)
            v["elapsed_s"] = round((v["ended_at"] or time.time()) - v["started_at"]) if v.get("started_at") else 0
            v["map_ready"] = os.path.exists(os.path.join(OUT, j["name"], "compiled", "map_manifest.json"))
            out.append(v)
        return out
JOBS = None

def make_yaml(spec):
    """웹 폼 → session_<name>.yaml (검증 포함). 반환: (yaml_path, 정규화된 spec)"""
    name = str(spec.get("name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,48}", name): raise ValueError("이름은 영문/숫자/_/- 2~49자")
    ds = next((d for d in scan_datasets() if d["path"] == spec.get("dataset")), None)
    if not ds: raise ValueError("데이터셋을 찾을 수 없음")
    a, b = int(spec.get("log_from", ds["log_min"])), int(spec.get("log_to", ds["log_max"]))
    if not (ds["log_min"] <= a <= b <= ds["log_max"]): raise ValueError("로그 범위가 데이터셋을 벗어남")
    if os.path.exists(os.path.join(OUT, name, "compiled", "map_manifest.json")) and not spec.get("overwrite"): raise ValueError("같은 이름의 맵이 이미 있음 (다른 이름을 쓰거나 덮어쓰기 선택)")
    opts = {}
    for k, (typ, dflt, rng) in OPTION_SPEC.items():
        v = spec.get("options", {}).get(k, dflt)
        try:
            v = {"bool": lambda x: bool(x) if not isinstance(x, str) else x.lower() in ("1", "true", "on", "yes"), "int": int, "float": float, "str": str}[typ](v)
        except Exception: raise ValueError(f"옵션 {k} 값이 잘못됨")
        if rng and not (rng[0] <= v <= rng[1]): raise ValueError(f"옵션 {k} 는 {rng[0]}~{rng[1]} 범위")
        opts[k] = v
    y = {"root": name, "mcap_dir": ds["path"], "logs": list(range(a, b + 1)),
         "options": {"map_id": spec.get("map_id") or name, "lidar_fusion": opts["lidar_fusion"], "mask_bottom_px": opts["mask_bottom_px"], "drop_classes": opts["drop_classes"],
                     "vlm_edges": opts["vlm_edges"], "vlm_max_pairs": opts["vlm_max_pairs"], "occupancy_res": opts["occupancy_res"], "prob_hit": opts["prob_hit"], "prob_miss": opts["prob_miss"],
                     "costmap": {"h_ground": opts["h_ground"], "h_obs": opts["h_obs"], "clearance": opts["clearance"], "band_min": opts["band_min"]},
                     "view_cubes": opts["view_cubes"], "view_coarse": opts["view_coarse"], "video_speed": opts["video_speed"], "video_pipeline": opts["video_pipeline"],
                     "classdict": opts["classdict"], "label_edge_px": opts["label_edge_px"], "require_depth": opts["require_depth"], "pose_graph": opts["pose_graph"]}}
    if opts["classdict"] != "default" and not os.path.exists(os.path.join(CD.DIR, f"{opts['classdict']}.json")): raise ValueError(f"클래스 사전 '{opts['classdict']}' 없음")
    if opts["geometry"] not in GEOMETRIES: raise ValueError(f"geometry 는 {GEOMETRIES} 중 하나")
    if opts["lidar_extrinsic"] not in ("calib", "slam"): raise ValueError("lidar_extrinsic 은 calib | slam")
    if str(opts["require_depth"]).lower() not in ("true", "false", "lidar"): raise ValueError("require_depth 는 true | false | lidar")
    y["options"].update({"lidar_extrinsic": opts["lidar_extrinsic"], "livox_flip": opts["livox_flip"], "depth_fill_range": opts["depth_fill_range"], "depth_fill_gap": opts["depth_fill_gap"]})
    if opts["geometry"] != "raw_scans":                          # 트랙 2: 로봇 저장 지도(.pcd/.txt) 필요
        rm = ds.get("robot_map")
        if not rm: raise ValueError("이 데이터셋에 로봇 저장 지도(<name>.pcd + <name>.txt, 보통 map_*/ 폴더)가 없음 — geometry=raw_scans(기존 방식) 로 빌드하거나 지도 폴더를 넣어 주세요")
        y["options"]["robot_map"] = {"pcd": rm["pcd"], "traj": rm["traj"], "geometry": opts["geometry"]}; y["options"]["pose_graph"] = True
    import yaml as _y
    path = os.path.join(SESS, f"session_{name}.yaml"); os.makedirs(SESS, exist_ok=True)
    open(path, "w", encoding="utf-8").write(f"# 웹 콘솔에서 생성 {time.strftime('%Y-%m-%d %H:%M')} · 데이터셋 {ds['name']}\n" + _y.safe_dump(y, allow_unicode=True, sort_keys=False))
    return path, {"name": name, "dataset": ds["path"], "dataset_name": ds["name"], "logs": [a, b], "options": y["options"]}

def scan_maps():
    maps = []
    if not os.path.isdir(OUT): return maps
    for name in sorted(os.listdir(OUT)):
        comp = os.path.join(OUT, name, "compiled"); man_p = os.path.join(comp, "map_manifest.json")
        if not os.path.exists(man_p): continue
        try: man = json.load(open(man_p))
        except Exception: continue
        summ = {}
        sp = os.path.join(comp, "summary.json")
        if os.path.exists(sp):
            try: summ = json.load(open(sp))
            except Exception: pass
        rel = lambda f: (f"output/{name}/compiled/{f}" if f and os.path.isfile(os.path.join(comp, f)) else None)   # f="" 면 compiled 디렉터리가 잡혀 빈 <video> 가 뜨던 버그
        vids = sorted(f for f in os.listdir(comp) if f.endswith(".mp4"))
        logs = man.get("source", {}).get("logs", [])
        size = sum(v.get("bytes", 0) for v in man.get("files", {}).values())
        build_out = os.path.join(ROOT, "sessions", f"{name}_build.out")
        qual = {}
        qp = os.path.join(comp, "data_quality.json")
        if os.path.exists(qp):
            try:
                q = json.load(open(qp)); tp = q.get("topics", {})
                qual = {"duration_s": q.get("duration_s"), "n_gaps": len(q.get("gaps", [])), "missing_total": sum(v.get("missing_est", 0) for v in tp.values()),
                        "topics": {k: {"count": v["count"], "expected": v["expected"], "missing_est": v["missing_est"], "rate_hz": v["rate_hz"], "dt_max_ms": v["dt_max_ms"], "bytes_mb": v["bytes_mb"], "n_gaps": v["n_gaps"],
                                       "late_start": [c["log"] for c in v.get("coverage", []) if (c.get("start_delay_s") or 0) > 1.0]} for k, v in tp.items()},
                        "log_gaps_s": q.get("log_gaps_s", [])}
            except Exception: pass
        tags = {}; edges_sample = []; classes = {}
        try:
            for o in json.load(open(os.path.join(comp, "session_objects.json"))):
                t = o.get("tag") or "?"; tags[t] = tags.get(t, 0) + 1
        except Exception: pass
        for fn, src in (("session_edges_vlm.json", "vlm"), ("voxel_session_relations.json", "terrain"), ("session_edges.json", "heuristic")):
            try:
                for e in json.load(open(os.path.join(comp, fn))):
                    if src == "vlm": d = f"{e['object_1']['tag']} {e['relation']} {e['object_2']['tag']}"; c = e.get("confidence")
                    elif src == "terrain": d = f"{e.get('tag','?')} on {e.get('on','?')}"; c = None
                    else: d = e.get("edge_description", ""); c = None
                    edges_sample.append({"src": src, "desc": d, "conf": c})
            except Exception: pass
        edges_sample = edges_sample[:40]
        if os.path.exists(qp):
            try: qual["log_bounds"] = json.load(open(qp)).get("log_bounds", [])
            except Exception: pass
        graph = None; gp = os.path.join(OUT, name, "graph", "graph.json")
        if os.path.exists(gp):
            try:
                import numpy as _np
                g = json.load(open(gp)); To = _np.array([n["T_odom"] for n in g["nodes"]])[:, :3, 3]; Tc = _np.array([n["T_corr"] for n in g["nodes"]])[:, :3, 3]
                sh = _np.linalg.norm(Tc - To, axis=1); ev = os.path.join(OUT, name, "graph", "eval_heldout.json")
                graph = {"source": g.get("source"), "keyframes": len(g["nodes"]), "loops": g.get("loops", 0), "shift_mean_m": round(float(sh.mean()), 3), "shift_max_m": round(float(sh.max()), 3),
                         "rrd": f"output/{name}/graph/graph_view.rrd" if os.path.exists(os.path.join(OUT, name, "graph", "graph_view.rrd")) else None,
                         "eval": json.load(open(ev)) if os.path.exists(ev) else None, "eval_rrd": f"output/{name}/graph/eval.rrd" if os.path.exists(os.path.join(OUT, name, "graph", "eval.rrd")) else None}
            except Exception as ex: graph = {"error": str(ex)}
        maps.append({"name": name, "quality": qual, "graph": graph, "quality_rrd": rel("data_quality.rrd"), "bev": rel("bev.png"), "tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])), "edges_sample": edges_sample, "map_id": man.get("map_id", name), "built": man.get("built", ""), "frame": man.get("frame", ""),
                     "mcap_dir": man.get("source", {}).get("mcap_dir", ""), "logs": logs, "n_logs": len(logs), "calib": man.get("source", {}).get("calib", ""),
                     "options": man.get("options", {}), "files": man.get("files", {}), "package_mb": round(size / 1e6, 1), "geometry_source": man.get("geometry_source", "raw scans"),
                     "summary": summ, "rrd": rel("map_package.rrd"), "manifest": f"output/{name}/compiled/map_manifest.json",
                     "voxel3d": man.get("voxel3d"), "voxel3d_rrd": rel("voxel3d_view.rrd"), "potree": (f"output/{name}/compiled/potree/index.html" if os.path.exists(os.path.join(comp, "potree", "index.html")) else None),   # tools/potree/potree_export.py
                     "voxel3d_variants": sorted({os.path.basename(f)[len("session_voxel3d_"):-len(".npz")] for f in glob.glob(os.path.join(comp, "session_voxel3d_*.npz")) if not os.path.basename(f).startswith("eval_") and not f.endswith("_summary.npz")}),   # 예: ["iso025"]
                     "voxel3d_variant_rrds": {os.path.basename(f)[len("voxel3d_"):-len("_view.rrd")]: f"output/{name}/compiled/{os.path.basename(f)}" for f in glob.glob(os.path.join(comp, "voxel3d_*_view.rrd"))}, "voxel3d_bev": rel("voxel3d_bev.png"), "voxel3d_tiles": (f"output/{name}/compiled/session_voxel3d_tiles/index.json" if os.path.exists(os.path.join(comp, "session_voxel3d_tiles", "index.json")) else None),
                     "voxel3d_eval": (json.load(open(os.path.join(comp, "voxel3d_eval.json"))) if os.path.exists(os.path.join(comp, "voxel3d_eval.json")) else None),
                     "videos": [f"output/{name}/compiled/{v}" for v in vids], "video_pipeline": rel(next((v for v in vids if "pipeline" in v), "")),
                     "video_rgb": rel(next((v for v in vids if "rgb" in v), "")),
                     "has_log": os.path.exists(build_out), "mtime": os.path.getmtime(man_p)})
    maps.sort(key=lambda m: -m["mtime"])
    return maps

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Cache-Control", "no-store"); super().end_headers()
    def log_message(self, fmt, *args): pass
    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode(); self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); p = u.path
        if p in ("/", "/index.html", "/portal"):
            b = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if p == "/api/maps":
            return self.send_json({"maps": scan_maps(), "root": ROOT, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "rerun_base": os.environ.get("RERUN_BASE", ""), "public_url": public_url(), "build_enabled": True, "token_required": bool(TOKEN),
                                   "option_spec": {k: {"type": t, "default": d, "range": r} for k, (t, d, r) in OPTION_SPEC.items()}})
        if p == "/api/datasets":
            maps = scan_maps(); ds = scan_datasets()
            for d in ds: d["maps"] = [m["name"] for m in maps if m.get("mcap_dir") == d["path"]]
            return self.send_json({"datasets": ds, "roots": DATASET_ROOTS})
        if p == "/api/jobs":
            return self.send_json({"jobs": JOBS.view(), "container": CONTAINER})
        if p == "/api/classdicts":
            return self.send_json({"classdicts": CD.list_all(), "roles": CD.ROLES, "role_cost": CD.ROLE_GROUND_COST, "ade_labels": ADE_LABELS,
                                   "models": CD.MODELS, "default_model": CD.DEFAULT_MODEL, "labels": LABELS})
        if p.startswith("/api/maps/") and p.endswith("/raisin"):           # raisin Load Map 용 zip: <맵이름>/<맵이름>.pcd + semantic/
            name = os.path.basename(p[len("/api/maps/"):-len("/raisin")]); q = urllib.parse.parse_qs(u.query)
            man = os.path.join(OUT, name, "compiled", "map_manifest.json")
            if not os.path.exists(man): return self.send_json({"error": "no map"}, 404)
            mname = q.get("map_name", [name])[0]; road = int(q.get("road", ["90"])[0]); cwx = q.get("cwext", ["0"])[0] == "1"
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", mname): return self.send_json({"error": "맵 이름은 영문/숫자/_/- 만"}, 400)
            if not (0 <= road <= 100): return self.send_json({"error": "road 0~100"}, 400)
            exp = os.path.join(OUT, name, "compiled", "export"); os.makedirs(exp, exist_ok=True); out = os.path.join(exp, f"{mname}_raisin_r{road}" + ("_cwext" if cwx else "") + ".zip")
            dep = os.path.join(OUT, name, "compiled", "session_costmap_occ.npz")
            if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(dep):
                try:
                    import export_pcd; export_pcd.export_raisin_map_zip(man, out, mname, road, cwx)
                except Exception as e: return self.send_json({"error": f"raisin 맵 생성 실패: {e}"}, 500)
            return self.send_json({"ok": True, "path": f"output/{name}/compiled/export/{os.path.basename(out)}", "bytes": os.path.getsize(out)})
        if p.startswith("/api/maps/") and p.endswith("/pcd"):
            name = os.path.basename(p[len("/api/maps/"):-len("/pcd")]); q = urllib.parse.parse_qs(u.query); src = q.get("source", ["voxels"])[0]
            if src not in ("voxels", "voxels_occ", "occupancy", "voxel3d", "voxel3d_tiles"): return self.send_json({"error": "source"}, 400)
            man = os.path.join(OUT, name, "compiled", "map_manifest.json")
            if not os.path.exists(man): return self.send_json({"error": "no map"}, 404)
            variant = re.sub(r"[^A-Za-z0-9_]", "", q.get("variant", [""])[0])[:32]; vsuf = ("_" + variant) if variant else ""   # voxel3d 변형 (예: iso025 = 2.5 cm 정육면체)
            if src == "voxel3d_tiles":                                          # 3D prior 타일 묶음 (zip) — 로봇 적재용
                idx = os.path.join(OUT, name, "compiled", f"session_voxel3d{vsuf}_tiles", "index.json")
                if not os.path.exists(idx): return self.send_json({"error": "이 지도에는 3D prior(voxel3d) 가 없습니다 (F 단계 옵션 voxel3d)"}, 404)
                exp = os.path.join(OUT, name, "compiled", "export"); os.makedirs(exp, exist_ok=True); out = os.path.join(exp, f"{name}_voxel3d{vsuf}_tiles.zip")
                if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(idx):
                    try:
                        import export_pcd; export_pcd.export_voxel3d_zip(man, out, variant)
                    except Exception as e: return self.send_json({"error": f"zip 생성 실패: {e}"}, 500)
                rel = f"output/{name}/compiled/export/{name}_voxel3d{vsuf}_tiles.zip"
                return self.send_json({"ok": True, "path": rel, "legend": None, "bytes": os.path.getsize(out)})
            exclude = tuple(x for x in q.get("exclude", [""])[0].split(",") if x); min_nsem = int(q.get("min_nsem", ["0"])[0]); occf = q.get("occ", [""])[0]; occ_filter = None if occf == "" else occf == "1"
            fmt = q.get("format", ["rgb"])[0]
            if fmt not in ("rgb", "robot"): return self.send_json({"error": "format"}, 400)
            import hashlib; tag = hashlib.md5(f"{src}|{variant}|{','.join(sorted(exclude))}|{min_nsem}|{occ_filter}|{fmt}".encode()).hexdigest()[:8]
            fname = f"{name}.pcd" if q.get("robot_name", ["0"])[0] == "1" else (f"{name}_{src}{vsuf}" + (f"_no-{'-'.join(sorted(exclude))}" if exclude else "") + (f"_n{min_nsem}" if min_nsem else "") + ("" if occ_filter is None else ("_occ" if occ_filter else "_noocc")) + (f"_{fmt}" if fmt != "rgb" else "") + f"_{tag}.pcd")
            exp = os.path.join(OUT, name, "compiled", "export"); os.makedirs(exp, exist_ok=True); out = os.path.join(exp, fname)
            dep = os.path.join(OUT, name, "compiled", f"session_voxel3d{vsuf}.npz") if src == "voxel3d" else man
            if src == "voxel3d" and not os.path.exists(dep): return self.send_json({"error": f"voxel3d 변형 '{variant or 'default'}' 파일이 없습니다"}, 404)
            if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(dep):
                try:
                    import export_pcd; export_pcd.export(man, out, src, exclude, min_nsem, occ_filter, fmt, variant)
                except Exception as e: return self.send_json({"error": f"PCD 생성 실패: {e}"}, 500)
            rel = f"output/{name}/compiled/export/{fname}"
            return self.send_json({"ok": True, "path": rel, "legend": rel[:-4] + "_legend.json", "bytes": os.path.getsize(out)})
        if p.startswith("/api/log/"):
            name = os.path.basename(p[len("/api/log/"):]); f = os.path.join(ROOT, "sessions", f"{name}_build.out")
            if not os.path.exists(f): return self.send_json({"error": "no log"}, 404)
            lines = open(f, errors="replace").read().splitlines()[-200:]
            return self.send_json({"name": name, "lines": lines})
        return self.serve_static()
    def serve_static(self):
        """정적 파일 + HTTP Range(206). Potree 2.0 뷰어는 octree.bin 을 노드마다 바이트 범위로 읽는다 (기본 핸들러는 Range 를 무시하고 전체를 보냄)."""
        rng = self.headers.get("Range"); path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip()) if rng else None
        if not m or not os.path.isfile(path) or (m.group(1) == "" and m.group(2) == ""): return super().do_GET()
        size = os.path.getsize(path)
        if m.group(1) == "": start, end = max(0, size - int(m.group(2))), size - 1
        else: start, end = int(m.group(1)), min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        if start > end or start >= size:
            self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers(); return
        self.send_response(206); self.send_header("Content-Type", self.guess_type(path)); self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}"); self.send_header("Content-Length", str(end - start + 1)); self.end_headers()
        with open(path, "rb") as f:
            f.seek(start); left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk: break
                self.wfile.write(chunk); left -= len(chunk)
    def do_POST(self):
        u = urllib.parse.urlparse(self.path); p = u.path
        if TOKEN and self.headers.get("X-Portal-Token", "") != TOKEN: return self.send_json({"error": "토큰 필요"}, 401)
        n = int(self.headers.get("Content-Length", "0") or 0); body = {}
        if n:
            try: body = json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception: return self.send_json({"error": "JSON 파싱 실패"}, 400)
        if p == "/api/build":
            try:
                path, spec = make_yaml(body); JOBS.add(spec["name"], path, spec)
            except ValueError as e: return self.send_json({"error": str(e)}, 400)
            except Exception as e: return self.send_json({"error": f"내부 오류: {e}"}, 500)
            return self.send_json({"ok": True, "name": spec["name"], "yaml": os.path.relpath(path, ROOT), "spec": spec})
        if p.startswith("/api/jobs/") and p.endswith("/cancel"):
            name = p[len("/api/jobs/"):-len("/cancel")]
            return self.send_json({"ok": JOBS.cancel(name)})
        if p == "/api/classdicts":
            try:
                path = CD.save(body); return self.send_json({"ok": True, "path": os.path.relpath(path, ROOT), "classdict": CD.load(body["name"])})
            except ValueError as e: return self.send_json({"error": str(e)}, 400)
            except Exception as e: return self.send_json({"error": f"내부 오류: {e}"}, 500)
        if p.startswith("/api/classdicts/") and p.endswith("/delete"):
            name = os.path.basename(p[len("/api/classdicts/"):-len("/delete")])
            if name == "default": return self.send_json({"error": "default 는 삭제 불가"}, 400)
            f = os.path.join(CD.DIR, f"{name}.json")
            if os.path.exists(f): os.remove(f)
            return self.send_json({"ok": True})
        if p == "/api/datasets/rescan":
            _ds_cache["t"] = 0; return self.send_json({"datasets": scan_datasets()})
        return self.send_json({"error": "unknown"}, 404)

JOBS = Jobs()
print(f"Map Portal: http://localhost:{PORT}/  (serving {ROOT}; datasets {DATASET_ROOTS}; builds in container {CONTAINER})", flush=True)
http.server.ThreadingHTTPServer(("0.0.0.0", PORT), functools.partial(H, directory=ROOT)).serve_forever()
