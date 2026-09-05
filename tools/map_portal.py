#!/usr/bin/env python3
"""Map Portal — 빌드된 Map Package 들을 한 페이지에서 보고 관리하는 로컬 웹 서버 (9092).
- /                         포털 페이지 (세션 카드: 데이터셋·로그·통계·rerun 링크·영상 자동재생·빌드 로그)
- /api/maps                 output/*/compiled/map_manifest.json 스캔 → JSON
- /api/log/<name>           빌드 로그 tail
- 그 외 경로                저장소 파일 서빙 (CORS) — rerun 뷰어가 .rrd 를 여기서 받는다 (rrd_http.py 대체)
사용: python tools/map_portal.py [저장소 루트=이 파일의 상위] [포트=9092]
"""
import http.server, functools, json, os, sys, time, urllib.parse

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9092
OUT = os.path.join(ROOT, "output")
HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_portal.html"), encoding="utf-8").read()

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
        rel = lambda f: f"output/{name}/compiled/{f}" if os.path.exists(os.path.join(comp, f)) else None
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
        maps.append({"name": name, "quality": qual, "quality_rrd": rel("data_quality.rrd"), "bev": rel("bev.png"), "tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])), "edges_sample": edges_sample, "map_id": man.get("map_id", name), "built": man.get("built", ""), "frame": man.get("frame", ""),
                     "mcap_dir": man.get("source", {}).get("mcap_dir", ""), "logs": logs, "n_logs": len(logs), "calib": man.get("source", {}).get("calib", ""),
                     "options": man.get("options", {}), "files": man.get("files", {}), "package_mb": round(size / 1e6, 1),
                     "summary": summ, "rrd": rel("map_package.rrd"), "manifest": f"output/{name}/compiled/map_manifest.json",
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
            return self.send_json({"maps": scan_maps(), "root": ROOT, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        if p.startswith("/api/log/"):
            name = os.path.basename(p[len("/api/log/"):]); f = os.path.join(ROOT, "sessions", f"{name}_build.out")
            if not os.path.exists(f): return self.send_json({"error": "no log"}, 404)
            lines = open(f, errors="replace").read().splitlines()[-200:]
            return self.send_json({"name": name, "lines": lines})
        return super().do_GET()

print(f"Map Portal: http://localhost:{PORT}/  (serving {ROOT})", flush=True)
http.server.ThreadingHTTPServer(("0.0.0.0", PORT), functools.partial(H, directory=ROOT)).serve_forever()
