#!/usr/bin/env python3
"""대화형 초기 위치 잡기 서버 (컨테이너 안, :9093). 지도 점군을 메모리에 들고, 클릭한 (x, y, yaw) 에서 스캔을 ICP 로 정합해 준다.

GET  /api/loc/info                      지도 bbox · BEV 메타(origin, res, 크기) · 스캔 목록(정답 자세 포함) · 소스
GET  /api/loc/bev.png                   정합용 지도 점군의 상단 뷰 (회색, 높이 밝기)
GET  /api/loc/scan?id=<i>               스캔 점(로봇 프레임, 다운샘플) + 정답 자세 {x,y,z,yaw}
POST /api/loc/localize {id,x,y,yaw_deg,multi_yaw}  → 단계별 자세, 최종 자세, fitness/rmse, 정답 대비 오차, ms
POST /api/loc/source {source}           지도 소스 교체 (voxels | voxels_occ | occupancy)
POST /api/loc/rrd {id,x,y,yaw_deg}      정합 결과 rrd 저장 → 경로 (포털 9092 가 서빙)
사용: python tools/localize_server.py <manifest> [--scans /data2/cg/data/raibo] [--logs 6,7,8,9,10] [--port 9093]
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # 저장소 루트 (패키지 import)
import argparse, glob, http.server, io, json, math, os, sys, threading, time, urllib.parse
import numpy as np, cv2
from package.load_map import MapPackage
from apps.localize import map_points, Localizer, yaw_T, pose_err, load_scans

ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("--scans", default="/data2/cg/data/raibo"); ap.add_argument("--logs", default="6,7,8,9,10")
ap.add_argument("--accum", type=int, default=5); ap.add_argument("--port", type=int, default=9093); ap.add_argument("--source", default="voxels_occ"); ap.add_argument("--exclude", default="person,car")
ap.add_argument("--stride", type=int, default=10, help="후보 스캔 간격")
a = ap.parse_args()
LOCK = threading.Lock()
STATE = {"source": a.source}


def build(source, manifest=None, logs=None):
    manifest = manifest or a.manifest; logs = logs or a.logs
    mp = MapPackage(manifest); ex = tuple(x for x in a.exclude.split(",") if x)       # 실패하면 여기서 예외 → 아래 상태 변경 없음
    a.manifest, a.logs = manifest, logs
    in_map = set(mp.man.get("source", {}).get("logs", []))
    pts = map_points(mp, source, ex); t = time.time()
    import open3d as o3d
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
    STATE.update(mp=mp, pts=pts, tree=tree, source=source, bbox=[pts.min(0).tolist(), pts.max(0).tolist()], n_pts=int(len(pts)), built_s=round(time.time() - t, 1))
    # BEV: 0.2m 격자, 열의 최고 z 를 밝기로
    R = 0.2; K = np.floor(pts[:, :2] / R).astype(np.int64); k0 = K.min(0); K -= k0; W, H = int(K[:, 0].max()) + 1, int(K[:, 1].max()) + 1
    order = np.argsort(pts[:, 2]); flat = K[order, 0] * H + K[order, 1]; zt = np.full(W * H, np.nan); zt[flat] = pts[order, 2]
    zt = zt.reshape(W, H).T; img = np.full((H, W), 28, np.uint8); m = np.isfinite(zt)
    if m.any():
        lo, hi = np.nanpercentile(zt, 2), np.nanpercentile(zt, 98); img[m] = (60 + 170 * np.clip((zt[m] - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)
    img = np.ascontiguousarray(img[::-1])                       # 위가 +y
    ok, png = cv2.imencode(".png", img); STATE["bev_png"] = png.tobytes()
    STATE["bev"] = {"x0": float(k0[0] * R), "y0": float(k0[1] * R), "res": R, "w": W, "h": H}     # 픽셀(u,v) → x = x0+(u+0.5)res, y = y0+(H-1-v+0.5)res
    # 스캔 후보 (보류 로그, 지도 범위 안, 커버리지)
    scans = []
    for lg in [int(x) for x in str(logs).split(",") if str(x).strip()]:
        files = sorted(glob.glob(os.path.join(a.scans, f"log_{lg:03d}", "lidar_scans", "*.npz")))
        for i in range(0, len(files) - a.accum, a.stride):
            P, org = load_scans(files, a.accum, i)
            lo_, hi_ = pts.min(0), pts.max(0)
            if not (np.all(org[:2] > lo_[:2] + 3) and np.all(org[:2] < hi_[:2] - 3)): continue
            Q = P[:: max(1, len(P) // 1500)]
            near = sum(1 for q in Q if tree.search_radius_vector_3d(q, 0.3)[0] > 0); ov = near / len(Q)
            if ov < 0.4: continue
            scans.append({"id": f"log_{lg:03d}_{i:06d}", "log": f"log_{lg:03d}", "idx": i, "files": files, "origin": org.tolist(), "overlap": round(ov, 2), "n_pts": int(len(P)),
                          "in_map": f"log_{lg:03d}" in in_map})
    STATE["scans"] = scans; STATE["scan_by_id"] = {s["id"]: s for s in scans}; STATE["in_map_logs"] = list(in_map)
    print(f"[loc] source={source} pts={len(pts):,} scans={len(scans)} (kd-tree {STATE['built_s']}s)", flush=True)


def scan_local(s):
    P, org = load_scans(s["files"], a.accum, s["idx"]); return P - org, org


def localize(s, x, y, yaw_deg, multi_yaw=False, crop_r=40.0):
    """스캔 s 를 (x,y,yaw) 초기 자세에서 정합. 지도는 초기 자세 주변 crop_r 안만 사용 (대화형 속도)."""
    import open3d as o3d
    P_local, org = scan_local(s); T_gt = yaw_T(org[0], org[1], 0, org[2]); T_init = yaw_T(x, y, math.radians(yaw_deg), z=org[2])
    pts = STATE["pts"]; d = np.hypot(pts[:, 0] - x, pts[:, 1] - y); sub = pts[d < crop_r]
    if len(sub) < 2000: return {"error": f"초기 위치 주변 {crop_r:.0f} m 안에 지도 점이 너무 적음 ({len(sub)})"}
    t0 = time.time(); loc = Localizer(sub)
    stages = []; T = T_init.copy()
    def run(Ti):
        Tc = Ti.copy(); st = []
        src_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_local))
        for (m, dist, vs), it in zip(loc.levels, (40, 40, 60)):
            src = src_full.voxel_down_sample(vs)
            res = o3d.pipelines.registration.registration_icp(src, m, dist, Tc, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
            Tc = np.asarray(res.transformation); st.append({"level_m": vs, "pose": pose_of(Tc), "fitness": round(float(res.fitness), 3), "rmse": round(float(res.inlier_rmse), 4)})
        return Tc, st, float(res.fitness), float(res.inlier_rmse)
    if multi_yaw:
        best = None
        for dy in (0, -30, 30, -60, 60, -90, 90, 180):
            Tc, st, fit, rmse = run(T_init @ yaw_T(0, 0, math.radians(dy)))
            if best is None or fit - 0.5 * rmse > best[4]: best = (Tc, st, fit, rmse, fit - 0.5 * rmse, dy)
        T, stages, fit, rmse, _, dy = best
    else:
        T, stages, fit, rmse = run(T_init); dy = 0
    em, ed = pose_err(np.linalg.inv(T_gt) @ T); ei_m, ei_d = pose_err(np.linalg.inv(T_gt) @ T_init)
    return {"init": pose_of(T_init), "stages": stages, "final": pose_of(T), "gt": pose_of(T_gt), "fitness": round(fit, 3), "rmse": round(rmse, 4),
            "err_init_m": round(ei_m, 2), "err_init_deg": round(ei_d, 1), "err_m": round(em, 3), "err_deg": round(ed, 2), "ok": bool(em < 0.3 and ed < 3),
            "ms": round((time.time() - t0) * 1000), "map_pts_used": int(len(sub)), "multi_yaw_pick_deg": dy}


def pose_of(T): return {"x": round(float(T[0, 3]), 3), "y": round(float(T[1, 3]), 3), "z": round(float(T[2, 3]), 3), "yaw_deg": round(math.degrees(math.atan2(T[1, 0], T[0, 0])), 2)}


def write_rrd(s, x, y, yaw_deg, out):
    """정합 결과를 rrd 로 (서버 프로세스 안이므로 전용 RecordingStream 을 만들고 끝에 닫아 flush)"""
    import rerun as rr, rerun.blueprint as rrb
    r = localize(s, x, y, yaw_deg)
    if "error" in r: return r
    P_local, org = scan_local(s); pts = STATE["pts"]
    rec = rr.new_recording("localize", make_default=False, make_thread_default=False); rec.save(out)
    sub = max(1, len(pts) // 1500000); rec.log("loc/map", rr.Points3D(pts[::sub], colors=[[140, 140, 140]], radii=0.04))
    def T_of(p): return yaw_T(p["x"], p["y"], math.radians(p["yaw_deg"]), p["z"])
    tf = lambda T: P_local @ T[:3, :3].T + T[:3, 3]
    for nm, p, col in (("init", r["init"], [255, 120, 40]), ("registered", r["final"], [40, 200, 255]), ("gt", r["gt"], [80, 255, 80])):
        T = T_of(p); rec.log(f"loc/scan_{nm}", rr.Points3D(tf(T), colors=[col], radii=0.06))
        rec.log(f"loc/pose_{nm}", rr.Arrows3D(origins=[T[:3, 3]], vectors=[T[:3, 0] * 2.0], colors=[col], radii=0.08, labels=[nm]))
    for k, st in enumerate(r["stages"]):
        rec.set_time_sequence("icp_level", k + 1); rec.log("loc/scan_progress", rr.Points3D(tf(T_of(st["pose"])), colors=[[255, 255, 0]], radii=0.05))
    rec.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="/loc", name="localize", contents=["+ /loc/**", "- /loc/scan_progress"]), rrb.TimePanel(expanded=False)), make_default=True)
    rec.disconnect()
    r["rrd"] = out; return r


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def _hdr(self, code, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Headers", "Content-Type"); self.send_header("Cache-Control", "no-store"); self.end_headers()
    def js(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode(); self._hdr(code, "application/json; charset=utf-8"); self.wfile.write(b)
    def do_OPTIONS(self): self._hdr(204, "text/plain")
    def scan(self, sid):
        s = STATE.get("scan_by_id", {}).get(str(sid))
        if s is None: raise KeyError(f"스캔 {sid} 없음 (지도 소스가 바뀌어 목록이 갱신됨 — 스캔을 다시 선택)")
        return s
    def do_GET(self):
        try: return self._get()
        except Exception as e: return self.js({"error": f"{type(e).__name__}: {e}"}, 500)
    def do_POST(self):
        try: return self._post()
        except Exception as e: return self.js({"error": f"{type(e).__name__}: {e}"}, 500)
    def _get(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == "/api/loc/info":
            return self.js({"source": STATE["source"], "n_pts": STATE["n_pts"], "bbox": STATE["bbox"], "bev": STATE["bev"], "manifest": a.manifest, "accum": a.accum, "logs": a.logs,
                            "map_name": os.path.basename(os.path.dirname(os.path.dirname(a.manifest))), "map_logs": sorted(STATE.get("in_map_logs", [])),
                            "scans": [{k: v for k, v in s.items() if k != "files"} for s in STATE["scans"]]})
        if u.path == "/api/loc/bev.png":
            self._hdr(200, "image/png"); self.wfile.write(STATE["bev_png"]); return
        if u.path == "/api/loc/scan":
            s = self.scan(q["id"][0]); P, org = scan_local(s); P = P[:: max(1, len(P) // 4000)]
            return self.js({"id": s["id"], "gt": pose_of(yaw_T(org[0], org[1], 0, org[2])), "points": np.round(P, 2).tolist(), "n_full": s["n_pts"]})
        self.js({"error": "unknown"}, 404)
    def _post(self):
        u = urllib.parse.urlparse(self.path); n = int(self.headers.get("Content-Length", "0") or 0); body = json.loads(self.rfile.read(n) or b"{}")
        with LOCK:
            if u.path == "/api/loc/localize":
                s = self.scan(body["id"]); return self.js(localize(s, float(body["x"]), float(body["y"]), float(body.get("yaw_deg", 0)), bool(body.get("multi_yaw", False))))
            if u.path == "/api/loc/rrd":
                s = self.scan(body["id"]); out = os.path.join(os.path.dirname(os.path.dirname(a.manifest)), "localize", f"loc_{s['id']}.rrd"); os.makedirs(os.path.dirname(out), exist_ok=True)
                return self.js(write_rrd(s, float(body["x"]), float(body["y"]), float(body.get("yaw_deg", 0)), out))
            if u.path == "/api/loc/source":
                try: build(body["source"]); return self.js({"ok": True, "source": STATE["source"], "n_pts": STATE["n_pts"], "scans": len(STATE["scans"])})
                except Exception as e: return self.js({"error": str(e)}, 400)
            if u.path == "/api/loc/map":                      # 지도·후보 로그 교체: {manifest (root 기준 상대/절대), logs "0,1,2"}
                try:
                    m = body["manifest"]; m = m if os.path.isabs(m) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), m)   # <repo>/…
                    build(body.get("source", STATE["source"]), manifest=m, logs=body.get("logs", a.logs)); return self.js({"ok": True, "manifest": a.manifest, "logs": a.logs, "scans": len(STATE["scans"])})
                except Exception as e: return self.js({"error": str(e)}, 400)
        self.js({"error": "unknown"}, 404)


build(a.source)
print(f"[loc] serving on :{a.port}", flush=True)
http.server.ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
