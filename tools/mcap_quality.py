#!/usr/bin/env python3
"""mcap 데이터 품질 검사 + rerun 재생 (rqt_bag 풍).
- 모든 토픽: 메시지 간격 dt(ms) · 크기(KB) · 초당 rate 시계열, 유실(gap) 이벤트, 토픽별 요약(개수·기대 개수·추정 유실·바이트)
- 재생: RGB(stride) · depth(stride) · LiDAR(1Hz, 1/4 점) · 오도메트리 궤적 — 같은 시간축(mcap_time)에서 스크럽
사용: python mcap_quality.py <mcap_dir> <out_prefix> --logs 0 1 2 [--img-stride 15] [--lidar-stride 10]
출력: <out_prefix>.rrd, <out_prefix>.json
"""
import argparse, collections, glob, json, os, struct, time, zlib
import numpy as np, cv2
import rerun as rr, rerun.blueprint as rrb
from mcap.reader import make_reader
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "geometry")); from pc2_reader import decode_livox

LIDAR_DT = np.dtype([("ot", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("refl", "u1"), ("tag", "u1"), ("line", "u1")])
ap = argparse.ArgumentParser()
ap.add_argument("mcap_dir"); ap.add_argument("out"); ap.add_argument("--logs", nargs="+", type=int, required=True)
ap.add_argument("--img-stride", type=int, default=15); ap.add_argument("--lidar-stride", type=int, default=10); ap.add_argument("--gap-factor", type=float, default=2.5)
ap.add_argument("--hi-rate", type=float, default=100.0, help="이 Hz 이상 토픽은 초 단위로 집계해 기록")
a = ap.parse_args()
rr.init("mcap_quality"); rr.save(a.out + ".rrd")
T = lambda s: s.replace("/", "_").strip("_")

# ---------- 1패스: 토픽별 타임스탬프·크기 ----------
files = [os.path.join(a.mcap_dir, f"log_{i:03d}.mcap") for i in a.logs]
per = collections.defaultdict(lambda: {"t": [], "n": [], "log": []}); enc = {}; bounds = []
for li, f in zip(a.logs, files):
    t0f = None; t1f = None
    with open(f, "rb") as fh:
        r = make_reader(fh); s = r.get_summary()
        for ch in s.channels.values(): enc[ch.topic] = s.schemas[ch.schema_id].name if ch.schema_id in s.schemas else "?"
        for sch, ch, msg in r.iter_messages():
            per[ch.topic]["t"].append(msg.log_time); per[ch.topic]["n"].append(len(msg.data)); per[ch.topic]["log"].append(li)
            t0f = msg.log_time if t0f is None else min(t0f, msg.log_time); t1f = msg.log_time if t1f is None else max(t1f, msg.log_time)
    bounds.append((li, t0f, t1f)); print(f"log_{li:03d}: {(t1f-t0f)/1e9:.1f}s", flush=True)
T0 = min(b[1] for b in bounds); sec = lambda ns: (ns - T0) / 1e9
summary = {"mcap_dir": a.mcap_dir, "logs": [f"log_{i:03d}" for i in a.logs], "duration_s": round((max(b[2] for b in bounds) - T0) / 1e9, 1), "topics": {}, "gaps": [],
           "log_bounds": [{"log": f"log_{li:03d}", "start_s": round(sec(t0f), 2), "end_s": round(sec(t1f), 2), "duration_s": round((t1f - t0f) / 1e9, 1)} for li, t0f, t1f in bounds]}

# 로그 경계 표시
for li, t0f, t1f in bounds:
    rr.set_time_seconds("mcap_time", sec(t0f)); rr.log("events", rr.TextLog(f"log_{li:03d} 시작", level="INFO")); rr.log("quality/log_index", rr.Scalar(li))
    rr.set_time_seconds("mcap_time", sec(t1f)); rr.log("events", rr.TextLog(f"log_{li:03d} 끝 ({(t1f-t0f)/1e9:.1f}s)", level="INFO"))
# 로그 사이 공백
for (la, _, e1), (lb, s2, _) in zip(bounds[:-1], bounds[1:]):
    g = (s2 - e1) / 1e9
    rr.set_time_seconds("mcap_time", sec(e1)); rr.log("events", rr.TextLog(f"log_{la:03d} → log_{lb:03d} 사이 공백 {g:.2f}s", level="WARN" if g > 1 else "INFO"))
    summary.setdefault("log_gaps_s", []).append({"from": f"log_{la:03d}", "to": f"log_{lb:03d}", "gap_s": round(g, 3)})

for topic, d in sorted(per.items()):
    t = np.array(d["t"], np.int64); n = np.array(d["n"], np.int64); lg = np.array(d["log"]); o = np.argsort(t); t, n, lg = t[o], n[o], lg[o]
    dt = np.diff(t) / 1e6                                     # ms
    same = lg[1:] == lg[:-1]                                  # 로그 경계는 gap 으로 세지 않음
    med = float(np.median(dt[same])) if same.any() else 0.0; rate = 1000.0 / med if med > 0 else 0.0
    exp_n = 0.0
    for li, t0f, t1f in bounds:
        m = lg == li
        if m.any() and med > 0: exp_n += ((t[m].max() - t[m].min()) / 1e6) / med + 1
    gaps = np.where(same & (dt > a.gap_factor * med))[0] if med > 0 else np.array([], int)
    name = T(topic); hi = rate >= a.hi_rate
    if not hi:
        for k in range(1, len(t)):
            rr.set_time_seconds("mcap_time", sec(t[k])); rr.log(f"quality/dt_ms/{name}", rr.Scalar(float(dt[k-1]) if same[k-1] else float("nan")))
            rr.log(f"quality/size_kb/{name}", rr.Scalar(float(n[k]) / 1024))
    # 초 단위 rate (모든 토픽), 고주파 토픽은 dt 최대·평균 크기도 초 단위로
    bins = np.floor((t - T0) / 1e9).astype(np.int64); ub = np.unique(bins)
    cnt = np.bincount(bins - bins.min(), minlength=bins.max() - bins.min() + 1)
    for b in ub:
        rr.set_time_seconds("mcap_time", float(b) + 0.5); rr.log(f"quality/rate_hz/{name}", rr.Scalar(float(cnt[b - bins.min()])))
        if hi:
            m = bins[1:] == b
            if m.any():
                rr.log(f"quality/dt_ms/{name}", rr.Scalar(float(dt[m & same].max()) if (m & same).any() else float("nan")))
                rr.log(f"quality/size_kb/{name}", rr.Scalar(float(n[1:][m].mean()) / 1024))
    glist = []
    for gi in gaps:
        rr.set_time_seconds("mcap_time", sec(t[gi + 1])); miss = int(round(dt[gi] / med)) - 1
        rr.log("events", rr.TextLog(f"[{topic}] 간격 {dt[gi]:.0f}ms (기대 {med:.0f}ms, 약 {miss}개 유실) @ log_{lg[gi+1]:03d}", level="WARN" if miss >= 3 else "INFO"))
        glist.append({"t_s": round(sec(t[gi + 1]), 3), "dt_ms": round(float(dt[gi]), 1), "missing_est": miss, "log": f"log_{lg[gi+1]:03d}"})
    # 로그별 커버리지: 로그 시작/끝 대비 첫/마지막 메시지 지연 (늦게 시작·일찍 끊긴 토픽 탐지)
    cov = []
    for li, t0f, t1f in bounds:
        m = lg == li
        if not m.any():
            cov.append({"log": f"log_{li:03d}", "count": 0, "start_delay_s": None, "end_early_s": None}); continue
        sd = (t[m].min() - t0f) / 1e9; ee = (t1f - t[m].max()) / 1e9
        cov.append({"log": f"log_{li:03d}", "count": int(m.sum()), "start_delay_s": round(float(sd), 2), "end_early_s": round(float(ee), 2)})
        if sd > 1.0:
            rr.set_time_seconds("mcap_time", sec(t0f)); rr.log("events", rr.TextLog(f"[{topic}] log_{li:03d} 시작 후 {sd:.1f}s 동안 메시지 없음", level="WARN"))
        if ee > 1.0:
            rr.set_time_seconds("mcap_time", sec(t[m].max())); rr.log("events", rr.TextLog(f"[{topic}] log_{li:03d} 끝나기 {ee:.1f}s 전에 마지막 메시지", level="WARN"))
    summary["topics"][topic] = {"schema": enc.get(topic, "?"), "count": int(len(t)), "expected": int(round(exp_n)), "missing_est": int(max(0, round(exp_n) - len(t))), "coverage": cov,
                                "rate_hz": round(rate, 1), "dt_med_ms": round(med, 2), "dt_max_ms": round(float(dt[same].max()) if same.any() else 0, 1),
                                "bytes_mb": round(float(n.sum()) / 1e6, 1), "mean_kb": round(float(n.mean()) / 1024, 1), "n_gaps": int(len(gaps))}
    summary["gaps"] += [{"topic": topic, **g} for g in glist]
    print(f"{topic:32s} n={len(t):6d} exp={exp_n:8.0f} miss≈{max(0, round(exp_n)-len(t)):4d} rate={rate:6.1f}Hz gaps={len(gaps):3d} {n.sum()/1e6:8.1f}MB", flush=True)

# ---------- 2패스: 재생 (RGB / depth / LiDAR / odom) ----------
# LiDAR 는 로봇(base) 프레임의 자식 엔티티로 로깅하고, base 의 Transform3D 를 odom 마다 갱신 → 스캔이 로봇을 따라다님
try:
    import yaml
    tr = yaml.safe_load(open(os.path.join(a.mcap_dir, "raisin_config", "calibration", "livox_lidar_0_lidar.yaml")))["transform"]
    t_bl = [float(tr["translation"][k]) for k in "xyz"]                                    # (표시용 오프셋; 정밀 값은 geometry/extrinsics.py)
except Exception:
    t_bl = [-0.0579, 0.0, 0.20932]
rr.log("playback/robot/lidar", rr.Transform3D(translation=t_bl), timeless=True)          # base → lidar 오프셋 (회전은 디코더에서 -x,-y 로 처리)
traj = []
for li, f in zip(a.logs, files):
    ci = di = si = 0
    with open(f, "rb") as fh:
        for sch, ch, msg in make_reader(fh).iter_messages(topics=["d455_front/rgb", "d455_front/depth", "/livox/lidar", "/Odometry/base"]):
            d = msg.data; rr.set_time_seconds("mcap_time", sec(msg.log_time))
            if ch.topic == "d455_front/rgb":
                ci += 1
                if ci % a.img_stride: continue
                s_, ns_, w, h, chn, rawlen, zlen = struct.unpack_from("<iIIIIII", d, 0)
                img = np.frombuffer(zlib.decompress(d[28:28 + zlen]), np.uint8).reshape(h, w, chn)
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                rr.log("playback/rgb", rr.ImageEncoded(contents=jpg.tobytes()))
            elif ch.topic == "d455_front/depth":
                di += 1
                if di % a.img_stride: continue
                s_, ns_, w, h, rawlen, zlen = struct.unpack_from("<iIIIII", d, 0)
                dep = np.frombuffer(zlib.decompress(d[24:24 + zlen]), np.uint16).reshape(h, w)
                rr.log("playback/depth", rr.DepthImage(dep, meter=1000.0))
            elif ch.topic == "/livox/lidar":
                si += 1
                if si % a.lidar_stride: continue
                xyz = decode_livox(d)[1][::4]                             # CustomMsg / PointCloud2 자동 (LIVOX_FLIP)
                z = xyz[:, 2]; c = np.clip((z - z.min()) / (np.ptp(z) + 1e-6), 0, 1)
                col = np.stack([(255 * c), (120 * np.ones_like(c)), (255 * (1 - c))], -1).astype(np.uint8)
                rr.log("playback/robot/lidar/points", rr.Points3D(xyz, colors=col, radii=0.03))
            else:
                v = struct.unpack_from("<7d", d, 8); traj.append(v[0:3])
                rr.log("playback/robot", rr.Transform3D(translation=v[0:3], rotation=rr.Quaternion(xyzw=[v[3], v[4], v[5], v[6]])))
                rr.log("playback/robot/marker", rr.Points3D([[0, 0, 0]], colors=[[255, 80, 80]], radii=0.15))
                if len(traj) % 10 == 0:
                    rr.log("playback/odom/path", rr.LineStrips3D([np.array(traj, np.float32)], colors=[[255, 200, 60]], radii=0.03))
    print(f"log_{li:03d} 재생 로깅 완료", flush=True)

bp = rrb.Blueprint(rrb.Tabs(
    rrb.Vertical(rrb.TimeSeriesView(origin="/quality/dt_ms", name="dt (ms) — 메시지 간격, 튀면 유실"),
                 rrb.TimeSeriesView(origin="/quality/rate_hz", name="rate (Hz, 1s)"),
                 rrb.TimeSeriesView(origin="/quality/size_kb", name="size (KB)"), name="Data quality"),
    rrb.TextLogView(origin="/events", name="Events (gaps, log boundaries)"),
    rrb.Horizontal(rrb.Vertical(rrb.Spatial2DView(origin="/playback/rgb", name="RGB"), rrb.Spatial2DView(origin="/playback/depth", name="Depth")),
                   rrb.Spatial3DView(origin="/playback", name="LiDAR + odometry (world)", contents=["+ /playback/robot/**", "+ /playback/odom/**"]), name="Playback")),
    rrb.TimePanel(expanded=True))
rr.send_blueprint(bp, make_default=True)
json.dump(summary, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)
print("저장:", a.out + ".rrd", a.out + ".json", f"({os.path.getsize(a.out + '.rrd')/1e6:.0f} MB)")
