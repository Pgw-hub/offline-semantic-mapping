#!/usr/bin/env python3
"""오프라인 SLAM 재생 1단계 — 녹화 mcap 에서 lidar_slam 플러그인 입력(/livox/lidar, /livox/imu)만 뽑아 호스트용 이진 파일로.

출력 <out>.bin : magic b"OSLD" + u32 version(1), 레코드 반복 { u8 kind(0 imu, 1 lidar), i64 log_time_ns, u32 n, n bytes }
               bytes 는 raisin 직렬화 메시지 본문 (레코더 v2 의 앞 8 바이트 수신 시각은 뺌 → getBuffer() 로 그대로 복원)
               레코드는 log_time(레코더 수신 시각) 순 = 실시간에 플러그인이 받던 순서.
     <out>.json: 입력 요약 + tf_static (base → livox_lidar_0:imu / :lidar, 호스트가 TF 로 주입)
사용: python export_sensor_dump.py <mcap_dir> <out_prefix> [--logs 0 1 2] [--t-end-s 760]
"""
import argparse, glob, json, os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from preprocess.geometry import raisin_msgs as RM
from mcap.reader import make_reader

ap = argparse.ArgumentParser(); ap.add_argument("mcap_dir"); ap.add_argument("out")
ap.add_argument("--logs", type=int, nargs="*"); ap.add_argument("--t-end-s", type=float, default=0, help="녹화 시작(첫 메시지) 기준 이 시각 이후는 버림, 0 이면 전부")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.mcap_dir, "log_[0-9][0-9][0-9].mcap")))
if a.logs is not None: files = [f for f in files if int(os.path.basename(f)[4:7]) in a.logs]
KIND = {"/livox/lidar": 1, "/livox/imu": 0}
recs, tf, n = [], {}, {0: 0, 1: 0}
for f in files:
    with open(f, "rb") as fh:
        rd = make_reader(fh); tft = [c.topic for c in rd.get_summary().channels.values() if c.topic.endswith("/tf_static")]   # <로봇 id>/tf_static (id 는 녹화마다 다를 수 있음)
        for sch, ch, m in rd.iter_messages(topics=list(KIND) + tft):
            if ch.topic.endswith("tf_static"):
                c = RM._Cur(m.data, 8); _, parent = c.header(); child = c.s(); t = c.u("<3d"); q = c.u("<4d")
                tf[child.split("/")[-1]] = {"parent": parent.split("/")[-1], "t": list(t), "q_xyzw": list(q)}
                continue
            recs.append((m.log_time, KIND[ch.topic], m.data[8:]))
recs.sort(key=lambda r: r[0])
t0 = recs[0][0]
if a.t_end_s > 0: recs = [r for r in recs if r[0] - t0 <= a.t_end_s * 1e9]
with open(a.out + ".bin", "wb") as o:
    o.write(b"OSLD" + struct.pack("<I", 1))
    for lt, k, b in recs:
        o.write(struct.pack("<BqI", k, lt, len(b))); o.write(b); n[k] += 1
info = {"mcap_dir": a.mcap_dir, "files": [os.path.basename(f) for f in files], "imu": n[0], "lidar": n[1],
        "t0_log_ns": t0, "t1_log_ns": recs[-1][0], "duration_s": round((recs[-1][0] - t0) / 1e9, 2), "tf_static": tf}
json.dump(info, open(a.out + ".json", "w"), indent=1)
print(f"imu {n[0]:,} · lidar {n[1]:,} · {info['duration_s']} s → {a.out}.bin")
for k in ("livox_lidar_0:imu", "livox_lidar_0:lidar"): print(" tf", k, tf.get(k))
