#!/usr/bin/env python3
"""오프라인 SLAM 재생 3단계 — 호스트 출력(slam_out.bin)을 원래 녹화와 같은 모양의 mcap 데이터셋으로.

원본 log_XXX.mcap 마다: SLAM 출력 토픽을 뺀 모든 메시지(카메라·tf_static·IMU·LiDAR·__recorder/*)를 그대로 복사하고,
그 로그의 시간 구간에 든 오프라인 SLAM 출력을 같은 토픽 이름·같은 스키마(원본 채널 것)로 넣는다 → 기존 맵 파이프라인을 그대로 쓴다.
config_snapshot 은 복사하고, 저장 맵 궤적(<map>.txt)이 있으면 함께 둔다.
사용: python import_outputs_to_mcap.py <원본 mcap_dir> <slam_out.bin> <out_dir> [--logs 0 1 2] [--map-dir <host 저장 맵 폴더>]
"""
import argparse, glob, json, os, shutil, struct
from mcap.reader import make_reader
from mcap.writer import Writer, CompressionType

TOPICS = {0: "/Odometry/base", 1: "/cloud_registered", 2: "/path/pgo", 3: "/pgo/keyframe_cloud", 4: "/Odometry/lio"}
REPLACED = set(TOPICS.values()) | {"/cloud_registered_compressed", "/Odometry/iekf_base", "/lidar_slam/operating_mode"}
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("slam_out"); ap.add_argument("out")
ap.add_argument("--logs", type=int, nargs="*"); ap.add_argument("--map-dir")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

recs = []
with open(a.slam_out, "rb") as f:
    assert f.read(4) == b"OSLO", "not a host dump"
    while True:
        h = f.read(13)
        if len(h) < 13: break
        k, t, n = struct.unpack("<BqI", h); recs.append((t, k, f.read(n)))
recs.sort(key=lambda r: r[0])
files = sorted(glob.glob(os.path.join(a.src, "log_[0-9][0-9][0-9].mcap")))
if a.logs is not None: files = [f for f in files if int(os.path.basename(f)[4:7]) in a.logs]
spans = []
for f in files:
    s = make_reader(open(f, "rb")).get_summary().statistics; spans.append((s.message_start_time, s.message_end_time))
schema_src = {}                                                     # 원본 채널 스키마 (레코더 v2 레이아웃 텍스트 포함)
for f in files:
    s = make_reader(open(f, "rb")).get_summary()
    for c in s.channels.values():
        if c.topic in TOPICS.values() and c.topic not in schema_src: schema_src[c.topic] = (s.schemas[c.schema_id], c)
stats = {}
for i, f in enumerate(files):
    lo = spans[i][0] if i else -2**62; hi = spans[i + 1][0] if i + 1 < len(files) else 2**62
    r = make_reader(open(f, "rb")); s = r.get_summary()
    out = open(os.path.join(a.out, os.path.basename(f)), "wb"); w = Writer(out, compression=CompressionType.ZSTD); w.start(profile="", library="offline_slam_import")
    sm, cm = {}, {}
    for sid, x in s.schemas.items(): sm[sid] = w.register_schema(x.name, x.encoding, x.data)
    for cid, c in s.channels.items():
        if c.topic not in REPLACED: cm[cid] = w.register_channel(c.topic, c.message_encoding, sm.get(c.schema_id, 0), c.metadata)
    slam_ch = {}
    for k, topic in TOPICS.items():
        if topic in schema_src:
            x, c = schema_src[topic]; sid = w.register_schema(x.name, x.encoding, x.data)
            slam_ch[k] = w.register_channel(topic, c.message_encoding, sid, {**c.metadata, "source": "offline_slam"})
    mine = [r_ for r_ in recs if lo <= r_[0] < hi and r_[1] in slam_ch]
    j = 0; n_copy = 0
    for sch, ch, m in r.iter_messages(log_time_order=True):
        while j < len(mine) and mine[j][0] <= m.log_time:
            w.add_message(slam_ch[mine[j][1]], mine[j][0], mine[j][2], mine[j][0]); j += 1
        if ch.id in cm: w.add_message(cm[ch.id], m.log_time, m.data, m.publish_time, m.sequence); n_copy += 1
    for t, k, b in mine[j:]: w.add_message(slam_ch[k], t, b, t)
    w.finish(); out.close()
    stats[os.path.basename(f)] = {"copied": n_copy, **{TOPICS[k]: sum(1 for x in mine if x[1] == k) for k in slam_ch}}
    print(os.path.basename(f), stats[os.path.basename(f)])
cs = os.path.join(a.src, "config_snapshot")
if os.path.isdir(cs) and not os.path.exists(os.path.join(a.out, "config_snapshot")): shutil.copytree(cs, os.path.join(a.out, "config_snapshot"))
if a.map_dir and os.path.isdir(a.map_dir): shutil.copytree(a.map_dir, os.path.join(a.out, "offline_slam_map"), dirs_exist_ok=True)
json.dump({"source_mcap_dir": a.src, "slam_out": a.slam_out, "logs": stats}, open(os.path.join(a.out, "offline_slam_import.json"), "w"), indent=1)
