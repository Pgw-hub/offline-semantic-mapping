#!/usr/bin/env python3
"""트랙 2 S 단계 — lidar_slam 의 /cloud_registered (deskew+정합된 world 점, PointCloud2) 를 스캔별 npz 로 (build_occupancy 입력 형식).
/cloud_registered 는 발행 시점의 실시간 오도메트리 프레임이므로, --pose-graph(로봇 최종 PGO graph/) 가 있으면 Δ(t)=T_corr·T_odom⁻¹ 로 최종 프레임에 맞춘다.
사용: python export_registered_scans.py <log.mcap> <calib_dir> <out_dir> [--pose-graph graph/] [--max-range 40]
출력: <out>/%06d.npz  xyz(float32, world) · origin(float32, 그 시각 LiDAR 위치) · stamp_ns
"""
import argparse, os, struct, sys
import numpy as np
import sys as _rs, os as _ro; _rs.path.insert(0, _ro.path.join(_ro.path.dirname(_ro.path.abspath(__file__)), "../../geometry")); import raisin_msgs as RM   # 레코더 v1/v2 레이아웃 공용 디코더
from mcap.reader import make_reader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "geometry"))
from pc2_reader import parse_pc2, xyz_of
from extrinsics import lidar_t_bl
from deskew import Odom

ap = argparse.ArgumentParser(); ap.add_argument("mcap"); ap.add_argument("calib"); ap.add_argument("out"); ap.add_argument("--pose-graph"); ap.add_argument("--max-range", type=float, default=40.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t_bl = lidar_t_bl(a.calib)
ns, pos, q, scans = [], [], [], []
with open(a.mcap, "rb") as f:
    for sch, ch, msg in make_reader(f).iter_messages(topics=["/Odometry/base", "/cloud_registered"]):
        if ch.topic == "/cloud_registered":
            h, P = RM.pc2(msg.data, sch); scans.append((h["ns"], xyz_of(P).astype(np.float32), h["frame"]))
        else:
            ns_, p_, q_ = RM.odom(msg.data, sch); ns.append(ns_); pos.append(p_); q.append([q_[3], q_[0], q_[1], q_[2]])
odom = Odom(np.array(ns, np.int64), np.array(pos), np.array(q)); pp = None
if a.pose_graph:
    from pose_provider import PoseProvider; pp = PoseProvider(a.pose_graph); print("pose graph:", a.pose_graph, pp.meta)
# frame_id 검사: lidar_slam 은 world 프레임(id_) 로 발행하지만, 드물게 센서 프레임(livox_lidar_0:imu) 클라우드가 같은 토픽에 섞여 나옴(오도메트리 공백 시각) → 제외
import collections; fc = collections.Counter(f for _, _, f in scans); world_frame = fc.most_common(1)[0][0]
skipped = [(ts / 1e9, f) for ts, _, f in scans if f != world_frame]
if skipped: print(f"frame_id 불일치 스캔 {len(skipped)}개 제외 (world='{world_frame}'): " + ", ".join(f"{t:.2f}s:{f}" for t, f in skipped[:8]))
n_ok = 0
for i, (ts, P, fr) in enumerate(scans):
    if fr != world_frame: continue
    if pp is not None:
        T = pp.at(ts)
        if T is None: continue
        D = pp.delta(np.array([ts], np.int64))[0]; P = P @ D[:3, :3].T.astype(np.float32) + D[:3, 3].astype(np.float32)
    else:
        R, p, ok = odom.at(np.array([ts], np.int64))
        if not ok[0]: continue
        T = np.eye(4); T[:3, :3] = R[0]; T[:3, 3] = p[0]
    origin = (T[:3, :3] @ t_bl + T[:3, 3]).astype(np.float32)
    r = np.linalg.norm(P - origin, axis=1); P = P[(r > 0.3) & (r < a.max_range)]
    np.savez(os.path.join(a.out, "%06d.npz" % i), xyz=P, origin=origin, stamp_ns=np.int64(ts)); n_ok += 1
print(f"스캔 {n_ok}/{len(scans)} 저장 → {a.out}")
