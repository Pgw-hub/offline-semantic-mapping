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
ap.add_argument("--aux-topic", default="/livox/lidar_1", help="보조 LiDAR 토픽. lidar_slam 의 aux_lidar 합치기가 빠진 스캔에 tf_static 자세로 직접 채움 (토픽 없으면 자동 생략)")
ap.add_argument("--aux-frame", default="livox_lidar_1:lidar", help="tf_static 에서 찾을 보조 LiDAR child frame 접미사")
ap.add_argument("--aux-mode", default="auto", choices=["auto", "always", "off"], help="auto: registered 점 수가 main 스캔의 1.3배 미만(=합쳐지지 않음)일 때만 추가 · always: 항상 추가 · off")
ap.add_argument("--aux-voxel", type=float, default=0.05, help="추가 전 보조 스캔 복셀 다운샘플 (lidar_slam cloud_registered_voxel_size 와 맞춤)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t_bl = lidar_t_bl(a.calib)
use_aux = a.aux_mode != "off" and RM.has_topic(a.mcap, a.aux_topic); T_b_aux = RM.tf_static(a.mcap, a.aux_frame) if use_aux else None
if use_aux and T_b_aux is None: print(f"보조 LiDAR {a.aux_topic}: tf_static 에 {a.aux_frame} 없음 → 생략"); use_aux = False
if use_aux: print(f"보조 LiDAR {a.aux_topic}: base→{T_b_aux[3].split('/')[-1]} t={np.round(T_b_aux[0],3).tolist()} · 모드 {a.aux_mode}")
ns, pos, q, scans = [], [], [], []; aux = {}; main_n = {}
with open(a.mcap, "rb") as f:
    for sch, ch, msg in make_reader(f).iter_messages(topics=["/Odometry/base", "/cloud_registered"] + (["/livox/lidar", a.aux_topic] if use_aux else [])):
        if ch.topic == "/cloud_registered":
            h, P = RM.pc2(msg.data, sch); scans.append((h["ns"], xyz_of(P).astype(np.float32), h["frame"]))
        elif ch.topic == "/Odometry/base":
            ns_, p_, q_ = RM.odom(msg.data, sch); ns.append(ns_); pos.append(p_); q.append([q_[3], q_[0], q_[1], q_[2]])
        elif ch.topic == a.aux_topic:
            ns_, xyz_, _, _ = RM.lidar_scan(msg.data, sch); aux[ns_] = xyz_
        else:
            main_n[RM.stamp_ns(msg.data, sch)] = RM.pc2(msg.data, sch)[0]["npts"]
odom = Odom(np.array(ns, np.int64), np.array(pos), np.array(q)); pp = None
def nearest(d, ts, tol_ns=50_000_000):
    if not d: return None
    ks = nearest.keys if getattr(nearest, "src", None) is d else np.array(sorted(d), np.int64); nearest.src, nearest.keys = d, ks
    i = np.clip(np.searchsorted(ks, ts), 0, len(ks) - 1); j = i - 1 if i > 0 and abs(ks[i-1] - ts) < abs(ks[i] - ts) else i
    return int(ks[j]) if abs(ks[j] - ts) <= tol_ns else None
n_aux_added = n_aux_merged = 0
if a.pose_graph:
    from pose_provider import PoseProvider; pp = PoseProvider(a.pose_graph); print("pose graph:", a.pose_graph, pp.meta)
# frame_id 검사: lidar_slam 은 world 프레임(id_) 로 발행하지만, 드물게 센서 프레임(livox_lidar_0:imu) 클라우드가 같은 토픽에 섞여 나옴(오도메트리 공백 시각) → 제외
import collections; fc = collections.Counter(f for _, _, f in scans); world_frame = fc.most_common(1)[0][0]
skipped = [(ts / 1e9, f) for ts, _, f in scans if f != world_frame]
if skipped: print(f"frame_id 불일치 스캔 {len(skipped)}개 제외 (world='{world_frame}'): " + ", ".join(f"{t:.2f}s:{f}" for t, f in skipped[:8]))
n_ok = 0
for i, (ts, P, fr) in enumerate(scans):
    if fr != world_frame: continue
    if use_aux:                                                              # lidar_slam 이 합치지 못한 스캔에 보조 LiDAR 점 추가 (실시간 오도메트리 프레임 = registered 와 같은 프레임)
        ka = nearest(aux, ts); km = nearest(main_n, ts)
        merged = (km is not None) and (len(P) >= 1.3 * main_n[km])
        if ka is not None and (a.aux_mode == "always" or not merged):
            Rw, pw, okw = odom.at(np.array([ts], np.int64))
            if okw[0]:
                A = aux[ka].astype(np.float64); A = A @ T_b_aux[1].T + T_b_aux[0]; A = A @ Rw[0].T + pw[0]
                if a.aux_voxel > 0:
                    k = np.floor(A / a.aux_voxel).astype(np.int64); _, ui = np.unique(k[:, 0] * 73856093 ^ k[:, 1] * 19349663 ^ k[:, 2] * 83492791, return_index=True); A = A[ui]
                P = np.concatenate([P, A.astype(np.float32)]); n_aux_added += 1
        elif merged: n_aux_merged += 1
        else: n_aux_skip = globals().get("n_aux_skip", 0) + 1; aux_skip_why = globals().get("aux_skip_why", {}); aux_skip_why["no_aux" if ka is None else ("no_odom" if not okw[0] else "other")] = aux_skip_why.get("no_aux" if ka is None else ("no_odom" if not okw[0] else "other"), 0) + 1; globals().update(n_aux_skip=n_aux_skip, aux_skip_why=aux_skip_why)
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
print(f"스캔 {n_ok}/{len(scans)} 저장 → {a.out}" + (f" · 보조 LiDAR: SLAM 이 합친 스캔 {n_aux_merged} · 파이프라인이 채운 스캔 {n_aux_added} · 못 채움 {globals().get('n_aux_skip', 0)} {globals().get('aux_skip_why', {})}" if use_aux else ""))
