#!/usr/bin/env python3
"""mcap 의 LiDAR 스캔을 world 좌표 점 + 센서 원점으로 내보내기 (occupancy 레이캐스팅용).
출력: <out_dir>/%06d.npz {xyz (N,3) float32 world, origin (3,) float32 world, stamp_ns}
사용: python3 export_lidar_scans.py <log.mcap> <calib_dir> <out_dir> [--max-range 40]
"""
import argparse, os, struct
import numpy as np, yaml
from mcap.reader import make_reader

LIDAR_DT = np.dtype([("ot", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("refl", "u1"), ("tag", "u1"), ("line", "u1")])


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def slerp(q0, q1, u):
    d = float(np.dot(q0, q1))
    if d < 0: q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u*(q1-q0); return q/np.linalg.norm(q)
    th = np.arccos(d)*u; q2 = q1 - q0*d; q2 /= np.linalg.norm(q2)
    return q0*np.cos(th) + q2*np.sin(th)


ap = argparse.ArgumentParser()
ap.add_argument("mcap"); ap.add_argument("calib"); ap.add_argument("out"); ap.add_argument("--max-range", type=float, default=40.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tr = yaml.safe_load(open(os.path.join(a.calib, "livox_lidar_0_lidar.yaml")))["transform"]
t_bl = np.array([tr["translation"][k] for k in "xyz"])       # 회전은 항등 (점의 180° 보정은 디코더에서)

gt_ns, gt_pos, gt_quat, scans = [], [], [], []
with open(a.mcap, "rb") as f:
    for sch, ch, msg in make_reader(f).iter_messages(topics=["/Odometry/base", "/livox/lidar"]):
        if ch.topic == "/livox/lidar":
            sec, nsec = struct.unpack_from("<iI", msg.data, 0); N, = struct.unpack_from("<I", msg.data, 24)
            p = np.frombuffer(msg.data, dtype=LIDAR_DT, count=N, offset=28)
            scans.append((sec*10**9+nsec, np.stack([-p["x"], -p["y"], p["z"]], -1).astype(np.float32)))
        else:
            v = struct.unpack_from("<7d", msg.data, 8); sec, nsec = struct.unpack_from("<iI", msg.data, 0)
            gt_ns.append(sec*10**9+nsec); gt_pos.append(v[0:3]); gt_quat.append([v[6], v[3], v[4], v[5]])
o = np.argsort(gt_ns); gt_ns = np.array(gt_ns)[o]; gt_pos = np.array(gt_pos)[o]; gt_quat = np.array(gt_quat)[o]
gt_quat /= np.linalg.norm(gt_quat, axis=1, keepdims=True)

n_ok = 0
for i, (ns, xyz) in enumerate(scans):
    k = np.searchsorted(gt_ns, ns)
    if k == 0 or k == len(gt_ns) or gt_ns[k]-gt_ns[k-1] > 250e6:
        continue
    u = (ns-gt_ns[k-1])/(gt_ns[k]-gt_ns[k-1])
    R = q2R(slerp(gt_quat[k-1], gt_quat[k], u)); t = gt_pos[k-1]*(1-u)+gt_pos[k]*u
    rng = np.linalg.norm(xyz, axis=1); keep = (rng > 0.6) & (rng < a.max_range)
    pw = xyz[keep] @ R.T + (R @ t_bl + t)
    origin = (R @ t_bl + t).astype(np.float32)
    np.savez(os.path.join(a.out, "%06d.npz" % i), xyz=pw.astype(np.float32), origin=origin, stamp_ns=np.int64(ns))
    n_ok += 1
print("스캔 %d/%d 저장 → %s" % (n_ok, len(scans), a.out))
