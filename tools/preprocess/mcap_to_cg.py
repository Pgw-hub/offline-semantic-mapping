#!/usr/bin/env python3
"""raisin_pod mcap -> ConceptGraphs(Replica 레이아웃) 데이터셋 변환.

출력: <out>/results/frame%06d.jpg (640x480 RGB), depth%06d.png (u16 mm, color K 격자로 워프),
      <out>/traj.txt (c2w 4x4 row-major, OpenCV 광학 프레임), stamps.npz

사용: python3 mcap_to_cg.py <log.mcap> <calib_dir> <out_dir> [--max-gap-ms 250]
"""
import argparse, os, struct, zlib

import cv2
import numpy as np
import yaml
from mcap.reader import make_reader


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def slerp(q0, q1, u):
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0); return q / np.linalg.norm(q)
    th0 = np.arccos(d); th = th0 * u
    q2 = q1 - q0 * d; q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def load_K(calib, name):
    ci = yaml.safe_load(open(os.path.join(calib, name)))["camera_info"]
    K = np.array(ci["K"], dtype=np.float64).reshape(3, 3)
    return K, ci["width"], ci["height"]


def load_T_base_cam(calib, name):
    tr = yaml.safe_load(open(os.path.join(calib, name)))["transform"]
    q = np.array([tr["rotation"][k] for k in "wxyz"], dtype=np.float64)
    q /= np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(q)
    T[:3, 3] = [tr["translation"][k] for k in "xyz"]
    return T


LIDAR_DT = np.dtype([("ot", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                     ("refl", "u1"), ("tag", "u1"), ("line", "u1")])


def decode_lidar(data):
    """raisin_pod CustomMsg -> (stamp_ns, Nx3 float32, refl). 좌표 180° 보정(-x,-y) 포함."""
    sec, nsec = struct.unpack_from("<iI", data, 0)
    N, = struct.unpack_from("<I", data, 24)
    pts = np.frombuffer(data, dtype=LIDAR_DT, count=N, offset=28)
    xyz = np.stack([-pts["x"], -pts["y"], pts["z"]], -1).astype(np.float32)
    return sec * 10**9 + nsec, xyz, pts["refl"].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap"); ap.add_argument("calib"); ap.add_argument("out")
    ap.add_argument("--max-gap-ms", type=float, default=250.0)
    ap.add_argument("--pair-ms", type=float, default=34.0, help="rgb-depth 최대 시차")
    ap.add_argument("--lidar", action="store_true", help="LiDAR 를 depth 에 융합 (D455 없는 픽셀 채움)")
    ap.add_argument("--lidar-win", type=float, default=0.4, help="프레임 주변 누적 창(초, ±)")
    ap.add_argument("--lidar-max", type=float, default=40.0, help="LiDAR depth 상한(m)")
    ap.add_argument("--pose-graph", help="geometry/build_graph.py 산출 graph/ 디렉터리 — 자세를 PGO 보정값으로 교체")
    a = ap.parse_args()

    Kc, Wc, Hc = load_K(a.calib, "d455_front_color_caminfo.yaml")
    Kd, Wd, Hd = load_K(a.calib, "d455_front_depth_aligned_caminfo.yaml")
    T_bc = load_T_base_cam(a.calib, "d455_front_color.yaml")
    print("color K:", Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2], (Wc, Hc))
    print("depth_aligned K:", Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2], (Wd, Hd))

    # depth(aligned, 자체 K) -> color K 격자 최근접 워프 맵 (외부 파라미터 동일 -> 순수 intrinsic)
    uc, vc = np.meshgrid(np.arange(Wc), np.arange(Hc))
    ud = np.round(Kd[0, 0] * (uc - Kc[0, 2]) / Kc[0, 0] + Kd[0, 2]).astype(np.int64)
    vd = np.round(Kd[1, 1] * (vc - Kc[1, 2]) / Kc[1, 1] + Kd[1, 2]).astype(np.int64)
    valid = (ud >= 0) & (ud < Wd) & (vd >= 0) & (vd < Hd)
    ud = np.clip(ud, 0, Wd - 1); vd = np.clip(vd, 0, Hd - 1)

    # base->lidar (항등회전 + 평행이동; 점의 180° 보정은 decode_lidar 에서)
    T_bl = np.eye(4); T_bl[:3, 3] = [-0.0579, 0.0, 0.20932]

    # --- 1pass: odometry + depth 압축본 (+lidar) 적재 ---
    gt_ns, gt_pos, gt_quat = [], [], []
    depth_z = {}   # stamp_ns -> compressed bytes
    lidar = {}     # stamp_ns -> (Nx3 lidar 좌표계, refl)
    topics = ["/Odometry/base", "d455_front/depth"] + (["/livox/lidar"] if a.lidar else [])
    with open(a.mcap, "rb") as f:
        r = make_reader(f)
        for sch, ch, msg in r.iter_messages(topics=topics):
            if ch.topic == "/livox/lidar":
                ns, xyz, refl = decode_lidar(msg.data)
                lidar[ns] = (xyz, refl)
            elif ch.topic == "/Odometry/base":
                v = struct.unpack_from("<7d", msg.data, 8)
                sec, nsec = struct.unpack_from("<iI", msg.data, 0)
                gt_ns.append(sec * 10**9 + nsec)
                gt_pos.append(v[0:3])
                gt_quat.append([v[6], v[3], v[4], v[5]])
            else:
                sec, nsec, w, h, rawlen, zlen = struct.unpack_from("<iIIIII", msg.data, 0)
                assert (w, h) == (Wd, Hd) and rawlen == w * h * 2, (w, h, rawlen)
                depth_z[sec * 10**9 + nsec] = msg.data[24:24 + zlen]
    order = np.argsort(gt_ns)
    gt_ns = np.array(gt_ns, dtype=np.int64)[order]
    gt_pos = np.array(gt_pos, dtype=np.float64)[order]
    gt_quat = np.array(gt_quat, dtype=np.float64)[order]
    gt_quat /= np.linalg.norm(gt_quat, axis=1, keepdims=True)
    d_stamps = np.array(sorted(depth_z), dtype=np.int64)
    print(f"odom {len(gt_ns)}, depth {len(d_stamps)}")

    def pose_at(ns):
        i = np.searchsorted(gt_ns, ns)
        if i == 0 or i == len(gt_ns):
            return None
        gap = (gt_ns[i] - gt_ns[i - 1]) / 1e6
        if gap > a.max_gap_ms:
            return None
        u = (ns - gt_ns[i - 1]) / (gt_ns[i] - gt_ns[i - 1])
        pos = gt_pos[i - 1] * (1 - u) + gt_pos[i] * u
        q = slerp(gt_quat[i - 1], gt_quat[i], u)
        T = np.eye(4); T[:3, :3] = quat_wxyz_to_R(q / np.linalg.norm(q)); T[:3, 3] = pos
        return T

    if a.pose_graph:                                   # PGO 보정 자세(geometry/build_graph.py 산출)로 교체
        import sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "geometry"))
        from pose_provider import PoseProvider
        _pp = PoseProvider(a.pose_graph); pose_at = _pp.at; print("pose graph:", a.pose_graph, _pp.meta)

    l_stamps = np.array(sorted(lidar), dtype=np.int64)
    if a.lidar:
        print(f"lidar 스캔 {len(l_stamps)}개")
        os.makedirs(os.path.join(a.out, "lidar_world"), exist_ok=True)

    def lidar_depth_and_world(frame_ns, T_wb_frame):
        """프레임 주변 스캔 누적 -> (카메라 depth mm u16, world 점, refl)."""
        win = int(a.lidar_win * 1e9)
        i0 = np.searchsorted(l_stamps, frame_ns - win)
        i1 = np.searchsorted(l_stamps, frame_ns + win)
        pw_all, rf_all = [], []
        for ns in l_stamps[i0:i1]:
            T_wb = pose_at(int(ns))
            if T_wb is None:
                continue
            xyz, refl = lidar[int(ns)]
            pw = xyz @ (T_wb[:3, :3] @ T_bl[:3, :3]).T + (T_wb[:3, :3] @ T_bl[:3, 3] + T_wb[:3, 3])
            pw_all.append(pw); rf_all.append(refl)
        if not pw_all:
            return None, None, None
        pw = np.concatenate(pw_all); rf = np.concatenate(rf_all)
        T_cw = np.linalg.inv(T_wb_frame @ T_bc)
        pc = pw @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = pc[:, 2]
        ok = (z > 0.25) & (z < a.lidar_max)
        u = np.round(Kc[0, 0] * pc[ok, 0] / z[ok] + Kc[0, 2]).astype(np.int64)
        v = np.round(Kc[1, 1] * pc[ok, 1] / z[ok] + Kc[1, 2]).astype(np.int64)
        inb = (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
        dep = np.full((Hc, Wc), np.inf, np.float32)
        np.minimum.at(dep, (v[inb], u[inb]), z[ok][inb])   # z-buffer
        dep_mm = np.where(np.isfinite(dep), dep * 1000.0, 0).astype(np.uint16)
        return dep_mm, pw, rf

    res = os.path.join(a.out, "results")
    os.makedirs(res, exist_ok=True)
    lines, stamps = [], []
    n_img = n_nopose = n_nodepth = 0
    n_lidar_fill = []
    med_depths = []
    with open(a.mcap, "rb") as f:
        r = make_reader(f)
        for sch, ch, msg in r.iter_messages(topics=["d455_front/rgb"]):
            sec, nsec, w, h, chn, rawlen, zlen = struct.unpack_from("<iIIIIII", msg.data, 0)
            ns = sec * 10**9 + nsec
            T_wb = pose_at(ns)
            if T_wb is None:
                n_nopose += 1; continue
            j = np.searchsorted(d_stamps, ns)
            cand = [k for k in (j - 1, j) if 0 <= k < len(d_stamps)]
            if not cand:
                n_nodepth += 1; continue
            k = min(cand, key=lambda k: abs(int(d_stamps[k] - ns)))
            if abs(int(d_stamps[k] - ns)) / 1e6 > a.pair_ms:
                n_nodepth += 1; continue

            rgb = np.frombuffer(zlib.decompress(msg.data[28:28 + zlen]), np.uint8).reshape(h, w, chn)
            dep = np.frombuffer(zlib.decompress(depth_z[d_stamps[k]]), np.uint16).reshape(Hd, Wd)
            dep_c = dep[vd, ud]
            dep_c[~valid] = 0
            i = n_img
            if a.lidar:
                ld, pw, rf = lidar_depth_and_world(ns, T_wb)
                if ld is not None:
                    fill = (dep_c == 0) & (ld > 0)
                    dep_c = dep_c.copy(); dep_c[fill] = ld[fill]
                    n_lidar_fill.append(int(fill.sum()))
                    if i % 10 == 0:   # rerun 컨텍스트용 world 점 (FOV 밖 포함, 서브샘플)
                        ss = pw[::6].astype(np.float16); rs = rf[::6]
                        np.savez_compressed(os.path.join(a.out, "lidar_world", "%06d.npz" % i),
                                            xyz=ss, refl=rs)
            cv2.imwrite(os.path.join(res, "frame%06d.jpg" % i),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(res, "depth%06d.png" % i), dep_c)
            c2w = T_wb @ T_bc
            lines.append(" ".join("%.9g" % x for x in c2w.reshape(-1)))
            stamps.append(ns)
            m = dep_c[dep_c > 0]
            if i % 200 == 0 and len(m):
                med_depths.append(float(np.median(m)))
            n_img += 1

    with open(os.path.join(a.out, "traj.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(os.path.join(a.out, "stamps.npz"), stamps=np.array(stamps, dtype=np.int64))
    print(f"완료: {n_img} 프레임 저장, pose스킵 {n_nopose}, depth스킵 {n_nodepth}")
    print("depth 중앙값 샘플(mm 가정):", [round(x) for x in med_depths])
    if a.lidar and n_lidar_fill:
        print(f"LiDAR 융합: 프레임당 채운 픽셀 중앙값 {int(np.median(n_lidar_fill))} "
              f"(D455 미관측 영역, 총 {Hc*Wc} 픽셀 중)")


if __name__ == "__main__":
    main()
