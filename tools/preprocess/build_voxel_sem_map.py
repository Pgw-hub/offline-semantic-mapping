#!/usr/bin/env python3
"""LiDAR + D455 depth + RGB -> semantic 3D 복셀 맵 (오프라인, ROS 불필요).

기하:   LiDAR 스캔(360°, ~40m) + D455 depth(전방, 조밀) 를 SLAM 자세로 world 변환 -> 0.1m 복셀 누적
의미:   각 semantic 프레임(10Hz, _sem/session) 시각에
        - depth 카메라 점: 픽셀의 semantic 확률을 그대로 (보이는 표면이므로 가림 문제 없음)
        - LiDAR 점: 카메라로 투영 후 depth 이미지와 z 비교 (z-buffer) -> 보이는 점만 라벨 수령
        카메라가 못 본 복셀은 기하만 유지 (unknown)
출력:   npz {ijk, n_pts, rgb, sem_mean(4), n_sem, first_frame} + 메타

사용: python3 build_voxel_sem_map.py <log.mcap> <calib_dir> <cg_dataset_dir> <sem_session_dir> <out.npz>
"""
import argparse, glob, os, struct, zlib
import cv2
import numpy as np
import yaml
from mcap.reader import make_reader

LIDAR_DT = np.dtype([("ot", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                     ("refl", "u1"), ("tag", "u1"), ("line", "u1")])
VOX = 0.10
CLASSES = ["grass", "path", "tree", "person"]   # sem_dir/classes.json 이 있으면 그것으로 대체 (클래스 사전)


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]], dtype=np.float64)


def slerp(q0, q1, u):
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u*(q1-q0); return q/np.linalg.norm(q)
    th0 = np.arccos(d); th = th0*u
    q2 = q1 - q0*d; q2 /= np.linalg.norm(q2)
    return q0*np.cos(th) + q2*np.sin(th)


def load_K(calib, name):
    ci = yaml.safe_load(open(os.path.join(calib, name)))["camera_info"]
    return np.array(ci["K"], dtype=np.float64).reshape(3, 3), ci["width"], ci["height"]


def load_T(calib, name):
    tr = yaml.safe_load(open(os.path.join(calib, name)))["transform"]
    q = np.array([tr["rotation"][k] for k in "wxyz"], dtype=np.float64); q /= np.linalg.norm(q)
    T = np.eye(4); T[:3, :3] = quat_wxyz_to_R(q); T[:3, 3] = [tr["translation"][k] for k in "xyz"]
    return T


def decode_lidar(data):
    sec, nsec = struct.unpack_from("<iI", data, 0)
    N, = struct.unpack_from("<I", data, 24)
    pts = np.frombuffer(data, dtype=LIDAR_DT, count=N, offset=28)
    xyz = np.stack([-pts["x"], -pts["y"], pts["z"]], -1).astype(np.float32)
    return sec*10**9 + nsec, xyz


def pack(ijk):
    """int32 3열 -> int64 키 (각 축 21비트, 오프셋 1<<20)."""
    o = 1 << 20
    return ((ijk[:, 0].astype(np.int64)+o) << 42) | ((ijk[:, 1].astype(np.int64)+o) << 21) | (ijk[:, 2].astype(np.int64)+o)


def unpack(key):
    o = 1 << 20
    m = (1 << 21) - 1
    return np.stack([(key >> 42) & m, (key >> 21) & m, key & m], -1).astype(np.int64) - o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap"); ap.add_argument("calib"); ap.add_argument("cg_dir"); ap.add_argument("sem_dir"); ap.add_argument("out")
    ap.add_argument("--depth-stride", type=int, default=3)
    ap.add_argument("--mask-bottom", type=int, default=60)
    ap.add_argument("--lidar-max", type=float, default=40.0)
    ap.add_argument("--edge-px", type=int, default=4, help="클래스 경계에서 이 픽셀 안의 라벨은 쓰지 않음 (실루엣 오염 방지, 0=끔)")
    ap.add_argument("--require-depth", type=int, default=1, help="1: 카메라 depth 가 없는 픽셀에는 LiDAR 점 라벨을 주지 않음 (가림 검사 불가 → 보류)")
    a = ap.parse_args()

    Kc, Wc, Hc = load_K(a.calib, "d455_front_color_caminfo.yaml")
    T_bc = load_T(a.calib, "d455_front_color.yaml")
    T_bl = np.eye(4); T_bl[:3, 3] = [-0.0579, 0.0, 0.20932]
    fx, fy, cx, cy = Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]

    # --- mcap: odom + lidar ---
    gt_ns, gt_pos, gt_quat, lidar = [], [], [], {}
    with open(a.mcap, "rb") as f:
        r = make_reader(f)
        for sch, ch, msg in r.iter_messages(topics=["/Odometry/base", "/livox/lidar"]):
            if ch.topic == "/livox/lidar":
                ns, xyz = decode_lidar(msg.data); lidar[ns] = xyz
            else:
                v = struct.unpack_from("<7d", msg.data, 8); sec, nsec = struct.unpack_from("<iI", msg.data, 0)
                gt_ns.append(sec*10**9+nsec); gt_pos.append(v[0:3]); gt_quat.append([v[6], v[3], v[4], v[5]])
    order = np.argsort(gt_ns)
    gt_ns = np.array(gt_ns, np.int64)[order]; gt_pos = np.array(gt_pos)[order]; gt_quat = np.array(gt_quat)[order]
    gt_quat /= np.linalg.norm(gt_quat, axis=1, keepdims=True)
    l_stamps = np.array(sorted(lidar), np.int64)
    print("odom %d, lidar %d" % (len(gt_ns), len(l_stamps)))

    def pose_at(ns):
        i = np.searchsorted(gt_ns, ns)
        if i == 0 or i == len(gt_ns) or (gt_ns[i]-gt_ns[i-1]) > 250e6:
            return None
        u = (ns-gt_ns[i-1])/(gt_ns[i]-gt_ns[i-1])
        T = np.eye(4); T[:3, :3] = quat_wxyz_to_R(slerp(gt_quat[i-1], gt_quat[i], u)); T[:3, 3] = gt_pos[i-1]*(1-u)+gt_pos[i]*u
        return T

    # --- cg 데이터셋 (D455 depth, rgb, c2w) ---
    st = np.load(os.path.join(a.cg_dir, "stamps.npz"))["stamps"]
    traj = np.loadtxt(os.path.join(a.cg_dir, "traj.txt")).reshape(-1, 4, 4)
    assert len(st) == len(traj)

    # --- semantic 인덱스: 이 로그 시간 범위 안의 것 ---
    idx = np.load(os.path.join(a.sem_dir, "rgb_index.npz"))
    classdict = None
    cj = os.path.join(a.sem_dir, "classes.json")
    if os.path.exists(cj):
        import json
        classdict = json.load(open(cj, encoding="utf-8")); CLASSES = [c["name"] for c in classdict["classes"]]
    NCH = len(CLASSES); print("클래스", NCH, CLASSES)
    rgb_ns = idx["rgb_stamp_ns"]
    sel = np.where((rgb_ns >= st[0]-50e6) & (rgb_ns <= st[-1]+50e6))[0]
    print("semantic 프레임 %d개 (세션 idx %d~%d)" % (len(sel), sel[0], sel[-1]))

    vv, uu = np.mgrid[0:Hc:a.depth_stride, 0:Wc:a.depth_stride]
    vv = vv.ravel(); uu = uu.ravel()
    keep_px = vv < Hc - a.mask_bottom
    vv = vv[keep_px]; uu = uu[keep_px]

    K_list, RGB_list, SEM_list, HAS_list, FR_list = [], [], [], [], []
    n_lidar_lab = n_lidar_tot = 0
    for fi, si in enumerate(sel):
        t = int(rgb_ns[si])
        j = int(np.argmin(np.abs(st - t)))
        if abs(int(st[j]) - t) > 40e6:
            continue
        T_wc = traj[j]; T_cw = np.linalg.inv(T_wc)
        dep = cv2.imread(os.path.join(a.cg_dir, "results", "depth%06d.png" % j), cv2.IMREAD_UNCHANGED).astype(np.float32)/1000.0
        rgb = cv2.imread(os.path.join(a.cg_dir, "results", "frame%06d.jpg" % j))[:, :, ::-1]
        sem = np.load(os.path.join(a.sem_dir, "sem", "%06d.npz" % si))["sem"]   # (480,640,N) uint8
        # 클래스 경계 마스크: argmax 라벨(무라벨=NCH)이 (2k+1) 창 안에서 바뀌는 픽셀 → 라벨 보류
        if a.edge_px > 0:
            lab_img = np.where(sem.max(-1) >= 77, sem.argmax(-1), NCH).astype(np.uint8)      # 0.3*255
            kern = np.ones((2 * a.edge_px + 1, 2 * a.edge_px + 1), np.uint8)
            edge = cv2.erode(lab_img, kern) != cv2.dilate(lab_img, kern)
        else:
            edge = np.zeros(sem.shape[:2], bool)

        # (a) depth 카메라 점 — 보이는 표면이므로 픽셀 semantic 직접
        z = dep[vv, uu]
        ok = (z > 0.3) & (z < 8.0)
        zc = z[ok]; uc = uu[ok]; vc = vv[ok]
        pc = np.stack([(uc-cx)/fx*zc, (vc-cy)/fy*zc, zc], -1)
        pw = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
        K_list.append(pack(np.floor(pw/VOX).astype(np.int32)))
        RGB_list.append(rgb[vc, uc]); SEM_list.append(sem[vc, uc]); HAS_list.append(~edge[vc, uc])   # 경계 픽셀은 라벨 보류
        FR_list.append(np.full(len(zc), fi, np.int32))

        # (b) LiDAR 점 — 카메라 z-buffer 로 가림 판정
        li = np.searchsorted(l_stamps, t - int(60e6)); lj = np.searchsorted(l_stamps, t + int(60e6))
        for ns in l_stamps[li:lj]:
            T_wb = pose_at(int(ns))
            if T_wb is None:
                continue
            xyz = lidar[int(ns)]
            rng_ = np.linalg.norm(xyz, axis=1)
            xyz = xyz[(rng_ > 0.6) & (rng_ < a.lidar_max)]
            pw = xyz @ (T_wb[:3, :3] @ T_bl[:3, :3]).T + (T_wb[:3, :3] @ T_bl[:3, 3] + T_wb[:3, 3])
            pc = pw @ T_cw[:3, :3].T + T_cw[:3, 3]
            zl = pc[:, 2]
            u = np.round(fx*pc[:, 0]/np.where(zl > 0, zl, 1)+cx).astype(np.int64)
            v = np.round(fy*pc[:, 1]/np.where(zl > 0, zl, 1)+cy).astype(np.int64)
            infov = (zl > 0.3) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc - a.mask_bottom)
            has = np.zeros(len(pw), bool)
            dcam = np.zeros(len(pw), np.float32)
            dcam[infov] = dep[v[infov], u[infov]]
            # 보임: lidar z 가 카메라 표면과 일치. depth 없는 픽셀은 (require_depth=0 일 때만) 통과, 클래스 경계 픽셀은 보류
            vis = infov & (((dcam <= 0) & (not a.require_depth)) | ((dcam > 0) & (np.abs(zl - dcam) < 0.25 + 0.05*zl)))
            vis[vis] &= ~edge[v[vis], u[vis]]
            has[vis] = True
            semv = np.zeros((len(pw), NCH), np.uint8); rgbv = np.zeros((len(pw), 3), np.uint8)
            semv[vis] = sem[v[vis], u[vis]]; rgbv[vis] = rgb[v[vis], u[vis]]
            rgbv[~vis] = 128
            K_list.append(pack(np.floor(pw/VOX).astype(np.int32)))
            RGB_list.append(rgbv); SEM_list.append(semv); HAS_list.append(has)
            FR_list.append(np.full(len(pw), fi, np.int32))
            n_lidar_tot += len(pw); n_lidar_lab += int(vis.sum())
        if fi % 100 == 0:
            print("  frame %d/%d" % (fi, len(sel)))

    keys = np.concatenate(K_list); rgbs = np.concatenate(RGB_list).astype(np.float32)
    sems = np.concatenate(SEM_list).astype(np.float32)/255.0; has = np.concatenate(HAS_list); frs = np.concatenate(FR_list)
    print("점 %s개 → 집계" % f"{len(keys):,}")
    uk, inv = np.unique(keys, return_inverse=True)
    n = len(uk)
    n_pts = np.bincount(inv, minlength=n).astype(np.int32)
    rgb_sum = np.stack([np.bincount(inv, weights=rgbs[:, c], minlength=n) for c in range(3)], -1)
    # rgb 는 라벨 받은 점(진짜 색) 우선, 없으면 회색
    hasf = has.astype(np.float32)
    rgb_lab = np.stack([np.bincount(inv, weights=rgbs[:, c]*hasf, minlength=n) for c in range(3)], -1)
    n_sem = np.bincount(inv, weights=hasf, minlength=n)
    sem_sum = np.stack([np.bincount(inv, weights=sems[:, c]*hasf, minlength=n) for c in range(NCH)], -1)
    first = np.full(n, 10**9, np.int32); np.minimum.at(first, inv, frs)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem_mean = np.where(n_sem[:, None] > 0, sem_sum/np.maximum(n_sem, 1)[:, None], 0).astype(np.float16)
        rgb_out = np.where(n_sem[:, None] > 0, rgb_lab/np.maximum(n_sem, 1)[:, None], rgb_sum/n_pts[:, None]).astype(np.uint8)
    ijk = unpack(uk).astype(np.int32)
    np.savez_compressed(a.out, ijk=ijk, n_pts=n_pts, rgb=rgb_out, sem_mean=sem_mean,
                        n_sem=n_sem.astype(np.int32), first_frame=first, vox=VOX, classes=np.array(CLASSES),
                        n_frames=len(sel), **({"classdict": np.array(__import__("json").dumps(classdict, ensure_ascii=False))} if classdict else {}))
    lab = int((n_sem > 0).sum())
    print("복셀 %s개, 라벨 있음 %s (%.0f%%), unknown %s" % (f"{n:,}", f"{lab:,}", 100*lab/n, f"{n-lab:,}"))
    print("LiDAR 점 라벨 수령율: %.1f%% (%s/%s)" % (100*n_lidar_lab/max(n_lidar_tot, 1), f"{n_lidar_lab:,}", f"{n_lidar_tot:,}"))
    am = sem_mean[n_sem > 0].astype(np.float32).argmax(1)
    print("라벨 분포:", {c: int((am == i).sum()) for i, c in enumerate(CLASSES)})
    print("저장:", a.out)


if __name__ == "__main__":
    main()
