#!/usr/bin/env python3
"""LiDAR 없는 옛 데이터셋용 — depth 카메라(AV1 압축 스트림) + robot_state 자세로 지도와 재생용 rrd 를 만든다.
20260908 처럼 /livox/lidar 와 /Odometry/base 가 비어 있는 mcap 에서, 남아 있는 것만으로 3D 를 복원한다:
  자세 : robot_state (10 Hz, 다리 로봇 상태추정기의 base_pos/base_quat — SLAM 아님, 드리프트 있음)
  점   : d455_front/depth_compressed, d430_{front,rear}/depth_compressed (AV1 8-bit gray = 선형 스케일된 depth)
         depth_m = gray * (far-near)/255 + near   (d455 near 0.1 far 6.0 / d430 near 0.1 far 3.0 — 플러그인 소스 값)
  색   : d455_front/rgb_compressed 를 같은 3D 점에 재투영
★ 점군이 두꺼운 주원인은 AV1 압축이다 (0916 데이터로 같은 시각 raw vs AV1 측정: 단일 프레임 국소 두께 0.9 cm → 3.4 cm, 3.6배. 3~5 m 구간은 2.6 → 5.2 cm).
  양자화 2.3 cm/level 과 자세 보간 오차(20프레임 누적 시 +0.9 cm)는 그보다 작다. raw depth(d455_front/depth) 가 녹화된 데이터셋이면 그쪽을 쓸 것.
사용: python tools/map_from_depth_cams.py <mcap_dir> <out_dir> [--logs 0,4] [--cameras d455_front,d430_front,d430_rear]
      [--vox 0.05] [--max-range 5] [--frame-stride 2] [--px-stride 2] [--rrd] [--pcd]
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # 저장소 루트 (패키지 import)
import argparse, glob, json, os, struct, subprocess, sys, tempfile, time
import numpy as np, yaml
from preprocess.geometry import raisin_msgs as RM   # 레코더 v1/v2 레이아웃 공용 디코더

NEAR_FAR = {"d455_front": (0.1, 6.0), "d430_front": (0.1, 3.0), "d430_rear": (0.1, 3.0)}
CALIB_DEPTH = {"d455_front": ("d455_front_depth_aligned_caminfo.yaml", "d455_front_depth_aligned.yaml"),
               "d430_front": ("d430_front_depth_caminfo.yaml", "d430_front_depth.yaml"),
               "d430_rear": ("d430_rear_depth_caminfo.yaml", "d430_rear_depth.yaml")}


def q2R(w, x, y, z):
    n = (w * w + x * x + y * y + z * z) ** 0.5; w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _R2q(R):
    tr = np.trace(R)
    if tr > 0: s = np.sqrt(tr + 1) * 2; return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    k = int(np.argmax(np.diag(R))); a_, b_, c_ = k, (k + 1) % 3, (k + 2) % 3
    s = np.sqrt(1 + R[a_, a_] - R[b_, b_] - R[c_, c_]) * 2; q = np.zeros(4)
    q[0] = (R[c_, b_] - R[b_, c_]) / s; q[1 + a_] = 0.25 * s; q[1 + b_] = (R[b_, a_] + R[a_, b_]) / s; q[1 + c_] = (R[c_, a_] + R[a_, c_]) / s
    return q / np.linalg.norm(q)


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def load_T(calib, name):
    tr = yaml.safe_load(open(os.path.join(calib, name)))["transform"]
    T = np.eye(4); T[:3, :3] = q2R(*[tr["rotation"][k] for k in "wxyz"]); T[:3, 3] = [tr["translation"][k] for k in "xyz"]; return T


def load_K(calib, name):
    ci = yaml.safe_load(open(os.path.join(calib, name)))["camera_info"]; K = np.array(ci["K"], float).reshape(3, 3)
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2], ci["width"], ci["height"]


def parse_pkt(d):
    """FfmpegPacket (레코더 layout) → (stamp_ns, w, h, payload)"""
    o = 0; sec, nsec = struct.unpack_from("<iI", d, o); o += 8
    w, h = struct.unpack_from("<ii", d, o); o += 8
    enclen, = struct.unpack_from("<I", d, o); o += 4 + enclen
    o += 8 + 1 + 1; dlen, = struct.unpack_from("<I", d, o); o += 4
    return sec * 10**9 + nsec, w, h, d[o:o + dlen]


def _leb128(b, o):
    v = s = 0
    while True:
        x = b[o]; o += 1; v |= (x & 0x7F) << s; s += 7
        if not (x & 0x80): return v, o


def has_seq_header(p):
    """AV1 시퀀스 헤더(OBU type 1) 포함 여부 — 디코딩은 이 패킷부터 시작해야 한다(녹화가 스트림 중간에 시작되므로)."""
    o = 0
    while o < len(p):
        b = p[o]; t = (b >> 3) & 0xF; ext = (b >> 2) & 1; hs = (b >> 1) & 1; o += 1
        if ext: o += 1
        if not hs: return t == 1
        sz, o = _leb128(p, o)
        if t == 1: return True
        o += sz
    return False


def decode_raw_depth(d):
    """레코더 depth_image 레이아웃: stamp; u32 w,h; u32 rawlen; u32 zlen; zlib(uint16 mm) → (ns, depth_m float32)"""
    import zlib
    sec, nsec = struct.unpack_from("<iI", d, 0); w, h, rawlen, zlen = struct.unpack_from("<4I", d, 8)
    z = np.frombuffer(zlib.decompress(d[24:24 + zlen]), np.uint16).reshape(h, w)
    return sec * 10**9 + nsec, z.astype(np.float32) / 1000.0


def decode_raw_rgb(d):
    """레코더 rgb_image 레이아웃: stamp; u32 w,h,ch; u32 rawlen; u32 zlen; zlib(uint8)"""
    import zlib
    sec, nsec = struct.unpack_from("<iI", d, 0); w, h, c, rawlen, zlen = struct.unpack_from("<5I", d, 8)
    return sec * 10**9 + nsec, np.frombuffer(zlib.decompress(d[28:28 + zlen]), np.uint8).reshape(h, w, c)


def decode_av1(pkts, tmpdir, tag, pix="gray"):
    """[(ns,w,h,payload)] → [(ns, gray HxW uint8)]. OBU 스트림으로 이어 붙여 한 번에 디코딩(패킷 단위 디코딩은 PyAV 에서 안 됨)."""
    import av
    start = next((i for i, (_, _, _, p) in enumerate(pkts) if p and has_seq_header(p)), None)
    if start is None: return []
    use = [x for x in pkts[start:] if x[3]]
    path = os.path.join(tmpdir, f"{tag}.obu")
    with open(path, "wb") as fh:
        for _, _, _, p in use: fh.write(p)
    out = []
    with av.open(path, format="obu") as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i >= len(use): break
            out.append((use[i][0], fr.to_ndarray(format=pix)))
    return out


class Pose:
    """robot_state 의 base_pos/base_quat 를 시각 보간 (위치 선형, 회전 slerp)."""
    def __init__(self, ns, pos, quat):
        o = np.argsort(ns); self.ns = np.asarray(ns)[o]; self.pos = np.asarray(pos)[o]; self.q = np.asarray(quat)[o]

    def at(self, t, max_gap_ns=250_000_000):
        i = int(np.searchsorted(self.ns, t))
        if i == 0 or i >= len(self.ns) or self.ns[i] - self.ns[i - 1] > max_gap_ns: return None
        u = (t - self.ns[i - 1]) / (self.ns[i] - self.ns[i - 1])
        q0, q1 = self.q[i - 1], self.q[i]                              # slerp (최근접 대체는 10 Hz·보행 진동에서 수 cm 오차)
        q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
        d = float(np.dot(q0, q1))
        if d < 0: q1 = -q1; d = -d
        if d > 0.9995: qi = q0 + u * (q1 - q0)
        else:
            th = np.arccos(np.clip(d, -1, 1)); st = np.sin(th)
            qi = (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / st
        T = np.eye(4); T[:3, :3] = q2R(*(qi / np.linalg.norm(qi))); T[:3, 3] = self.pos[i - 1] * (1 - u) + self.pos[i] * u
        return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mcap_dir"); ap.add_argument("out"); ap.add_argument("--logs")
    ap.add_argument("--cameras", default="d455_front,d430_front,d430_rear"); ap.add_argument("--vox", type=float, default=0.05)
    ap.add_argument("--max-range", type=float, default=5.0); ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--px-stride", type=int, default=2); ap.add_argument("--rrd", action="store_true"); ap.add_argument("--pcd", action="store_true")
    ap.add_argument("--play-stride", type=int, default=5, help="rrd 재생 프레임 간격")
    ap.add_argument("--level", action="store_true", help="궤적 최적합 평면의 법선을 +z 로 맞추는 전역 회전 (다리 로봇 상태추정기의 일정한 pitch 편향 보정; 평지 주행 가정)")
    ap.add_argument("--bev", action="store_true", help="위에서 본 PNG 도 저장")
    ap.add_argument("--rrd-vox", type=float, default=0.15, help="rrd 에 넣을 지도 복셀 크기 (파일 크기용; 0 이면 원본)")
    ap.add_argument("--pose", default="robot_state", choices=["robot_state", "odom"], help="자세 출처: robot_state(다리 상태추정기) | odom(/Odometry/base, lidar_slam)")
    ap.add_argument("--pose-graph", help="graph/ 디렉터리 — 로봇/오프라인 PGO 보정 자세 사용 (가장 정확)")
    ap.add_argument("--force-compressed", action="store_true", help="raw 가 있어도 AV1 압축본을 사용 (raw 대비 비교용)")
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    from mcap.reader import make_reader
    calib = os.path.join(a.mcap_dir, "raisin_config", "calibration")
    cams = [c for c in a.cameras.split(",") if c]
    files = sorted(glob.glob(os.path.join(a.mcap_dir, "log_*.mcap")))
    if a.logs:
        lo, hi = (a.logs.split(",") + [None])[:2]; rng = range(int(lo), int(hi or lo) + 1)
        files = [f for f in files if int(os.path.basename(f)[4:7]) in rng]
    # raw(무압축 zlib) 가 녹화돼 있으면 그쪽을 쓴다 — AV1 압축본은 depth 국소 두께가 3.6 배(0.9→3.4 cm) 두꺼워짐
    topics = ["robot_state", "/Odometry/base", "d455_front/rgb", "d455_front/rgb_compressed"] + [f"{c}/depth" for c in cams] + [f"{c}/depth_compressed" for c in cams]
    ns_p, pos_p, q_p = [], [], []; raw = {t: [] for t in topics if t != "robot_state"}
    for f in files:
        for sch, ch, m in make_reader(open(f, "rb")).iter_messages(topics=topics):
            if ch.topic in ("robot_state", "/Odometry/base"):
                want = "robot_state" if a.pose == "robot_state" else "/Odometry/base"
                if ch.topic != want: continue
                d = m.data
                if want == "robot_state": ns_v, p_v, pq = RM.robot_state(d, sch); v = (*p_v, *pq)          # base_pos xyz + quat wxyz
                else: ns_v, p_v, q_v = RM.odom(d, sch); v = (*p_v, *q_v); pq = (q_v[3], q_v[0], q_v[1], q_v[2])   # Odometry: quat xyzw → wxyz
                sec, nsec = divmod(ns_v, 10**9)
                ns_p.append(sec * 10**9 + nsec); pos_p.append(v[:3]); q_p.append(pq)
            elif ch.topic.endswith("_compressed"): raw[ch.topic].append(parse_pkt(m.data))
            else: raw[ch.topic].append((m.data, sch))                  # raw Image 는 나중에 디코딩 (v1 zlib / v2 raw)
    if a.pose_graph:                                                   # PoseProvider (로봇 PGO 그래프) — Δ(t) 보정 포함
        from preprocess.geometry.pose_provider import PoseProvider
        _pp = PoseProvider(a.pose_graph); print("pose graph:", a.pose_graph, _pp.meta)
        class _PG:
            ns = _pp.ns; pos = np.array([T[:3, 3] for T in _pp.at_many(_pp.ns)[0]])
            def at(self, t, **k): return _pp.at(int(t))
        pose = _PG(); pose_src = f"pose_graph:{os.path.basename(a.pose_graph.rstrip('/'))}"
    else:
        if not ns_p: sys.exit(f"{a.pose} 가 없습니다 — 자세를 만들 수 없음")
        pose = Pose(ns_p, pos_p, q_p); pose_src = a.pose
    if a.level and not a.pose_graph and len(pose.pos) > 50:                                # 궤적 평면의 법선 → +z 로 보내는 전역 회전
        Q = pose.pos - pose.pos.mean(0); n_ = np.linalg.svd(Q, full_matrices=False)[2][-1]
        if n_[2] < 0: n_ = -n_
        v = np.cross(n_, [0, 0, 1.0]); sn = np.linalg.norm(v); cs = float(np.dot(n_, [0, 0, 1.0]))
        if sn > 1e-9:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); R_lv = np.eye(3) + vx + vx @ vx * ((1 - cs) / sn**2)
        else: R_lv = np.eye(3)
        tilt = np.degrees(np.arccos(np.clip(cs, -1, 1))); p0 = pose.pos[0].copy()
        pose.pos = (pose.pos - p0) @ R_lv.T; pose.q = np.array([_qmul(_R2q(R_lv), q) for q in pose.q])
        LEVEL = (R_lv, p0); print(f"레벨링: 궤적 평면 기울기 {tilt:.1f}° 를 수평으로 보정")
    else: LEVEL = None
    span = (pose.ns[-1] - pose.ns[0]) / 1e9
    print(f"로그 {len(files)} · 자세[{pose_src}] {len(pose.ns)} ({span:.0f}s, 이동범위 {np.ptp(pose.pos, 0).round(1)} m) [{time.time()-t0:.0f}s]")

    tmp = tempfile.mkdtemp()
    if raw.get("d455_front/rgb") and not a.force_compressed:
        rgb_frames = [RM.rgb_image(d, s_, order="rgb") for d, s_ in raw["d455_front/rgb"]]; rgb_src = "raw"
    else:
        rgb_frames = decode_av1(raw.get("d455_front/rgb_compressed", []), tmp, "rgb", pix="rgb24"); rgb_src = "AV1"
    rgb_ns = np.array([x[0] for x in rgb_frames], np.int64) if rgb_frames else np.zeros(0, np.int64)
    Kc = load_K(calib, "d455_front_color_caminfo.yaml") if os.path.exists(os.path.join(calib, "d455_front_color_caminfo.yaml")) else None
    Tbc_color = load_T(calib, "d455_front_color.yaml") if os.path.exists(os.path.join(calib, "d455_front_color.yaml")) else None
    print(f"RGB 프레임 {len(rgb_frames)} ({rgb_src})")

    acc = {}                                                         # voxel key → [sum_rgb(3), count]
    per_cam, play = {}, []
    for cam in cams:
        near, far = NEAR_FAR[cam]
        if raw.get(f"{cam}/depth") and not a.force_compressed:         # raw: 이미 미터 단위(스케일 복원 불필요)
            fr = [RM.depth_image(d, s_) for d, s_ in raw[f"{cam}/depth"]]; src = "raw"
        elif raw.get(f"{cam}/depth_compressed"):
            fr = [(t, g.astype(np.float32) * (far - near) / 255.0 + near) for t, g in decode_av1(raw[f"{cam}/depth_compressed"], tmp, cam)]
            fr = [(t, np.where(z <= near + 1e-6, 0.0, z).astype(np.float32)) for t, z in fr]; src = "AV1"
        else: print(f"{cam}: depth 없음"); continue
        ci, ex = CALIB_DEPTH[cam]; fx, fy, cx, cy, W, H = load_K(calib, ci); T_bc = load_T(calib, ex)
        Hf, Wf = fr[0][1].shape[:2]
        if (Wf, Hf) != (W, H):                                         # 실제 프레임 크기에 맞춰 K 스케일 (raw/압축 해상도가 다를 수 있음)
            sx, sy = Wf / W, Hf / H; fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy; W, H = Wf, Hf
            print(f"  {cam}: caminfo {int(W/sx)}x{int(H/sy)} → 프레임 {Wf}x{Hf}, K 스케일 적용")
        vv, uu = np.mgrid[0:H:a.px_stride, 0:W:a.px_stride]
        n_pts = 0; n_use = 0
        for i, (ns, zimg) in enumerate(fr):
            if i % a.frame_stride: continue
            T_wb = pose.at(ns)
            if T_wb is None: continue
            z = zimg[vv, uu]; ok = (z > 0.05) & (z < a.max_range)
            if not ok.any(): continue
            zc = z[ok]; uc = uu[ok]; vc = vv[ok]
            pc = np.stack([(uc - cx) / fx * zc, (vc - cy) / fy * zc, zc], -1)
            T_wc = T_wb @ T_bc; pw = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
            col = np.full((len(pw), 3), 160, np.uint8)
            if cam == "d455_front" and rgb_frames and Kc is not None and Tbc_color is not None:
                j = int(np.argmin(np.abs(rgb_ns - ns)))
                if abs(int(rgb_ns[j]) - ns) < 60_000_000:
                    img = rgb_frames[j][1]
                    if img.ndim == 2: img = np.stack([img] * 3, -1)
                    pcol = (pw - Tbc_color[:3, 3] - T_wb[:3, 3]) if False else (np.linalg.inv(T_wb @ Tbc_color)[:3, :3] @ pw.T).T + np.linalg.inv(T_wb @ Tbc_color)[:3, 3]
                    fxc, fyc, cxc, cyc, Wc_, Hc_ = Kc; zz = pcol[:, 2]
                    ui = np.round(fxc * pcol[:, 0] / np.where(zz > 0, zz, 1) + cxc).astype(int)
                    vi = np.round(fyc * pcol[:, 1] / np.where(zz > 0, zz, 1) + cyc).astype(int)
                    m_ = (zz > 0) & (ui >= 0) & (ui < Wc_) & (vi >= 0) & (vi < Hc_)
                    col[m_] = img[vi[m_], ui[m_]][:, :3]
            K_ = np.floor(pw / a.vox).astype(np.int64); o_ = 1 << 20
            key = ((K_[:, 0] + o_) << 42) | ((K_[:, 1] + o_) << 21) | (K_[:, 2] + o_)
            uk, inv = np.unique(key, return_inverse=True)
            s = np.stack([np.bincount(inv, weights=col[:, c].astype(np.float64), minlength=len(uk)) for c in range(3)], -1)
            c_ = np.bincount(inv, minlength=len(uk))
            for k, sv, cv in zip(uk, s, c_):
                e = acc.get(k)
                if e is None: acc[k] = [sv, cv]
                else: e[0] += sv; e[1] += cv
            n_pts += len(pw); n_use += 1
            if cam == "d455_front" and (n_use % a.play_stride == 0): play.append((ns, pw.astype(np.float32), col.copy(), T_wb.copy()))
        per_cam[cam] = (len(fr), n_use, n_pts, src)
        print(f"{cam}: {src} {len(fr)} 프레임 디코딩 · 사용 {n_use} · 점 {n_pts:,} [{time.time()-t0:.0f}s]")

    if not acc: sys.exit("누적된 점이 없습니다")
    keys = np.fromiter(acc.keys(), np.int64, len(acc)); m = (1 << 21) - 1; o_ = 1 << 20
    ijk = np.stack([(keys >> 42) & m, (keys >> 21) & m, keys & m], -1) - o_
    P = ((ijk + 0.5) * a.vox).astype(np.float32)
    C = np.stack([np.array([acc[k][0][c] / acc[k][1] for k in keys]) for c in range(3)], -1).astype(np.uint8)
    traj = pose.pos.astype(np.float32)
    print(f"지도 복셀 {len(P):,} ({a.vox} m) · bbox {P.min(0).round(1)}..{P.max(0).round(1)}")
    if a.bev:
        import cv2
        zl = np.percentile(P[:, 2], 2); m_ = (P[:, 2] > zl + 0.2) & (P[:, 2] < zl + 3.0)
        Q = P[m_]; Cq = C[m_]; res = 0.1; x0, y0 = Q[:, :2].min(0) - 1; W = int((Q[:, 0].max() - x0) / res) + 20; H = int((Q[:, 1].max() - y0) / res) + 20
        img = np.full((H, W, 3), 255, np.uint8); o2 = np.argsort(Q[:, 2])
        u = ((Q[o2, 0] - x0) / res).astype(int); v = (H - 1 - (Q[o2, 1] - y0) / res).astype(int)
        ok2 = (u >= 0) & (u < W) & (v >= 0) & (v < H); img[v[ok2], u[ok2]] = Cq[o2][ok2][:, ::-1]
        tu = ((traj[:, 0] - x0) / res).astype(int); tv = (H - 1 - (traj[:, 1] - y0) / res).astype(int)
        for i in range(1, len(tu)): cv2.line(img, (tu[i-1], tv[i-1]), (tu[i], tv[i]), (0, 140, 255), 2)
        cv2.imwrite(os.path.join(a.out, "bev.png"), img); print("bev:", os.path.join(a.out, "bev.png"), img.shape)

    summary = {"mcap_dir": a.mcap_dir, "logs": [os.path.basename(f) for f in files], "pose_source": pose_src,
               "duration_s": round(span, 1), "voxels": int(len(P)), "vox_m": a.vox, "max_range_m": a.max_range,
               "cameras": {c: {"frames_decoded": v[0], "frames_used": v[1], "points": v[2], "depth_source": v[3]} for c, v in per_cam.items()},
               "rgb_frames": len(rgb_frames), "depth_scale": {c: NEAR_FAR[c] for c in per_cam}}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)

    if a.pcd:
        from package.export_pcd import write_pcd
        write_pcd(os.path.join(a.out, "map.pcd"), P.astype(np.float64), C); print("pcd:", os.path.join(a.out, "map.pcd"))
    if a.rrd:
        import rerun as rr, rerun.blueprint as rrb
        rr.init("map_from_depth"); rr.save(os.path.join(a.out, "map_from_depth.rrd"))
        Pv, Cv, vv_ = P, C, a.vox
        if a.rrd_vox > a.vox:                                          # 뷰어용 재격자화 (파일 크기)
            Kv = np.floor(P / a.rrd_vox).astype(np.int64); ov = 1 << 20
            kv = ((Kv[:, 0] + ov) << 42) | ((Kv[:, 1] + ov) << 21) | (Kv[:, 2] + ov)
            uk2, inv2 = np.unique(kv, return_inverse=True); cnt2 = np.bincount(inv2, minlength=len(uk2))
            m2 = (1 << 21) - 1; ij2 = np.stack([(uk2 >> 42) & m2, (uk2 >> 21) & m2, uk2 & m2], -1) - ov
            Pv = ((ij2 + 0.5) * a.rrd_vox).astype(np.float32); vv_ = a.rrd_vox
            Cv = np.stack([np.bincount(inv2, weights=C[:, c].astype(np.float64), minlength=len(uk2)) / cnt2 for c in range(3)], -1).astype(np.uint8)
            print(f"rrd 지도 복셀 {len(P):,} → {len(Pv):,} ({a.rrd_vox} m)")
        rr.log("map/points", rr.Points3D(Pv, colors=Cv, radii=vv_ * 0.5), timeless=True)
        rr.log("map/trajectory", rr.LineStrips3D([traj], colors=[[255, 200, 40]], radii=0.03), timeless=True)
        for i, (ns, pw, col, T_wb) in enumerate(play):
            rr.set_time_nanos("stamp", int(ns))
            rr.log("live/frame_points", rr.Points3D(pw[::4], colors=col[::4], radii=a.vox * 1.2))
            rr.log("live/robot", rr.Transform3D(translation=T_wb[:3, 3], mat3x3=T_wb[:3, :3]))
            if rgb_frames:
                j = int(np.argmin(np.abs(rgb_ns - ns)))
                img = rgb_frames[j][1]
                rr.log("live/rgb", rr.Image(np.stack([img] * 3, -1) if img.ndim == 2 else img))
        rr.send_blueprint(rrb.Blueprint(rrb.Tabs(
            rrb.Spatial3DView(origin="/", name="Map (depth cams + robot_state)", contents=["+ /map/**"]),
            rrb.Horizontal(rrb.Spatial3DView(origin="/", name="Playback 3D", contents=["+ /map/trajectory", "+ /live/**"]),
                           rrb.Spatial2DView(origin="/live/rgb", name="RGB"), name="Playback"),
        ), collapse_panels=True))
        print("rrd:", os.path.join(a.out, "map_from_depth.rrd"))
    print(f"완료 [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
