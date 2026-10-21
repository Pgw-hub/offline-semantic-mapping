"""raisin mcap 메시지 디코더 — 레코더 두 세대의 레이아웃을 한 API 로.
  v1 (2026-09-16 까지): 압축 레이아웃. Odometry = stamp + 7d(pos, quat xyzw) + 6d, PointCloud2 = stamp + frame + h,w,ps,rs + …, Image = zlib(uint16 mm / rgb8).
  v2 (2026-09-23 이후, raisin_pod 전체 ROS 레이아웃): 모든 메시지가 `stamp{i32 sec; u32 nsec}`(레코더 수신 시각) + ROS 메시지 본문(header{stamp; str frame_id} 포함).
     Image 는 압축 없이 raw (depth 32FC1 float m 또는 16UC1 mm), 문자열은 u32 길이 + bytes.
  판별: 채널 스키마 텍스트에 'nanosec' 가 있으면 v2 (v1 스키마는 nsec 만 씀).
모든 헬퍼는 (헤더 stamp ns, …) 를 돌려준다. v2 의 헤더 stamp 는 발행자 시각(offset 8)이고 offset 0 의 stamp 는 수신 시각이라 sim 에서는 다를 수 있다.
사용: from preprocess.geometry.raisin_msgs import is_v2, odom, odom13, path_poses, pc2, depth_image, rgb_image, robot_state, stamp_ns
      for sch, ch, m in reader.iter_messages(...): ns, pos, q = odom(m.data, sch)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))  # scripts/ (패키지 import)
import struct, zlib
import numpy as np

_V2_CACHE = {}
def is_v2(sch):
    """sch: mcap Schema (또는 스키마 텍스트/bytes). v2(ROS 전체 레이아웃) 여부."""
    if sch is None: return False
    key = id(sch)
    if key in _V2_CACHE: return _V2_CACHE[key]
    txt = sch if isinstance(sch, (bytes, str)) else getattr(sch, "data", b"")
    if isinstance(txt, bytes): txt = txt.decode(errors="replace")
    v = "nanosec" in txt; _V2_CACHE[key] = v; return v

class _Cur:
    __slots__ = ("d", "o")
    def __init__(self, d, o=0): self.d = d; self.o = o
    def u(self, fmt):
        v = struct.unpack_from(fmt, self.d, self.o); self.o += struct.calcsize(fmt); return v
    def s(self):
        n, = struct.unpack_from("<I", self.d, self.o); self.o += 4; v = self.d[self.o:self.o + n]; self.o += n; return v.decode(errors="replace")
    def raw(self, n):
        v = self.d[self.o:self.o + n]; self.o += n; return v
    def skip(self, n): self.o += n
    def header(self):
        sec, nsec = self.u("<iI"); frame = self.s(); return sec * 10**9 + nsec, frame

def stamp_ns(d, sch=None):
    """헤더(발행자) stamp ns. v1: offset 0. v2: header.stamp (offset 8; 앞 8 B 는 수신 시각). robot_state v2 는 timestamp 필드."""
    if is_v2(sch):
        sec, nsec = struct.unpack_from("<iI", d, 8); return sec * 10**9 + nsec
    sec, nsec = struct.unpack_from("<iI", d, 0); return sec * 10**9 + nsec
def recv_ns(d):
    sec, nsec = struct.unpack_from("<iI", d, 0); return sec * 10**9 + nsec

# ---------------- nav_msgs/Odometry ----------------
def odom13(d, sch=None):
    """→ (ns, v[13]) v = pos xyz, quat xyzw, lin xyz, ang xyz (v1 레이아웃 순서 그대로)"""
    if not is_v2(sch):
        sec, nsec = struct.unpack_from("<iI", d, 0); return sec * 10**9 + nsec, struct.unpack_from("<13d", d, 8)
    c = _Cur(d, 8); ns, _ = c.header(); c.s()                       # child_frame_id
    pose = c.u("<7d"); c.skip(36 * 8); tw = c.u("<6d")
    return ns, pose + tw
def odom(d, sch=None):
    """→ (ns, pos(3,), quat xyzw(4,))"""
    ns, v = odom13(d, sch); return ns, np.array(v[:3]), np.array(v[3:7])

# ---------------- nav_msgs/Path ----------------
def path_poses(d, sch=None):
    """→ (P (N,7) [x y z qx qy qz qw], pose_ns (N,) int64 또는 None(v1: 자세별 stamp 없음))"""
    if not is_v2(sch):
        N = struct.unpack_from("<I", d, 8)[0]; return np.frombuffer(d, "<f8", N * 7, 12).reshape(N, 7).copy(), None
    c = _Cur(d, 8); c.header(); N, = c.u("<I"); P = np.empty((N, 7)); T = np.empty(N, np.int64)
    for i in range(N):
        T[i], _ = c.header(); P[i] = c.u("<7d")
    return P, T

# ---------------- sensor_msgs/PointCloud2 ----------------
_DT = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
def _pc2_body(c):
    """cursor 가 header 앞에 있을 때 PointCloud2 본문 파싱 → (hdr dict, structured array)"""
    ns, frame = c.header(); h, w, nf = c.u("<3I"); names, fmts, offs = [], [], []
    for _ in range(nf):
        name = c.s(); off, = c.u("<I"); dt, = c.u("<B"); cnt, = c.u("<I")
        names.append(name); offs.append(off); fmts.append(_DT[dt] if cnt == 1 else f"({cnt},){_DT[dt]}")
    be, = c.u("<B"); ps, rs = c.u("<2I"); dl, = c.u("<I"); data = c.raw(dl); npts = h * w
    dtype = np.dtype({"names": names, "formats": fmts, "offsets": offs, "itemsize": ps})
    P = np.frombuffer(data, dtype=dtype, count=min(npts, dl // ps)) if ps else np.zeros(0, dtype)
    return {"ns": ns, "frame": frame, "npts": len(P), "fields": names, "point_step": ps}, P
def pc2(d, sch=None):
    """→ (hdr{ns, frame, npts, fields}, P structured array) — v1 은 pc2_reader.parse_pc2 와 동일 결과"""
    if not is_v2(sch):
        from preprocess.geometry.pc2_reader import parse_pc2; return parse_pc2(d)
    return _pc2_body(_Cur(d, 8))
def keyframe_cloud(d, sch=None):
    """raisin_interfaces/KeyframeCloud → (idx, hdr, P). v1 레이아웃은 pc2_reader 쪽 규약(idx + PointCloud2) 을 따른다."""
    if not is_v2(sch):
        idx, = struct.unpack_from("<Q", d, 8); from preprocess.geometry.pc2_reader import parse_pc2; h, P = parse_pc2(d[16:]); return idx, h, P
    c = _Cur(d, 8); idx, = c.u("<Q"); h, P = _pc2_body(c); return idx, h, P

# ---------------- sensor_msgs/Image ----------------
def depth_image(d, sch=None):
    """→ (ns, depth float32 (H,W) [m]). v1: zlib uint16 mm. v2: raw 32FC1(m) / 16UC1(mm)."""
    if not is_v2(sch):
        sec, nsec, w, h, rawlen, zlen = struct.unpack_from("<iIIIII", d, 0)
        dep = np.frombuffer(zlib.decompress(d[24:24 + zlen]), np.uint16).reshape(h, w).astype(np.float32) / 1000.0
        return sec * 10**9 + nsec, dep
    c = _Cur(d, 8); ns, _ = c.header(); h, w = c.u("<2I"); enc = c.s(); c.u("<B"); step, = c.u("<I"); n, = c.u("<I"); buf = c.raw(n)
    if enc in ("32FC1",): dep = np.frombuffer(buf, "<f4", h * w).reshape(h, w).astype(np.float32)
    elif enc in ("16UC1", "mono16"): dep = np.frombuffer(buf, "<u2", h * w).reshape(h, w).astype(np.float32) / 1000.0
    else: raise ValueError(f"depth encoding {enc}")
    dep = np.nan_to_num(dep, nan=0.0, posinf=0.0, neginf=0.0); return ns, dep
def depth_image_mm(d, sch=None):
    """→ (ns, uint16 mm (H,W)) — 옛 코드 호환"""
    ns, dep = depth_image(d, sch); return ns, np.clip(dep * 1000.0, 0, 65535).astype(np.uint16)
def rgb_image(d, sch=None, order=None):
    """→ (ns, uint8 (H,W,3)) — 채널 순서는 녹화된 그대로(v1: raw rgb8 로 기록됐다고 명시되어 있었으나 실제 파일은 BGR 로 다뤄 왔음; 호출 측 관례 유지). v2: encoding 이 bgr8 이면 그대로, rgb8 도 그대로 돌려주고 enc 를 함께 반환하지 않는다."""
    if not is_v2(sch):
        sec, nsec, w, h, chn, rawlen, zlen = struct.unpack_from("<iIIIIII", d, 0)
        return sec * 10**9 + nsec, np.frombuffer(zlib.decompress(d[28:28 + zlen]), np.uint8).reshape(h, w, chn)
    c = _Cur(d, 8); ns, _ = c.header(); h, w = c.u("<2I"); enc = c.s(); c.u("<B"); step, = c.u("<I"); n, = c.u("<I"); buf = c.raw(n)
    if enc in ("rgb8", "bgr8"): img = np.frombuffer(buf, np.uint8, h * w * 3).reshape(h, w, 3)
    elif enc in ("rgba8", "bgra8"): img = np.frombuffer(buf, np.uint8, h * w * 4).reshape(h, w, 4)[:, :, :3]
    elif enc in ("mono8",): img = np.repeat(np.frombuffer(buf, np.uint8, h * w).reshape(h, w, 1), 3, 2)
    else: raise ValueError(f"rgb encoding {enc}")
    if order and enc[:3] in ("rgb", "bgr") and enc[:3] != order: img = np.ascontiguousarray(img[:, :, ::-1])   # 요청 채널 순서로 (v1 은 파일 그대로 — 기존 관례 유지)
    return ns, img
def image_encoding(d, sch=None):
    if not is_v2(sch): return None
    c = _Cur(d, 8); c.header(); c.u("<2I"); return c.s()

# ---------------- raisin_interfaces/FfmpegPacket ----------------
def ffmpeg_packet(d, sch=None):
    """→ (ns, payload bytes, w, h, encoding, pts, flags)"""
    if not is_v2(sch):
        c = _Cur(d, 0); sec, nsec = c.u("<iI"); w, h = c.u("<ii"); enc = c.s(); pts, = c.u("<q"); flags, = c.u("<B"); c.u("<B"); n, = c.u("<I")
        return sec * 10**9 + nsec, c.raw(n), w, h, enc, pts, flags
    c = _Cur(d, 8); ns, _ = c.header(); n, = c.u("<I"); payload = c.raw(n); w, h = c.u("<ii"); enc = c.s(); pts, = c.u("<Q"); flags, = c.u("<B")
    return ns, payload, w, h, enc, pts, flags

# ---------------- raisin_interfaces/RobotState ----------------
def robot_state(d, sch=None):
    """→ (ns, base_pos(3,), base_quat wxyz(4,))"""
    if not is_v2(sch):
        sec, nsec = struct.unpack_from("<iI", d, 0); v = struct.unpack_from("<7d", d, 12); return sec * 10**9 + nsec, np.array(v[:3]), np.array(v[3:7])
    c = _Cur(d, 8); sec, nsec = c.u("<iI"); N, = c.u("<I")
    for _ in range(N): c.s(); c.u("<3d"); c.u("<H"); c.u("<d")
    v = c.u("<7d"); return sec * 10**9 + nsec, np.array(v[:3]), np.array(v[3:7])

# ---------------- LiDAR 스캔 (livox CustomMsg / PointCloud2) ----------------
def lidar_scan(d, sch=None, flip=None):
    """→ (ns, xyz f32 (N,3), offset_time ns (N,) int64|None, reflectivity|None). v1 은 pc2_reader.decode_livox 그대로.
    v2 PointCloud2 필드: x y z intensity [tag line] [timestamp|t|time|offset_time]. LIVOX_FLIP 은 pc2_reader 와 같은 규약."""
    if not is_v2(sch):
        from preprocess.geometry.pc2_reader import decode_livox; return decode_livox(d, flip)
    h, P = _pc2_body(_Cur(d, 8)); names = P.dtype.names
    xyz = np.stack([P["x"], P["y"], P["z"]], -1).astype(np.float32)
    from preprocess.geometry.pc2_reader import livox_flip
    if (livox_flip() if flip is None else flip): xyz[:, 0] *= -1; xyz[:, 1] *= -1
    ot = None
    for k in ("t", "time", "offset_time", "timestamp"):
        if k in names:
            v = P[k].astype(np.float64)
            ot = (v - (v.min() if k == "timestamp" else 0)).astype(np.int64) if v.max() > 1e6 else (v * 1e9).astype(np.int64)   # ns 단위 정수 vs 초 단위 float
            break
    refl = P["intensity"].astype(np.float32) if "intensity" in names else (P["reflectivity"].astype(np.float32) if "reflectivity" in names else None)
    return h["ns"], xyz, ot, refl


# ---------------- tf_static ----------------
def tf_static(mcap_path, child_suffix, topic_suffix="/tf_static"):
    """mcap 의 <nick>/tf_static 에서 child frame 이 child_suffix 로 끝나는 변환 → (t(3,), R(3,3), parent, child) 또는 None"""
    from mcap.reader import make_reader
    with open(mcap_path, "rb") as fh:
        r = make_reader(fh); s = r.get_summary(); topics = [ch.topic for ch in s.channels.values() if ch.topic.endswith(topic_suffix)]
        if not topics: return None
        for sch, ch, m in r.iter_messages(topics=topics):
            c = _Cur(m.data, 8); ns, parent = c.header(); child = c.s(); t = np.array(c.u("<3d")); x, y, z, w = c.u("<4d")
            if child.endswith(child_suffix):
                R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
                return t, R, parent, child
    return None
def has_topic(mcap_path, topic):
    from mcap.reader import make_reader
    with open(mcap_path, "rb") as fh:
        s = make_reader(fh).get_summary(); return any(ch.topic == topic for ch in s.channels.values())

# ---------------- nav_msgs/Odometry frame_id ----------------
def odom_frame(d, sch=None):
    """→ header.frame_id (v2) 또는 None (v1 레이아웃엔 없음). 위치추정 모드로 바뀌면 저장 맵 이름으로 바뀐다."""
    if not is_v2(sch): return None
    return _Cur(d, 8).header()[1]

# ---------------- 레코더 자기 진단 (raisin_mcap_recorder_plugin, raisin_pod 레이아웃) ----------------
def recorder_diagnostics(d):
    """__recorder/diagnostics → dict(ns, enqueued, dropped, queue_depth, queue_cap, topics{topic: (received, written, published, pub_state, bytes)}, write_failed, open_failed, manifest_failed, free_mb).
    dropped 는 큐 넘침 · 직렬화 실패 · 저장 오류로 취소된 작업을 겹쳐 센 값이라 다른 카운터와 더하지 않는다 (스키마 주석)."""
    c = _Cur(d, 0); sec, nsec = c.u("<iI"); enq, drop = c.u("<2Q"); qd, qc, N = c.u("<3I"); topics = {}
    for _ in range(N):
        t = c.s(); rcv, wr, pub = c.u("<3Q"); ps, = c.u("<B"); nb, lw = c.u("<2Q"); c.u("<2B"); topics[t] = (rcv, wr, pub, ps, nb)
    wf, of, mf, free = c.u("<4Q")
    return {"ns": sec * 10**9 + nsec, "enqueued": enq, "dropped": drop, "queue_depth": qd, "queue_cap": qc, "topics": topics,
            "write_failed": wf, "open_failed": of, "manifest_failed": mf, "free_mb": free}
def recorder_system(d):
    """__recorder/system → dict. cpu/disk 값은 직전 표본 대비 증분, psi_* 는 10 s 평균 %(some), wake_late_ms 는 표본 스레드가 늦게 깬 최대 시간."""
    c = _Cur(d, 0); sec, nsec = c.u("<iI"); wall, up = c.u("<2Q"); l1, l5, l15, cpu, pc, pi, pm = c.u("<7f")
    mt, ma, st, sf, dirty, wb, rss = c.u("<7Q"); thr, = c.u("<I"); scpu, vcs, nvcs, wl, ws, dt, da, dr, dw = c.u("<9Q")
    return {"ns": sec * 10**9 + nsec, "load1": l1, "cpu_busy_pct": cpu, "psi_cpu": pc, "psi_io": pi, "psi_mem": pm,
            "mem_total_mb": mt / 1024, "mem_avail_mb": ma / 1024, "dirty_mb": dirty / 1024, "writeback_mb": wb / 1024,
            "wake_late_ms": wl / 1e6, "disk_avail_mb": da, "disk_read_kb": dr, "disk_write_kb": dw}
