"""mcap 레코더의 `pointcloud2` typekey 레이아웃 파서 (20260914 sim 녹화에서 확인).
레이아웃: stamp{i32 sec; u32 nsec}; u32 flen; frame_id; u32 h,w,point_step,row_step; u8 is_bigendian; u8 is_dense;
          u32 F; F*{u32 nlen; name; u32 offset; u8 datatype; u32 count}; u32 len; data
사용: hdr, P = parse_pc2(msg.data)  → hdr(dict: ns, frame, npts, fields), P(structured ndarray: x,y,z,… 필드명 그대로)
"""
import struct
import numpy as np
_DT = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}

def parse_pc2(d):
    o = 0; sec, nsec = struct.unpack_from('<iI', d, o); o += 8
    flen, = struct.unpack_from('<I', d, o); o += 4; frame = d[o:o + flen].decode(errors='replace'); o += flen
    h, w, ps, rs = struct.unpack_from('<4I', d, o); o += 16; be, dense = d[o], d[o + 1]; o += 2
    nf, = struct.unpack_from('<I', d, o); o += 4; names, fmts, offs = [], [], []
    for _ in range(nf):
        nlen, = struct.unpack_from('<I', d, o); o += 4; name = d[o:o + nlen].decode(); o += nlen
        off, = struct.unpack_from('<I', d, o); o += 4; dt = d[o]; o += 1; cnt, = struct.unpack_from('<I', d, o); o += 4
        names.append(name); fmts.append(_DT[dt] if cnt == 1 else f'{cnt}{_DT[dt]}'); offs.append(off)
    dl, = struct.unpack_from('<I', d, o); o += 4; npts = h * w
    dtype = np.dtype({'names': names, 'formats': fmts, 'offsets': offs, 'itemsize': ps})
    P = np.frombuffer(d, dtype, npts, o)
    return {'ns': sec * 10**9 + nsec, 'frame': frame, 'npts': npts, 'point_step': ps, 'fields': names, 'is_dense': bool(dense), 'bigendian': bool(be)}, P

def xyz_of(P):
    return np.stack([P['x'], P['y'], P['z']], 1).astype(np.float64)


# ---- /livox/lidar 공용 디코더: raisin CustomMsg(구 레코더 `pointcloud`) 또는 PointCloud2(`pointcloud2`, 2026-09-14 이후) 자동 판별
import os as _os
LIVOX_DT = np.dtype([("ot", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("refl", "u1"), ("tag", "u1"), ("line", "u1")])

def livox_flip():
    """점 좌표 180° yaw 보정(-x,-y) 여부. 실로봇 Livox 드라이버 프레임은 base 와 180° 어긋나 있어 기본 켬; sim(raisim livox) 은 항등 → LIVOX_FLIP=0."""
    return _os.environ.get("LIVOX_FLIP", "1") not in ("0", "false", "False")

def is_pc2(data):
    flen, = struct.unpack_from("<I", data, 8)
    return 0 < flen < 128 and all(32 <= b < 127 for b in data[12:12 + flen])

def decode_livox(data, flip=None):
    """→ (stamp_ns, xyz float32 (N,3) [flip 적용], ot_ns int64 (N,) 점별 시각 오프셋, refl uint8 (N,))"""
    flip = livox_flip() if flip is None else flip
    sec, nsec = struct.unpack_from("<iI", data, 0); ns = sec * 10**9 + nsec
    if is_pc2(data):
        h, P = parse_pc2(data); x, y, z = P["x"].astype(np.float32), P["y"].astype(np.float32), P["z"].astype(np.float32)
        ot = P["t"].astype(np.int64) if "t" in P.dtype.names else np.zeros(len(P), np.int64)
        if "t" in P.dtype.names and ot.max() > 10**9: ot = ot - ot.min()             # 절대 ns 로 온 경우 → 오프셋
        refl = P["intensity"].astype(np.uint8) if "intensity" in P.dtype.names else np.zeros(len(P), np.uint8)
    else:
        N, = struct.unpack_from("<I", data, 24); p = np.frombuffer(data, dtype=LIVOX_DT, count=N, offset=28)
        x, y, z, ot, refl = p["x"], p["y"], p["z"], p["ot"].astype(np.int64), p["refl"].copy()
    xyz = np.stack([-x, -y, z], -1) if flip else np.stack([x, y, z], -1)
    return ns, xyz.astype(np.float32), ot, refl
