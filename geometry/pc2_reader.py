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
