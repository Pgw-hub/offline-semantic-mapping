#!/usr/bin/env python3
"""voxel3d 타일 폴더(voxel3d_tiles v2) → session_voxel3d_<이름>.npz (포털 변형·Potree 레이어·PCD 내보내기가 읽는 형식).
계단 채움 결과(session_voxel3d_stair_tiles/, options.stair_complete)는 타일로만 나오므로 이걸로 npz 를 만들면 Potree 에 'voxel3d_stair' 레이어로 보인다.
  python voxel3d_tiles_to_npz.py <tiles_dir> <out.npz> [--base session_voxel3d.npz]   # base: res·classes·params 를 가져올 본 맵 npz (기본 같은 폴더의 session_voxel3d.npz)
"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("tiles"); ap.add_argument("out"); ap.add_argument("--base", default=None); a = ap.parse_args()
idx = json.load(open(os.path.join(a.tiles, "index.json"))); assert idx["format"] == "voxel3d_tiles v2", idx["format"]
REC = np.dtype([(n, t) for n, t in idx["record"]])
R = np.concatenate([np.fromfile(os.path.join(a.tiles, t["file"]), REC) for t in idx["tiles"]])
base = a.base or os.path.join(os.path.dirname(os.path.abspath(a.tiles)), "session_voxel3d.npz")
B = np.load(base, allow_pickle=True) if os.path.exists(base) else None
classes = B["classes"] if B is not None else np.array(idx.get("classes", []), "<U16")
params = json.loads(str(B["params"])) if B is not None else {}
params["source_codes"] = idx.get("source_codes", params.get("source_codes", {})); params["from_tiles"] = os.path.basename(a.tiles)
np.savez_compressed(a.out, ijk=np.stack([R["ix"].astype(np.int32), R["iy"].astype(np.int32), R["iz"].astype(np.int32)], 1),
                    mean_z=R["mean_z"].astype(np.float32), max_z=R["max_z"].astype(np.float32), count=R["count"].astype(np.uint16),
                    logodds=R["logodds"].astype(np.float16), cls=R["cls"].astype(np.uint8), source=R["src"].astype(np.uint8),
                    res_xy=np.float64(idx["res_xy"]), res_z=np.float64(idx["res_z"]), prob_res=np.float64(idx.get("prob_res", idx["res_z"])),
                    classes=classes, params=json.dumps(params, ensure_ascii=False))
print(f"{len(R):,} 셀 → {a.out} (src: { {int(s): int((R['src']==s).sum()) for s in np.unique(R['src'])} })")
