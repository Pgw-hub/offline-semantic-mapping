#!/usr/bin/env python3
"""voxel3d 산출물에서 채움 셀(기본 평탄 채움 source 3, --drop-source 2 3 이면 depth 채움까지)을 뺀 변형을 만든다 → session_voxel3d[_<base>]_<suffix>.npz + _tiles/ (포털이 변형으로 자동 인식).
  python tools/voxel3d_drop_fill.py <compiled_dir> [--base ""|iso025] [--drop-source 3]
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("compiled"); ap.add_argument("--base", default=""); ap.add_argument("--drop-source", type=int, nargs="+", default=[3]); ap.add_argument("--suffix", default="nofill", help="변형 이름 접미사 (예: lidaronly = depth·평탄 채움 모두 제외)"); a = ap.parse_args()
src_name = "session_voxel3d" + (f"_{a.base}" if a.base else ""); dst_name = src_name + "_" + a.suffix
v = dict(np.load(os.path.join(a.compiled, src_name + ".npz"))); keep = ~np.isin(v["source"], a.drop_source)
n = len(keep)
for k, arr in list(v.items()):
    if isinstance(arr, np.ndarray) and arr.ndim >= 1 and len(arr) == n and k not in ("coarse_ijk",): v[k] = arr[keep]
prm = json.loads(str(v["params"])); prm["dropped_sources"] = a.drop_source; prm["fill_flat"] = 0; v["params"] = np.array(json.dumps(prm, ensure_ascii=False))
np.savez_compressed(os.path.join(a.compiled, dst_name + ".npz"), **v)
td = os.path.join(a.compiled, dst_name + "_tiles"); shutil.rmtree(td, ignore_errors=True)
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "voxel3d_tiles_filter.py"), os.path.join(a.compiled, src_name + "_tiles"), td, "--drop-source", *map(str, a.drop_source)], check=True)
s = os.path.join(a.compiled, src_name + "_summary.json")
if os.path.exists(s):
    j = json.load(open(s)); j.update(cells=int(keep.sum()), cells_flat_fill=0, **({"cells_depth": 0} if 2 in a.drop_source else {}), total_mb=round(int(keep.sum()) * 24 / 1e6, 1)); json.dump(j, open(os.path.join(a.compiled, dst_name + "_summary.json"), "w"), indent=1, ensure_ascii=False)
print(f"{dst_name}: {n:,} → {int(keep.sum()):,} 셀 ({int(keep.sum())*24/1e6:.1f} MB)")
