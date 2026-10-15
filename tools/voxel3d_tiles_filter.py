#!/usr/bin/env python3
"""voxel3d_tiles v2 → 같은 형식, 지정한 source 의 셀을 뺀 사본 (예: 평탄 채움 source 3 제거).

build_voxel3d.py 의 --fill-flat 은 셀을 '추가'만 하므로 source 3 을 빼면 --fill-flat 0 으로 다시 돌린 것과 같다.
사용: voxel3d_tiles_filter.py <in_tiles_dir> <out_tiles_dir> [--drop-source 3] [--frame-note TEXT]
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir"); ap.add_argument("out_dir")
    ap.add_argument("--drop-source", type=int, nargs="+", default=[3])
    ap.add_argument("--frame-note", default=None, help="index.json 의 frame 설명 덮어쓰기")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.in_dir, "index.json")))
    assert idx["format"] == "voxel3d_tiles v2", idx["format"]
    rec = np.dtype([(n, t) for n, t in idx["record"]])
    assert rec.itemsize == idx["record_bytes"]
    os.makedirs(a.out_dir, exist_ok=True)
    tiles, start, kept_total, dropped_total = [], 0, 0, 0
    for t in idx["tiles"]:
        R = np.fromfile(os.path.join(a.in_dir, t["file"]), rec)
        keep = ~np.isin(R["src"], a.drop_source)
        dropped_total += int((~keep).sum())
        R = R[keep]
        if len(R) == 0:
            continue
        R.tofile(os.path.join(a.out_dir, t["file"]))
        tiles.append({**t, "start": start, "n": int(len(R)), "z_min": round(float(R["mean_z"].min()), 2),
                      "z_max": round(float(R["mean_z"].max()), 2), "bytes": int(len(R) * rec.itemsize)})
        start += len(R)
        kept_total += len(R)
    out = {**idx, "n_cells": kept_total, "total_bytes": kept_total * rec.itemsize, "tiles": tiles,
           "derived_from": os.path.abspath(a.in_dir), "dropped_sources": a.drop_source}
    if 3 in a.drop_source:
        out["fill_flat"] = 0
    if a.frame_note:
        out["frame"] = a.frame_note
    json.dump(out, open(os.path.join(a.out_dir, "index.json"), "w"), indent=1, ensure_ascii=False)
    print(f"kept {kept_total:,} cells ({kept_total * rec.itemsize / 1e6:.1f} MB, {len(tiles)} tiles), dropped {dropped_total:,}")


if __name__ == "__main__":
    main()
