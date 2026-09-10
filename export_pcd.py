#!/usr/bin/env python3
"""Map Package → .pcd (binary). 필드 x y z rgb(packed float) label(uint32; 클래스 사전 순서, other=N, unknown=N+1)
source: voxels(0.1m 전체) | voxels_occ(정합용 정적 표면: occupancy free·동적 제거, person/car 제외) | occupancy(0.2m occupied 셀, rgb 없음)
사용: python export_pcd.py <manifest> <out.pcd> [--source voxels] [--exclude person,car]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_map import MapPackage
import classdict as CD


def write_pcd(path, xyz, rgb=None, label=None):
    n = len(xyz); fields = ["x", "y", "z"]; types = ["F"] * 3; arrs = [xyz[:, 0].astype("<f4"), xyz[:, 1].astype("<f4"), xyz[:, 2].astype("<f4")]
    if rgb is not None:
        packed = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
        fields.append("rgb"); types.append("F"); arrs.append(packed.astype("<u4").view("<f4"))          # PCL 관례: uint32 를 float 비트로
    if label is not None:
        fields.append("label"); types.append("U"); arrs.append(label.astype("<u4"))
    hdr = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS " + " ".join(fields) + "\nSIZE " + " ".join(["4"] * len(fields)) +
           "\nTYPE " + " ".join(types) + "\nCOUNT " + " ".join(["1"] * len(fields)) + f"\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    rec = np.empty(n, dtype=[(f, "<f4" if t == "F" else "<u4") for f, t in zip(fields, types)])
    for f, a in zip(fields, arrs): rec[f] = a
    with open(path, "wb") as fh: fh.write(hdr.encode("ascii")); fh.write(rec.tobytes())
    return n


def export(manifest, out, source="voxels", exclude=("person", "car")):
    mp = MapPackage(manifest); names = mp.classes; lab = mp.labels(); n_cls = len(names)
    if source in ("voxels", "voxels_occ"):
        keep = np.ones(len(lab), bool)
        if source == "voxels_occ":
            ex = [names.index(c) for c in exclude if c in names]; keep &= ~np.isin(lab, ex)
            if mp.occ is not None: keep &= (mp.surf_state != 0) & ~mp.surf_removed
        xyz = mp.voxel_centers()[keep]; rgb = mp.rgb[keep]; label = lab[keep]
    elif source == "occupancy":
        O = mp.occ; occm = O["state"] == 1; xyz = ((O["ijk"] + 0.5) * O["res"])[occm]; rgb = None; label = None
    else:
        raise ValueError(source)
    n = write_pcd(out, xyz, rgb, label)
    legend = {"source": source, "points": int(n), "frame": mp.man.get("frame"), "label_ids": {**{i: c for i, c in enumerate(names)}, n_cls: "other", n_cls + 1: "unknown"},
              "classdict": mp.classdict.get("name"), "excluded": list(exclude) if source == "voxels_occ" else [], "rgb": "packed uint32 (r<<16|g<<8|b) stored as float bits (PCL convention)"}
    json.dump(legend, open(os.path.splitext(out)[0] + "_legend.json", "w"), ensure_ascii=False, indent=1)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("out"); ap.add_argument("--source", default="voxels", choices=["voxels", "voxels_occ", "occupancy"]); ap.add_argument("--exclude", default="person,car")
    a = ap.parse_args(); t = time.time(); n = export(a.manifest, a.out, a.source, tuple(x for x in a.exclude.split(",") if x))
    print(f"{n:,} points → {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB, {time.time()-t:.1f}s)")
