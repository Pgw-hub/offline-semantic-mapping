#!/usr/bin/env python3
"""두 Map Package 를 한 rrd 에 나란히 (before/ · after/) — 같은 데이터셋의 두 빌드를 비교.
사용: python compare_maps_rerun.py <manifest_A> <manifest_B> <out.rrd> [--label-a before] [--label-b after] [--sub 1]
탭: 'Side by side' (좌 A, 우 B 3D) · 'Overlay' (A 회색 반투명 + B 색) · 각 단독.
"""
import argparse, os, sys
import numpy as np
import rerun as rr, rerun.blueprint as rrb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_map import MapPackage
import classdict as CD

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("out")
ap.add_argument("--label-a", default="before"); ap.add_argument("--label-b", default="after"); ap.add_argument("--sub", type=int, default=1)
a = ap.parse_args()
rr.init("compare_maps"); rr.save(a.out)

def log_map(prefix, mp, sub):
    ctr = mp.voxel_centers(); lab = mp.labels(); n_cls = len(mp.classes)
    pal = mp.colors + [CD.hex2rgb(CD.OTHER_COLOR), CD.hex2rgb(CD.UNKNOWN_COLOR)]
    idx = np.arange(len(ctr))[::sub]
    for ci, cname in enumerate(mp.classes):
        sel = idx[lab[idx] == ci]
        if len(sel): rr.log(f"{prefix}/voxels/semantic/{cname}", rr.Points3D(ctr[sel], colors=[list(pal[ci])], radii=mp.vox * 0.45))
    oth = idx[lab[idx] == n_cls]; unk = idx[lab[idx] == n_cls + 1]
    rr.log(f"{prefix}/voxels/other", rr.Points3D(ctr[oth], colors=[list(pal[n_cls])], radii=mp.vox * 0.45))
    rr.log(f"{prefix}/voxels/unknown", rr.Points3D(ctr[unk], colors=[list(pal[n_cls + 1])], radii=mp.vox * 0.35))
    for i, o in enumerate(mp.objects):
        tag = o.get("tag") or "?"
        if tag == "person": continue
        cen = np.array(o.get("center_w"), float); ext = np.array(o.get("extent") or [0.5] * 3, float)
        rr.log(f"{prefix}/objects/{tag.replace(' ', '_')}_{i}", rr.Boxes3D(centers=[cen], half_sizes=[ext / 2], labels=[tag], colors=[[255, 200, 40]]))
    return {n: int((lab == i).sum()) for i, n in enumerate(mp.classes + ["other", "unknown"])}

A = MapPackage(a.a); B = MapPackage(a.b)
sa = log_map(a.label_a, A, a.sub); sb = log_map(a.label_b, B, a.sub)
print(a.label_a, sa); print(a.label_b, sb)
hide = lambda p: [f"- /{p}/voxels/unknown", f"- /{p}/voxels/other"]
va = rrb.Spatial3DView(origin=f"/{a.label_a}", name=f"{a.label_a}", contents=[f"+ /{a.label_a}/**"] + hide(a.label_a))
vb = rrb.Spatial3DView(origin=f"/{a.label_b}", name=f"{a.label_b}", contents=[f"+ /{a.label_b}/**"] + hide(a.label_b))
bp = rrb.Blueprint(rrb.Tabs(
    rrb.Horizontal(va, vb, name="Side by side"),
    rrb.Spatial3DView(origin="/", name="Both (toggle in tree)", contents=[f"+ /{a.label_a}/voxels/semantic/**", f"+ /{a.label_b}/voxels/semantic/**"]),
    rrb.Spatial3DView(origin=f"/{a.label_a}", name=f"{a.label_a} only", contents=[f"+ /{a.label_a}/**"] + hide(a.label_a)),
    rrb.Spatial3DView(origin=f"/{a.label_b}", name=f"{a.label_b} only", contents=[f"+ /{a.label_b}/**"] + hide(a.label_b))),
    rrb.TimePanel(expanded=False), rrb.SelectionPanel(expanded=False))
rr.send_blueprint(bp, make_default=True)
print("저장:", a.out, f"({os.path.getsize(a.out)/1e6:.0f} MB)")
