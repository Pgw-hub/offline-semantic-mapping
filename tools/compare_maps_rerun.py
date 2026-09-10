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
    if mp.occ is not None:                                   # 확률적 occupancy 의 occupied 셀 — 높이 색. 기하 번짐(deskew/PGO) 차이는 여기서 가장 잘 보임
        O = mp.occ; occ = O["state"] == 1; C = (O["ijk"][occ].astype(np.float64) + 0.5) * O["res"]
        z = C[:, 2]; zn = np.clip((z - np.percentile(z, 2)) / max(np.percentile(z, 98) - np.percentile(z, 2), 1e-3), 0, 1)
        col = (np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], 1) * 255).astype(np.uint8)
        rr.log(f"{prefix}/occupancy/occupied", rr.Points3D(C.astype(np.float32), colors=col, radii=O["res"] * 0.45))
        print(f"  {prefix}: occupancy occupied {int(occ.sum()):,} cells")
    return {n: int((lab == i).sum()) for i, n in enumerate(mp.classes + ["other", "unknown"])}

A = MapPackage(a.a); B = MapPackage(a.b)
sa = log_map(a.label_a, A, a.sub); sb = log_map(a.label_b, B, a.sub)
if A.occ is not None and B.occ is not None and A.occ["res"] == B.occ["res"]:      # Difference: 한쪽에만 occupied 인 셀
    R = A.occ["res"]
    def keys(O): I = O["ijk"][O["state"] == 1].astype(np.int64) + (1 << 20); return (I[:, 0] << 42) | (I[:, 1] << 21) | I[:, 2]
    ka, kb = keys(A.occ), keys(B.occ); only_a = np.setdiff1d(ka, kb); only_b = np.setdiff1d(kb, ka)
    def xyz(k): m = (1 << 21) - 1; return (np.stack([(k >> 42) & m, (k >> 21) & m, k & m], -1).astype(np.float64) - (1 << 20) + 0.5) * R
    rr.log("difference/only_in_" + a.label_a.split()[0], rr.Points3D(xyz(only_a).astype(np.float32), colors=[[230, 60, 60]], radii=R * 0.45))
    rr.log("difference/only_in_" + a.label_b.split()[0], rr.Points3D(xyz(only_b).astype(np.float32), colors=[[60, 120, 255]], radii=R * 0.45))
    rr.log("difference/common", rr.Points3D(xyz(np.intersect1d(ka, kb)).astype(np.float32), colors=[[150, 150, 150]], radii=R * 0.2))
    print(f"occupancy 차이: 공통 {len(np.intersect1d(ka, kb)):,} · {a.label_a} 만 {len(only_a):,} · {a.label_b} 만 {len(only_b):,}")
print(a.label_a, sa); print(a.label_b, sb)
hide = lambda p: [f"- /{p}/voxels/unknown", f"- /{p}/voxels/other", f"- /{p}/occupancy/**"]
va = rrb.Spatial3DView(origin=f"/{a.label_a}", name=f"{a.label_a}", contents=[f"+ /{a.label_a}/**"] + hide(a.label_a))
vb = rrb.Spatial3DView(origin=f"/{a.label_b}", name=f"{a.label_b}", contents=[f"+ /{a.label_b}/**"] + hide(a.label_b))
oa = rrb.Spatial3DView(origin=f"/{a.label_a}", name=f"{a.label_a} occupancy", contents=[f"+ /{a.label_a}/occupancy/**"])
ob = rrb.Spatial3DView(origin=f"/{a.label_b}", name=f"{a.label_b} occupancy", contents=[f"+ /{a.label_b}/occupancy/**"])
bp = rrb.Blueprint(rrb.Tabs(
    rrb.Horizontal(va, vb, name="Side by side"),
    rrb.Horizontal(oa, ob, name="Occupancy side by side"),
    rrb.Spatial3DView(origin="/difference", name="Difference (red=old only, blue=new only)"),
    rrb.Spatial3DView(origin="/", name="Both (toggle in tree)", contents=[f"+ /{a.label_a}/voxels/semantic/**", f"+ /{a.label_b}/voxels/semantic/**"]),
    rrb.Spatial3DView(origin=f"/{a.label_a}", name=f"{a.label_a} only", contents=[f"+ /{a.label_a}/**"] + hide(a.label_a)),
    rrb.Spatial3DView(origin=f"/{a.label_b}", name=f"{a.label_b} only", contents=[f"+ /{a.label_b}/**"] + hide(a.label_b))),
    rrb.TimePanel(expanded=False), rrb.SelectionPanel(expanded=False))
rr.send_blueprint(bp, make_default=True)
print("저장:", a.out, f"({os.path.getsize(a.out)/1e6:.0f} MB)")
