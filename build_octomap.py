#!/usr/bin/env python3
"""LiDAR 스캔(점 + 센서 원점) → OctoMap 확률적 occupancy (레이캐스팅, log-odds) → semantic 조인 → npz + rerun.

사용: python build_octomap.py <lidar_scans_dir> <session_voxsem.npz> <out_prefix> [--res 0.2] [--rrd out.rrd] [--stride 1]
출력: <out>_octomap.bt (OctoMap 바이너리) · <out>_occupancy.npz {xyz, prob, state(1 occ/0 free), sem_label}
"""
import argparse, glob, os, time
import numpy as np
import octomap

ap = argparse.ArgumentParser()
ap.add_argument("scans"); ap.add_argument("voxsem"); ap.add_argument("out")
ap.add_argument("--res", type=float, default=0.2)
ap.add_argument("--stride", type=int, default=1, help="스캔 간격 (1=전부)")
ap.add_argument("--max-range", type=float, default=40.0, help="복셀 빌더의 LiDAR 상한(40m)과 일치시켜야 함")
ap.add_argument("--rrd", default=None)
ap.add_argument("--alpha", type=float, default=1.0, help="큐브 투명도 0~1 (rrd 에 굽힘)")
ap.add_argument("--objects", default=None, help="맵 패키지 objects.json → 3D 박스+라벨 겹침 (occ/objects/*)")
a = ap.parse_args()

ap2 = argparse.ArgumentParser(add_help=False)
tree = octomap.OcTree(a.res)
# 희소 야외 LiDAR 의 지면 침식·식생 통과 완화: miss 를 약하게(0.5 에 가깝게), hit 을 강하게
PH = float(os.environ.get("OCTO_PROB_HIT", "0.7")); PM = float(os.environ.get("OCTO_PROB_MISS", "0.4"))
tree.setProbHit(PH); tree.setProbMiss(PM)
print(f"OctoMap res {a.res} m · prob_hit {PH} · prob_miss {PM}")
files = sorted(glob.glob(os.path.join(a.scans, "*.npz")))[:: a.stride]
t0 = time.time(); npts = 0
for k, f in enumerate(files):
    d = np.load(f)
    pts = d["xyz"].astype(np.float64); org = d["origin"].astype(np.float64)
    # insertPointCloud: 원점→각 점 레이를 따라 free 업데이트(miss), 끝점 occupied(hit). 프레임당 중복 셀 1회 처리.
    tree.insertPointCloud(pts, org, maxrange=a.max_range, lazy_eval=True)
    npts += len(pts)
    if k % 100 == 0:
        print(f"  scan {k}/{len(files)}  {time.time()-t0:.0f}s")
tree.updateInnerOccupancy()
print(f"삽입 완료: 스캔 {len(files)}, 점 {npts:,}, {time.time()-t0:.0f}s")
tree.write(a.out + "_octomap.bt".encode() if isinstance(a.out, bytes) else (a.out + "_octomap.bt").encode())

# 리프 순회 → 좌표·확률·상태
occ_xyz, occ_p, free_xyz, free_p = [], [], [], []
for node in tree.begin_leafs():
    c = node.getCoordinate(); p = node.getOccupancy(); s = node.getSize()
    if tree.isNodeOccupied(node):
        occ_xyz.append(c); occ_p.append(p)
    else:
        free_xyz.append(c); free_p.append(p)
occ_xyz = np.array(occ_xyz, np.float32); free_xyz = np.array(free_xyz, np.float32)
occ_p = np.array(occ_p, np.float32); free_p = np.array(free_p, np.float32)
print(f"occupied {len(occ_xyz):,} · free {len(free_xyz):,} (리프 기준; 큰 free 리프는 병합됨)")

# semantic 조인: 우리 0.1m semantic 복셀 → 이 해상도 키로 다수결
v = np.load(a.voxsem)
V = float(v["vox"]); classes = [str(c) for c in v["classes"]]
sem = v["sem_mean"].astype(np.float32); has = v["n_sem"] > 0
lab = sem.argmax(1); lab = np.where(has & (sem.max(1) < 0.3), len(classes), lab); lab = np.where(has, lab, len(classes) + 1)
ctr = (v["ijk"] + 0.5) * V
key = lambda P: np.floor(P / a.res).astype(np.int64)
K = key(ctr); Kp = (K[:, 0] << 42) + (K[:, 1] << 21) + K[:, 2] + (1 << 62)
order = np.argsort(Kp); Kp_s = Kp[order]; lab_s = lab[order]
uk, st = np.unique(Kp_s, return_index=True); en = np.r_[st[1:], len(Kp_s)]
maj = {}
n_cls = len(classes) + 2
for u, s0, s1 in zip(uk, st, en):
    maj[int(u)] = int(np.bincount(lab_s[s0:s1], minlength=n_cls).argmax())
Ko = key(occ_xyz); Kop = (Ko[:, 0] << 42) + (Ko[:, 1] << 21) + Ko[:, 2] + (1 << 62)
occ_lab = np.array([maj.get(int(k), n_cls - 1) for k in Kop], np.int16)   # 우리 복셀에 없으면 unknown-라벨
# 비교: 우리 semantic 복셀 각각의 OctoMap 상태 — tree.search 로 정확 조인 (pruned 리프도 올바르게 처리)
state = np.full(len(ctr), -1, np.int8)     # -1 unknown(미관측), 0 free, 1 occupied
t1 = time.time()
for i, p in enumerate(ctr.astype(np.float64)):
    n = tree.search(p)
    if n is not None:
        try:
            state[i] = 1 if tree.isNodeOccupied(n) else 0
        except Exception:
            pass
ours_in_free = state == 0
print(f"우리 복셀 {len(ctr):,} → OctoMap 상태: occupied {int((state==1).sum()):,} · free {int(ours_in_free.sum()):,} "
      f"({100*ours_in_free.mean():.1f}%) · unknown {int((state==-1).sum()):,}   [{time.time()-t1:.0f}s]")
for nm, sel in [("free 안 우리 복셀", ours_in_free), ("occupied 일치", state == 1)]:
    pl = lab[sel]; print(f"   {nm} 라벨:", {n: int((pl == i).sum()) for i, n in enumerate(classes + ['other', 'unknown'])})

np.savez_compressed(a.out + "_occupancy.npz", occ_xyz=occ_xyz, occ_prob=occ_p, occ_label=occ_lab,
                    free_xyz=free_xyz, free_prob=free_p, res=a.res, classes=np.array(classes + ["other", "unknown"]),
                    ours_state=state)
print("저장:", a.out + "_occupancy.npz", "/", a.out + "_octomap.bt")

if a.rrd:
    import rerun as rr, rerun.blueprint as rrb
    rr.init("octomap"); rr.save(a.rrd)
    bp = rrb.Blueprint(rrb.Tabs(
        rrb.Spatial3DView(origin="/occ", name="Occupied (semantic)", contents=["+ /occ/occupied/**", "+ /occ/objects/**"]),
        rrb.Spatial3DView(origin="/occ", name="Occupied (no edges)", contents=["+ /occ/occupied/**", "- /occ/occupied/*/wire", "+ /occ/objects/**"]),
        rrb.Spatial3DView(origin="/occ", name="Cleared voxels", contents=["+ /occ/occupied/**", "+ /occ/cleared/**"]),
        rrb.Spatial3DView(origin="/occ", name="Free space", contents=["+ /occ/free", "+ /occ/occupied/**"]),
        rrb.Spatial3DView(origin="/occ", name="All", contents="+ /occ/**")), rrb.TimePanel(expanded=False))
    rr.send_blueprint(bp, make_default=True)
    pal = np.array([(95, 170, 95), (200, 165, 120), (40, 110, 60), (220, 60, 60), (120, 135, 160), (110, 110, 110)], np.uint8)

    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cube_mesh import log_cubes   # 면별 음영 큐브 메시 + 검은 모서리(<entity>/wire)

    for i, nm in enumerate(classes + ["other", "unknown"]):
        m = occ_lab == i
        if m.any():
            log_cubes(f"occ/occupied/{nm}", occ_xyz[m], a.res, pal[i], alpha=a.alpha)
    sub = np.arange(len(free_xyz))[::4]
    rr.log("occ/free", rr.Points3D(free_xyz[sub], colors=[[80, 160, 255, 40]], radii=a.res * 0.25))
    if a.objects:   # 씬그래프 객체(맵 패키지 objects.json) 를 같은 world 프레임에 겹침 — 이 OctoMap 범위 안의 것만
        import json
        raw = json.load(open(a.objects)); objs = list(raw.values()) if isinstance(raw, dict) else raw
        lo, hi = occ_xyz.min(0) - 1.0, occ_xyz.max(0) + 1.0; n_obj = 0
        for i, o in enumerate(objs):
            tag = o.get("tag") or o.get("object_tag")
            if tag == "person":
                continue
            cen = np.array(o.get("center_w") or o["bbox_center"], float)
            if not (np.all(cen >= lo) and np.all(cen <= hi)):
                continue
            ext = np.array(o.get("extent") or o.get("bbox_extent") or [0.5] * 3, float)
            rr.log(f"occ/objects/{tag.replace(' ', '_')}_{i}", rr.Boxes3D(centers=[cen], half_sizes=[ext / 2], labels=[tag], colors=[[255, 200, 40]]))
            n_obj += 1
        print(f"objects 겹침: {n_obj}/{len(objs)} (OctoMap 범위 내)")
    log_cubes("occ/cleared", ctr[ours_in_free][::2], V * 0.9, (255, 60, 200), edges=False)   # OctoMap 이 비웠다고 본 우리 복셀
    print("rrd:", a.rrd)
