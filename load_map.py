#!/usr/bin/env python3
"""map_manifest.json 하나로 맵 전체를 로드 → 요약 → rerun 시각화.  (재계산 없음, CPU 전용)

사용: python load_map.py /data2/cg/compiled/map_manifest.json [--rrd out.rrd] [--sub 2]
"""
import argparse, json, os, time
import numpy as np


class MapPackage:
    """매니페스트가 가리키는 파일들을 메모리에 올린 맵. Map Compiler 가 읽는 단위."""

    def __init__(self, manifest_path, costmap_key="costmap_2d"):
        t0 = time.time()
        self.man = json.load(open(manifest_path))
        self.costmap_key = costmap_key if costmap_key in self.man["files"] else "costmap_2d"
        root = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))   # …/cg
        P = lambda k: os.path.join(root, self.man["files"][k]["path"])
        self.vox = float(self.man["voxel_m"])
        v = np.load(P("terrain_voxels"))
        self.ijk, self.sem, self.rgb = v["ijk"], v["sem_mean"].astype(np.float32), v["rgb"]
        self.n_sem, self.n_pts = v["n_sem"], v["n_pts"]
        self.classes = [str(c) for c in v["classes"]]
        raw = json.load(open(P("objects")))
        self.objects = list(raw.values()) if isinstance(raw, dict) else raw
        self.clip = np.load(P("objects_clip"))["clip"].astype(np.float32)
        self.edges = {"heuristic": json.load(open(P("edges_heuristic"))),
                      "object_terrain": json.load(open(P("edges_object_terrain"))),
                      "vlm": json.load(open(P("edges_vlm")))}
        if self.costmap_key in self.man["files"]:
            cm = np.load(P(self.costmap_key))
            self.costmap, self.cm_x0, self.cm_y0, self.cm_res = cm["cost"], float(cm["x0"]), float(cm["y0"]), float(cm["res"])
        else:   # 코스트맵이 아직 없는 패키지(빌드 중) 도 로드 가능
            self.costmap, self.cm_x0, self.cm_y0, self.cm_res = None, 0.0, 0.0, 0.0
        # 확률적 occupancy (레이캐스팅 log-odds + 표면 하이브리드). 없으면 None.
        self.occ = None
        if "occupancy_probabilistic" in self.man["files"]:
            o = np.load(P("occupancy_probabilistic"))
            self.occ = {"ijk": o["ijk"], "res": float(o["res"]), "logodds": o["logodds"], "state": o["state"], "source": o["source"],
                        "hits": o["hits"], "misses": o["misses"]}
            self.surf_state, self.surf_removed = o["surf_state"], o["surf_removed"]
        self.load_s = time.time() - t0

    def occupancy_state(self, xyz):
        """world 점들 → 1 occupied / 0 free / -1 unknown (CPU, 정렬 키 이진탐색)"""
        if self.occ is None:
            return np.full(len(xyz), -1, np.int8)
        R = self.occ["res"]; K = np.floor(np.asarray(xyz) / R).astype(np.int64)
        if not hasattr(self, "_occ_keys"):
            I = self.occ["ijk"].astype(np.int64) + (1 << 20)
            self._occ_keys = (I[:, 0] << 42) | (I[:, 1] << 21) | I[:, 2]
            self._occ_order = np.argsort(self._occ_keys); self._occ_keys = self._occ_keys[self._occ_order]
        K = K + (1 << 20); k = (K[:, 0] << 42) | (K[:, 1] << 21) | K[:, 2]
        p = np.searchsorted(self._occ_keys, k); pc = np.minimum(p, len(self._occ_keys) - 1); f = self._occ_keys[pc] == k
        out = np.full(len(k), -1, np.int8); out[f] = self.occ["state"][self._occ_order[pc[f]]]
        return out

    def coarsen(self, res):
        """복셀 격자를 res(m) 로 재격자화 (서브샘플이 아니라 병합: semantic 은 n_sem 가중 평균, rgb 는 n_pts 가중 평균)."""
        if res <= self.vox:
            return self
        ctr = self.voxel_centers(); K = np.floor(ctr / res).astype(np.int64)
        uk, inv = np.unique(K, axis=0, return_inverse=True); inv = inv.ravel(); n = len(uk)
        w = self.n_sem.astype(np.float64)
        sem = np.zeros((n, self.sem.shape[1])); np.add.at(sem, inv, self.sem * w[:, None])
        n_sem = np.bincount(inv, weights=w, minlength=n); sem /= np.maximum(n_sem, 1)[:, None]
        wp = self.n_pts.astype(np.float64)
        rgb = np.zeros((n, 3)); np.add.at(rgb, inv, self.rgb.astype(np.float64) * wp[:, None])
        n_pts = np.bincount(inv, weights=wp, minlength=n); rgb /= np.maximum(n_pts, 1)[:, None]
        self.ijk, self.vox = uk, float(res)
        self.sem, self.n_sem = sem.astype(np.float32), n_sem.astype(np.int32)
        self.rgb, self.n_pts = rgb.clip(0, 255).astype(np.uint8), n_pts.astype(np.int32)
        return self

    # --- 편의 질의 (CPU) ---
    def voxel_centers(self):
        return (self.ijk + 0.5) * self.vox

    def labels(self):
        lab = self.sem.argmax(1); has = self.n_sem > 0; other = has & (self.sem.max(1) < 0.3)
        lab = np.where(other, len(self.classes), lab)          # other
        return np.where(has, lab, len(self.classes) + 1)       # unknown

    def summary(self):
        lab = self.labels(); names = self.classes + ["other", "unknown(LiDAR only)"]
        cnt = {n: int((lab == i).sum()) for i, n in enumerate(names)}
        c = self.costmap if self.costmap is not None else np.full((1, 1), 255, np.uint8)
        occ = None
        if self.occ is not None:
            s = self.occ["state"]
            occ = {"res_m": self.occ["res"], "cells": int(len(s)), "occupied": int((s == 1).sum()), "free": int((s == 0).sum()),
                   "surface_voxels_removed_dynamic": int(self.surf_removed.sum())}
        return {"load_seconds": round(self.load_s, 2), "voxels": int(len(self.ijk)), "voxel_labels": cnt, "occupancy": occ,
                "objects": len(self.objects), "clip_dim": int(self.clip.shape[1]),
                "edges": {k: len(v) for k, v in self.edges.items()},
                "costmap": {"shape": list(c.shape), "res_m": self.cm_res,
                            "free_path": int((c == 10).sum()), "grass": int((c == 90).sum()),
                            "uncertain": int((c == 200).sum()), "lethal": int((c == 254).sum())}}


def default_blueprint():
    """뷰어가 열릴 때의 기본 레이아웃/가시성. 0.16 은 엔티티별 on/off 대신 뷰의 contents 포함·제외로 제어."""
    import rerun.blueprint as rrb
    main = rrb.Spatial3DView(origin="/map", name="Map (default)", contents=[
        "+ /map/**",                  # unknown(LiDAR 만 본 회색 복셀) 은 기본 켜짐 — 지형 문맥
        "- /map/voxels/rgb",          # 실제색 복셀: semantic 과 겹치므로 기본 꺼짐
        "- /map/voxels/other",
        "- /map/edges/heuristic/**",  # 671 개 근사 엣지: 기본 꺼짐 (노이즈)
        "- /map/costmap2d",
        "- /map/occupancy/**",        # 확률적 occupancy 는 별도 탭
    ])
    occ = rrb.Spatial3DView(origin="/map", name="Occupancy", contents=[
        "+ /map/occupancy/occupied/**", "+ /map/objects/**"])
    occ_free = rrb.Spatial3DView(origin="/map", name="Occupancy + free", contents=["+ /map/occupancy/**"])
    cost = rrb.Spatial3DView(origin="/map", name="Costmap", contents=[
        "+ /map/costmap2d", "+ /map/objects/**", "+ /map/voxels/unknown"])
    rgb = rrb.Spatial3DView(origin="/map", name="RGB", contents=["+ /map/voxels/rgb", "+ /map/objects/**"])
    everything = rrb.Spatial3DView(origin="/map", name="All layers", contents="+ /map/**")
    return rrb.Blueprint(rrb.Tabs(main, occ, occ_free, cost, rgb, everything),
                         rrb.TimePanel(expanded=False),
                         rrb.SelectionPanel(expanded=False))


def to_rerun(mp, out, sub=2, cubes=False, edges=True, alpha=1.0, occ_view_res=0.4):
    import rerun as rr
    rr.init("map_package"); rr.save(out)
    rr.send_blueprint(default_blueprint(), make_active=True, make_default=True)
    ctr = mp.voxel_centers(); lab = mp.labels()
    pal = np.array([(95, 170, 95), (200, 165, 120), (40, 110, 60), (220, 60, 60), (120, 135, 160), (110, 110, 110)], np.uint8)
    idx = np.arange(len(ctr))[::sub]
    n_cls = len(mp.classes)
    known = idx[lab[idx] < n_cls]; oth = idx[lab[idx] == n_cls]; unk = idx[lab[idx] == n_cls + 1]
    # 이전 뷰와 같은 구조: 카메라가 본 복셀(semantic/rgb) · 어휘 밖(other) · LiDAR 만(unknown) 을 각각 토글 가능하게
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cube_mesh import log_cubes   # 면별 음영 큐브 메시 + 검은 모서리(<entity>/wire, 토글 가능)

    # 동적 클래스 표면 복셀(free 공간 안) 제거 — occupancy 레이어가 있을 때만 (coarsen 뒤에는 인덱스가 달라 생략)
    if mp.occ is not None and len(mp.surf_removed) == len(ctr):
        idx = idx[~mp.surf_removed[idx]]
    # 클래스별 엔티티 → 뷰어에서 grass/path/tree/person 을 개별 토글 (--cubes 면 정육면체 메시)
    for ci, cname in enumerate(mp.classes):
        sel = idx[lab[idx] == ci]
        if len(sel):
            if cubes:
                log_cubes(f"map/voxels/semantic/{cname}", ctr[sel], mp.vox, pal[ci], edges=edges, alpha=alpha)
            else:
                rr.log(f"map/voxels/semantic/{cname}", rr.Points3D(ctr[sel], colors=[pal[ci].tolist()], radii=mp.vox * 0.45))
    rr.log("map/voxels/rgb", rr.Points3D(ctr[np.concatenate([known, oth])], colors=mp.rgb[np.concatenate([known, oth])], radii=mp.vox * 0.45))
    rr.log("map/voxels/other", rr.Points3D(ctr[oth], colors=[[120, 135, 160]], radii=mp.vox * 0.45))
    rr.log("map/voxels/unknown", rr.Points3D(ctr[unk], colors=[[110, 110, 110]], radii=mp.vox * 0.35))
    # 확률적 occupancy 레이어 (0.2m): occupied 큐브는 출처별 색, free 는 반투명 점
    if mp.occ is not None:
        O = mp.occ; oc = (O["ijk"] + 0.5) * O["res"]; src = O["source"]; st = O["state"]
        vr = max(occ_view_res, O["res"])            # 표시용 병합 해상도 (데이터는 그대로)
        occm = st == 1
        K = np.floor(oc[occm] / vr).astype(np.int64); uk, inv = np.unique(K, axis=0, return_inverse=True); inv = inv.ravel()
        vsrc = np.zeros(len(uk), np.int8); np.maximum.at(vsrc, inv, src[occm])     # 3(둘 다) > 2(표면복원) > 1(레이만)
        vctr = (uk + 0.5) * vr
        for code, nm, col in [(3, "occupied/ray_and_surface", (120, 160, 120)), (2, "occupied/surface_restored", (230, 170, 60)),
                              (1, "occupied/ray_only", (220, 70, 70))]:
            m = vsrc == code
            if m.any():
                log_cubes(f"map/occupancy/{nm}", vctr[m], vr, col, edges=edges, alpha=alpha)
        fr = np.where(st == 0)[0][::max(4, sub * 4)]
        rr.log("map/occupancy/free", rr.Points3D(oc[fr], colors=[[80, 160, 255, 40]], radii=O["res"] * 0.25))
    # costmap → 지면 위 색 점 (row=x, col=y)
    c = mp.costmap; ii, jj = np.where(c < 255)
    cx = mp.cm_x0 + (ii + 0.5) * mp.cm_res; cy = mp.cm_y0 + (jj + 0.5) * mp.cm_res
    # 지면 z: 해당 xy 의 최저 복셀 근사 → 간단히 전역 복셀 z 최소 percentile 사용
    zg = np.percentile(ctr[:, 2], 5)
    cpal = {10: (170, 140, 110), 90: (95, 150, 95), 60: (120, 120, 120), 150: (120, 130, 150), 200: (220, 120, 40), 254: (200, 40, 40)}
    cc = np.array([cpal.get(int(v), (255, 255, 255)) for v in c[ii, jj]], np.uint8)
    rr.log("map/costmap2d", rr.Points3D(np.stack([cx, cy, np.full_like(cx, zg)], -1)[::sub], colors=cc[::sub], radii=mp.cm_res * 0.5))
    # 객체
    byid = {}
    for i, o in enumerate(mp.objects):
        tag = o.get("tag") or o.get("object_tag")
        if tag == "person":
            continue
        cen = np.array(o.get("center_w") or o["bbox_center"], float); ext = np.array(o.get("extent") or o.get("bbox_extent") or [0.5] * 3, float)
        rr.log(f"map/objects/{tag.replace(' ', '_')}_{i}", rr.Boxes3D(centers=[cen], half_sizes=[ext / 2], labels=[tag], colors=[[255, 200, 40]]))
        byid[i] = cen
    # 엣지 3종 (좌표가 있는 것만)
    COL = {"heuristic": (150, 150, 150), "object_terrain": (80, 200, 255), "vlm": (255, 140, 60)}
    for src, E in mp.edges.items():
        n = 0
        for e in E:
            if src == "heuristic":
                a, b = e.get("object_1_center"), e.get("object_2_center")
            elif src == "object_terrain":
                a = e.get("center"); b = [a[0], a[1], a[2] - 0.5] if a else None
            else:
                a, b = e["object_1"]["center"], e["object_2"]["center"]
            if a is None or b is None:
                continue
            lbl = e.get("edge_description") or (f"on {e['on']}" if "on" in e else f"{e['object_1']['tag']} {e['relation']} {e['object_2']['tag']}")
            rr.log(f"map/edges/{src}/{n}", rr.LineStrips3D([[a, b]], colors=[COL[src]], radii=0.03, labels=[lbl]))
            n += 1
    return out


def render_bev(mp, out_png, res=0.2, max_w=1280):
    """상단 뷰 썸네일: 셀별 최고 z 복셀의 클래스 색 (grass/path/tree/person/other/unknown), 객체 박스 윤곽, 어두운 배경."""
    import cv2
    ctr = mp.voxel_centers(); lab = mp.labels()
    pal = {0: (111, 145, 121), 1: (141, 147, 155), 2: (79, 122, 92), 3: (209, 133, 47), 4: (92, 153, 199), 5: (70, 74, 80)}   # 디자인 팔레트 (RGB)
    K = np.floor(ctr[:, :2] / res).astype(np.int64); k0 = K.min(0); K -= k0; W, H = int(K[:, 0].max()) + 1, int(K[:, 1].max()) + 1
    scale = min(1.0, max_w / max(W, H))
    order = np.argsort(ctr[:, 2]); flat = K[order, 0] * H + K[order, 1]
    top = np.full(W * H, -1, np.int64); top[flat] = lab[order]          # 마지막 대입 = 최고 z
    img = np.full((H, W, 3), (37, 37, 39), np.uint8)
    for c, col in pal.items():
        m = (top == c).reshape(W, H).T; img[m] = col
    img = np.ascontiguousarray(img[::-1])                                 # +y 위
    if scale < 1: img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    for o in mp.objects:
        if (o.get("tag") or "") == "person": continue
        c = o.get("center_w"); e = o.get("extent") or [0.5, 0.5, 0.5]
        if not c: continue
        x0 = int(((c[0] - e[0] / 2) / res - k0[0]) * scale); x1 = int(((c[0] + e[0] / 2) / res - k0[0]) * scale)
        y0 = int(img.shape[0] - ((c[1] + e[1] / 2) / res - k0[1]) * scale); y1 = int(img.shape[0] - ((c[1] - e[1] / 2) / res - k0[1]) * scale)
        cv2.rectangle(img, (x0, y0), (max(x1, x0 + 2), max(y1, y0 + 2)), (41, 151, 255), 1)
    cv2.imwrite(out_png, img[:, :, ::-1]); return img.shape


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--rrd", default=None); ap.add_argument("--sub", type=int, default=2)
    ap.add_argument("--cubes", action="store_true", help="semantic 복셀을 정육면체 메시로 (OctoMap 스타일, 무거움 → --sub 4~8 권장)")
    ap.add_argument("--no-edges", action="store_true", help="--cubes 에서 검은 모서리 와이어 생략")
    ap.add_argument("--alpha", type=float, default=1.0, help="큐브 투명도 0~1 (rrd 에 굽힘; 0.16 뷰어에 런타임 슬라이더 없음)")
    ap.add_argument("--coarse", type=float, default=None, help="복셀을 이 해상도(m)로 병합해서 표시 (예: 0.3)")
    ap.add_argument("--bev", default=None, help="상단 뷰 PNG 저장 경로 (포털 썸네일)")
    ap.add_argument("--occ-view", type=float, default=0.4, help="occupancy 레이어 표시 해상도(m) (데이터 0.2m 는 그대로)")
    a = ap.parse_args()
    mp = MapPackage(a.manifest)
    print(json.dumps(mp.summary(), ensure_ascii=False, indent=1))
    if a.coarse:
        t0 = time.time(); mp.coarsen(a.coarse)
        print(f"coarsen → {a.coarse} m: {len(mp.ijk):,} voxels ({time.time()-t0:.1f}s)")
    if a.bev:
        print("bev", render_bev(mp, a.bev), "→", a.bev)
    if a.rrd:
        t0 = time.time(); to_rerun(mp, a.rrd, a.sub, a.cubes, not a.no_edges, a.alpha, a.occ_view)
        print(f"rerun 저장 {a.rrd} ({time.time()-t0:.1f}s)")
