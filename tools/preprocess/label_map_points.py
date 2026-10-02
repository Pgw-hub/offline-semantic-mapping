#!/usr/bin/env python3
"""트랙 2 C 단계 — 로봇 lidar_slam 이 저장한 기하 지도(<name>.pcd, 최종 PGO 프레임) 의 점 하나하나에 SegFormer 라벨을 투영해 session_voxsem.npz 를 만든다.
기하는 로봇 지도 그대로(점을 추가하지 않음), 의미만 입힌다. 프레임별로: 카메라 자세(traj.txt, 로봇 PGO 그래프로 보간) → 지도 점을 카메라에 투영 → depth z-buffer 로 가림 판정
→ 클래스 경계 픽셀 제외 → 보이는 점에 픽셀 semantic 누적. 여러 로그를 한 번에 처리해 세션 파일을 바로 쓴다 (merge_voxels 불필요).
사용: python label_map_points.py <map.pcd> <calib_dir> <sem_dir> <out_session_voxsem.npz> --datasets <ds/log_000> [<ds/log_001> ...]
                                 [--vox 0.1 --mask-bottom 60 --edge-px 4 --require-depth 1 --max-range 40 --pcd-out compiled/robot_map_semantic.pcd]
출력 npz 형식은 build_voxel_sem_map/merge_voxels 와 동일 (ijk, n_pts, rgb, sem_mean, n_sem, first_frame, vox, classes, classdict, n_frames) + map_xyz(원 점 좌표).
"""
import argparse, glob, json, os, sys
import numpy as np, cv2, yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); import classdict as CD

ap = argparse.ArgumentParser(); ap.add_argument("pcd"); ap.add_argument("calib"); ap.add_argument("sem_dir"); ap.add_argument("out"); ap.add_argument("--datasets", nargs="+", required=True)
ap.add_argument("--vox", type=float, default=0.1); ap.add_argument("--mask-bottom", type=int, default=60); ap.add_argument("--edge-px", type=int, default=4)
ap.add_argument("--require-depth", type=int, default=1, help="1: depth 없는 픽셀의 점 제외 · 0: 가림 판정 없이 허용(벽 뒤 점에 라벨 샘) · 2: depth 없으면 LiDAR 지도 점 자체 z-buffer(픽셀 블록 최소 거리)로 가림 판정 — 원거리 프레임도 투표")
ap.add_argument("--zbuf-px", type=int, default=3, help="require-depth 2 의 z-buffer 픽셀 블록"); ap.add_argument("--zbuf-tol", type=float, default=0.3, help="블록 최소 거리 + tol + 0.05·z 안이면 보임")
ap.add_argument("--zbuf-range", type=float, default=15.0, help="require-depth 2: depth 없는 점은 이 거리(m) 안에서만 투표 (원거리 지면이 지평선 픽셀의 하늘/건물 라벨을 받는 것 방지)")
ap.add_argument("--max-range", type=float, default=40.0); ap.add_argument("--pcd-out")
ap.add_argument("--depth-fill-range", type=float, default=0.0, help="depth 카메라 점으로 LiDAR 빈 곳 채우기 (카메라 거리 상한 m, 0=끔)")
ap.add_argument("--depth-gap", type=float, default=0.15, help="LiDAR 지도 점이 이 거리 안에 있으면 depth 점 추가 안 함 (표면 이중화 방지)")
ap.add_argument("--depth-px-stride", type=int, default=3); ap.add_argument("--depth-frame-stride", type=int, default=2); ap.add_argument("--depth-edge", type=float, default=0.12, help="이웃 픽셀과 depth 차 (m) 이상이면 경계(flying pixel) 로 버림")
ap.add_argument("--depth-shadow", type=float, default=0.5, help="같은 픽셀 방향에 LiDAR 표면이 이 거리(m) 안에 있으면 '같은 표면의 어긋난 복제' 로 보고 버림. LiDAR 표면보다 이만큼 앞에 있는 점(LiDAR 가 놓친 물체)만 통과")
a = ap.parse_args()

import open3d as o3d
from scipy.spatial import cKDTree
M = np.asarray(o3d.io.read_point_cloud(a.pcd).points).astype(np.float64); N = len(M); tree = cKDTree(M)
ci = yaml.safe_load(open(os.path.join(a.calib, "d455_front_color_caminfo.yaml")))["camera_info"]; K = np.array(ci["K"], float).reshape(3, 3); Wc, Hc = ci["width"], ci["height"]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
cj = os.path.join(a.sem_dir, "classes.json"); classdict = json.load(open(cj, encoding="utf-8")) if os.path.exists(cj) else None
CLASSES = [c["name"] for c in classdict["classes"]] if classdict else None
idx = np.load(os.path.join(a.sem_dir, "rgb_index.npz")); rgb_ns = idx["rgb_stamp_ns"]
print(f"지도 점 {N:,} · bbox {M.min(0).round(1)}..{M.max(0).round(1)} · 카메라 {Wc}x{Hc} fx {fx:.1f}")

sem_sum = None; n_sem = np.zeros(N, np.float32); rgb_sum = np.zeros((N, 3), np.float32); first = np.full(N, 10**9, np.int64); n_vis_tot = 0; n_frames = 0
DF = {"key": [], "sem": [], "rgb": [], "fr": [], "xyz": []}; n_df_cand = n_df_kept = 0                       # depth 채움 후보 (0.1 m 격자 키로 집계)
vv_g, uu_g = np.mgrid[0:Hc:a.depth_px_stride, 0:Wc:a.depth_px_stride]
for ds in a.datasets:
    st = np.load(os.path.join(ds, "stamps.npz"))["stamps"]; traj = np.loadtxt(os.path.join(ds, "traj.txt")).reshape(-1, 4, 4); assert len(st) == len(traj)
    li = int(os.path.basename(ds.rstrip("/"))[-3:]) if os.path.basename(ds.rstrip("/"))[-3:].isdigit() else 0
    sel = np.where((rgb_ns >= st[0] - 50e6) & (rgb_ns <= st[-1] + 50e6))[0]
    print(f"{os.path.basename(ds)}: 프레임 {len(st)} · semantic 프레임 {len(sel)}")
    for fi, si in enumerate(sel):
        t = int(rgb_ns[si]); j = int(np.argmin(np.abs(st - t)))
        if abs(int(st[j]) - t) > 40e6: continue
        semf = os.path.join(a.sem_dir, "sem", "%06d.npz" % si)
        if not os.path.exists(semf): continue
        T_wc = traj[j]; T_cw = np.linalg.inv(T_wc); cam = T_wc[:3, 3]
        dep = cv2.imread(os.path.join(ds, "results", "depth%06d.png" % j), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        rgb = cv2.imread(os.path.join(ds, "results", "frame%06d.jpg" % j))[:, :, ::-1]
        sem = np.load(semf)["sem"]; NCH = sem.shape[-1]
        if sem_sum is None: sem_sum = np.zeros((N, NCH), np.float32)
        if a.edge_px > 0:
            lab_img = np.where(sem.max(-1) >= 77, sem.argmax(-1), NCH).astype(np.uint8); kern = np.ones((2 * a.edge_px + 1, 2 * a.edge_px + 1), np.uint8)
            edge = cv2.erode(lab_img, kern) != cv2.dilate(lab_img, kern)
        else: edge = np.zeros(sem.shape[:2], bool)
        cand = np.array(tree.query_ball_point(cam, a.max_range), dtype=np.int64)
        if len(cand) == 0: continue
        pc = M[cand] @ T_cw[:3, :3].T + T_cw[:3, 3]; zl = pc[:, 2]
        u = np.round(fx * pc[:, 0] / np.where(zl > 0, zl, 1) + cx).astype(np.int64); v = np.round(fy * pc[:, 1] / np.where(zl > 0, zl, 1) + cy).astype(np.int64)
        infov = (zl > 0.3) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc - a.mask_bottom)
        dcam = np.zeros(len(cand), np.float32); dcam[infov] = dep[v[infov], u[infov]]
        vis = infov & (dcam > 0) & (np.abs(zl - dcam) < 0.25 + 0.05 * zl)                       # depth 있는 픽셀: D455 z-buffer
        nd = infov & (dcam <= 0)
        if a.require_depth == 2 and nd.any():                                                      # depth 없는 픽셀: LiDAR 지도 점끼리 z-buffer (같은 픽셀 블록의 최소 거리 표면만 보임)
            zb = np.full((Hc // a.zbuf_px + 1, Wc // a.zbuf_px + 1), np.inf, np.float32)
            np.minimum.at(zb, (v[infov] // a.zbuf_px, u[infov] // a.zbuf_px), zl[infov].astype(np.float32))
            zbp = zb[v[nd] // a.zbuf_px, u[nd] // a.zbuf_px]
            vis[nd] = (zl[nd] < zbp + a.zbuf_tol + 0.05 * zl[nd]) & (zl[nd] < a.zbuf_range)
        elif a.require_depth == 0: vis |= nd                                                       # 가림 판정 없이 허용
        vis[vis] &= ~edge[v[vis], u[vis]]
        pi = cand[vis]; uu, vv = u[vis], v[vis]
        sem_sum[pi] += sem[vv, uu].astype(np.float32) / 255.0; n_sem[pi] += 1; rgb_sum[pi] += rgb[vv, uu]
        np.minimum.at(first, pi, fi + li * 600); n_vis_tot += int(vis.sum()); n_frames += 1
        if a.depth_fill_range > 0 and fi % a.depth_frame_stride == 0:                    # ---- depth 카메라 점으로 LiDAR 빈 곳 채우기
            zc = dep[vv_g, uu_g]; okd = (zc > 0.3) & (zc < a.depth_fill_range)
            # 경계(flying pixel): 4-이웃 중 depth 차가 depth_edge 이상인 픽셀 제외
            dz = np.zeros_like(dep); dz[:, 1:] = np.maximum(dz[:, 1:], np.abs(dep[:, 1:] - dep[:, :-1])); dz[1:, :] = np.maximum(dz[1:, :], np.abs(dep[1:, :] - dep[:-1, :]))
            okd &= dz[vv_g, uu_g] < a.depth_edge; okd &= ~edge[vv_g, uu_g]; okd &= vv_g < Hc - a.mask_bottom
            zc, uc, vc = zc[okd], uu_g[okd], vv_g[okd]
            if len(zc):
                # LiDAR z-buffer (시야 안 LiDAR 지도 점 전부, 픽셀 stride 격자, 3×3 이웃 최소): 같은 방향에 LiDAR 표면이 있으면 depth 점은 그 표면의 복제(정렬 오차) 일 확률이 높음
                zb = np.full((Hc // a.depth_px_stride + 1, Wc // a.depth_px_stride + 1), np.inf, np.float32)
                infz = infov & (zl < a.depth_fill_range + a.depth_shadow + 1.0)
                np.minimum.at(zb, (v[infz] // a.depth_px_stride, u[infz] // a.depth_px_stride), zl[infz].astype(np.float32))
                zb = cv2.erode(zb, np.ones((3, 3), np.uint8))                                          # 3×3 최소 (이웃 픽셀의 LiDAR 도 같은 표면으로 간주)
                zbp = zb[vc // a.depth_px_stride, uc // a.depth_px_stride]
                okz = ~np.isfinite(zbp) | (zc < zbp - a.depth_shadow)                                  # LiDAR 없음 또는 LiDAR 표면보다 shadow 만큼 앞(LiDAR 가 놓친 물체)
                zc, uc, vc = zc[okz], uc[okz], vc[okz]
            if len(zc):
                pcam = np.stack([(uc - cx) / fx * zc, (vc - cy) / fy * zc, zc], -1); pw = pcam @ T_wc[:3, :3].T + T_wc[:3, 3]
                far = tree.query(pw, distance_upper_bound=a.depth_gap)[0] > a.depth_gap              # LiDAR 점이 gap 안에 없는 곳만
                n_df_cand += len(pw); n_df_kept += int(far.sum())
                if far.any():
                    K_ = np.floor(pw[far] / a.vox).astype(np.int64); o_ = 1 << 20; DF["key"].append(((K_[:, 0] + o_) << 42) | ((K_[:, 1] + o_) << 21) | (K_[:, 2] + o_))
                    DF["sem"].append(sem[vc[far], uc[far]].astype(np.float32) / 255.0); DF["rgb"].append(rgb[vc[far], uc[far]].astype(np.float32)); DF["fr"].append(np.full(int(far.sum()), fi + li * 600, np.int64)); DF["xyz"].append(pw[far].astype(np.float32))   # 3D prior(voxel3d) 용 원점 좌표
        if fi % 200 == 0: print(f"  frame {fi}/{len(sel)} 보이는 지도 점 {int(vis.sum()):,}")
if sem_sum is None: sys.exit("semantic 프레임 없음")
NCH = sem_sum.shape[1]; CLASSES = CLASSES or [str(i) for i in range(NCH)]
has = n_sem > 0
sem_mean_pt = np.where(has[:, None], sem_sum / np.maximum(n_sem, 1)[:, None], 0)
rgb_pt = np.where(has[:, None], rgb_sum / np.maximum(n_sem, 1)[:, None], 128).astype(np.uint8)
# 0.1 m 격자로 집계 (지도 점이 이미 그 격자면 1:1)
key = np.floor(M / a.vox).astype(np.int64); o = 1 << 20; k = ((key[:, 0] + o) << 42) | ((key[:, 1] + o) << 21) | (key[:, 2] + o)
uk, inv = np.unique(k, return_inverse=True); n = len(uk)
n_pts = np.bincount(inv, minlength=n); ns_ = np.bincount(inv, weights=n_sem, minlength=n)
sem_v = np.stack([np.bincount(inv, weights=sem_sum[:, c], minlength=n) for c in range(NCH)], -1) / np.maximum(ns_, 1)[:, None]
rgb_v = np.stack([np.bincount(inv, weights=rgb_sum[:, c], minlength=n) for c in range(3)], -1) / np.maximum(ns_, 1)[:, None]
rgb_v = np.where(ns_[:, None] > 0, rgb_v, 128).astype(np.uint8); ff = np.full(n, 10**9, np.int64); np.minimum.at(ff, inv, first)
source_v = np.ones(n, np.uint8)                                                                  # 1 = LiDAR(로봇 지도/registered), 2 = depth 채움
if DF["key"]:
    dk = np.concatenate(DF["key"]); dsem = np.concatenate(DF["sem"]); drgb = np.concatenate(DF["rgb"]); dfr = np.concatenate(DF["fr"])
    duk, dinv = np.unique(dk, return_inverse=True); new_mask = ~np.isin(duk, uk); duk = duk[new_mask]                   # LiDAR 복셀과 같은 셀은 제외
    remap = {k: i for i, k in enumerate(duk)}; keep = np.array([k in remap for k in dk]) if len(duk) else np.zeros(len(dk), bool)
    if keep.any():
        idx2 = np.array([remap[k] for k in dk[keep]]); nd = len(duk); cnt = np.bincount(idx2, minlength=nd)
        sem_d = np.stack([np.bincount(idx2, weights=dsem[keep][:, c], minlength=nd) for c in range(NCH)], -1) / cnt[:, None]
        rgb_d = (np.stack([np.bincount(idx2, weights=drgb[keep][:, c], minlength=nd) for c in range(3)], -1) / cnt[:, None]).astype(np.uint8)
        ff_d = np.full(nd, 10**9, np.int64); np.minimum.at(ff_d, idx2, dfr[keep])
        uk = np.concatenate([uk, duk]); n_pts = np.concatenate([n_pts, cnt]); ns_ = np.concatenate([ns_, cnt.astype(np.float64)]); sem_v = np.concatenate([sem_v, sem_d]); rgb_v = np.concatenate([rgb_v, rgb_d]); ff = np.concatenate([ff, ff_d])
        source_v = np.concatenate([source_v, np.full(nd, 2, np.uint8)]); n = len(uk)
        print(f"depth 채움: 후보 {n_df_cand:,} 점 중 LiDAR {a.depth_gap} m 밖 {n_df_kept:,} → 새 복셀 {nd:,} (LiDAR 복셀 {int((source_v==1).sum()):,} 의 {100*nd/max((source_v==1).sum(),1):.0f}%)")
m = (1 << 21) - 1; ijk = (np.stack([(uk >> 42) & m, (uk >> 21) & m, uk & m], -1) - o).astype(np.int32)
np.savez_compressed(a.out, ijk=ijk, n_pts=n_pts.astype(np.int32), rgb=rgb_v, sem_mean=sem_v.astype(np.float16), n_sem=ns_.astype(np.int32), first_frame=ff.astype(np.int32), source=source_v,
                    vox=a.vox, classes=np.array(CLASSES), n_frames=n_frames, map_xyz=M.astype(np.float32), map_sem=sem_mean_pt.astype(np.float16), map_n_sem=n_sem.astype(np.int32),
                    geometry_source=np.array(f"robot map {os.path.basename(a.pcd)}"),
                    depth_xyz=(np.concatenate(DF["xyz"]) if DF["xyz"] else np.zeros((0, 3), np.float32)),   # ← depth 채움 점(LiDAR gap 밖, 필터 통과) 원점 좌표; build_voxel3d.py 가 씀
                    **({"classdict": np.array(json.dumps(classdict, ensure_ascii=False))} if classdict else {}))
lab = int(has.sum()); am = sem_mean_pt[has].argmax(1)
print(f"지도 점 {N:,} 중 라벨 받음 {lab:,} ({100*lab/N:.0f}%) · 프레임 {n_frames} · 프레임당 보이는 점 평균 {n_vis_tot/max(n_frames,1):,.0f}")
print("라벨 분포:", {c: int((am == i).sum()) for i, c in enumerate(CLASSES)}); print("복셀", n, "저장:", a.out)
if a.pcd_out:
    pal = np.array([CD.hex2rgb(c["color"]) for c in classdict["classes"]] + [CD.hex2rgb(CD.OTHER_COLOR), CD.hex2rgb(CD.UNKNOWN_COLOR)], np.float64) / 255.0 if classdict else None
    lab_all = np.where(has, sem_mean_pt.argmax(1), NCH + 1); lab_all[has & (sem_mean_pt.max(1) < 0.3)] = NCH
    col = pal[lab_all] if pal is not None else np.tile([0.5, 0.5, 0.5], (N, 1))
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(M); pc.colors = o3d.utility.Vector3dVector(col)
    if (source_v == 2).any():                                                                   # LiDAR-only pcd 는 따로, 기본 pcd 는 depth 채움 포함(복셀 중심)
        o3d.io.write_point_cloud(a.pcd_out.replace(".pcd", "_lidar_only.pcd"), pc)
        sv = sem_v.astype(np.float32); hv = ns_ > 0; lv = np.where(hv, sv.argmax(1), NCH + 1); lv[hv & (sv.max(1) < 0.3)] = NCH
        pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(((ijk + 0.5) * a.vox).astype(np.float64)); pc.colors = o3d.utility.Vector3dVector(pal[lv] if pal is not None else np.tile([0.5, 0.5, 0.5], (n, 1)))
    o3d.io.write_point_cloud(a.pcd_out, pc); print("semantic pcd:", a.pcd_out)
