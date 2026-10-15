#!/usr/bin/env python3
"""voxel3d 사전맵 rerun 뷰 + BEV 높이맵 PNG.
  python voxel3d_view.py compiled/session_voxel3d.npz compiled/voxel3d_view.rrd [--voxsem voxel/session_voxsem.npz] [--sub 1] [--bev compiled/voxel3d_bev.png]
엔티티: prior/lidar/<class>(클래스 색, 클래스별 토글) · prior/lidar_height(높이 색) · prior/depth_fill · prior/flat_fill(평탄 채움, 연한 클래스 색) · removed/free(확률 free 로 제거) · removed/dynamic · free_space(확률 격자 free, 1/30 표본) · tiles(경계)
"""
import argparse, json, os, sys
import numpy as np, rerun as rr, rerun.blueprint as rrb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import classdict as CD
ap = argparse.ArgumentParser(); ap.add_argument("prior"); ap.add_argument("out"); ap.add_argument("--voxsem"); ap.add_argument("--sub", type=int, default=1); ap.add_argument("--bev"); ap.add_argument("--max-pts", type=int, default=6_000_000)
ap.add_argument("--ref-pcd", help="비교용 로봇 저장 SLAM 지도(.pcd) — robot_slam_map/ 엔티티로 추가")
ap.add_argument("--boxes", action="store_true", help="점 대신 속이 찬 상자(OctoMap/RViz 풍)로. 세부 셀은 res_xy×res_xy×res_z 상자, 확률 격자(0.1 m)는 정육면체 상자 레이어 추가")
ap.add_argument("--box-max", type=int, default=600_000, help="세부 셀 상자 최대 개수 (넘으면 균등 간격 표본; rrd 크기·뷰어 부하 제한)")
ap.add_argument("--box-crop", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"), help="세부 셀 상자를 이 xy 영역만 그림 (영역 안은 표본 없이 전부)")
a = ap.parse_args()
v = np.load(a.prior); RX, RZ, RP = float(v["res_xy"]), float(v["res_z"]), float(v["prob_res"]); prm = json.loads(str(v["params"])); classes = [str(c) for c in v["classes"]]
I = v["ijk"]; xyz = np.stack([(I[:, 0] + 0.5) * RX, (I[:, 1] + 0.5) * RX, v["mean_z"]], -1).astype(np.float32); cls = v["cls"]; src = v["source"]
pal = None
if a.voxsem:
    vs = np.load(a.voxsem, allow_pickle=True)
    if "classdict" in vs.files: cd = json.loads(str(vs["classdict"])); pal = {i: CD.hex2rgb(c["color"]) for i, c in enumerate(cd["classes"])}
if pal is None: pal = {i: tuple(int(x) for x in np.array([[46,204,113],[241,196,15],[52,152,219],[155,89,182],[230,126,34],[26,188,156],[231,76,60],[149,165,166]][i % 8])) for i in range(len(classes))}
pal[254] = CD.hex2rgb(CD.OTHER_COLOR); pal[255] = CD.hex2rgb(CD.UNKNOWN_COLOR)
col_cls = np.array([pal.get(int(c), (128, 128, 128)) for c in range(256)], np.uint8)[cls]
def turbo(z):
    t = np.clip((z - np.percentile(z, 2)) / max(np.percentile(z, 98) - np.percentile(z, 2), 1e-6), 0, 1)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1); return (np.stack([r, g, b], -1) * 255).astype(np.uint8)
sub = max(a.sub, int(np.ceil(len(xyz) / a.max_pts)))
# ---- 상자 메시 (cube_mesh.py 와 같은 면별 음영, 비등방 크기 지원) ----
_C = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], np.float32)
_F = [([4,5,6,7],1.0),([0,3,2,1],0.5),([0,1,5,4],0.72),([2,3,7,6],0.72),([1,2,6,5],0.86),([3,0,4,7],0.86)]
_FO = np.stack([_C[f] for f,_ in _F]); _SH = np.array([[sh]*4 for _,sh in _F], np.float32)[...,None]; _FT = np.array([[0,1,2],[0,2,3]], np.uint32)
def boxes_mesh(centers, half, colors):
    """centers (n,3) · half (3,) 반크기 · colors (n,3) → 면별 음영 상자 메시"""
    c = np.asarray(centers, np.float32).reshape(-1,3); n = len(c); h = np.asarray(half, np.float32)
    verts = (c[:,None,None,:] + (_FO*h)[None]).reshape(-1,3)
    tris = ((_FT[None] + (np.arange(6,dtype=np.uint32)*4)[:,None,None]).reshape(-1,3)[None] + (np.arange(n,dtype=np.uint32)*24)[:,None,None]).reshape(-1,3)
    col = (np.asarray(colors, np.float32)[:,None,None,:] * _SH[None]).clip(0,255).astype(np.uint8).reshape(-1,3)
    return rr.Mesh3D(vertex_positions=verts, triangle_indices=tris, vertex_colors=col)
def box_sel():
    """상자로 그릴 세부 셀 선택: crop 영역 안 전부, 아니면 box_max 이하로 균등 표본"""
    keep = np.ones(len(xyz), bool)
    if a.box_crop: x0,x1,y0,y1 = a.box_crop; keep &= (xyz[:,0]>=x0)&(xyz[:,0]<=x1)&(xyz[:,1]>=y0)&(xyz[:,1]<=y1)
    idx = np.flatnonzero(keep)
    if len(idx) > a.box_max: idx = idx[np.linspace(0, len(idx)-1, a.box_max).astype(int)]
    return idx
rad = RX * 0.5
L = src == 1; D = src == 2; FF = src == 3
def cname(c): return classes[c] if c < len(classes) else ("other" if c == 254 else "unlabeled")
def log_tiles():
    # 타일 경계
    idx_path = a.prior.replace(".npz", "_tiles/index.json")
    if os.path.exists(idx_path):
        ti = json.load(open(idx_path)); T = float(ti["tile_m"]); strips = []
        for t in ti["tiles"]:
            x0, y0, z = t["x0"], t["y0"], t["z_min"]; strips.append([[x0, y0, z], [x0 + T, y0, z], [x0 + T, y0 + T, z], [x0, y0 + T, z], [x0, y0, z]])
        rr.log("tiles", rr.LineStrips3D(strips, colors=[255, 200, 0], radii=0.01), static=True)
def log_bev():
    global img
    # BEV 높이맵: 컬럼별 최상단(mean_z) — LiDAR 원점 높이 아래 지형만 (max_z = 지형 상한: 전체 z 5퍼센타일 + 1.5 m)
    zl = np.percentile(xyz[:, 2], 3); maxz = zl + 1.5; k = xyz[:, 2] < maxz
    res = 0.05; x0, y0 = xyz[:, 0].min(), xyz[:, 1].min(); W = int((xyz[:, 0].max() - x0) / res) + 1; H = int((xyz[:, 1].max() - y0) / res) + 1
    u = np.clip(((xyz[k, 0] - x0) / res).astype(int), 0, W - 1); vv = np.clip(((xyz[k, 1] - y0) / res).astype(int), 0, H - 1); hm = np.full((H, W), -np.inf, np.float32); np.maximum.at(hm, (vv, u), xyz[k, 2])
    img = np.full((H, W, 3), 255, np.uint8); ok = np.isfinite(hm); img[ok] = turbo(hm[ok]); img = img[::-1]
    rr.log("bev_height", rr.Image(img), static=True)
def log_ref():
    if not a.ref_pcd: return
    import open3d as o3d
    Pr = np.asarray(o3d.io.read_point_cloud(a.ref_pcd).points).astype(np.float32)
    rr.log("robot_slam_map/height", rr.Points3D(Pr, colors=turbo(Pr[:, 2]), radii=RX * 0.6), static=True)
    rr.log("robot_slam_map/red", rr.Points3D(Pr, colors=[230, 50, 50], radii=RX * 0.6), static=True)
    print(f"robot SLAM map {len(Pr):,} 점 ← {a.ref_pcd}")
def log_points():
    # 클래스별 엔티티 — rerun 왼쪽 패널에서 클래스 하나씩 켜고 끌 수 있게 (prior/lidar/<class>, prior/flat_fill/<class>)
    for c in np.unique(cls):
        kc = L & (cls == c)
        if kc.any(): rr.log(f"prior/lidar/{cname(int(c))}", rr.Points3D(xyz[kc][::sub], colors=col_cls[kc][::sub], radii=rad), static=True)
        kf = FF & (cls == c)
        if kf.any(): rr.log(f"prior/flat_fill/{cname(int(c))}", rr.Points3D(xyz[kf][::sub], colors=col_cls[kf][::sub] // 2 + 100, radii=rad * 0.8), static=True)   # 평탄 채움: 클래스 색을 연하게
    rr.log("prior/lidar_height", rr.Points3D(xyz[L][::sub], colors=turbo(xyz[L, 2])[::sub], radii=rad), static=True)
    if D.any(): rr.log("prior/depth_fill", rr.Points3D(xyz[D][::sub], colors=[0, 200, 255], radii=rad), static=True)
    if "removed_ijk" in v.files and len(v["removed_ijk"]):
        R = v["removed_ijk"]; rx = np.stack([(R[:, 0] + 0.5) * RX, (R[:, 1] + 0.5) * RX, (R[:, 2] + 0.5) * RZ], -1).astype(np.float32); w = v["removed_why"]
        rr.log("removed/free", rr.Points3D(rx[w == 1][::sub], colors=[255, 60, 60], radii=rad), static=True)
        if (w == 2).any(): rr.log("removed/dynamic", rr.Points3D(rx[w == 2][::sub], colors=[255, 0, 255], radii=rad), static=True)
    if "coarse_ijk" in v.files:
        C = v["coarse_ijk"]; lo = v["coarse_logodds"].astype(np.float32); fr = lo < 0
        cx = ((C[fr] + 0.5) * RP).astype(np.float32); s2 = max(30, int(np.ceil(len(cx) / 300000)))
        rr.log("free_space", rr.Points3D(cx[::s2], colors=[170, 170, 170], radii=RP * 0.15), static=True)
def log_fine_boxes():
    # 세부 셀 상자: 중심은 셀 중심(z 는 셀 중심, 평균 z 아님), 크기 res_xy×res_xy×res_z, 색 = 클래스(채움 셀은 연하게, depth 는 하늘색)
    bi = box_sel(); cen = np.stack([(I[bi,0]+0.5)*RX, (I[bi,1]+0.5)*RX, (I[bi,2]+0.5)*RZ], -1).astype(np.float32)
    bcol = col_cls[bi].astype(np.float32); bcol[src[bi]==3] = bcol[src[bi]==3]//2 + 100; bcol[src[bi]==2] = [0,200,255]
    for c in np.unique(cls[bi]):
        k = cls[bi]==c
        rr.log(f"prior_boxes/{cname(int(c))}", boxes_mesh(cen[k], [RX/2, RX/2, RZ/2], bcol[k]), static=True)
    print(f"세부 셀 상자 {len(bi):,}개 (전체 {len(xyz):,}" + (f", crop {a.box_crop}" if a.box_crop else f", box_max {a.box_max:,}") + ")")
def log_coarse_boxes():
    # 확률 격자(OctoMap 층) 상자: 점유 0.1 m 셀 전부, 색 = 그 셀 안 세부 셀의 다수 클래스
    if "coarse_ijk" in v.files:
        C = v["coarse_ijk"]; lo = v["coarse_logodds"].astype(np.float32); occ_c = lo > 0; Cc = C[occ_c]
        OFF = 1 << 20; ck = ((Cc[:,0].astype(np.int64)+OFF)<<42)|((Cc[:,1].astype(np.int64)+OFF)<<21)|(Cc[:,2].astype(np.int64)+OFF)
        pk = np.floor(xyz / RP).astype(np.int64); fk = ((pk[:,0]+OFF)<<42)|((pk[:,1]+OFF)<<21)|(pk[:,2]+OFF)
        order = np.argsort(ck); cks = ck[order]; pos = np.searchsorted(cks, fk); pos = np.minimum(pos, len(cks)-1); hit = cks[pos]==fk
        maj = np.full(len(Cc), 255, np.int64)
        if hit.any():
            key = pos[hit]*256 + cls[hit].astype(np.int64); uk, cnt = np.unique(key, return_counts=True); o = np.lexsort((-cnt, uk//256)); uk, cnt = uk[o], cnt[o]
            first = np.unique(uk//256, return_index=True)[1]; maj[order[uk[first]//256]] = uk[first]%256
        ccen = ((Cc + 0.5) * RP).astype(np.float32); ccol = np.array([pal.get(int(c),(128,128,128)) for c in range(256)], np.uint8)[maj]
        for c in np.unique(maj):
            k = maj==c; rr.log(f"prior_coarse_boxes/{cname(int(c))}", boxes_mesh(ccen[k], [RP/2]*3, ccol[k].astype(np.float32)), static=True)
        print(f"확률 격자 점유 상자 {len(Cc):,}개 ({RP} m)")

POINT_TABS = lambda: [
    rrb.Spatial3DView(origin="/", name="3D prior (class colors)", contents=["+ /prior/lidar/**", "+ /prior/depth_fill", "+ /prior/flat_fill/**", "+ /tiles"]),
    rrb.Spatial3DView(origin="/", name="height colors", contents=["+ /prior/lidar_height", "+ /prior/depth_fill", "+ /tiles"]),
    rrb.Spatial3DView(origin="/", name="removed (red) + free space", contents=["+ /removed/**", "+ /free_space", "+ /prior/lidar/**"]),
    rrb.Spatial2DView(origin="/bev_height", name="BEV height")]
if a.boxes:
    # 렌더 부하 분리: 메인 파일 = 세부 셀 상자만(클래스별 토글), 점·제거셀·free·0.1 m 확률 격자 상자는 <out>_points 파일로
    rr.init("voxel3d"); rr.save(a.out); log_fine_boxes(); log_tiles(); log_bev()
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(
        rrb.Spatial3DView(origin="/", name=f"boxes {RX*100:g}×{RX*100:g}×{RZ*100:g} cm (class colors)", contents=["+ /prior_boxes/**", "+ /tiles"]),
        rrb.Spatial2DView(origin="/bev_height", name="BEV height")), collapse_panels=False))
    pout = a.out.replace("_view.rrd", "_points_view.rrd") if a.out.endswith("_view.rrd") else a.out.replace(".rrd", "_points.rrd")
    rr.init("voxel3d_points"); rr.save(pout); log_points(); log_coarse_boxes(); log_tiles(); log_bev()
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*POINT_TABS(), rrb.Spatial3DView(origin="/", name=f"boxes {RP*100:g} cm (확률 격자)", contents=["+ /prior_coarse_boxes/**", "+ /tiles"])), collapse_panels=False))
    print("points rrd", pout)
else:
    rr.init("voxel3d"); rr.save(a.out); log_points(); log_tiles(); log_bev(); log_ref()
    ref_tabs = [rrb.Spatial3DView(origin="/", name="robot SLAM map (height)", contents=["+ /robot_slam_map/height", "+ /tiles"]),
                rrb.Spatial3DView(origin="/", name="ours (height) vs robot (red)", contents=["+ /prior/lidar_height", "+ /robot_slam_map/red"])] if a.ref_pcd else []
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*POINT_TABS(), *ref_tabs), collapse_panels=False))
if a.bev:
    import cv2; cv2.imwrite(a.bev, img[:, :, ::-1])
n = len(xyz); print(f"셀 {n:,} (표본 1/{sub}) · LiDAR {int(L.sum()):,} · depth {int(D.sum()):,} · 평탄채움 {int(FF.sum()):,} · rrd {a.out}" + (f" · bev {a.bev}" if a.bev else ""))
