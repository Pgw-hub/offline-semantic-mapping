#!/usr/bin/env python3
"""실외 테스트베드 GT (Claude Design HTML 기하 포팅) → 격자 라벨 → 지면 높이로 맵 프레임 정합 → 코스트맵 평가. README 참조.
디자인 좌표 (x, z): Y-up, −z = 북. 여기서는 2D 평면 (x_d, z_d) 만 쓴다."""
import json, os, sys, math
import numpy as np
from shapely.geometry import Polygon, Point, box as sbox, LineString
from shapely.ops import unary_union
from shapely import affinity

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
RES = 0.2
SW = 0.15
# ---------------- HTML 기하 상수 (그대로) ----------------
U0 = [-4, -3.5]; E = [-0.875, 0.485]; Mv = [0.485, 0.875]; R_END = 41
def pt(off, r): return [U0[0] + off * Mv[0] + r * E[0], U0[1] + off * Mv[1] + r * E[1]]
RY_DIAG = math.atan2(-E[1], E[0])
roadPoly = [[-4,-11],[4,-11],[4,-3],[11.4,-3],[11.4,3],[4,3],[4,15.1],[-4,15.1]]
diagPoly = [[-4, pt(6, 0)[1] + 6 * 0.554 * E[1]], pt(6, R_END), pt(0, R_END), [-4, -3.5]]
sitePoly = [[-42.3,16.1],[-27,-2.5],[12,-31.6],[21,-31.6],[21,-10],[16.5,-10],[16.5,21.5],[-27,21.5],[-38,24]]
P0 = [-22.5, -1.5]; RY_SLOPE = 0.641

def rbox(sx, sz, x, z, ry=0.0):
    """three.js boxGeo(sx,·,sz).rotateY(ry).translate(x,·,z) 의 XZ 발자국. rotateY: x' = x cos + z sin, z' = −x sin + z cos"""
    b = sbox(-sx / 2, -sz / 2, sx / 2, sz / 2)                           # shapely (x, z)
    b = affinity.rotate(b, -math.degrees(ry), origin=(0, 0))              # shapely 는 반시계(+) 회전; three.js rotateY(+) 는 XZ 평면에서 시계 방향
    return affinity.translate(b, x, z)
def gbox(sx, sz, lx, lz, gx, gz, gry):
    """group(position gx,gz, rotation.y gry) 안의 로컬 박스 (lx, lz)"""
    b = sbox(lx - sx / 2, lz - sz / 2, lx + sx / 2, lz + sz / 2)
    b = affinity.rotate(b, -math.degrees(gry), origin=(0, 0)); return affinity.translate(b, gx, gz)
def gpoint(lx, lz, gx, gz, gry):
    c, s = math.cos(gry), math.sin(gry); return (gx + lx * c + lz * s, gz - lx * s + lz * c)

def build_shapes():
    road = unary_union([Polygon(roadPoly), Polygon(diagPoly)])
    site = Polygon(sitePoly)
    # 횡단보도: 북·남 테이블 [-4,4]×[z0,z1] (램프 ±1.2 는 차도), 동쪽 줄무늬 x 6.25~9.25, z −2.75~2.75
    cross = unary_union([sbox(-4, -8.2, 4, -4.2), sbox(-4, 6.5, 4, 10.5), sbox(6.25, -2.75, 9.25, 2.75)])
    # 화단 (grass)
    beds = [sbox(-27, 17.6, -8, 21.3), sbox(-3, 17.6, 16.3, 21.3), sbox(13.2, -7.7, 16.3, 17.4)]
    # 등판시험로 (별도 클래스: 평가 제외) + 측면 식재
    SL = 0.6 + sum((2 * 0.4 * s + 1.5 + 0.9) if s else (4 + 0.9) for s in [12, 8, 6, 0, 4])
    slope = gbox(SL, 1.5, SL / 2, 0, P0[0], P0[1], RY_SLOPE)
    slope_plant = gbox(SL + 6, 2.1, SL / 2 + 1, -2.35, P0[0], P0[1], RY_SLOPE)
    grass = unary_union(beds + [slope_plant])
    # 연석: 도로 가장자리 바깥 0.2 m (인도 쪽)
    curb = road.buffer(0.2).difference(road)
    # 장애물 발자국
    obs = []
    obs += [gbox(8, 4, 0, 0, 13.4, -24.5, RY_SLOPE), gbox(1.5, 1.5, 2.5, 2.9, 13.4, -24.5, RY_SLOPE)]        # 관제실 + 계단
    obs += [gbox(5, 3, 0, 0, 7.9, -18.5, RY_SLOPE)]                                                           # 창고
    obs += [sbox(18.5, -8.5, 30.5, 15.5)]                                                                     # 이웃 건물
    for x in [-24, -18, -12, 1, 7, 13]: obs.append(Point(x, 19.5).buffer(0.13 * 1.1 + 0.1))                   # 가로수 줄기
    for z in [-4, 2, 8, 14]: obs.append(Point(14.8, z).buffer(0.13 * 0.85 + 0.1))
    for x in np.arange(-26, 16, 3):                                                                           # 관목 (icosahedron r0.5, 1.6×0.8×1 스케일, z=18.3)
        if x < -8.5 or x > -2.5: obs.append(affinity.scale(Point(x + 1.5, 18.3).buffer(0.5), 1.6, 1.0))
    for z in np.arange(-6.5, 17, 2.2): obs.append(affinity.scale(Point(13.75, z).buffer(0.45), 1.0, 1.6))
    for x in np.arange(0.5, SL + 3, 2.6): obs.append(affinity.scale(Point(*gpoint(x, -2.4, P0[0], P0[1], RY_SLOPE)).buffer(0.6), 1.5, 0.9))  # 등판로 측 관목(회전 무시, 근사)
    for x in [-7.2, -5.5, -3.8]: obs.append(Point(x, 20.9).buffer(0.1))                                        # 볼라드
    for z in [-9.5, -8.2]: obs.append(Point(16.2, z).buffer(0.1))
    for x, z in [(-5.4,-3.9),(5.4,-8.7),(-5.4,6.1),(5.4,11.0),(10.0,-3.6),(5.6,3.6)]: obs.append(Point(x, z).buffer(0.06))   # 보행 신호등
    for x, z in [(5.3,12.6),(-5.3,-10.3),(12.2,3.6)]: obs.append(Point(x, z).buffer(0.12))                     # 차량 신호등
    for x, z in [(5.8,-9.4),(5.8,14.2),(-5.8,13.2)]: obs.append(Point(x, z).buffer(0.16))                      # 가로등
    for r in [10, 22, 34]: c = pt(7.0, r); obs.append(Point(c[0], c[1]).buffer(0.16))
    a, b = pt(6.55, 3), pt(6.55, 40); obs.append(LineString([a, b]).buffer(0.08))                              # 울타리 2 (사선 도로 하단, 파이프 0.35~0.85 m = 밴드 장애물)
    f0, f1 = gpoint(0, 1.2, P0[0], P0[1], RY_SLOPE), gpoint(SL, 1.2, P0[0], P0[1], RY_SLOPE); obs.append(LineString([f0, f1]).buffer(0.05))   # 울타리 1 (등판로 안쪽)
    obstacle = unary_union(obs)
    sidewalk = site.difference(road).difference(grass).difference(slope)
    return dict(sidewalk=sidewalk, grass=grass, road=road, crosswalk=cross, curb=curb, slope=slope, obstacle=obstacle, site=site)

CLASSES = ["outside", "sidewalk", "grass", "road", "crosswalk", "curb", "slope", "obstacle"]   # 뒤가 우선 (덮어쓰기)
COLORS = {"outside": (60,60,60), "sidewalk": (255,255,255), "grass": (95,170,95), "road": (128,64,128), "crosswalk": (240,210,74), "curb": (0,200,255), "slope": (200,120,200), "obstacle": (0,0,0)}

def rasterize(shapes, xs, ys, transform=None):
    """격자 셀 중심 (xs[i], ys[j]) → 라벨 id. transform: (x_d,z_d)=f(x_m,y_m) 역변환 함수(맵 격자에 GT 를 찍을 때)."""
    import shapely
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if transform is not None: X, Y = transform(X, Y)
    lab = np.zeros(X.shape, np.uint8)
    for k, name in enumerate(CLASSES):
        if name == "outside": continue
        m = shapely.contains_xy(shapes[name], X.ravel(), Y.ravel()).reshape(X.shape); lab[m] = k
    return lab

def preview(lab, path, scale=2):
    from PIL import Image
    img = np.zeros(lab.shape + (3,), np.uint8)
    for k, name in enumerate(CLASSES): img[lab == k] = COLORS[name]
    Image.fromarray(np.flipud(img.transpose(1, 0, 2))).resize((lab.shape[0] * scale, lab.shape[1] * scale), Image.NEAREST).save(path)

def cmd_build():
    sh = build_shapes(); xs = np.arange(-44, 23, RES) + RES / 2; zs = np.arange(-33, 26, RES) + RES / 2
    lab = rasterize(sh, xs, zs)
    np.savez_compressed(os.path.join(HERE, "gt_design.npz"), label=lab, classes=np.array(CLASSES), x0=xs[0] - RES / 2, z0=zs[0] - RES / 2, res=RES)
    preview(lab, os.path.join(HERE, "gt_design.png"))
    print("GT 격자", lab.shape, {n: int((lab == k).sum()) for k, n in enumerate(CLASSES)})

# ---------------- 정합: 맵 지면 높이 (차도 = 인도 − 0.15 m) ----------------
def map_ground(root):
    """session_voxsem.npz → 0.2 m 셀별 지면 z (n_pts≥3 복셀의 최저 z). 반환 (gz(H,W), x0, y0)"""
    z = np.load(os.path.join(root, "voxel", "session_voxsem.npz"), allow_pickle=True)
    vox = float(z["vox"]); xyz = z["ijk"].astype(np.float32) * vox                  # 로봇 지도 점은 복셀당 1점(n_pts=1) → 점 수 필터 없음, 셀 최저 z = 지면
    cm = np.load(os.path.join(root, "compiled", "session_costmap_occ.npz"), allow_pickle=True); x0, y0 = float(cm["x0"]), float(cm["y0"]); H, W = cm["cost"].shape
    gi = ((xyz[:, 0] - x0) / RES).astype(int); gj = ((xyz[:, 1] - y0) / RES).astype(int); m = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
    gz = np.full((H, W), np.inf, np.float32); np.minimum.at(gz, (gi[m], gj[m]), xyz[m, 2]); gz[~np.isfinite(gz)] = np.nan
    return gz, x0, y0

def register(gt_road_pts, low, high, x0, y0, H, W):
    """GT 도로 셀 중심 (디자인 좌표) 를 맵 격자 low(차도 높이) 마스크에 맞추는 2D 변환 탐색: 거울 ±, yaw 0..359°, 평행이동 = FFT 상관.
    디자인 (x_d, z_d) → RaiSim/맵 후보 (x, y) = (x_d, −z_d) (−z 북 → +y 북) 또는 거울 (x_d, z_d)."""
    from scipy.signal import fftconvolve
    best = None
    tgt = low.astype(np.float32) - 0.5 * high.astype(np.float32)             # 도로 위면 +1, 인도 위면 −0.5
    for mirror in (1, -1):
        px, py = gt_road_pts[:, 0], -mirror * gt_road_pts[:, 1]
        for yaw in np.arange(0, 360, 1.0):
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)); rx = c * px - s * py; ry = s * px + c * py
            # GT 점을 임시 캔버스에 찍고 상관 → 최적 시프트
            gi = np.round(rx / RES).astype(int); gj = np.round(ry / RES).astype(int); gi -= gi.min(); gj -= gj.min()
            src = np.zeros((gi.max() + 1, gj.max() + 1), np.float32); src[gi, gj] = 1
            corr = fftconvolve(tgt, src[::-1, ::-1], mode="full")
            k = np.unravel_index(np.argmax(corr), corr.shape); score = float(corr[k])
            if best is None or score > best[0]:
                # 시프트: corr index k ↔ src 원점이 tgt 의 (k - src.shape + 1) 에 놓임
                oi, oj = k[0] - src.shape[0] + 1, k[1] - src.shape[1] + 1
                tx = x0 + (oi - 0) * RES - (rx.min()); ty = y0 + (oj - 0) * RES - (ry.min())
                best = (score, mirror, yaw, tx, ty)
    return best

def make_transform(mirror, yaw, tx, ty):
    """(x_d, z_d) → (x_m, y_m) 과 역변환"""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    def fwd(xd, zd):
        px, py = xd, -mirror * zd; return c * px - s * py + tx, s * px + c * py + ty
    def inv(xm, ym):
        px, py = xm - tx, ym - ty; rx = c * px + s * py; ry = -s * px + c * py; return rx, -mirror * ry
    return fwd, inv

COST2CLS = {10: "sidewalk", 20: "crosswalk", 230: "road", 90: "grass", 60: "unlabeled", 150: "context", 200: "uncertain", 254: "lethal", 255: "unknown"}

def cmd_eval(roots):
    sh = build_shapes(); g = np.load(os.path.join(HERE, "gt_design.npz"), allow_pickle=True); lab_d = g["label"]; xs = g["x0"] + RES * (np.arange(lab_d.shape[0]) + 0.5); zs = g["z0"] + RES * (np.arange(lab_d.shape[1]) + 0.5)
    ri = CLASSES.index("road"); ci = CLASSES.index("crosswalk"); si = CLASSES.index("sidewalk")
    roadcells = np.argwhere((lab_d == ri) | (lab_d == ci)); gt_road_pts = np.stack([xs[roadcells[:, 0]], zs[roadcells[:, 1]]], 1)
    gz, x0, y0 = map_ground(roots[0]); H, W = gz.shape; ok = np.isfinite(gz)
    h, e = np.histogram(gz[ok], bins=np.arange(-2, 2, 0.05)); top = np.argsort(h)[::-1]
    peaks = sorted([e[top[0]] + 0.025] + [e[t] + 0.025 for t in top[1:12] if abs(e[t] - e[top[0]]) > 0.1][:1]); thr = float(np.mean(peaks)) if len(peaks) == 2 else float(np.median(gz[ok]))
    low = ok & (gz < thr); high = ok & (gz >= thr); print(f"지면 높이 봉우리 {np.round(peaks,2)} → 차도 임계 {thr:.2f} m · 낮은 셀 {int(low.sum())} 높은 셀 {int(high.sum())}")
    score, mirror, yaw, tx, ty = register(gt_road_pts, low, high, x0, y0, H, W)
    fwd, inv = make_transform(mirror, yaw, tx, ty); print(f"정합: mirror {mirror} · yaw {yaw:.0f}° · t ({tx:.2f}, {ty:.2f}) · score {score:.0f}")
    xm = x0 + RES * (np.arange(H) + 0.5); ym = y0 + RES * (np.arange(W) + 0.5)
    gt_m = rasterize(sh, xm, ym, transform=inv)                                   # 맵 격자 위 GT
    # 정합 품질: GT 도로 ∩ 낮은 셀 / GT 도로 ∩ 관측 셀
    gtroad = (gt_m == ri) | (gt_m == ci); print(f"정합 검증: GT 도로 셀 중 지면 관측 {int((gtroad & ok).sum())}, 그중 낮은(차도 높이) 비율 {(gtroad & low).sum() / max((gtroad & ok).sum(),1):.0%} · GT 인도 셀 중 높은 비율 {((gt_m==si) & high).sum() / max(((gt_m==si) & ok).sum(),1):.0%}")
    np.savez_compressed(os.path.join(HERE, "gt_map_frame.npz"), label=gt_m, classes=np.array(CLASSES), x0=x0, y0=y0, res=RES, mirror=mirror, yaw=yaw, tx=tx, ty=ty, root=roots[0])
    preview(gt_m, os.path.join(roots[0], "compiled", "gt_map_frame.png"))
    from PIL import Image
    for root in roots:
        cost = np.load(os.path.join(root, "compiled", "session_costmap_occ.npz"), allow_pickle=True)["cost"]; assert cost.shape == gt_m.shape
        rep = {}
        for gname in ["sidewalk", "crosswalk", "road", "grass", "curb", "obstacle"]:
            gm = gt_m == CLASSES.index(gname); n = int(gm.sum()); seen = gm & (cost != 255)
            dist = {COST2CLS.get(int(v), str(int(v))): int(c) for v, c in zip(*np.unique(cost[seen], return_counts=True))}
            rep[gname] = {"gt_cells": n, "observed": int(seen.sum()), "map_label_dist": dist}
        # IoU (관측 셀만, GT ground 4클래스 vs 맵 라벨)
        iou = {}
        for gname, cv in [("sidewalk", 10), ("crosswalk", 20), ("road", 230), ("grass", 90)]:
            gm = (gt_m == CLASSES.index(gname)) & (cost != 255); pm = (cost == cv) & (gt_m != 0); inter = (gm & pm).sum(); union = (gm | pm).sum()
            iou[gname] = {"iou": round(float(inter / max(union, 1)), 3), "recall": round(float(inter / max(gm.sum(), 1)), 3), "precision": round(float(inter / max(pm.sum(), 1)), 3)}
        sw = (gt_m == si) & (cost != 255); rd = (gt_m == ri) & (cost != 255)
        key = {"sidewalk_as_road": round(float(((cost == 230) & sw).sum() / max(sw.sum(), 1)), 3), "road_as_sidewalk_or_crosswalk": round(float((((cost == 10) | (cost == 20)) & rd).sum() / max(rd.sum(), 1)), 3),
               "crosswalk_recall": iou["crosswalk"]["recall"], "obstacle_as_lethal": round(float(((cost == 254) & (gt_m == CLASSES.index("obstacle")) & (cost != 255)).sum() / max(((gt_m == CLASSES.index("obstacle")) & (cost != 255)).sum(), 1)), 3)}
        out = {"root": os.path.basename(root), "transform": {"mirror": mirror, "yaw_deg": yaw, "tx": tx, "ty": ty}, "iou": iou, "key": key, "per_gt_class": rep}
        json.dump(out, open(os.path.join(root, "compiled", "gt_eval.json"), "w"), indent=1, ensure_ascii=False)
        print(f"== {os.path.basename(root)}: IoU sidewalk {iou['sidewalk']['iou']} crosswalk {iou['crosswalk']['iou']} road {iou['road']['iou']} grass {iou['grass']['iou']} | 인도→차도 {key['sidewalk_as_road']:.1%} · 차도→인도/횡단 {key['road_as_sidewalk_or_crosswalk']:.1%} · 횡단보도 재현 {key['crosswalk_recall']:.1%} · 장애물 lethal {key['obstacle_as_lethal']:.1%}")
        # 오버레이: 맵 라벨 vs GT 불일치 (빨강 = GT 인도인데 맵 차도, 파랑 = GT 차도인데 맵 인도/횡단, 주황 = GT 횡단보도 미검출)
        img = np.zeros(cost.shape + (3,), np.uint8)
        for k, name in enumerate(CLASSES): img[gt_m == k] = tuple(int(c * 0.45) for c in COLORS[name])
        img[(cost == 230) & sw] = (255, 40, 40); img[((cost == 10) | (cost == 20)) & rd] = (60, 120, 255); img[(gt_m == ci) & (cost != 20) & (cost != 255)] = (255, 160, 0)
        Image.fromarray(np.flipud(img.transpose(1, 0, 2))).resize((H * 2, W * 2), Image.NEAREST).save(os.path.join(root, "compiled", "gt_overlay.png"))

if __name__ == "__main__":
    if sys.argv[1] == "build": cmd_build()
    elif sys.argv[1] == "eval": cmd_eval([os.path.abspath(r) for r in sys.argv[2:]])
