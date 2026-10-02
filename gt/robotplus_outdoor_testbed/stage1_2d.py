#!/usr/bin/env python3
"""1단계: 2D 분할 정확도 — 프레임별 Mapillary(사전 그룹) 출력 vs GT 를 카메라로 역투영한 2D 정답.
2D GT: 지도 점(LiDAR)을 카메라에 투영, 픽셀별 최근접 점(z-buffer)의 GT 셀 라벨 → 픽셀 라벨(점 사이 2 px 채움). 지면 4클래스만 평가.
사용: python stage1_2d.py <map_root> [--stride 1] [--panels 8]  → root/compiled/stage1_2d.json, stage1_panels.jpg"""
import argparse, glob, json, os, sys
import numpy as np, cv2, yaml
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--panels", type=int, default=8); ap.add_argument("--video", help="프레임별 4분할(RGB|pred|GT|errors) 영상 mp4 경로"); ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--write-oracle", help="2D GT 를 사전 채널 one-hot npz 로 저장할 sem_dir (오라클 맵 빌드용: sem/ 에 npz, classes.json·rgb_index.npz·rgb 는 원본 링크)")
ap.add_argument("--calib", default="/data2/20261002/railab_raibo_20261002_054154/config_snapshot/raisin_root/calibration")
a = ap.parse_args(); root = os.path.abspath(a.root)
g = np.load(os.path.join(HERE, "gt_map_frame.npz"), allow_pickle=True); gt = g["label"]; GCL = list(g["classes"]); gx0, gy0, R = float(g["x0"]), float(g["y0"]), float(g["res"])
EVAL = ["sidewalk", "road", "crosswalk", "grass"]; gid = {n: GCL.index(n) for n in EVAL}
cd = json.load(open(root + "/sem/classes.json")); names = [c["name"] for c in cd["classes"]]; NCH = len(names)
pred_map = {names.index(n): k for k, n in enumerate(EVAL) if n in names}                    # 예측 채널 → 평가 클래스 idx (그 외 → 4 'other')
ci = yaml.safe_load(open(os.path.join(a.calib, "d455_front_color_caminfo.yaml")))["camera_info"]; K = np.array(ci["K"], float).reshape(3, 3); Wc, Hc = ci["width"], ci["height"]; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
z = np.load(root + "/voxel/session_voxsem.npz", allow_pickle=True); M = z["map_xyz"].astype(np.float64)
gi = ((M[:, 0] - gx0) / R).astype(int); gj = ((M[:, 1] - gy0) / R).astype(int); ok = (gi >= 0) & (gi < gt.shape[0]) & (gj >= 0) & (gj < gt.shape[1])
gzc = np.full(gt.shape, np.inf, np.float32); np.minimum.at(gzc, (gi[ok], gj[ok]), M[ok, 2].astype(np.float32))   # 셀 지면 z (최저점)
near_ground = np.zeros(len(M), bool); near_ground[ok] = M[ok, 2] <= gzc[gi[ok], gj[ok]] + 0.3
pt_gt = np.full(len(M), -1, np.int16); pt_gt[ok & near_ground] = gt[gi[ok & near_ground], gj[ok & near_ground]]
pt_cls = np.full(len(M), -1, np.int16)                                                   # 평가 클래스 idx (0..3), 그 외 -1
for k, n in enumerate(EVAL): pt_cls[pt_gt == gid[n]] = k
from scipy.spatial import cKDTree; tree = cKDTree(M)
rgb_ns = np.load(root + "/sem/rgb_index.npz")["rgb_stamp_ns"]
logs = [(ds, np.load(ds + "/stamps.npz")["stamps"], np.loadtxt(ds + "/traj.txt").reshape(-1, 4, 4)) for ds in sorted(glob.glob(root + "/dataset/log_*"))]
BANDS = [(0, 5), (5, 10), (10, 20), (20, 60)]; NB = len(BANDS)
conf = np.zeros((2, NB, 4, 5), np.int64)
chan = np.zeros((NB, NCH + 1), np.int64)                                                 # GT 인도 픽셀의 예측 채널 분포 (마지막 = 확률<0.3 none)                                                 # [로봇 위치(0 인도,1 차도/기타), 거리대, GT, pred(0..3 + other)]
per_frame = []; MASK_BOTTOM = 60
prev = None                                                                         # (지도점 id, 예측 클래스) 직전 프레임 — 깜빡임 측정
sem_files = sorted(glob.glob(root + "/sem/sem/*.npz"))[::a.stride]
if a.video:
    import tempfile, subprocess; vdir = tempfile.mkdtemp(prefix="stage1_video_", dir=root + "/compiled")
if a.write_oracle:
    os.makedirs(os.path.join(a.write_oracle, "sem"), exist_ok=True)
    for fn in ["classes.json", "rgb_index.npz"]:
        dst = os.path.join(a.write_oracle, fn); os.path.exists(dst) or os.symlink(os.path.join(root, "sem", fn), dst)
    os.path.exists(os.path.join(a.write_oracle, "rgb")) or os.symlink(os.path.join(root, "sem", "rgb"), os.path.join(a.write_oracle, "rgb"))
COL = np.array([[255, 255, 255], [128, 64, 128], [240, 210, 74], [95, 170, 95], [60, 60, 60]], np.uint8)
panels = []
for f in sem_files:
    si = int(os.path.basename(f)[:6]); t = int(rgb_ns[si]); ds, st, traj = min(logs, key=lambda L: np.abs(L[1] - t).min()); j = int(np.argmin(np.abs(st - t)))
    if abs(int(st[j]) - t) > 40e6: continue
    T_wc = traj[j]; T_cw = np.linalg.inv(T_wc); cam = T_wc[:3, 3]
    cand = np.array(tree.query_ball_point(cam, 60.0), dtype=np.int64); pc = M[cand] @ T_cw[:3, :3].T + T_cw[:3, 3]; zl = pc[:, 2]
    u = np.round(fx * pc[:, 0] / np.where(zl > 0, zl, 1) + cx).astype(int); v = np.round(fy * pc[:, 1] / np.where(zl > 0, zl, 1) + cy).astype(int)
    infov = (zl > 0.3) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc - MASK_BOTTOM)
    zb = np.full((Hc, Wc), np.inf, np.float32); np.minimum.at(zb, (v[infov], u[infov]), zl[infov].astype(np.float32))
    vis = infov & (zl <= zb[np.clip(v, 0, Hc - 1), np.clip(u, 0, Wc - 1)] + 0.3 + 0.05 * zl)
    gimg = np.full((Hc, Wc), -1, np.int16); dimg = np.full((Hc, Wc), np.inf, np.float32)
    sel = vis & (pt_cls[cand] >= 0); gimg[v[sel], u[sel]] = pt_cls[cand[sel]]; dimg[v[sel], u[sel]] = zl[sel]
    # 점 사이 채움(2 px): 라벨은 최근접 전파, 거리는 최소
    has = gimg >= 0; kern = np.ones((5, 5), np.uint8)
    gfill = cv2.dilate((gimg + 1).astype(np.uint8), kern).astype(np.int16) - 1; dfill = -cv2.dilate(-np.where(np.isfinite(dimg), dimg, -1e9).astype(np.float32), kern)
    gimg = np.where(has, gimg, gfill); dimg = np.where(has, dimg, dfill); valid = gimg >= 0
    s = np.load(f)["sem"]; am = s.argmax(-1); mx = s.max(-1); pred = np.full((Hc, Wc), 4, np.int16)
    for ch, k in pred_map.items(): pred[(am == ch) & (mx >= 77)] = k
    # 로봇 위치의 GT
    ri_, rj_ = int((cam[0] - gx0) / R), int((cam[1] - gy0) / R); robot_on = 0 if (0 <= ri_ < gt.shape[0] and 0 <= rj_ < gt.shape[1] and gt[ri_, rj_] == gid["sidewalk"]) else 1
    fr = {"frame": si, "robot_on_sidewalk": robot_on == 0, "iou": {}}
    # 프레임 특징: 근거리 비율, 보이는 차도/연석 맥락, 검은 하늘 비율, 예측 분포(인도 GT 픽셀)
    gsw = valid & (gimg == 0); fr["gt_sidewalk_px"] = int(gsw.sum()); fr["near5_share"] = round(float((gsw & (dimg < 5)).sum() / max(gsw.sum(), 1)), 3)
    fr["gt_road_px"] = int((valid & (gimg == 1)).sum()); fr["gt_crosswalk_px"] = int((valid & (gimg == 2)).sum())
    rgb_ = np.asarray(Image.open(root + f"/sem/rgb/{si:06d}.png").convert("RGB")); fr["black_sky_share"] = round(float((rgb_[:Hc // 2].max(-1) < 25).mean()), 3)
    fr["sidewalk_pred"] = {("other" if p == 4 else EVAL[p]): round(float((gsw & (pred == p)).sum() / max(gsw.sum(), 1)), 3) for p in range(5)} if gsw.sum() > 200 else None
    # 깜빡임: 직전 프레임과 공유하는 지면 점 중 예측 클래스가 바뀐 비율 (GT 인도 점만)
    ids = cand[sel]; lab_now = pred[v[sel], u[sel]]; gt_now = pt_cls[cand[sel]]
    if prev is not None:
        common, ia, ib = np.intersect1d(ids, prev[0], return_indices=True); m_ = (gt_now[ia] == 0) & (lab_now[ia] != 4) & (prev[1][ib] != 4)
        fr["flicker_sidewalk"] = round(float((lab_now[ia][m_] != prev[1][ib][m_]).mean()), 3) if m_.sum() > 100 else None; fr["shared_pts"] = int(len(common))
    prev = (ids, lab_now)
    for b, (d0, d1) in enumerate(BANDS):
        m = valid & (dimg >= d0) & (dimg < d1)
        if m.any(): np.add.at(conf[robot_on, b], (gimg[m], pred[m]), 1)
        ms = m & (gimg == 0)
        if ms.any(): np.add.at(chan[b], np.where(mx[ms] >= 77, am[ms], NCH), 1)
    for k, n in enumerate(EVAL):
        gm = valid & (gimg == k); pm = valid & (pred == k); inter = (gm & pm).sum(); uni = (gm | pm).sum()
        if gm.sum() > 200: fr["iou"][n] = round(float(inter / max(uni, 1)), 3)
    per_frame.append(fr)
    if a.video:
        rgb = np.asarray(Image.open(root + f"/sem/rgb/{si:06d}.png").convert("RGB"))
        p_img = (0.4 * rgb + 0.6 * COL[np.where(valid, pred, 4)]).astype(np.uint8); g_img = (0.4 * rgb + 0.6 * COL[np.where(valid, gimg, 4)]).astype(np.uint8)
        err = (0.35 * rgb).astype(np.uint8); okp = valid & (gimg == pred); wrong = valid & (gimg != pred)
        err[okp] = (0.4 * rgb[okp] + (0, 110, 0)).astype(np.uint8); err[wrong] = (0.3 * rgb[wrong] + (170, 0, 0)).astype(np.uint8)
        top = np.concatenate([rgb, p_img], 1); bot = np.concatenate([g_img, err], 1); fr_img = np.concatenate([top, bot], 0)
        fr_img = cv2.cvtColor(fr_img, cv2.COLOR_RGB2BGR)
        def txt(img, t, x, y, col=(255, 255, 255)):
            cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        pos = "robot ON SIDEWALK" if robot_on == 0 else "robot on road/other"
        txt(fr_img, f"#{si:06d}  {os.path.basename(ds)} f{j}   {pos}", 8, 22); txt(fr_img, "model prediction (Mapillary -> dict)", Wc + 8, 22)
        txt(fr_img, "2D GT (back-projected from testbed GT)", 8, Hc + 22)
        iou_t = "  ".join(f"{k} IoU {v:.2f}" for k, v in fr["iou"].items()); txt(fr_img, f"errors: green=correct red=wrong   {iou_t}", Wc + 8, Hc + 22)
        for k, n in enumerate(EVAL): cv2.rectangle(fr_img, (Wc + 8 + k * 110, Hc - 22), (Wc + 24 + k * 110, Hc - 8), tuple(int(c) for c in COL[k][::-1]), -1); txt(fr_img, n, Wc + 28 + k * 110, Hc - 10)
        cv2.imwrite(os.path.join(vdir, f"{len(per_frame):06d}.png"), fr_img)
    if a.write_oracle:                                                                   # GT one-hot (평가 클래스만, 나머지 채널 0; GT 없는 픽셀 전부 0 → other/미라벨)
        oh = np.zeros((Hc, Wc, NCH), np.uint8)
        for ch, k in pred_map.items(): oh[..., ch][valid & (gimg == k)] = 255
        np.savez_compressed(os.path.join(a.write_oracle, "sem", f"{si:06d}.npz"), sem=oh)
    if a.panels and (len(per_frame) - 1) % max(1, len(sem_files) // a.panels) == 0 and len(panels) < a.panels:
        rgb = np.asarray(Image.open(root + f"/sem/rgb/{si:06d}.png").convert("RGB"))
        p_img = (0.4 * rgb + 0.6 * COL[np.where(valid, pred, 4)]).astype(np.uint8); g_img = (0.4 * rgb + 0.6 * COL[np.where(valid, gimg, 4)]).astype(np.uint8)
        err = rgb.copy(); wrong = valid & (gimg != pred) & (gimg < 4); err[wrong] = (255, 40, 40); err[valid & (gimg == pred)] = (0.5 * rgb[valid & (gimg == pred)] + (0, 90, 0)).astype(np.uint8)
        panels.append((si, fr["iou"].get("sidewalk"), rgb, p_img, g_img, err))
# ---- 요약 ----
def iou_from(c):   # c: (4 GT, 5 pred)
    out = {}
    for k, n in enumerate(EVAL):
        inter = c[k, k]; gtn = c[k].sum(); prn = c[:, k].sum(); out[n] = {"iou": round(float(inter / max(gtn + prn - inter, 1)), 3), "recall": round(float(inter / max(gtn, 1)), 3), "gt_px": int(gtn)}
    return out
summary = {"frames": len(per_frame), "overall": iou_from(conf.sum((0, 1))), "by_range": {f"{d0}-{d1}m": iou_from(conf[:, b].sum(0)) for b, (d0, d1) in enumerate(BANDS)},
           "by_robot_pos": {"robot_on_sidewalk": iou_from(conf[0].sum(0)), "robot_on_road_or_other": iou_from(conf[1].sum(0))},
           "sidewalk_gt_pred_channel_by_range": {f"{d0}-{d1}m": {(names + ["none(<0.3)"])[c]: round(float(chan[b, c] / max(chan[b].sum(), 1)), 3) for c in range(NCH + 1) if chan[b, c] / max(chan[b].sum(), 1) >= 0.02} for b, (d0, d1) in enumerate(BANDS)},
           "sidewalk_gt_pred_dist_by_range": {f"{d0}-{d1}m": {("other" if p == 4 else EVAL[p]): round(float(conf[:, b, 0, p].sum() / max(conf[:, b, 0].sum(), 1)), 3) for p in range(5)} for b, (d0, d1) in enumerate(BANDS)},
           "sidewalk_gt_pred_dist_by_robot_pos": {rp: {("other" if p == 4 else EVAL[p]): round(float(conf[r, :, 0, p].sum() / max(conf[r, :, 0].sum(), 1)), 3) for p in range(5)} for r, rp in enumerate(["on_sidewalk", "on_road_or_other"])}}
json.dump({"summary": summary, "per_frame": per_frame}, open(root + "/compiled/stage1_2d.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
# ---- 패널 ----
if panels:
    W_, H_ = 320, 240; rows = panels[:a.panels]; sheet = Image.new("RGB", (W_ * 4, (H_ + 18) * len(rows)), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for r, (si, iou_sw, *ims) in enumerate(rows):
        for k, im in enumerate(ims): sheet.paste(Image.fromarray(im).resize((W_, H_)), (k * W_, r * (H_ + 18) + 18))
        d.text((4, r * (H_ + 18) + 3), f"#{si:06d}  RGB | pred | GT(2D) | errors(red)   sidewalk IoU {iou_sw}", fill=(0, 0, 0))
    sheet.save(root + "/compiled/stage1_panels.jpg", quality=85)
if a.video:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(vdir, "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.video], check=True)
    import shutil; shutil.rmtree(vdir); print("video:", a.video, os.path.getsize(a.video) // 1_000_000, "MB")
