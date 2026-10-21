#!/usr/bin/env python3
"""RGB | 검출(ConceptGraphs YOLO-World+SAM, exps/s_detections_stride10/vis) | 세그멘테이션(SegFormer 4채널, sem/sem) 3분할 배속 영상.
사용: python make_pipeline_video.py <session_root> <out.mp4> [--speed 15] [--logs log_000 ...]
검출 패널: stride 10 프레임에서 가장 가까운 vis 이미지(≤5 프레임 차). 세그 패널: 프레임 시각에 가장 가까운 10Hz sem 프레임을 RGB 위에 확률 가중 색으로 합성.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # 저장소 루트 (패키지 import)
import argparse, glob, os, shutil, subprocess, tempfile, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--speed", type=int, default=15); ap.add_argument("--logs", nargs="*", default=None)
ap.add_argument("--src-fps", type=float, default=30.0); ap.add_argument("--crf", type=int, default=20); ap.add_argument("--det-exp", default="s_detections_stride10")
a = ap.parse_args(); R = a.root
DS = os.path.join(R, "dataset"); SEM = os.path.join(R, "sem")
logs = a.logs or sorted(os.path.basename(p) for p in glob.glob(os.path.join(DS, "log_*")) if os.path.isdir(os.path.join(p, "results")))
idx = np.load(os.path.join(SEM, "rgb_index.npz")); sem_stamps = idx["rgb_stamp_ns"].astype(np.int64)
# 클래스 사전(sem/classes.json)에서 이름·색을 읽음 (없으면 초기 4클래스)
_cj = os.path.join(SEM, "classes.json")
if os.path.exists(_cj):
    import json; _cd = json.load(open(_cj, encoding="utf-8"))
    NAMES = [c["name"] for c in _cd["classes"]]
    from common import classdict as _CD
    SEG_TITLE = "Segmentation: " + _CD.MODELS.get(_cd.get("model") or _CD.DEFAULT_MODEL, {}).get("title", "SegFormer-B4 · ADE20K").split(" (")[0].replace(" · ", " - ") + f"  [{_cd.get('name')}]"   # cv2 Hershey 폰트는 "·" 미지원
    COL = np.array([[int(c["color"][1:3], 16), int(c["color"][3:5], 16), int(c["color"][5:7], 16)] for c in _cd["classes"]], np.float32)[:, ::-1]   # BGR
else:
    COL = np.array([(95, 200, 95), (210, 170, 110), (40, 120, 60), (230, 60, 60)], np.float32)[:, ::-1]
    NAMES = ["grass", "path", "tree", "person"]
    SEG_TITLE = "Segmentation: SegFormer-B4 (ADE20K)"
W, H, BAR = 640, 480, 40

def panel_title(img, text):
    bar = np.zeros((BAR, img.shape[1], 3), np.uint8); cv2.putText(bar, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])

def seg_overlay(rgb, sem):
    if sem.shape[:2] != rgb.shape[:2]:                        # 세그 결과는 긴 변 640·입력 비율 유지 (16:9 면 640x360) → 패널 크기로 맞춤
        sem = np.stack([cv2.resize(sem[..., c], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR) for c in range(sem.shape[-1])], -1)
    p = sem.astype(np.float32) / 255.0                       # (H,W,N)
    color = (p[..., :, None] * COL[None, None]).sum(2)       # 확률 가중 색 (H,W,3)
    alpha = np.clip(p.sum(-1, keepdims=True), 0, 1) * 0.65
    out = (rgb.astype(np.float32) * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    y = 24
    for n, c in zip(NAMES, COL):
        cv2.rectangle(out, (10, y - 12), (26, y + 2), tuple(int(v) for v in c), -1); cv2.putText(out, n, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); y += 20
    return out

tmp = tempfile.mkdtemp(prefix="pipevid_"); k = 0; t_sess = 0.0; t0 = time.time(); n_det_hit = 0; n_seg_hit = 0
for lg in logs:
    d = os.path.join(DS, lg); fs = sorted(glob.glob(os.path.join(d, "results", "frame*.jpg"))); n = len(fs)
    stamps = np.load(os.path.join(d, "stamps.npz"))["stamps"].astype(np.int64)
    visdir = os.path.join(d, "exps", a.det_exp, "vis"); vis_frames = sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(os.path.join(visdir, "frame??????.jpg")))
    vis_arr = np.array(vis_frames) if vis_frames else None
    fps_log = (n - 1) / max((stamps[-1] - stamps[0]) / 1e9, 1e-6) if n > 1 else a.src_fps     # 실제 소스 fps (sim 은 ~8 Hz, 실로봇 30 Hz)
    step = max(1, int(round(a.speed * fps_log / 30.0)))                                        # 출력 30 fps 에서 x{speed} 배속이 되는 프레임 간격
    for i in range(0, n, step):
        rgb = cv2.imread(fs[i])
        # 검출 패널
        det = None
        if vis_arr is not None and len(vis_arr):
            j = vis_arr[np.argmin(np.abs(vis_arr - i))]
            if abs(j - i) <= max(step, 10) // 2 + 1:
                det = cv2.imread(os.path.join(visdir, f"frame{j:06d}.jpg")); n_det_hit += 1
        if det is None:
            det = rgb.copy(); cv2.putText(det, "no detection frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        det = cv2.resize(det, (W, H))
        # 세그 패널
        si = int(np.argmin(np.abs(sem_stamps - stamps[i])))
        seg = rgb.copy()
        if abs(int(sem_stamps[si]) - int(stamps[i])) < 80_000_000:
            sp = os.path.join(SEM, "sem", f"{si:06d}.npz"); rp = os.path.join(SEM, "rgb", f"{si:06d}.png")
            if os.path.exists(sp):
                base = cv2.imread(rp) if os.path.exists(rp) else rgb
                seg = seg_overlay(cv2.resize(base, (W, H)), np.load(sp)["sem"]); n_seg_hit += 1
        t = t_sess + i / fps_log; mm, ss = divmod(int(t), 60)
        frame = np.hstack([panel_title(cv2.resize(rgb, (W, H)), f"RGB   {lg}  frame {i:04d}  {mm:02d}:{ss:02d}  x{a.speed}"),
                           panel_title(det, "Detection: YOLO-World + SAM (stride 10)"),
                           panel_title(seg, SEG_TITLE)])
        cv2.imwrite(os.path.join(tmp, f"{k:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]); k += 1
    t_sess += n / fps_log; print(f"  {lg}: {n} frames @ {fps_log:.1f} fps, step {step} → 누적 {k}", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "30", "-i", os.path.join(tmp, "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-movflags", "+faststart", a.out], check=True)
shutil.rmtree(tmp)
print(f"영상 {a.out}: {k} 프레임 = {k/30:.1f}s (x{a.speed}) · 검출 패널 {n_det_hit}/{k} · 세그 패널 {n_seg_hit}/{k} · {os.path.getsize(a.out)/1e6:.1f} MB [{time.time()-t0:.0f}s]")
