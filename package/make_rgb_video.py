#!/usr/bin/env python3
"""세션 RGB 프레임(dataset/log_XXX/results/frame*.jpg, 30fps) → 배속 영상 (H.264 mp4).
사용: python make_rgb_video.py <dataset_dir> <out.mp4> [--speed 15] [--width 960] [--logs log_000 log_001 ...]
배속 = 프레임 stride (30fps 원본에서 speed 장마다 1장 → 30fps 재생). 상단에 로그명·프레임·세션 경과·배속 오버레이.
"""
import argparse, glob, os, shutil, subprocess, tempfile, time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("out")
ap.add_argument("--speed", type=int, default=15); ap.add_argument("--width", type=int, default=960); ap.add_argument("--src-fps", type=float, default=30.0)
ap.add_argument("--logs", nargs="*", default=None); ap.add_argument("--crf", type=int, default=20)
a = ap.parse_args()
logs = a.logs or sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.dataset, "log_*")) if os.path.isdir(os.path.join(p, "results")))
tmp = tempfile.mkdtemp(prefix="rgbvid_"); k = 0; t_sess = 0.0; t0 = time.time()
for lg in logs:
    fs = sorted(glob.glob(os.path.join(a.dataset, lg, "results", "frame*.jpg"))); n = len(fs)
    for i in range(0, n, a.speed):
        im = cv2.imread(fs[i]); h, w = im.shape[:2]; im = cv2.resize(im, (a.width, int(h * a.width / w)), interpolation=cv2.INTER_AREA)
        t = t_sess + i / a.src_fps; mm, ss = divmod(int(t), 60)
        bar = im.copy(); cv2.rectangle(bar, (0, 0), (a.width, 34), (0, 0, 0), -1); im = cv2.addWeighted(bar, 0.55, im, 0.45, 0)
        cv2.putText(im, f"{lg}   frame {i:04d}/{n}   session {mm:02d}:{ss:02d}   x{a.speed}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(tmp, f"{k:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90]); k += 1
    t_sess += n / a.src_fps
    print(f"  {lg}: {n} frames → 누적 {k}", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "30", "-i", os.path.join(tmp, "%05d.jpg"),
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-movflags", "+faststart", a.out]
subprocess.run(cmd, check=True); shutil.rmtree(tmp)
print(f"영상 {a.out}: {k} 프레임 = {k/30:.1f}s (원본 {t_sess/60:.1f}분, x{a.speed}), {os.path.getsize(a.out)/1e6:.1f} MB [{time.time()-t0:.0f}s]")
