#!/usr/bin/env python3
"""세그 모델/추론 설정 실험 하네스 — base root 의 dataset/graph/rgb 를 공유하고 sem 만 새로 만들어 2D(GT 역투영) 평가, 옵션으로 3D 빌드+GT 평가.
사용: python run_exp.py <base_root> <exp_name> [--pre "--size 640 --tta-flip"] [--sem-dir <기존 sem 디렉터리(전처리 생략)>] [--build] [--classdict mapillary_road_v1]
결과: output/exp_<base>_<name>/compiled/{stage1_2d.json, gt_eval.json}, 요약 1줄을 gt/robotplus_outdoor_testbed/experiments.jsonl 에 append"""
import argparse, json, os, subprocess, sys, time, glob
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(REPO, "output")
ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("name"); ap.add_argument("--pre", default=""); ap.add_argument("--sem-dir"); ap.add_argument("--build", action="store_true"); ap.add_argument("--classdict", default="mapillary_road_v1"); ap.add_argument("--stride", type=int, default=2)
a = ap.parse_args(); base = os.path.join(OUT, a.base); root = os.path.join(OUT, f"exp_{a.base}_{a.name}"); os.makedirs(root + "/compiled", exist_ok=True)
for d in ["dataset", "graph"]:
    if not os.path.exists(f"{root}/{d}"): os.symlink(f"../{a.base}/{d}", f"{root}/{d}")
sem = root + "/sem"
if a.sem_dir:
    if os.path.islink(sem) or os.path.exists(sem): pass
    else: os.symlink(os.path.abspath(a.sem_dir), sem)
else:
    os.makedirs(sem, exist_ok=True)
    for f in ["rgb", "rgb_index.npz"]:
        if not os.path.exists(f"{sem}/{f}"): os.symlink(f"../../{a.base}/sem/{f}", f"{sem}/{f}")
    if not glob.glob(sem + "/sem/*.npz"):
        cmd = [sys.executable, f"{REPO}/tools/preprocess/precompute_semantic.py", sem, "--classdict", a.classdict] + a.pre.split()
        t0 = time.time(); print("B:", " ".join(cmd[2:]), flush=True); subprocess.run(cmd, check=True, stdout=open(root + "/compiled/B.log", "w"), stderr=subprocess.STDOUT); print(f"B done {time.time()-t0:.0f}s", flush=True)
# GT 정합(맵 프레임)은 base 것을 복사 (같은 데이터셋)
import shutil; shutil.copy(f"{base}/compiled/gt_map_frame.npz", f"{root}/compiled/gt_map_frame.npz")
subprocess.run([sys.executable, f"{HERE}/stage1_2d.py", root, "--stride", str(a.stride), "--panels", "0"], check=True, stdout=open(root + "/compiled/stage1.log", "w"), stderr=subprocess.STDOUT)
s2 = json.load(open(root + "/compiled/stage1_2d.json"))["summary"]
row = {"exp": os.path.basename(root), "base": a.base, "pre": a.pre, "sem_dir": a.sem_dir, "iou2d": {k: v["iou"] for k, v in s2["overall"].items()}, "sw_by_range": {k: v["sidewalk"]["iou"] for k, v in s2["by_range"].items()},
       "sw_pred": {k: s2["sidewalk_gt_pred_dist_by_robot_pos"][k] for k in s2["sidewalk_gt_pred_dist_by_robot_pos"]}}
if a.build:
    y = open(f"{REPO}/sessions/session_{a.base}.yaml").read().replace(f"root: {a.base}\n", f"root: {os.path.basename(root)}\n"); yp = f"{REPO}/sessions/session_{os.path.basename(root)}.yaml"; open(yp, "w").write(y)
    t0 = time.time(); subprocess.run([sys.executable, f"{REPO}/sessions/build_map_package.py", os.path.basename(yp), "--steps", "C,E,F,G,H"], cwd=f"{REPO}/sessions", check=True, stdout=open(root + "/compiled/build.log", "w"), stderr=subprocess.STDOUT)
    subprocess.run([sys.executable, f"{HERE}/gt_eval.py", "eval", root], check=True, stdout=open(root + "/compiled/gt_eval.log", "w"), stderr=subprocess.STDOUT)
    g = json.load(open(root + "/compiled/gt_eval.json")); row["iou3d"] = {k: v["iou"] for k, v in g["iou"].items()}; row["key3d"] = g["key"]; row["build_s"] = round(time.time() - t0)
with open(HERE + "/experiments.jsonl", "a") as fh: fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False))
