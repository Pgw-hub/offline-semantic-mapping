#!/usr/bin/env python3
"""/Odometry/base 를 TUM 궤적(# timestamp tx ty tz qx qy qz qw)으로 — robot_graph.py --traj 대용.
로봇 SLAM 이 저장 지도(.txt)를 안 남겼고 /path/pgo 도 쓸 수 없을 때(예: 20261002 sim 은 /path/pgo 가 항상 pose 2개) 사용.
오도메트리 자체를 키프레임 자세로 쓰므로 PGO 보정 Δ(t)=0 — /Odometry/base 에 루프클로저 점프가 없을 때만 타당(robot_graph 로그의 '점프 샘플' 확인).
  python geometry/odom_traj.py <mcap_dir> <out.txt> [--every 4] [--logs 0,1,2]
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))  # 저장소 루트 (패키지 import)
import argparse, glob, os, sys
import numpy as np
from preprocess.geometry import raisin_msgs as RM
from mcap.reader import make_reader

ap = argparse.ArgumentParser(); ap.add_argument("mcap_dir"); ap.add_argument("out"); ap.add_argument("--every", type=int, default=4, help="오도메트리 N 샘플마다 1 키프레임 (10 Hz → 0.4 s)")
ap.add_argument("--logs", help="예: 0,1,2 (기본 전체)")
a = ap.parse_args(); logs = None if not a.logs else {int(x) for x in a.logs.split(",")}
rows = []
for f in sorted(glob.glob(os.path.join(a.mcap_dir, "log_*.mcap"))):
    if logs is not None and int(os.path.basename(f)[4:7]) not in logs: continue
    for sch, ch, m in make_reader(open(f, "rb")).iter_messages(topics=["/Odometry/base"]):
        ns, p, q = RM.odom(m.data, sch); rows.append((ns, *p, *q))
rows.sort(); rows = rows[::a.every]
step = np.linalg.norm(np.diff(np.array([r[1:4] for r in rows]), axis=0), axis=1) if len(rows) > 1 else np.zeros(0)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as fh:
    fh.write("# timestamp tx ty tz qx qy qz qw   (from /Odometry/base, every %d; PGO correction = 0)\n" % a.every)
    for r in rows: fh.write("%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % (r[0] / 1e9, *r[1:]))
print(f"키프레임 {len(rows)} 개 → {a.out} · 간격 중앙 {np.median(step)*100 if len(step) else 0:.1f} cm 최대 {step.max()*100 if len(step) else 0:.0f} cm · 시각 {rows[0][0]/1e9:.2f}..{rows[-1][0]/1e9:.2f} s")
