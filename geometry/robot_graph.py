#!/usr/bin/env python3
"""로봇 lidar_slam 의 최종 PGO 키프레임 자세를 우리 파이프라인의 graph/ (PoseProvider 규약) 로 변환.
자세 출처 (둘 중 하나):
  --traj <name>.txt   로봇 save_point_cloud_map 이 저장한 TUM 궤적 (# timestamp tx ty tz qx qy qz qw) — 시각 정확 (권장)
  (없으면)            mcap 의 /path/pgo 마지막 메시지. pose 별 stamp 가 없어 오도메트리와 상대 이동을 맞춰 시각 복원
                      (/Odometry/base 는 Mapping 모드에서 PGO 보정이 실시간 반영돼 루프클로저 때 수 m 점프 → 점프를 제거한 연속 오도메트리로 매칭)
  python geometry/robot_graph.py <mcap_dir> <out_graph_dir> [--traj robot_map.txt] [--logs 0,13] [--rrd]
산출: graph.json (nodes: stamp_ns, T_odom(원 /Odometry/base), T_corr(로봇 PGO), jump_ns), odom_log_XXX.npz, robot_graph_view.rrd
"""
import argparse, glob, json, os, struct, sys, time
import numpy as np
from mcap.reader import make_reader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deskew import q2R as _q2R_wxyz

def q2R(q):  # xyzw
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def T_of(p, q): T = np.eye(4); T[:3, :3] = q2R(q); T[:3, 3] = p; return T
def yaw_of(T): return np.arctan2(T[1, 0], T[0, 0])
def wrap(a): return np.angle(np.exp(1j * a))

def read(mcap_dir, logs):
    files = sorted(glob.glob(os.path.join(mcap_dir, "log_*.mcap")))
    if logs is not None: files = [f for f in files if int(os.path.basename(f)[4:7]) in logs]
    per_log, last_path = {}, None
    for f in files:
        name = os.path.basename(f)[:-5]; ns, pos, qx = [], [], []
        for sch, ch, m in make_reader(open(f, "rb")).iter_messages(topics=["/Odometry/base", "/path/pgo"]):
            d = m.data; sec, nsec = struct.unpack_from("<iI", d, 0)
            if ch.topic == "/Odometry/base":
                v = struct.unpack_from("<13d", d, 8); ns.append(sec * 10**9 + nsec); pos.append(v[:3]); qx.append(v[3:7])
            else:
                N = struct.unpack_from("<I", d, 8)[0]; P = np.frombuffer(d, "<f8", N * 7, 12).reshape(N, 7)
                if last_path is None or len(P) >= len(last_path): last_path = P.copy()
        per_log[name] = (np.array(ns, np.int64), np.array(pos), np.array(qx))
    ns = np.concatenate([v[0] for v in per_log.values()]); pos = np.concatenate([v[1] for v in per_log.values()]); qx = np.concatenate([v[2] for v in per_log.values()])
    o = np.argsort(ns); T = np.array([T_of(p, q) for p, q in zip(pos[o], qx[o])])
    return per_log, ns[o], T, last_path

def dejump(T, ns, jump_mps=4.0, jump_degps=200.0):
    """루프클로저 점프(속도 >4m/s 또는 yaw 속도 >200°/s 로 나타나는 불연속)를 직전 샘플의 상대 이동으로 대체한 연속 오도메트리. 점프 시각 목록도 반환."""
    C = np.empty_like(T); C[0] = T[0]; rel_prev = np.eye(4); jumps = []
    for i in range(1, len(T)):
        rel = np.linalg.inv(T[i-1]) @ T[i]; dt = max((ns[i] - ns[i-1]) / 1e9, 1e-3)
        if np.linalg.norm(rel[:3, 3]) / dt > jump_mps or abs(np.degrees(yaw_of(rel))) / dt > jump_degps:
            jumps.append(int(ns[i])); rel = rel_prev
        C[i] = C[i-1] @ rel; rel_prev = rel
    return C, np.array(jumps, np.int64)

def match_times(ns, C, P, T_live, min_gap_s=0.3, max_gap_s=60.0, w_yaw=0.5):
    """PGO 연속 키프레임 상대 이동 ↔ 연속 오도메트리 상대 이동 순차 매칭 → 키프레임 시각 인덱스."""
    Tp = np.array([T_of(p[:3], p[3:7]) for p in P]); n = len(P); j = np.zeros(n, int); err = np.zeros(n); t = ns / 1e9
    j[0] = int(np.argmin(np.linalg.norm(T_live[:min(300, len(t)), :3, 3] - P[0, :3], axis=1)))  # 시작점: 실시간 오도메트리 위치가 PGO 0번과 가장 가까운 샘플
    for k in range(1, n):
        rel = np.linalg.inv(Tp[k-1]) @ Tp[k]; dp, dy = rel[:3, 3], yaw_of(rel); j0 = j[k-1]
        a0 = int(np.searchsorted(t, t[j0] + min_gap_s)); a1 = int(np.searchsorted(t, t[j0] + max_gap_s)); cand = np.arange(max(a0, j0 + 1), max(a1, j0 + 2))
        cand = cand[cand < len(t)]
        if len(cand) == 0: j[k:] = len(t) - 1; err[k:] = np.inf; break
        Ti = np.linalg.inv(C[j0]); p = C[cand, :3, 3] @ Ti[:3, :3].T + Ti[:3, 3]; R = np.einsum("ij,njk->nik", Ti[:3, :3], C[cand, :3, :3]); y = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        e = np.linalg.norm(p - dp, axis=1) + w_yaw * np.abs(wrap(y - dy)); a = int(np.argmin(e)); j[k] = cand[a]; err[k] = e[a]
    return Tp, j, err

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mcap_dir"); ap.add_argument("out"); ap.add_argument("--traj"); ap.add_argument("--logs"); ap.add_argument("--rrd", action="store_true")
    ap.add_argument("--tmax", type=float, help="시작 후 이 초 이후의 키프레임은 버림 (시각 복원이 불확실한 구간 제외)"); ap.add_argument("--max-err", type=float, default=0.15, help="상대이동 매칭 오차가 이보다 큰 노드는 버림 (m)")
    a = ap.parse_args(); logs = None
    if a.logs: lo, hi = (a.logs.split(",") + [None])[:2]; logs = set(range(int(lo), int(hi or lo) + 1))
    t0 = time.time(); per_log, ns, T, P = read(a.mcap_dir, logs); C, jumps = dejump(T, ns)
    print(f"오도메트리 {len(ns)} 샘플 {(ns[-1]-ns[0])/1e9:.0f}s · 루프클로저 점프 샘플 {len(jumps)} · /path/pgo 마지막 poses {0 if P is None else len(P)} [{time.time()-t0:.0f}s]")
    if a.traj:
        D = np.loadtxt(a.traj, comments="#"); kf_ns = (D[:, 0] * 1e9).round().astype(np.int64); Tp = np.array([T_of(r[1:4], r[4:8]) for r in D]); src = f"robot lidar_slam saveMap traj ({os.path.basename(a.traj)})"
        k = np.clip(np.searchsorted(ns, kf_ns), 1, len(ns) - 1); k = np.where(np.abs(ns[k - 1] - kf_ns) < np.abs(ns[k] - kf_ns), k - 1, k)   # 가장 가까운 오도메트리 샘플
        near = np.abs(ns[k] - kf_ns) / 1e6; print(f"TUM 궤적 {len(D)} 키프레임 · 오도메트리 stamp 와 차이 중앙 {np.median(near):.1f}ms 최대 {near.max():.1f}ms (>50ms {int((near>50).sum())}개)")
        keep = near <= 100                                            # 녹화 시작 전/후의 키프레임(오도메트리 없음) 제외 — 매핑을 녹화보다 먼저 시작한 경우
        if (~keep).any(): print(f"  오도메트리 범위 밖 키프레임 {int((~keep).sum())}개 제외: stamp 오프셋 {np.round((kf_ns[~keep] - ns[0]) / 1e9, 1).tolist()[:6]} s")
        kf_ns, Tp, k = kf_ns[keep], Tp[keep], k[keep]
        j = k; err = np.zeros(len(kf_ns))
    else:
        assert P is not None, "/path/pgo 없음"
        Tp, j, err = match_times(ns, C, P, T); kf_ns = ns[j]; src = "robot lidar_slam /path/pgo (last msg; stamps recovered by odometry matching)"
        gaps = np.diff(kf_ns) / 1e9; bad = err > 0.15
        print(f"키프레임 시각 복원: 상대이동 오차 중앙 {np.median(err[1:])*100:.1f}cm p90 {np.percentile(err[1:],90)*100:.1f}cm · >15cm {bad.sum()}개 (kf {np.where(bad)[0][:12].tolist()}) · 간격 중앙 {np.median(gaps):.2f}s 최소 {gaps.min():.2f}s 최대 {gaps.max():.1f}s")
    if not a.traj:                                           # 신뢰 구간만 남김
        keep = np.isfinite(err) & (err <= a.max_err); keep[0] = True
        if a.tmax: keep &= (kf_ns - ns[0]) / 1e9 <= a.tmax
        print(f"노드 필터: {len(keep)} → {keep.sum()} (오차>{a.max_err}m 또는 tmax 초과 제외)"); Tp, j, err, kf_ns = Tp[keep], j[keep], err[keep], kf_ns[keep]
    # 연속 노드 간 상대이동 일관성: PGO 상대이동 vs 실시간 오도메트리 상대이동 (점프 없는 구간은 수 cm 여야 함)
    To = T[j]; rp = np.einsum("nij,njk->nik", np.linalg.inv(Tp[:-1]), Tp[1:]); ro = np.einsum("nij,njk->nik", np.linalg.inv(To[:-1]), To[1:])
    dpos = np.linalg.norm(rp[:, :3, 3] - ro[:, :3, 3], axis=1); dyaw = np.abs(np.degrees(wrap(np.arctan2(rp[:, 1, 0], rp[:, 0, 0]) - np.arctan2(ro[:, 1, 0], ro[:, 0, 0]))))
    seg_jump = np.array([((jumps > kf_ns[i]) & (jumps <= kf_ns[i+1])).any() for i in range(len(kf_ns) - 1)]); ok_ = ~seg_jump
    print(f"연속 노드 상대이동 (PGO vs 오도메트리, 점프 없는 {ok_.sum()}구간): 위치 차 중앙 {np.median(dpos[ok_])*100:.1f}cm p90 {np.percentile(dpos[ok_],90)*100:.1f}cm 최대 {dpos[ok_].max()*100:.0f}cm · yaw 차 중앙 {np.median(dyaw[ok_]):.2f}° 최대 {dyaw[ok_].max():.1f}° · 점프 걸친 구간 {seg_jump.sum()}개")
    corr = np.linalg.norm(Tp[:, :3, 3] - To[:, :3, 3], axis=1); print(f"최종 PGO − 실시간 /Odometry/base: 위치 차 중앙 {np.median(corr)*100:.0f}cm p90 {np.percentile(corr,90)*100:.0f}cm 최대 {corr.max()*100:.0f}cm")
    os.makedirs(a.out, exist_ok=True)
    for name, (n_, p_, qx_) in per_log.items():
        np.savez_compressed(os.path.join(a.out, f"odom_{name}.npz"), ns=n_, pos=p_, quat_wxyz=np.stack([qx_[:, 3], qx_[:, 0], qx_[:, 1], qx_[:, 2]], 1))
    log_of = lambda t: next((nm for nm, v in per_log.items() if len(v[0]) and v[0][0] <= t <= v[0][-1]), "")
    nodes = []
    for i in range(len(kf_ns)):
        jn = jumps[(jumps > kf_ns[i-1]) & (jumps <= kf_ns[i])] if i else np.array([], np.int64)
        nodes.append({"id": i, "stamp_ns": int(kf_ns[i]), "log": log_of(kf_ns[i]), "T_odom": To[i].tolist(), "T_corr": Tp[i].tolist(), "cloud": None, "n_pts": 0, "match_err_m": float(err[i]),
                      **({"jump_ns": int(jn[0])} if len(jn) else {})})
    g = {"format_version": 1, "source": src, "frame": "robot lidar_slam PGO world (final /path/pgo)", "time_base": "header stamp ns", "extrinsics": {}, "nodes": nodes,
         "edges": [{"i": i, "j": i + 1, "type": "odom"} for i in range(len(nodes) - 1)], "odom_files": {nm: f"odom_{nm}.npz" for nm in per_log}, "loops": None,
         "optimized_at": time.strftime("%Y-%m-%d %H:%M"), "mcap_dir": a.mcap_dir, "logs": sorted(per_log), "keyframe": {"source": "robot", "n": len(nodes)},
         "odom_jumps_ns": jumps.tolist(), "pose_change_m": {"median": float(np.median(corr)), "max": float(corr.max())}}
    json.dump(g, open(os.path.join(a.out, "graph.json"), "w"), ensure_ascii=False)
    if a.rrd:
        import rerun as rr, rerun.blueprint as rrb
        rr.init("robot_graph"); rr.save(os.path.join(a.out, "graph_view.rrd"))
        rr.log("odometry_live", rr.LineStrips3D([T[:, :3, 3].astype(np.float32)], colors=[[150, 150, 150]], radii=0.05))
        rr.log("robot_pgo", rr.LineStrips3D([Tp[:, :3, 3].astype(np.float32)], colors=[[60, 120, 255]], radii=0.08))
        rr.log("robot_pgo/keyframes", rr.Points3D(Tp[:, :3, 3].astype(np.float32), colors=[[60, 120, 255]], radii=0.15))
        rr.log("correction", rr.LineStrips3D(np.stack([To[:, :3, 3], Tp[:, :3, 3]], 1).astype(np.float32), colors=[[230, 60, 60]], radii=0.03))
        if len(jumps): rr.log("odometry_live/jumps", rr.Points3D(T[np.searchsorted(ns, jumps), :3, 3].astype(np.float32), colors=[[255, 160, 0]], radii=0.4))
        rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="/", name="live odom(grey, jumps orange) vs robot final PGO(blue), correction(red)"), collapse_panels=True))
    print("저장:", os.path.join(a.out, "graph.json"), f"· 노드 {len(nodes)} [{time.time()-t0:.0f}s]")
