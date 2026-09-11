#!/usr/bin/env python3
"""트랙 1: 서버 오프라인 PGO — mcap(/Odometry/base + /livox/lidar) → deskew → 키프레임 → 루프클로저(GICP) → GTSAM 최적화 → graph/ (GRAPH_FORMAT.md).

사용:
  python geometry/build_graph.py sessions/x.yaml                 # mcap_dir/logs/root 는 yaml 에서, 출력 output/<root>/graph
  python geometry/build_graph.py <mcap_dir> --logs 0-10 --out DIR
옵션: --kf-dist 1.0 --kf-rot 10 (키프레임 간격) --accum 2 (앞뒤 2스캔씩 누적) --loop-radius 8 --loop-min-gap 40 --rounds 2 --no-loops --rrd
출력: graph/graph.json, clouds/%06d.npz(base 프레임 xyz float32), odom_log_XXX.npz(원 오도메트리 전체 — 자세 보간용), graph_view.rrd(--rrd)
"""
import argparse, glob, json, os, sys, time
import numpy as np, yaml, open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deskew import Odom, read_mcap, deskew_scan, R_batch
import struct
from mcap.reader import make_reader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def T_of(R, p): T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p; return T
def inv(T): Ti = np.eye(4); Ti[:3, :3] = T[:3, :3].T; Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]; return Ti
def rot_deg(T): return float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
def pcd(xyz): return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, np.float64)))
def down(xyz, v): return np.asarray(pcd(xyz).voxel_down_sample(v).points)
def parse_logs(s):
    out = []
    for part in str(s).split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out


# ---------------------------------------------------------------- 1. 키프레임 추출 (로그별 스트리밍)
def dejump_odometry(mcap_dir, logs, jump_mps=4.0, jump_degps=200.0, lookback=4):
    """전 로그 /Odometry/base 를 읽어 루프클로저 점프(Mapping 모드에서 PGO 보정이 실시간 반영돼 생기는 불연속, 두 프레임을 번갈아 발행하는 창 포함)를 제거한
    연속 오도메트리를 만든다. 샘플 i 는 직전 lookback 개 중 속도가 정상인 가장 가까운 샘플 m 의 상대이동으로 이어 붙임(번갈아 발행 → m=i-2).
    반환: ns(정렬), pos, q(wxyz), 점프 시각 목록."""
    from pose_provider import _R2q
    ns, pos, q = [], [], []
    for li in logs:
        with open(os.path.join(mcap_dir, "log_%03d.mcap" % li), "rb") as f:
            for sch, ch, msg in make_reader(f).iter_messages(topics=["/Odometry/base"]):
                v = struct.unpack_from("<7d", msg.data, 8); sec, nsec = struct.unpack_from("<iI", msg.data, 0)
                ns.append(sec * 10**9 + nsec); pos.append(v[0:3]); q.append([v[6], v[3], v[4], v[5]])
    ns = np.array(ns, np.int64); o = np.argsort(ns); ns = ns[o]; pos = np.array(pos)[o]; q = np.array(q)[o]; q /= np.linalg.norm(q, axis=1, keepdims=True)
    T = np.tile(np.eye(4), (len(ns), 1, 1)); T[:, :3, :3] = R_batch(q); T[:, :3, 3] = pos
    C = T.copy(); jumps = []; rel_prev = np.eye(4)
    def ok(rel, dt): return np.linalg.norm(rel[:3, 3]) / dt <= jump_mps and abs(np.degrees(np.arctan2(rel[1, 0], rel[0, 0]))) / dt <= jump_degps
    for i in range(1, len(ns)):
        placed = False
        for m in range(i - 1, max(i - 1 - lookback, -1), -1):
            rel = inv(T[m]) @ T[i]; dt = max((ns[i] - ns[m]) / 1e9, 1e-3)
            if ok(rel, dt):
                C[i] = C[m] @ rel; placed = True
                if m == i - 1: rel_prev = rel
                else: jumps.append(int(ns[i]))
                break
        if not placed:                                            # 이어 붙일 정상 샘플 없음 → 직전 속도(병진만, 회전 누적 방지)로 외삽
            ext = np.eye(4); ext[:3, 3] = rel_prev[:3, 3]; C[i] = C[i-1] @ ext; jumps.append(int(ns[i]))
    jumps = np.array(sorted(set(jumps)), np.int64)
    if len(jumps):                                                # 1초 이상 이어지는 점프 창 = 실시간 자세가 수 초간 미끄러지는 구간(PGO 대보정). 이 안의 오도메트리는 외삽값이라 신뢰 불가
        br = np.where(np.diff(jumps) > int(1e9))[0]; starts = np.r_[0, br + 1]; ends = np.r_[br, len(jumps) - 1]
        for s_, e_ in zip(starts, ends):
            if jumps[e_] - jumps[s_] >= int(1e9): print(f"    ★ dejump 경고: +{(jumps[s_]-ns[0])/1e9:.1f}~{(jumps[e_]-ns[0])/1e9:.1f}s 오도메트리 미끄러짐 창 ({e_-s_+1} 샘플) — 이 구간 노드/엣지는 신뢰 불가", flush=True)
    qC = _R2q(C[:, :3, :3]); return ns, C[:, :3, 3].copy(), qC, jumps


def extract_keyframes(mcap_dir, logs, t_bl, kf_dist, kf_rot_deg, accum, rmin, rmax, vox, dj=None):
    nodes, clouds, odom_all = [], [], {}
    last_T = None
    for li in logs:
        name = "log_%03d" % li; path = os.path.join(mcap_dir, name + ".mcap"); t = time.time()
        odom, scans = read_mcap(path)
        if dj is not None:                                            # 점프 제거 오도메트리로 교체 (ns 일치 샘플)
            dns, dpos, dq = dj; k = np.searchsorted(dns, odom.ns); k = np.clip(k, 0, len(dns) - 1); hit = dns[k] == odom.ns
            odom.pos = np.where(hit[:, None], dpos[k], odom.pos); odom.q = np.where(hit[:, None], dq[k], odom.q)
        jump = np.linalg.norm(np.diff(odom.pos, axis=0), axis=1); bad = np.where(jump > 5.0)[0]   # SLAM 붕괴(172935 log_011~) 감지
        if len(bad):
            cut = odom.ns[bad[0]]; print(f"  ★ {name}: 오도메트리 점프 {jump[bad[0]]:.1f}m @ +{(cut-odom.ns[0])/1e9:.0f}s → 이후 스캔 제외, 다음 로그부터 중단", flush=True)
            scans = [s for s in scans if s[0] < cut]; logs = list(logs[: list(logs).index(li) + 1]) if li in logs else list(logs)
        odom_all[name] = odom
        # 스캔별 deskew 결과 캐시(누적용): (Pb0, R0, p0)
        dsk = [None] * len(scans)
        def get(i):
            if dsk[i] is None: dsk[i] = deskew_scan(odom, scans[i][0], scans[i][1], scans[i][2], t_bl) or False
            return dsk[i]
        n_kf0 = len(nodes)
        for i in range(len(scans)):
            d = get(i)
            if d is False: continue
            Pb0, R0, p0 = d; T = T_of(R0, p0)
            if last_T is not None:
                dT = inv(last_T) @ T
                if np.linalg.norm(dT[:3, 3]) < kf_dist and rot_deg(dT) < kf_rot_deg: continue
            last_T = T
            # 누적: 앞뒤 accum 스캔을 kf base 프레임으로
            parts = []
            for k in range(max(0, i - accum), min(len(scans), i + accum + 1)):
                dk = get(k)
                if dk is False: continue
                Pk, Rk, pk = dk; Pw = Pk @ Rk.T + pk; parts.append((Pw - p0) @ R0)
            P = np.concatenate(parts); r = np.linalg.norm(P, axis=1); P = P[(r > rmin) & (r < rmax)]
            P = down(P, vox).astype(np.float32)
            nodes.append({"id": len(nodes), "stamp_ns": int(scans[i][0]), "log": name, "T_odom": T}); clouds.append(P)
            for k in range(max(0, i - 2 * accum - 1), i - accum): dsk[k] = None  # 메모리 해제
        print(f"  {name}: 스캔 {len(scans)} → 키프레임 {len(nodes)-n_kf0} (누적 {len(nodes)}) [{time.time()-t:.0f}s]", flush=True)
        del scans, dsk
    return nodes, clouds, odom_all


# ---------------------------------------------------------------- 2. 루프클로저
def register(src_xyz, tgt_xyz, T_init, vox=0.1):
    """src(kf i 프레임) → tgt(kf j 프레임). 반환 T_ji(추정), fitness, rmse"""
    s = pcd(down(src_xyz, vox)); tg = pcd(down(tgt_xyz, vox))
    prm = o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=25)
    s.estimate_normals(prm); tg.estimate_normals(prm); s.estimate_covariances(prm); tg.estimate_covariances(prm)
    reg = o3d.pipelines.registration
    T = T_init
    for mc in (1.5, 0.6):
        r = reg.registration_generalized_icp(s, tg, mc, T, reg.TransformationEstimationForGeneralizedICP(), reg.ICPConvergenceCriteria(max_iteration=60)); T = r.transformation
    r = reg.registration_icp(s, tg, 0.3, T, reg.TransformationEstimationPointToPlane(), reg.ICPConvergenceCriteria(max_iteration=40))
    return np.asarray(r.transformation), float(r.fitness), float(r.inlier_rmse)


def find_loops(nodes, clouds, T_est, existing, radius, min_gap, sub=4, fit_min=0.55, rmse_max=0.10, max_shift=5.0, max_rot=20.0):
    P = np.array([T[:3, 3] for T in T_est]); n = len(nodes); new = []
    tree = o3d.geometry.KDTreeFlann(pcd(P))
    tried = set((a, b) for a, b, *_ in existing)
    for i in range(n):
        k, idx, _ = tree.search_radius_vector_3d(P[i], radius)
        cand = [j for j in idx if i - j >= min_gap]
        if not cand: continue
        j = min(cand, key=lambda j: np.linalg.norm(P[i] - P[j]))
        if any(abs(a - i) < 5 and abs(b - j) < 5 for a, b in tried): continue
        tried.add((i, j))
        # 목표 서브맵: j 주변 kf 들을 j 프레임으로
        Tj_inv = inv(T_est[j]); parts = []
        for k2 in range(max(0, j - sub), min(n, j + sub + 1)):
            if k2 - j > 0 and i - k2 < min_gap // 2: break
            Tk = Tj_inv @ T_est[k2]; parts.append(clouds[k2] @ Tk[:3, :3].T + Tk[:3, 3])
        tgt = np.concatenate(parts); T_init = Tj_inv @ T_est[i]
        T_ji, fit, rmse = register(clouds[i], tgt, T_init)
        dT = inv(T_init) @ T_ji; shift = float(np.linalg.norm(dT[:3, 3])); rot = rot_deg(dT)
        ok = fit >= fit_min and rmse <= rmse_max and shift <= max_shift and rot <= max_rot
        print(f"    loop {i:4d}←{j:4d} ({nodes[i]['log']}↔{nodes[j]['log']}) d0={np.linalg.norm(P[i]-P[j]):.1f}m fit={fit:.2f} rmse={rmse*100:.1f}cm shift={shift:.2f}m rot={rot:.1f}° {'OK' if ok else 'reject'}", flush=True)
        if ok: new.append((i, j, T_ji, fit, rmse))
    return new


# ---------------------------------------------------------------- 3. GTSAM
def optimize(nodes, odom_edges, loops, T_init):
    import gtsam
    g = gtsam.NonlinearFactorGraph(); v = gtsam.Values()
    g.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(T_init[0]), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 3 + [1e-3] * 3))))
    for a, b, T_rel, sig in odom_edges:
        g.add(gtsam.BetweenFactorPose3(a, b, gtsam.Pose3(T_rel), gtsam.noiseModel.Diagonal.Sigmas(np.array(sig))))
    huber = gtsam.noiseModel.mEstimator.Huber.Create(1.0)
    for i, j, T_ji, fit, rmse in loops:   # T_ji: i 프레임 → j 프레임 = j 에서 본 i 의 자세 → Between(j, i)
        base = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.radians(0.5)] * 3 + [0.05] * 3))
        g.add(gtsam.BetweenFactorPose3(j, i, gtsam.Pose3(T_ji), gtsam.noiseModel.Robust.Create(huber, base)))
    for k, T in enumerate(T_init): v.insert(k, gtsam.Pose3(T))
    prm = gtsam.LevenbergMarquardtParams(); prm.setMaxIterations(100); prm.setRelativeErrorTol(1e-6)
    opt = gtsam.LevenbergMarquardtOptimizer(g, v, prm); e0 = g.error(v); res = opt.optimize(); e1 = g.error(res)
    return [res.atPose3(k).matrix() for k in range(len(T_init))], e0, e1


# ---------------------------------------------------------------- 4. 저장 / 시각화
def save_graph(out, nodes, clouds, odom_edges, loops, T_corr, odom_all, t_bl, meta):
    os.makedirs(os.path.join(out, "clouds"), exist_ok=True)
    for k, P in enumerate(clouds): np.savez_compressed(os.path.join(out, "clouds", "%06d.npz" % k), xyz=P)
    for name, od in odom_all.items(): np.savez_compressed(os.path.join(out, f"odom_{name}.npz"), ns=od.ns, pos=od.pos, quat_wxyz=od.q)
    T_bl = np.eye(4); T_bl[:3, 3] = t_bl
    g = {"format_version": 1, "source": "server_pgo", "frame": "map", "time_base": "header_stamp_ns", "extrinsics": {"T_base_lidar": T_bl.tolist()},
         "nodes": [dict(id=n["id"], stamp_ns=n["stamp_ns"], log=n["log"], T_odom=np.round(n["T_odom"], 6).tolist(), T_corr=np.round(T_corr[k], 6).tolist(),
                        cloud="clouds/%06d.npz" % k, n_pts=int(len(clouds[k]))) for k, n in enumerate(nodes)],
         "edges": [dict(**{"from": a, "to": b}, type="odom", T_rel=np.round(T, 6).tolist(), info=(1 / np.array(sig) ** 2).round(2).tolist()) for a, b, T, sig in odom_edges]
                + [dict(**{"from": j, "to": i}, type="loop", T_rel=np.round(T, 6).tolist(), info=[1 / np.radians(0.5) ** 2] * 3 + [1 / 0.05 ** 2] * 3, fitness=round(f, 3), rmse_m=round(r, 4)) for i, j, T, f, r in loops],
         "odom_files": {name: f"odom_{name}.npz" for name in odom_all}, "loops": len(loops), "dejump": bool(a.dejump), **meta}
    json.dump(g, open(os.path.join(out, "graph.json"), "w"), ensure_ascii=False)
    st = os.stat(REPO)                                   # 컨테이너(root) 산출물 → 저장소 소유자로
    for d, dirs, files in os.walk(out):
        for x in [d] + [os.path.join(d, f) for f in files]:
            try: os.chown(x, st.st_uid, st.st_gid)
            except PermissionError: pass
    print("저장:", os.path.join(out, "graph.json"), f"· 노드 {len(nodes)} · odom 엣지 {len(odom_edges)} · loop {len(loops)}")


def write_rrd(out, nodes, clouds, T_odom, T_corr, loops, vox=0.3):
    import rerun as rr, rerun.blueprint as rrb
    rr.init("pgo_graph")
    def merged(Ts):
        parts = [(clouds[k] @ Ts[k][:3, :3].T + Ts[k][:3, 3]) for k in range(0, len(clouds), 1)]
        return down(np.concatenate(parts), vox)
    for tag, Ts in (("before", T_odom), ("after", T_corr)):
        M = merged(Ts); z = M[:, 2]; zn = np.clip((z - np.percentile(z, 2)) / max(np.percentile(z, 98) - np.percentile(z, 2), 1e-3), 0, 1)
        col = (np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], 1) * 255).astype(np.uint8)
        rr.log(f"{tag}/map", rr.Points3D(M.astype(np.float32), colors=col, radii=0.06), timeless=True)
        P = np.array([T[:3, 3] for T in Ts])
        rr.log(f"{tag}/path", rr.LineStrips3D([P.astype(np.float32)], colors=[[255, 255, 255]], radii=0.12), timeless=True)
        rr.log(f"{tag}/keyframes", rr.Points3D(P.astype(np.float32), colors=[[255, 200, 0]], radii=0.2), timeless=True)
        if loops:
            segs = [np.stack([P[i], P[j]]).astype(np.float32) for i, j, *_ in loops]
            rr.log(f"{tag}/loops", rr.LineStrips3D(segs, colors=[[255, 0, 80]], radii=0.1), timeless=True)
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(rrb.Spatial3DView(origin="before", name="odometry only"), rrb.Spatial3DView(origin="after", name="PGO")), collapse_panels=True))
    rr.save(os.path.join(out, "graph_view.rrd")); print("rrd:", os.path.join(out, "graph_view.rrd"))


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("src", help="session.yaml 또는 mcap_dir"); ap.add_argument("--logs"); ap.add_argument("--out")
    ap.add_argument("--kf-dist", type=float, default=1.0); ap.add_argument("--kf-rot", type=float, default=10.0); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--rmin", type=float, default=0.8); ap.add_argument("--rmax", type=float, default=50.0); ap.add_argument("--vox", type=float, default=0.05)
    ap.add_argument("--loop-radius", type=float, default=8.0); ap.add_argument("--loop-min-gap", type=int, default=40); ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-shift", type=float, default=5.0, help="루프 GICP 결과가 초기 추정에서 이보다 많이 움직이면 거부 (m). 드리프트가 큰 로그는 올릴 것"); ap.add_argument("--max-rot", type=float, default=20.0)
    ap.add_argument("--no-loops", action="store_true"); ap.add_argument("--rrd", action="store_true")
    ap.add_argument("--dejump", action="store_true", help="[실험] /Odometry/base 의 실시간 PGO 보정 점프(루프클로저)를 제거한 연속 오도메트리 사용. 수 초간 미끄러지는 대보정 창은 복원 못함 → 그런 데이터셋은 robot_graph.py(로봇 PGO 자세) 사용")
    a = ap.parse_args(); t_all = time.time()
    if a.src.endswith((".yaml", ".yml")):
        s = yaml.safe_load(open(a.src)); mcap_dir = s["mcap_dir"]; logs = [int(x) for x in s["logs"]] if a.logs is None else parse_logs(a.logs)
        out = a.out or os.path.join(REPO, "output", s["root"], "graph")
    else:
        mcap_dir = a.src; logs = parse_logs(a.logs or "0"); out = a.out or os.path.join(mcap_dir, "graph")
    if a.logs: logs = parse_logs(a.logs)
    tr = yaml.safe_load(open(os.path.join(mcap_dir, "raisin_config/calibration/livox_lidar_0_lidar.yaml")))["transform"]; t_bl = np.array([tr["translation"][k] for k in "xyz"])
    print(f"[1] 키프레임 추출: {mcap_dir} logs={logs} (kf {a.kf_dist}m/{a.kf_rot}°, accum ±{a.accum})", flush=True)
    dj = None
    if a.dejump:
        dns, dpos, dq, jumps = dejump_odometry(mcap_dir, logs); dj = (dns, dpos, dq)
        print(f"    dejump: 오도메트리 {len(dns)} 샘플, 점프 제거 {len(jumps)} 샘플 (시각 +{np.round((jumps - dns[0]) / 1e9, 1)[:12].tolist()}{'…' if len(jumps) > 12 else ''} s)", flush=True)
    nodes, clouds, odom_all = extract_keyframes(mcap_dir, logs, t_bl, a.kf_dist, a.kf_rot, a.accum, a.rmin, a.rmax, a.vox, dj)
    T_odom = [n["T_odom"] for n in nodes]
    odom_edges = []
    for k in range(len(nodes) - 1):
        T_rel = inv(T_odom[k]) @ T_odom[k + 1]; d = np.linalg.norm(T_rel[:3, 3])
        odom_edges.append((k, k + 1, T_rel, [np.radians(0.2) + np.radians(0.5) * d] * 3 + [0.02 + 0.01 * d] * 3))
    loops = []; T_est = list(T_odom)
    if not a.no_loops:
        for rd in range(a.rounds):
            print(f"[2] 루프 탐색 round {rd+1}/{a.rounds} (radius {a.loop_radius}m, gap≥{a.loop_min_gap} kf)", flush=True)
            new = find_loops(nodes, clouds, T_est, loops, a.loop_radius, a.loop_min_gap, max_shift=a.max_shift, max_rot=a.max_rot)
            if not new: print("    새 루프 없음"); break
            loops += new
            print(f"[3] GTSAM 최적화 (loop {len(loops)})", flush=True)
            T_est, e0, e1 = optimize(nodes, odom_edges, loops, T_est)
            print(f"    error {e0:.1f} → {e1:.1f}", flush=True)
    T_corr = T_est
    dpos = np.array([np.linalg.norm(T_corr[k][:3, 3] - T_odom[k][:3, 3]) for k in range(len(nodes))])
    # 루프 잔차: 최적화 후 루프 엣지 자세 vs 측정
    res = [np.linalg.norm((inv(inv(T_corr[j]) @ T_corr[i]) @ T)[:3, 3]) for i, j, T, *_ in loops]
    meta = {"optimized_at": time.strftime("%Y-%m-%d %H:%M"), "mcap_dir": mcap_dir, "logs": logs, "keyframe": {"dist_m": a.kf_dist, "rot_deg": a.kf_rot, "accum": a.accum, "vox_m": a.vox},
            "loop_residual_rms_m": float(np.sqrt(np.mean(np.square(res)))) if res else None, "pose_change_m": {"max": float(dpos.max()), "mean": float(dpos.mean())}}
    print(f"[4] 결과: 자세 변화 평균 {dpos.mean():.2f}m 최대 {dpos.max():.2f}m · 루프 잔차 rms {meta['loop_residual_rms_m']} m · {time.time()-t_all:.0f}s")
    save_graph(out, nodes, clouds, odom_edges, loops, T_corr, odom_all, t_bl, meta)
    if a.rrd: write_rrd(out, nodes, clouds, T_odom, T_corr, loops)
