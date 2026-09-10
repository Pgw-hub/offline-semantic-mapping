"""보정 자세 제공자 — build_graph.py 산출 graph/ 를 읽어 임의 시각(헤더 ns)의 PGO 보정 자세 world←base 를 준다.
   T(t) = Δ(t) · T_odom(t),  Δ_k = T_corr(k)·T_odom(k)⁻¹ (노드별 world 보정),  Δ(t) 는 t 를 감싸는 두 노드의 Δ 를 보간(위치 선형·회전 slerp) → 노드 경계 점프 없음.
   T_odom(t) 는 graph/odom_log_XXX.npz(원 /Odometry/base 전체) slerp 보간.
사용:  from geometry.pose_provider import PoseProvider;  pp = PoseProvider("output/x/graph");  T = pp.at(ns)  # None = 범위 밖/오도메트리 공백
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deskew import Odom, slerp_batch, R_batch

def _R2q(R):  # (N,3,3) → (N,4) wxyz
    N = len(R); q = np.empty((N, 4)); tr = np.trace(R, axis1=1, axis2=2)
    for i in range(N):
        m = R[i]
        if tr[i] > 0: s = np.sqrt(tr[i] + 1) * 2; q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        else:
            k = int(np.argmax(np.diag(m))); a, b, c = k, (k + 1) % 3, (k + 2) % 3
            s = np.sqrt(1 + m[a, a] - m[b, b] - m[c, c]) * 2; v = np.zeros(4); v[0] = (m[c, b] - m[b, c]) / s; v[1 + a] = 0.25 * s; v[1 + b] = (m[b, a] + m[a, b]) / s; v[1 + c] = (m[c, a] + m[a, c]) / s; q[i] = v
    return q / np.linalg.norm(q, axis=1, keepdims=True)

class PoseProvider:
    def __init__(self, graph_dir, max_gap_ms=250):
        g = json.load(open(os.path.join(graph_dir, "graph.json"))); self.meta = {k: g.get(k) for k in ("source", "frame", "loops", "optimized_at")}
        self.ns = np.array([n["stamp_ns"] for n in g["nodes"]], np.int64)
        To = np.array([n["T_odom"] for n in g["nodes"]]); Tc = np.array([n["T_corr"] for n in g["nodes"]])
        D = Tc @ np.linalg.inv(To); self.dq = _R2q(D[:, :3, :3]); self.dp = D[:, :3, 3]
        ns, pos, q = [], [], []
        for f in g["odom_files"].values():
            o = np.load(os.path.join(graph_dir, f)); ns.append(o["ns"]); pos.append(o["pos"]); q.append(o["quat_wxyz"])
        self.odom = Odom(np.concatenate(ns), np.concatenate(pos), np.concatenate(q)); self.max_gap = int(max_gap_ms * 1e6)
        self.t_min, self.t_max = int(self.ns[0]), int(self.ns[-1])
    def delta(self, t):
        """t (N,) → Δ(t) (N,4,4): 감싸는 두 노드 Δ 의 보간. 첫/마지막 노드 밖은 끝 노드 값."""
        t = np.asarray(t, np.int64); k = np.clip(np.searchsorted(self.ns, t), 1, len(self.ns) - 1)
        u = np.clip((t - self.ns[k - 1]) / np.maximum(self.ns[k] - self.ns[k - 1], 1), 0, 1).astype(float)
        R = R_batch(slerp_batch(self.dq[k - 1], self.dq[k], u)); p = self.dp[k - 1] * (1 - u)[:, None] + self.dp[k] * u[:, None]
        D = np.tile(np.eye(4), (len(t), 1, 1)); D[:, :3, :3] = R; D[:, :3, 3] = p; return D
    def at_many(self, t):
        t = np.asarray(t, np.int64); R, p, ok = self.odom.at(t); T = np.tile(np.eye(4), (len(t), 1, 1)); T[:, :3, :3] = R; T[:, :3, 3] = p
        return np.einsum("nij,njk->nik", self.delta(t), T), ok
    def at(self, ns):
        T, ok = self.at_many([int(ns)]); return T[0] if ok[0] else None
    def at_odom(self, ns):
        """보정 전 오도메트리 자세(비교용)"""
        R, p, ok = self.odom.at(np.array([int(ns)], np.int64))
        if not ok[0]: return None
        T = np.eye(4); T[:3, :3] = R[0]; T[:3, 3] = p[0]; return T
