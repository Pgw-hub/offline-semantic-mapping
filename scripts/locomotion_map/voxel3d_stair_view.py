#!/usr/bin/env python3
"""계단 디딤판 채움 확인 그림 (voxel3d_stair_complete.py 결과 검수용, 빌드 F 단계에서 자동 생성).
위: 측면도(계단 축 t vs 높이 z, 계단 폭 안 셀) — 채움 전 / 채움 후. 회색 = 원래 셀(LiDAR·평탄 채움; depth 채움 src 2 는 그리지 않음), 초록 = 디딤판·앞면 채움(src 5), 점선 = 모델 디딤판 높이.
아래: 채움 후 윗면도(맵 좌표) — 디딤판 사각형 안 칸마다 파랑 = 디딤판 높이(±3 cm)에 셀 있음, 빨강 = 디딤판 사이 높이의 원래 맵 윗면(가짜 중간 단·구름, 앞면 판 제외).
글자는 영어(노트북 이미지에 한글 글꼴 없음).
사용: voxel3d_stair_view.py <stair.json> <out.png> <before_tiles> <after_tiles>"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voxel3d_stair_complete import read_tiles, top_cells


def load(d):
    idx, R = read_tiles(d); r = idx["res_xy"]
    R = R[R["src"] != 2]                      # depth 채움 셀(src 2)은 그림에서 뺀다 — 계단 검수는 LiDAR 셀과 채움 셀만 (2026-11-02)
    x = (R["ix"] + 0.5) * r; y = (R["iy"] + 0.5) * r
    z = np.where(R["max_z"] - R["mean_z"] > 0.03, R["max_z"], R["mean_z"]).astype(np.float64)
    return R, r, x, y, z


def side(a, M, R, x, y, z, title):
    u, v = np.array(M["u"]), np.array(M["v"]); H = [t["h"] for t in M["treads"]]; N = [t["nose"] for t in M["treads"]]
    t_lo, t_hi = min(N) - 0.6, max(N) + M["depth"] + 0.6; z_lo, z_hi = min(H) - 0.45, max(H) + 0.45
    t = np.c_[x, y] @ u; l = np.c_[x, y] @ v
    band = (l >= M["l0"]) & (l <= M["l1"]) & (t > t_lo) & (t < t_hi) & (z > z_lo) & (z < z_hi)
    for m, col, lab in [(band & (R["src"] != 5), "#8a8f94", "map cells (no depth fill)"), (band & (R["src"] == 5), "#1f9a55", "tread / riser fill")]:
        if m.any(): a.scatter(t[m], z[m], s=1.4, c=col, lw=0, label=f"{lab} {m.sum():,}")
    for h, n in zip(H, N): a.plot([n, n + M["depth"]], [h, h], "k--", lw=0.6, alpha=0.6)
    a.set_xlim(t_lo, t_hi); a.set_ylim(z_lo, z_hi); a.set_aspect("equal"); a.set_title(title, fontsize=11)
    a.set_xlabel("stair axis [m]"); a.set_ylabel("z [m]"); a.legend(loc="upper left", fontsize=8, markerscale=6)


def top(a, M, R, r, x, y, z):
    u, v = np.array(M["u"]), np.array(M["v"]); H = np.array([t["h"] for t in M["treads"]]); N = np.array([t["nose"] for t in M["treads"]])
    tp = top_cells(x, y, z, r)
    corners = np.array([[N.min(), M["l0"]], [N.min(), M["l1"]], [N.max() + M["depth"], M["l0"]], [N.max() + M["depth"], M["l1"]]]) @ np.array([u, v])
    ix0, iy0 = np.floor(corners.min(0) / r).astype(int) - 2; ix1, iy1 = np.floor(corners.max(0) / r).astype(int) + 2
    GX, GY = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1), indexing="ij")
    ct = ((GX + 0.5) * r) * u[0] + ((GY + 0.5) * r) * u[1]; cl = ((GX + 0.5) * r) * v[0] + ((GY + 0.5) * r) * v[1]
    kk = np.full(GX.shape, -1)
    for k, n in enumerate(N): kk[(ct >= n) & (ct < n + M["depth"]) & (cl >= M["l0"]) & (cl <= M["l1"])] = k
    on = np.zeros(GX.shape, bool); off = np.zeros(GX.shape, bool)
    a_ = R["ix"] - ix0; b_ = R["iy"] - iy0; ok = (a_ >= 0) & (a_ < GX.shape[0]) & (b_ >= 0) & (b_ < GX.shape[1])
    for i_, j_, z_, t_, s_ in zip(a_[ok], b_[ok], z[ok], tp[ok], R["src"][ok]):
        k = kk[i_, j_]
        if k < 0: continue
        if abs(z_ - H[k]) <= 0.03: on[i_, j_] = True
        elif t_ and s_ != 5 and H[k] + 0.035 < z_ < H[k] + M["rise"] - 0.035: off[i_, j_] = True   # 앞면 판(src 5)은 일부러 넣은 것이라 빼고 셈
    img = np.full(GX.shape + (3,), 0.93); inside = kk >= 0; img[inside] = 1.0
    img[on] = (0.20, 0.42, 0.75); img[off & ~on] = (0.85, 0.25, 0.2); img[off & on] = (0.55, 0.30, 0.55)
    a.imshow(img.transpose(1, 0, 2), origin="lower", extent=(ix0 * r, (ix1 + 1) * r, iy0 * r, (iy1 + 1) * r), interpolation="nearest")
    for n in N:
        q = np.array([[n, M["l0"]], [n + M["depth"], M["l0"]], [n + M["depth"], M["l1"]], [n, M["l1"]], [n, M["l0"]]]) @ np.array([u, v])
        a.plot(q[:, 0], q[:, 1], "k-", lw=0.4, alpha=0.5)
    a.set_title(f"after: tread-height cells {on[inside].mean() * 100:.0f} %, between-tread tops {off[inside].mean() * 100:.1f} %", fontsize=10)
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")


def main():
    js, out, before, after = sys.argv[1:5]
    M = json.load(open(js))
    fig = plt.figure(figsize=(13, 9.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    for c, (d, title) in enumerate([(before, "before (map as built)"), (after, "after (stair fill + risers)")]):
        R, r, x, y, z = load(d); side(fig.add_subplot(gs[0, c]), M, R, x, y, z, title)
    top(fig.add_subplot(gs[1, :]), M, R, r, x, y, z)
    fig.suptitle(f"{len(M['treads'])} treads, rise {M['rise'] * 100:.1f} cm, run {M['run'] * 100:.1f} cm, depth {M['depth'] * 100:.1f} cm, "
                 f"width {(M['l1'] - M['l0']) * 100:.0f} cm, line-fit residual h {M['fit_residual_cm']['h']:.1f} / nose {M['fit_residual_cm']['nose']:.1f} cm", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=100)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
