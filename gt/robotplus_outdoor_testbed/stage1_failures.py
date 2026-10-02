#!/usr/bin/env python3
"""stage1_2d.json(per_frame) → 언제·왜 2D 분할이 실패하는가: 타임라인 PNG, 실패 에피소드 표, 조건별 평균 IoU.
사용: python stage1_failures.py <map_root>  → compiled/stage1_timeline.png, stage1_failures.json"""
import json, os, sys
import numpy as np
root = os.path.abspath(sys.argv[1]); d = json.load(open(root + "/compiled/stage1_2d.json")); pf = [f for f in d["per_frame"] if f.get("gt_sidewalk_px", 0) > 200]
fr = np.array([f["frame"] for f in pf]); iou = np.array([f["iou"].get("sidewalk", np.nan) for f in pf], float)
on_sw = np.array([f["robot_on_sidewalk"] for f in pf]); near = np.array([f["near5_share"] for f in pf]); sky = np.array([f["black_sky_share"] for f in pf])
road_ctx = np.array([f["gt_road_px"] > 300 for f in pf]); fl = np.array([f.get("flicker_sidewalk") if f.get("flicker_sidewalk") is not None else np.nan for f in pf], float)
pred_road = np.array([(f["sidewalk_pred"] or {}).get("road", np.nan) for f in pf], float); pred_other = np.array([(f["sidewalk_pred"] or {}).get("other", np.nan) for f in pf], float)
def grp(mask, name): m = mask & np.isfinite(iou); return {"frames": int(m.sum()), "sidewalk_iou_mean": round(float(np.nanmean(iou[m])), 3) if m.any() else None, "pred_road_mean": round(float(np.nanmean(pred_road[m])), 3) if m.any() else None, "pred_other_mean": round(float(np.nanmean(pred_other[m])), 3) if m.any() else None}
cond = {
    "robot_on_sidewalk & road_visible": grp(on_sw & road_ctx, ""), "robot_on_sidewalk & no_road_visible": grp(on_sw & ~road_ctx, ""),
    "robot_on_road & road_visible": grp(~on_sw & road_ctx, ""), "robot_on_road & no_road_visible": grp(~on_sw & ~road_ctx, ""),
    "near5_share>0.5 (발밑 위주)": grp(near > 0.5, ""), "near5_share<=0.2 (원거리 위주)": grp(near <= 0.2, ""),
    "black_sky>0.6": grp(sky > 0.6, ""), "black_sky<=0.3": grp(sky <= 0.3, ""),
}
# 실패 에피소드: 연속 프레임에서 인도 IoU < 0.25
bad = np.isfinite(iou) & (iou < 0.25); eps = []; i = 0
while i < len(bad):
    if bad[i]:
        j = i
        while j + 1 < len(bad) and bad[j + 1]: j += 1
        if j - i + 1 >= 3:
            sl = slice(i, j + 1); eps.append({"frames": [int(fr[i]), int(fr[j])], "n": int(j - i + 1), "sidewalk_iou_mean": round(float(np.nanmean(iou[sl])), 3), "robot_on_sidewalk": round(float(on_sw[sl].mean()), 2),
                                             "road_visible": round(float(road_ctx[sl].mean()), 2), "near5_share": round(float(near[sl].mean()), 2), "black_sky": round(float(sky[sl].mean()), 2),
                                             "pred_road": round(float(np.nanmean(pred_road[sl])), 2), "pred_other": round(float(np.nanmean(pred_other[sl])), 2)})
        i = j + 1
    else: i += 1
for e in eps:   # 원인 분류
    e["cause"] = ("발밑 인도→road (맥락 없음)" if e["pred_road"] >= e["pred_other"] and e["road_visible"] < 0.5 else "road 라벨 (차도 보임)" if e["pred_road"] >= e["pred_other"] else "building/기타 (하늘 번짐·원거리)" if e["black_sky"] > 0.5 or e["near5_share"] < 0.3 else "기타")
out = {"frames_evaluated": len(pf), "sidewalk_iou_mean": round(float(np.nanmean(iou)), 3), "flicker_median": round(float(np.nanmedian(fl)), 3), "flicker_p90": round(float(np.nanpercentile(fl, 90)), 3),
       "flicker_by_pos": {"on_sidewalk": round(float(np.nanmean(fl[on_sw])), 3), "on_road": round(float(np.nanmean(fl[~on_sw])), 3)},
       "conditions": cond, "episodes": sorted(eps, key=lambda e: -e["n"])}
json.dump(out, open(root + "/compiled/stage1_failures.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=1, ensure_ascii=False)); print("episodes:", len(eps))
for e in out["episodes"][:12]: print(" ", e)
# 타임라인
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
ax[0].fill_between(fr, 0, 1, where=on_sw, color="#00e5ff", alpha=0.18, step="mid", label="robot on sidewalk"); ax[0].plot(fr, iou, lw=0.8, color="k", label="sidewalk IoU (2D)"); ax[0].set_ylim(0, 1); ax[0].legend(loc="upper right", fontsize=8); ax[0].set_ylabel("IoU")
ax[1].plot(fr, pred_road, lw=0.8, color="#6a0dad", label="GT sidewalk px → road"); ax[1].plot(fr, pred_other, lw=0.8, color="#607d8b", label="→ other (building…)"); ax[1].set_ylim(0, 1); ax[1].legend(loc="upper right", fontsize=8); ax[1].set_ylabel("share")
ax[2].plot(fr, fl, lw=0.8, color="#d32f2f", label="flicker: label change on shared sidewalk pts"); ax[2].plot(fr, near, lw=0.8, color="#ff9800", label="near(<5m) share"); ax[2].plot(fr, sky, lw=0.8, color="#333", label="black sky share"); ax[2].set_ylim(0, 1); ax[2].legend(loc="upper right", fontsize=8); ax[2].set_xlabel("sem frame"); ax[2].set_ylabel("ratio")
for e in eps: ax[0].axvspan(e["frames"][0], e["frames"][1], color="red", alpha=0.12)
plt.tight_layout(); plt.savefig(root + "/compiled/stage1_timeline.png", dpi=110); print("timeline saved")
