#!/usr/bin/env python3
"""CPU 로봇 semantic navigation 유스케이스 데모.

서버가 만든 맵(dense semantic + scene graph)을 Map Compiler 가 로봇용으로 가공했을 때,
CPU 만으로 기존 geometric-only 자율주행의 세 가지 실제 문제가 해결됨을 검증한다.

  A. 잔디 위 점검 지점 이동  — 기하만: 잔디=거친표면=장애물 → 도달 불가 / semantic: 식생 통과 허용
  B. 산책로 준수 장거리 주행 — 순진한 기하: 평평하면 직진(잔디 가로지름) / semantic: 경로 준수 정책
  C. 텍스트 goal            — 기존: 좌표 waypoint 만 / semantic: "벤치로 가" → CLIP(CPU) → goal 자동

출력: /data2/cg/compiled/usecase/ (PNG 렌더 + results.json)
전부 CPU 연산 (torch CPU 강제). 좌표 규약: grid row=x, col=y, world = origin + idx*res.
"""
import heapq, json, os, time

import cv2
import numpy as np

OUT = "/data2/cg/compiled/usecase"
os.makedirs(OUT, exist_ok=True)
RES = 0.10   # 데모 해상도 (0.05 원본을 2x 다운샘플)

# ---------------- 1. 맵 로드 ----------------
d = np.load("/data2/20260827/_peek/global_session_full.npz")
res0 = float(d["gres"]); gx0 = float(d["gy0"]) * 0 + float(d["gx0"]); gy0 = float(d["gy0"])
cnt = d["el_cnt"]

def ds(a, f=2, how="mean"):
    h, w = a.shape
    a = a[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f)
    return a.mean(axis=(1, 3)) if how == "mean" else a.max(axis=(1, 3))

cnt2 = ds(cnt)
valid = cnt2 > 0
el = ds(np.where(cnt > 0, d["elevation"] / np.maximum(cnt, 1), np.nan))
probs = {k: ds(d[f"sem_{k}"] / np.maximum(cnt, 1)) for k in ["grass", "path", "tree"]}
S = np.stack([probs["grass"], probs["path"], probs["tree"]])
lab = np.full(valid.shape, -1, np.int8)
lab[valid] = S[:, valid].argmax(0)
N1, N2 = lab.shape
print(f"맵 {N1}x{N2} @ {RES}m, 유효 {valid.sum():,}셀")

# 기하 지표: 국소 거칠기(3x3 표준편차)와 경사
el_f = np.where(np.isfinite(el), el, 0)
k = np.ones((3, 3), np.float32) / 9
mean = cv2.filter2D(el_f, -1, k)
sq = cv2.filter2D(el_f * el_f, -1, k)
rough = np.sqrt(np.maximum(sq - mean * mean, 0))
# 맵 구멍(NaN)이 gradient 로 전파되지 않도록 정규화 컨볼루션으로 채운 뒤 경사 계산
fin = np.isfinite(el).astype(np.float32)
el_z = np.where(np.isfinite(el), el, 0).astype(np.float32)
K5 = np.ones((5, 5), np.float32)
el_fill = cv2.filter2D(el_z, -1, K5) / np.maximum(cv2.filter2D(fin, -1, K5), 1e-6)
el_s = np.where(np.isfinite(el), el, el_fill)
gyx = np.gradient(el_s, RES)
slope = np.hypot(gyx[0], gyx[1])
rough[~valid] = np.inf
slope_v = np.where(np.isfinite(slope), slope, np.inf)
slope_v[~valid] = np.inf

LETHAL = 255

# 주의: 오프라인 융합 elevation 은 이미 평활화되어 잔디의 라이다 거칠기가 지워져 있다
# (실측: path 0.4cm vs grass 0.5cm). 따라서 기존 로봇의 동작은 물리 재현이 아니라
# 사용자가 보고한 실패 모드 그대로를 costmap 정책으로 모델링한다:
#   보수적 기하 = 식생(잔디·수풀)이 라이다에 장애물로 잡힘 → 잔디 차단
#   순진한 기하 = 평평하면 어디든 주행 → 잔디·경로 구분 없이 직진
GEO_CONS = np.full(lab.shape, LETHAL, np.float32)
GEO_CONS[valid & (lab == 1) & (slope_v < 0.35)] = 10           # 경로만 통행
GEO_NAIVE = np.full(lab.shape, LETHAL, np.float32)
GEO_NAIVE[valid & (lab != 2) & (slope_v < 0.35)] = 10          # 나무 빼고 평지 전부 동일 비용

def sem_costmap(w_grass):
    """semantic 정책 costmap: path 10, grass w_grass, tree/미관측 lethal, 급경사 lethal."""
    c = np.full(lab.shape, LETHAL, np.float32)
    c[valid & (lab == 1)] = 10
    c[valid & (lab == 0)] = w_grass
    c[valid & (slope_v > 0.6)] = LETHAL       # 강둑 급경사는 식생이어도 차단
    return c

SEM_CROSS = sem_costmap(60)     # A: 잔디 통과 허용 모드
SEM_KEEP = sem_costmap(220)     # B: 경로 준수 모드 (통과 가능하되 강한 페널티)

# ---------------- 2. A* (8-이웃, 순수 CPU) ----------------
SQ2 = 2 ** 0.5

def astar(cost, s, g):
    t0 = time.time()
    open_ = [(0.0, s)]
    gsc = {s: 0.0}; came = {}
    nb = [(-1,0,1),(1,0,1),(0,-1,1),(0,1,1),(-1,-1,SQ2),(-1,1,SQ2),(1,-1,SQ2),(1,1,SQ2)]
    while open_:
        _, cur = heapq.heappop(open_)
        if cur == g:
            path = [cur]
            while cur in came: cur = came[cur]; path.append(cur)
            path.reverse()
            L = sum(SQ2 if (abs(a[0]-b[0])+abs(a[1]-b[1])==2) else 1 for a,b in zip(path,path[1:])) * RES
            return path, L, time.time()-t0
        cx = gsc[cur]
        for dx, dy, w in nb:
            nx, ny = cur[0]+dx, cur[1]+dy
            if not (0 <= nx < N1 and 0 <= ny < N2): continue
            cc = cost[nx, ny]
            if cc >= LETHAL: continue
            ng = cx + w * (cc + cost[cur]) / 2 * RES
            if ng < gsc.get((nx, ny), 1e18):
                gsc[(nx, ny)] = ng; came[(nx, ny)] = cur
                h = np.hypot(nx-g[0], ny-g[1]) * 10 * RES   # admissible (최소비용 10)
                heapq.heappush(open_, (ng + h, (nx, ny)))
    return None, None, time.time()-t0

def w2c(x, y): return (int(round((x - gx0) / RES)), int(round((y - gy0) / RES)))
def c2w(c): return (gx0 + c[0] * RES, gy0 + c[1] * RES)

def nearest(mask, x, y, r=120):
    ci, cj = w2c(x, y)
    ii, jj = np.where(mask[max(ci-r,0):ci+r, max(cj-r,0):cj+r])
    if len(ii) == 0: raise RuntimeError(f"근처에 없음 ({x},{y})")
    ii += max(ci-r,0); jj += max(cj-r,0)
    k = np.argmin((ii-ci)**2 + (jj-cj)**2)
    return (int(ii[k]), int(jj[k]))

# ---------------- 3. 렌더 ----------------
BG = {  -1:(45,45,45), 0:(95,150,95), 1:(110,140,170), 2:(60,95,60) }  # BGR: grass 초록, path 황갈, tree 짙초록
def render(cost, paths, marks, fname, box):
    (x0,x1,y0,y1) = box
    i0,j0 = w2c(x0,y0); i1,j1 = w2c(x1,y1)
    img = np.zeros((i1-i0, j1-j0, 3), np.uint8)
    for k,v in BG.items(): img[lab[i0:i1, j0:j1]==k] = v
    img[(cost[i0:i1, j0:j1]>=LETHAL) & valid[i0:i1,j0:j1]] //= 2   # 차단영역 어둡게
    for pth, col, th in paths:
        if pth is None: continue
        pts = np.array([[p[1]-j0, p[0]-i0] for p in pth])
        cv2.polylines(img, [pts], False, col, th)
    for (x,y), col, _ in marks:
        ci,cj = w2c(x,y)
        cv2.circle(img, (cj-j0, ci-i0), 7, col, -1)
    img = img.transpose(1,0,2)[::-1].copy()   # 가로=x, 세로=y(위+)
    H2 = img.shape[0]
    for (x,y), col, lbl in marks:              # 텍스트는 변환 후 좌표로 (영문만)
        ci,cj = w2c(x,y)
        px, py = ci - i0, H2 - 1 - (cj - j0)
        cv2.putText(img, lbl, (px+10, py+5), 0, 0.6, (0,0,0), 3)
        cv2.putText(img, lbl, (px+10, py+5), 0, 0.6, (255,255,255), 1)
    img = cv2.resize(img, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT, fname), img)
    return fname

R = {}

# ---------------- 시나리오 A: 잔디 위 점검 지점 ----------------
start_A = nearest(valid & (lab==1) & (GEO_CONS<LETHAL), 15, 32)
# 목표: 경로에서 5m 이상 떨어진 잔디 내부 (강둑 잔디밭 점검 지점)
dist_to_path = cv2.distanceTransform((lab!=1).astype(np.uint8), cv2.DIST_L2, 3) * RES
goal_mask_A = valid & (lab==0) & (dist_to_path>5) & (SEM_CROSS<LETHAL)
goal_A = nearest(goal_mask_A, -5, 55, r=150)
pA_geo, LA_geo, tA_geo = astar(GEO_CONS, start_A, goal_A)
pA_sem, LA_sem, tA_sem = astar(SEM_CROSS, start_A, goal_A)
R["A"] = dict(start=c2w(start_A), goal=c2w(goal_A),
              geo=dict(ok=pA_geo is not None, length=LA_geo, t=round(tA_geo,2)),
              sem=dict(ok=pA_sem is not None, length=None if LA_sem is None else round(LA_sem,1), t=round(tA_sem,2)))
render(SEM_CROSS, [(pA_geo,(60,60,230),2),(pA_sem,(230,180,50),3)],
       [(c2w(start_A),(255,255,255),"START"),(c2w(goal_A),(60,60,230),"GOAL (on grass)")],
       "scenA.png", (-45,35,15,75))
print("A:", R["A"])

# ---------------- 시나리오 B: 산책로 준수 장거리 ----------------
start_B = nearest(valid & (lab==1), 22, 28)
goal_B  = nearest(valid & (lab==1), -38, 58)
pB_nv, LB_nv, tB_nv = astar(GEO_NAIVE, start_B, goal_B)
pB_kp, LB_kp, tB_kp = astar(SEM_KEEP, start_B, goal_B)
def onpath(p): return None if p is None else round(float(np.mean([lab[c]==1 for c in p]))*100,1)
R["B"] = dict(start=c2w(start_B), goal=c2w(goal_B),
              naive=dict(ok=pB_nv is not None, length=None if LB_nv is None else round(LB_nv,1), on_path_pct=onpath(pB_nv), t=round(tB_nv,2)),
              sem=dict(ok=pB_kp is not None, length=None if LB_kp is None else round(LB_kp,1), on_path_pct=onpath(pB_kp), t=round(tB_kp,2)))
render(SEM_KEEP, [(pB_nv,(60,60,230),2),(pB_kp,(230,180,50),3)],
       [(c2w(start_B),(255,255,255),"START"),(c2w(goal_B),(255,255,255),"GOAL")],
       "scenB.png", (-50,35,15,75))
print("B:", R["B"])

# ---------------- 시나리오 C: 텍스트 goal ("벤치로 가") ----------------
import torch, open_clip
torch.set_num_threads(8)
objs = json.load(open("/data2/cg/compiled/session_objects.json"))
F = np.load("/data2/cg/compiled/session_objects_clip.npz")["clip"].astype(np.float32)
F /= np.linalg.norm(F, axis=1, keepdims=True)
t0 = time.time()
model,_,_ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
model = model.eval()
tok = open_clip.get_tokenizer("ViT-H-14")
t_load = time.time()-t0
robot_xy = c2w(start_B)
t0 = time.time()
with torch.no_grad():
    q = model.encode_text(tok(["a park bench to sit on"])).numpy().ravel()
q /= np.linalg.norm(q)
sim = F @ q
t_clip = time.time()-t0
# 유사도 상위 중 로봇에서 가장 가까운 것
order = np.argsort(-sim)[:8]
cands = [(i, objs[i], float(sim[i]), np.hypot(objs[i]["center_w"][0]-robot_xy[0], objs[i]["center_w"][1]-robot_xy[1])) for i in order]
# 실내(log_000/001) 객체는 야외 시나리오에서 제외하고, 유사도 임계 넘는 것 중 최근접
outdoor = [c for c in cands if not set(c[1]["logs"]) & {"log_000", "log_001"} and c[2] > 0.24]
best = min(outdoor or cands, key=lambda c: c[3])
bx, by, _ = best[1]["center_w"]
# 접근 goal: 로봇 위치에서 "도달 가능한" 통행 셀 중 벤치 최근접 (고립 영역 배제)
from collections import deque
free = SEM_KEEP < LETHAL
reach = np.zeros_like(free)
dq = deque([start_B]); reach[start_B] = True
while dq:
    cx, cy = dq.popleft()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < N1 and 0 <= ny < N2 and free[nx, ny] and not reach[nx, ny]:
                reach[nx, ny] = True; dq.append((nx, ny))
goal_C = nearest(reach, bx, by, r=100)
t0 = time.time()
pC, LC, tC = astar(SEM_KEEP, start_B, goal_C)
R["C"] = dict(robot=robot_xy, query="a park bench to sit on",
              top=[dict(tag=c[1]["tag"], sim=round(c[2],3), dist=round(c[3],1)) for c in cands],
              picked=dict(tag=best[1]["tag"], center=best[1]["center_w"], sim=round(best[2],3)),
              goal=c2w(goal_C), clip_ms=round(t_clip*1000), clip_load_s=round(t_load,1),
              path=dict(ok=pC is not None, length=None if LC is None else round(LC,1), t=round(tC,2)))
render(SEM_KEEP, [(pC,(230,180,50),3)],
       [(robot_xy,(255,255,255),"ROBOT"),((bx,by),(60,230,230),"BENCH"),(c2w(goal_C),(230,180,50),"GOAL")],
       "scenC.png", (min(robot_xy[0],bx)-12, max(robot_xy[0],bx)+12, min(robot_xy[1],by)-12, max(robot_xy[1],by)+12))
print("C:", json.dumps(R["C"], ensure_ascii=False))

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=1)
print("저장:", OUT)
