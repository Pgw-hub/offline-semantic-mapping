#!/usr/bin/env python3
"""녹화 커버리지 모니터 — mcap 을 읽으며 "어디가 매핑됐고 어디가 비었는지" 를 맵 빌더와 같은 기준으로 보여준다.
녹화 중(--live): 레코더가 쓰고 있는 log_*.mcap 을 뒤따라 읽는다 (방금 쓴 데이터는 페이지 캐시에 있어 디스크 읽기가 거의 없다). 녹화 후: 같은 계산을 빠르게 재생.

레이어 (0.1 m 위에서 본 격자)
  보행용 맵   /cloud_registered 점 중 센서(LiDAR 원점) --near-range(5 m, build_voxel3d max_range) 이내 → 스캔 수 --ok-scans 이상 초록 · 미만 노랑
              그 밖 --far-range(40 m) 이내에서만 보인 칸 회색 (= 더 가까이 가야 보행용 맵에 들어감) · 못 본 칸 검정
  카메라      보행용 LiDAR 복셀(0.1 m)을 depth 프레임에 투영해 label_map_points 와 같은 판정(시야 안 · 그 픽셀 depth 있음 · |z − depth| < 0.25 + 0.05 z)을 통과한 복셀
              = semantic 라벨이 붙을 곳. RGB 짝이 있는 프레임만. 통과 못 한 보행용 칸은 보라. (클래스 경계 4 px 제외는 세그 결과가 없어 생략)
  3D          두 화면: LiDAR 관측 수(초록/노랑) · 카메라 확인 여부(파랑/보라)를 같은 LiDAR 맵 위에. 카메라 자세 · 프러스텀 · RGB 영상
              판정은 0.1 m 복셀, 화면은 --view-vox(0.25 m) 로 묶어 10 m 타일 중 바뀐 것만 0.5 s 마다 보냄 (녹화 중엔 마지막 값만 남겨 메모리 일정)
  상태        토픽 끊김 · SLAM 지연 · 좌표계 전환(위치추정 모드) · 레코더 유실/큐/IO 압력/남은 디스크 → 경고
빌드와 같은 제외 규칙: world 가 아닌 frame_id 의 클라우드, /Odometry/base frame_id 가 바뀐 뒤(저장 맵 위치추정)의 스캔 (export_registered_scans 와 동일).
자세는 녹화 중 SLAM 자세(/Odometry/base) — 빌드는 최종 PGO 자세라 루프 보정이 큰 세션에서는 수 cm~수십 cm 어긋날 수 있다.

사용
  python coverage_monitor.py <mcap 폴더 또는 .mcap>                 # 녹화 끝난 데이터 재생 (rerun 창)
  python coverage_monitor.py <녹화 중인 폴더> --live                 # 녹화 중 따라 읽기 (Ctrl+C 로 끝)
  python coverage_monitor.py <시뮬 녹화 폴더> --live --sim           # 시뮬 (LiDAR 외부 파라미터 slam · RGB/depth 짝 250 ms)
  --view spawn|connect|save|none  rerun 창 띄우기 · 떠 있는 `rerun --serve`(포털 9091 웹뷰어) 로 보내기 · .rrd 저장 · 없음
  --out <폴더>                     coverage_summary.json · coverage_bev.png · coverage_camera.png · coverage_grid.npz (기본 output/_coverage/<폴더 이름>)
녹화 영향 최소화: 시작 시 nice 19 · ionice idle · 스레드 1 개. 처리가 밀리면 depth 를 건너뛰고, 레코더 유실/큐가 늘면 depth 처리를 멈춘다.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # scripts/ (패키지 import)
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): _os.environ.setdefault(_k, "1")   # 녹화 장비 CPU 를 덜 쓰도록 (numpy import 전)
import argparse, bisect, collections, glob, io, json, os, re, shutil, subprocess, sys, time
import numpy as np, cv2, yaml
from preprocess.geometry import raisin_msgs as RM
from preprocess.geometry.extrinsics import lidar_t_bl
from preprocess.geometry.deskew import q2R, slerp_batch
from preprocess.mcap_to_cg import load_K, load_T_base_cam

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C_UNSEEN, C_FAR, C_WEAK, C_OK = (24, 24, 24), (95, 95, 95), (235, 190, 40), (60, 175, 85)
C_LIDAR_ONLY, C_TRAJ, C_POS, C_FOV, C_CAM = (200, 60, 200), (255, 255, 255), (255, 60, 60), (80, 220, 240), (60, 140, 235)


# ---------------- 입력: 재생(인덱스) / 녹화 중(파일 꼬리 따라 읽기) ----------------
def log_files(src):
    if os.path.isfile(src): return [src]
    return sorted(glob.glob(os.path.join(src, "log_*.mcap")), key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))

def replay_source(files, topics):
    from mcap.reader import make_reader
    for p in files:
        with open(p, "rb") as f:
            for sch, ch, m in make_reader(f).iter_messages(topics=topics): yield p, sch, ch.topic, m.log_time, m.data

class _Tail(io.RawIOBase):
    """커지는 파일을 끝까지 읽고, 끝에 닿으면 더 써질 때까지 기다린다. stop() 이 참이면 EOF."""
    def __init__(self, path, stop, poll=0.2): self.f = open(path, "rb"); self.stop = stop; self.poll = poll; self.path = path
    def readable(self): return True
    def readinto(self, b):
        while True:
            d = self.f.read(len(b))
            if d: b[:len(d)] = d; return len(d)
            if self.stop(self): return 0
            time.sleep(self.poll)
    def backlog(self): return os.path.getsize(self.path) - self.f.tell()

def live_source(src, topics, state, idle_s=10.0, done_idle=0.0):
    """레코더가 쓰는 log_*.mcap 을 순서대로 따라간다. 다음 로그가 생기고 지금 파일이 idle_s 동안 안 커지면 다음으로.
    done_idle > 0: 마지막 파일을 끝까지(푸터) 읽은 뒤 done_idle 초 동안 새 로그가 없으면 녹화가 끝난 것으로 보고 마친다."""
    from mcap.stream_reader import StreamReader
    from mcap.records import Schema, Channel, Message, Footer
    want = set(topics); i = 0
    while not state["quit"]:
        files = log_files(src)
        if i >= len(files):
            if done_idle > 0 and state.get("closed_at") and time.time() - state["closed_at"] > done_idle: print(f"새 로그 없이 {done_idle:.0f} s — 녹화 끝으로 보고 마침", flush=True); return
            time.sleep(0.5); continue
        p = files[i]; grow = {"size": -1, "t": time.time()}
        def stop(t, p=p, g=grow):
            if state["quit"]: return True
            s = os.path.getsize(p)
            if s != g["size"]: g["size"], g["t"] = s, time.time(); return False
            return len(log_files(src)) > i + 1 and time.time() - g["t"] > idle_s
        tail = _Tail(p, stop); state["tail"] = tail; sch, ch = {}, {}
        try:
            for rec in StreamReader(io.BufferedReader(tail, 1 << 20), skip_magic=False, emit_chunks=False).records:
                if isinstance(rec, Schema): sch[rec.id] = rec
                elif isinstance(rec, Channel): ch[rec.id] = rec
                elif isinstance(rec, Message):
                    c = ch.get(rec.channel_id)
                    if c is not None and c.topic in want: yield p, sch.get(c.schema_id), c.topic, rec.log_time, rec.data
                elif isinstance(rec, Footer): state["closed_at"] = time.time(); break
        except Exception as e:                                                         # 레코더가 끝맺지 못한 파일 (EndOfFile 등) → 다음 로그
            if not state["quit"]: state["warn"](f"{os.path.basename(p)} 읽기 끝: {type(e).__name__} {str(e)[:60]}", f"{os.path.basename(p)} ended: {type(e).__name__}")
        i += 1


# ---------------- 실시간 자세 (/Odometry/base 보간) ----------------
class LiveOdom:
    def __init__(self): self.ns, self.p, self.q = [], [], []
    def add(self, ns, p, q_xyzw):
        if self.ns and ns <= self.ns[-1]: return
        self.ns.append(ns); self.p.append(p); self.q.append(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]))
    def last(self): return self.ns[-1] if self.ns else -1
    def at(self, t, max_gap=250_000_000):
        """t ns → (R, p) 또는 None (범위 밖 · 오도메트리 공백)"""
        k = bisect.bisect_left(self.ns, t)
        if k == 0 or k >= len(self.ns) or self.ns[k] - self.ns[k - 1] > max_gap: return None
        u = (t - self.ns[k - 1]) / max(self.ns[k] - self.ns[k - 1], 1)
        q = slerp_batch(self.q[k - 1][None], self.q[k][None], np.array([u]))[0]
        return q2R(q), self.p[k - 1] * (1 - u) + self.p[k] * u


# ---------------- 복셀 키 (21 bit × 3) ----------------
_O = 1 << 20; _M = (1 << 21) - 1
def vkey(P, vox): k = np.floor(P / vox).astype(np.int64); return ((k[:, 0] + _O) << 42) | ((k[:, 1] + _O) << 21) | (k[:, 2] + _O)
def vctr(keys, vox): return ((np.stack([(keys >> 42) & _M, (keys >> 21) & _M, keys & _M], -1) - _O + 0.5) * vox).astype(np.float32)


# ---------------- 타일별 복셀 집합 (맵이 커져도 바뀐 타일만 합치고 · 화면에 보냄) ----------------
class Tiles:
    def __init__(self, vox, tile_m):
        self.vox = vox; self.n = max(1, int(round(tile_m / vox))); self.tile_m = self.n * vox
        self.T = {}; self.buf = collections.defaultdict(list); self.dirty = set()             # tid → (keys 정렬, cnt, ctr)
    def tid(self, keys):
        ix = ((keys >> 42) & _M) - _O; iy = ((keys >> 21) & _M) - _O; return (ix // self.n + 50001) * 100003 + (iy // self.n + 50001)
    def add(self, keys):
        """keys: 한 스캔(또는 프레임)의 고유 복셀 키 → 타일별 대기열"""
        if len(keys) == 0: return
        t = self.tid(keys); ut, inv = np.unique(t, return_inverse=True)
        for i, u in enumerate(ut): self.buf[int(u)].append(keys[inv == i])
    def flush(self):
        """대기열을 타일에 합침 → 바뀐 타일 id 목록"""
        changed = list(self.buf)
        for t in changed:
            old = self.T.get(t); k = np.concatenate(([old[0]] if old else []) + self.buf[t]); w = np.concatenate(([old[1]] if old else []) + [np.ones(len(b), np.int32) for b in self.buf[t]])
            uk, inv = np.unique(k, return_inverse=True); self.T[t] = (uk, np.bincount(inv, weights=w).astype(np.int32), vctr(uk, self.vox))
        self.buf.clear(); self.dirty.update(changed); return changed
    def xy(self, t): return np.array([t // 100003 - 50001, t % 100003 - 50001]) * self.tile_m              # 타일 왼쪽 아래 (m)
    def near(self, c, r):
        """중심 c 에서 r 안에 걸치는 타일들의 복셀 중심 (아직 안 합친 대기열 포함 — 중복은 판정에 영향 없음)"""
        hit = lambda t: np.all(np.abs(self.xy(t) + self.tile_m / 2 - c[:2]) < r + self.tile_m / 2)
        out = [v[2] for t, v in self.T.items() if hit(t)] + [vctr(b, self.vox) for t, L in self.buf.items() if hit(t) for b in L]
        return np.concatenate(out) if out else np.zeros((0, 3), np.float32)
    def count(self): return sum(len(v[0]) for v in self.T.values())
    def all(self):
        if not self.T: return np.zeros(0, np.int64), np.zeros(0, np.int32)
        return np.concatenate([v[0] for v in self.T.values()]), np.concatenate([v[1] for v in self.T.values()])


# ---------------- 자라는 2D 격자 ----------------
class Grid:
    def __init__(self, res, layers): self.res = res; self.x0 = self.y0 = None; self.L = {k: None for k in layers}; self.W = self.H = 0
    def _fit(self, ix, iy, pad=100):
        lo_x, hi_x, lo_y, hi_y = int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())
        if self.x0 is None:
            self.x0, self.y0 = lo_x - pad, lo_y - pad; self.W, self.H = hi_x - lo_x + 1 + 2 * pad, hi_y - lo_y + 1 + 2 * pad
            for k in self.L: self.L[k] = np.zeros((self.H, self.W), np.int32)
            return
        if lo_x >= self.x0 and lo_y >= self.y0 and hi_x < self.x0 + self.W and hi_y < self.y0 + self.H: return
        nx0, ny0 = min(self.x0, lo_x - pad), min(self.y0, lo_y - pad); nW = max(self.x0 + self.W, hi_x + 1 + pad) - nx0; nH = max(self.y0 + self.H, hi_y + 1 + pad) - ny0
        for k, a in self.L.items():
            b = np.zeros((nH, nW), np.int32); b[self.y0 - ny0:self.y0 - ny0 + self.H, self.x0 - nx0:self.x0 - nx0 + self.W] = a; self.L[k] = b
        self.x0, self.y0, self.W, self.H = nx0, ny0, nW, nH
    def cells(self, xy):
        """(N,2) m → 고유 칸 (ix, iy)"""
        k = np.floor(xy / self.res).astype(np.int64); u = np.unique((k[:, 0] + _O) << 21 | (k[:, 1] + _O)); return (u >> 21) - _O, (u & _M) - _O   # 2D 키 하나로 (unique(axis=0) 보다 빠름)
    def add(self, layer, xy):
        if len(xy) == 0: return
        ix, iy = self.cells(xy); self._fit(ix, iy); self.L[layer][iy - self.y0, ix - self.x0] += 1
    def area(self, mask): return float(mask.sum()) * self.res * self.res


# ---------------- 본체 ----------------
class Monitor:
    def __init__(self, a, calib, warn):
        self.a = a; self.warn = warn; self.t_bl = lidar_t_bl(calib, a.lidar_extrinsic)
        self.odom = LiveOdom(); self.g = Grid(a.res, ("near", "far", "cam")); self.traj = []
        cam = a.camera; self.cam = None
        try:
            Kd, Wd, Hd = load_K(calib, f"{cam}_depth_aligned_caminfo.yaml"); T_bc = load_T_base_cam(calib, f"{cam}_color.yaml")
            vv, uu = np.mgrid[0:Hd - a.mask_bottom:a.depth_px, 0:Wd:a.depth_px]
            self.cam = dict(K=Kd, W=Wd, H=Hd, T_bc=T_bc, uu=uu.ravel(), vv=vv.ravel(), rx=(uu.ravel() - Kd[0, 2]) / Kd[0, 0], ry=(vv.ravel() - Kd[1, 2]) / Kd[1, 1])
            self.cam["half_fov"] = float(np.arctan2(Wd / 2, Kd[0, 0]))                                # 수평 반화각 (depth_aligned = color 와 같은 시야)
        except Exception as e: warn(f"카메라 {cam} 캘리브레이션 없음 → 카메라 레이어 생략 ({type(e).__name__})", f"no calibration for camera {cam} - camera layer off")
        self.world_frame = None; self.odom_main = None; self.relocalized = False
        self.pend_scan = collections.deque(); self.pend_depth = collections.deque(); self.rgb_ns = collections.deque(maxlen=200)
        self.cam_pose = None; self.last_rgb = None                                                    # 마지막으로 처리한 depth 프레임의 카메라 자세 (c2w, OpenCV 광학 프레임) · 마지막 RGB 메시지
        self.vox = Tiles(a.vox3d, a.tile_m); self.camv = Tiles(a.vox3d, a.tile_m)                   # 보행용 LiDAR 복셀 · 카메라가 depth 일치로 본 복셀 (= semantic 라벨이 붙을 복셀)
        self.n = collections.Counter(); self.depth_i = 0; self.shed_depth_until = 0.0
        self.topic_t = collections.defaultdict(list); self.gaps = collections.Counter(); self.lat = []
        self.rec = {"dropped0": None, "dropped": 0, "queue_max": 0, "psi_io_max": 0.0, "cpu_max": 0.0, "free_mb": None, "mem_avail_mb": None}
        self.t0 = None; self.t_now = None

    # ---- 메시지 하나 ----
    def feed(self, sch, topic, log_ns, d):
        if self.t0 is None: self.t0 = log_ns
        self.n[topic] += 1; self._gap(topic, log_ns)
        if topic == "/Odometry/base":
            ns, p, q = RM.odom(d, sch); fr = RM.odom_frame(d, sch)
            if fr is not None:
                if self.odom_main is None: self.odom_main = fr
                elif fr != self.odom_main and not self.relocalized:
                    self.relocalized = True; self.warn(f"/Odometry/base frame_id '{self.odom_main}' → '{fr}' (저장 맵 위치추정 모드) — 이후 스캔은 빌드에서 제외되므로 커버리지에도 넣지 않음", f"/Odometry/base frame_id '{self.odom_main}' -> '{fr}' (localization on saved map): later scans are excluded from the build, coverage stops here")
            if not self.relocalized: self.odom.add(ns, np.asarray(p, float), np.asarray(q, float))
            self._drain()
        elif topic == "/cloud_registered":
            h, P = RM.pc2(d, sch); lat = (log_ns - h["ns"]) / 1e6
            if self.n["/cloud_registered"] == 1 and abs(lat) > 60_000:                               # 헤더 시각이 다른 시계 (시뮬 시계) → 지연 판정 불가
                self.clock_other = True; self.warn("헤더 시각이 수신 시각과 다른 시계 (시뮬?) → SLAM 지연 검사 끔", "header stamps use another clock (sim?) - SLAM latency check off")
            if not getattr(self, "clock_other", False): self.lat.append(lat)
            if self.lat and self.lat[-1] > self.a.max_latency_ms and self.n["lat_warn"] % 50 == 0: self.warn(f"SLAM 지연 {self.lat[-1]:.0f} ms (> {self.a.max_latency_ms:.0f})", f"SLAM latency {self.lat[-1]:.0f} ms (> {self.a.max_latency_ms:.0f})")
            if self.lat and self.lat[-1] > self.a.max_latency_ms: self.n["lat_warn"] += 1
            if self.world_frame is None and self.odom.ns: self.world_frame = h["frame"]
            if h["frame"] != self.world_frame: self.n["scan_frame_skip"] += 1; return
            if self.relocalized: self.n["scan_reloc_skip"] += 1; return
            self.pend_scan.append((h["ns"], np.stack([P["x"], P["y"], P["z"]], -1).astype(np.float32))); self._drain()
        elif topic == f"{self.a.camera}/rgb": self.rgb_ns.append(RM.stamp_ns(d, sch)); self.last_rgb = (d, sch)   # 디코드는 화면에 올릴 때만
        elif topic == f"{self.a.camera}/depth" and self.cam is not None:
            self.depth_i += 1
            if self.depth_i % self.a.depth_stride or time.time() < self.shed_depth_until or self.relocalized: return
            ns, dep = RM.depth_image(d, sch); self.pend_depth.append((ns, dep)); self._drain()
        elif topic == "__recorder/diagnostics":
            r = RM.recorder_diagnostics(d); R = self.rec
            if R["dropped0"] is None: R["dropped0"] = r["dropped"]
            nd = r["dropped"] - R["dropped0"]
            if nd > R["dropped"]: self.warn(f"레코더 유실 +{nd - R['dropped']} (누적 {nd})", f"recorder dropped +{nd - R['dropped']} (total {nd})"); self._shed()
            R["dropped"] = nd; R["queue_max"] = max(R["queue_max"], r["queue_depth"]); R["free_mb"] = r["free_mb"]
            if r["queue_depth"] > 0.25 * max(r["queue_cap"], 1): self.warn(f"레코더 쓰기 큐 {r['queue_depth']}/{r['queue_cap']}", f"recorder write queue {r['queue_depth']}/{r['queue_cap']}"); self._shed()
            if r["free_mb"] < self.a.min_free_gb * 1024 and self.n["free_warn"] == 0: self.n["free_warn"] += 1; self.warn(f"남은 디스크 {r['free_mb']/1024:.1f} GB", f"disk free {r['free_mb']/1024:.1f} GB")
        elif topic == "__recorder/system":
            s = RM.recorder_system(d); R = self.rec; R["psi_io_max"] = max(R["psi_io_max"], s["psi_io"]); R["cpu_max"] = max(R["cpu_max"], s["cpu_busy_pct"]); R["mem_avail_mb"] = s["mem_avail_mb"]
            R["last_sys"] = s
            if s["psi_io"] > self.a.max_psi_io and self.n["psi_warn"] % 30 == 0: self.warn(f"IO 대기 압력 {s['psi_io']:.1f} %", f"IO pressure {s['psi_io']:.1f} %"); self._shed()
            if s["psi_io"] > self.a.max_psi_io: self.n["psi_warn"] += 1

    def _shed(self):
        """녹화 중일 때만: 레코더가 힘들어하면 depth 처리를 30 s 멈춘다 (재생 때 레코더 값은 과거 기록이라 무관)"""
        if self.a.live: self.shed_depth_until = time.time() + 30; self.warn("녹화 보호: depth 처리 30 s 멈춤", "protecting the recorder: depth processing paused 30 s")

    def _gap(self, topic, ns):
        L = self.topic_t[topic]; L.append(ns)
        if len(L) > 400: del L[:200]
        if len(L) >= 20:
            dt = np.diff(L[-20:]); med = float(np.median(dt)); g = L[-1] - L[-2]
            if g > max(5 * med, self.a.min_gap_s * 1e9) and topic not in ("__recorder/system",):
                self.gaps[topic] += 1
                if self.gaps[topic] <= 20: self.warn(f"{topic} 끊김 {g/1e9:.2f} s (평소 {med/1e6:.0f} ms 간격)", f"{topic} gap {g/1e9:.2f} s (usually {med/1e6:.0f} ms)")

    # ---- 자세가 준비된 스캔 / depth 처리 ----
    def _drain(self):
        last = self.odom.last()
        while self.pend_scan and (self.pend_scan[0][0] <= last or len(self.pend_scan) > 50):          # 자세가 따라올 때까지 대기 (너무 쌓이면 버림)
            ts, P = self.pend_scan.popleft(); T = self.odom.at(ts)
            if T is None: self.n["scan_nopose"] += 1; continue
            self._scan(ts, P, T)
        while self.pend_depth and (self.pend_depth[0][0] <= last or len(self.pend_depth) > 30):
            ts, dep = self.pend_depth.popleft(); T = self.odom.at(ts)
            if T is None: self.n["depth_nopose"] += 1; continue
            if not any(abs(r - ts) <= self.a.pair_ms * 1e6 for r in list(self.rgb_ns)[-20:]): self.n["depth_norgb"] += 1; continue
            self._depth(ts, dep, T)

    def _scan(self, ts, P, T):
        R, p = T; o = R @ self.t_bl + p; r = np.linalg.norm(P - o.astype(np.float32), axis=1)
        near = (r > 0.3) & (r < self.a.near_range); far = (r >= self.a.near_range) & (r < self.a.far_range)
        self.g.add("near", P[near, :2]); self.g.add("far", P[far, :2]); self.n["scans"] += 1
        self.traj.append((ts, o)); self.vox.add(np.unique(vkey(P[near], self.a.vox3d)))

    def _depth(self, ts, dep, T):
        c = self.cam; R, p = T; Twc = np.eye(4); Twc[:3, :3] = R; Twc[:3, 3] = p; Twc = Twc @ c["T_bc"]; self.cam_pose = Twc
        if dep.shape != (c["H"], c["W"]):
            if self.n["depth_shape_warn"] == 0: self.warn(f"depth 크기 {dep.shape} ≠ caminfo {(c['H'], c['W'])} → 카메라 레이어 생략", f"depth size {dep.shape} != caminfo {(c['H'], c['W'])} - camera layer off")
            self.n["depth_shape_warn"] += 1; return
        # label_map_points 와 같은 판정: LiDAR 복셀을 카메라에 투영 → 시야 안 · 그 픽셀 depth 있음 · |z − depth| < 0.25 + 0.05 z 면 라벨이 붙는다
        P = self.vox.near(Twc[:3, 3], self.a.cam_range); self.n["depth_frames"] += 1                      # 카메라 주변 타일만 (라벨은 실측상 10 m 밖에선 안 붙음)
        if len(P) == 0: return
        P = P[np.sum((P - Twc[:3, 3].astype(np.float32)) ** 2, 1) < self.a.cam_range ** 2]; Tcw = np.linalg.inv(Twc)
        pc = P @ Tcw[:3, :3].T.astype(np.float32) + Tcw[:3, 3].astype(np.float32); zl = pc[:, 2]; K = c["K"]; zs = np.where(zl > 0, zl, 1)
        u = np.round(K[0, 0] * pc[:, 0] / zs + K[0, 2]).astype(np.int64); v = np.round(K[1, 1] * pc[:, 1] / zs + K[1, 2]).astype(np.int64)
        inf = (zl > 0.3) & (u >= 0) & (u < c["W"]) & (v >= 0) & (v < c["H"] - self.a.mask_bottom)
        dc = np.zeros(len(P), np.float32); dc[inf] = dep[v[inf], u[inf]]
        vis = inf & (dc > 0) & (np.abs(zl - dc) < 0.25 + 0.05 * zl)
        if vis.any(): self.camv.add(np.unique(vkey(P[vis], self.a.vox3d)))

    def flush(self):
        """대기 중인 복셀을 타일에 합치고, 카메라 2D 칸은 바뀐 카메라 타일만 갱신 (횟수는 늘기만 하므로 최댓값 갱신)"""
        self.vox.flush(); g = self.g
        for t in self.camv.flush():
            k, c, ctr = self.camv.T[t]
            if g.x0 is None: continue
            ix, iy = (np.floor(ctr[:, :2] / g.res).astype(np.int64)).T; g._fit(ix, iy); np.maximum.at(g.L["cam"], (iy - g.y0, ix - g.x0), c)

    def tile_view(self, t, view_vox):
        """타일 하나를 화면용 굵은 복셀로: (중심, LiDAR 관측 수 최대, 카메라 확인 비율)"""
        k, c, ctr = self.vox.T[t]; ck = self.camv.T.get(t); seen = np.isin(k, ck[0], assume_unique=True) if ck else np.zeros(len(k), bool)
        uq, inv = np.unique(vkey(ctr, view_vox), return_inverse=True)
        cmax = np.zeros(len(uq), np.int32); np.maximum.at(cmax, inv, c); frac = np.bincount(inv, weights=seen) / np.bincount(inv)
        return vctr(uq, view_vox), cmax, frac

    # ---- 화면 ----
    def bev(self):
        g = self.g
        if g.x0 is None: return None, None
        near, far, cam = g.L["near"], g.L["far"], g.L["cam"]
        img = np.empty((g.H, g.W, 3), np.uint8); img[:] = C_UNSEEN
        img[(far > 0) & (near == 0)] = C_FAR; img[(near > 0) & (near < self.a.ok_scans)] = C_WEAK; img[near >= self.a.ok_scans] = C_OK
        seen = (near > 0) | (far > 0); camd = cam > 0
        ci = np.empty_like(img); ci[:] = C_UNSEEN; ci[seen] = (70, 70, 70)
        lv = np.clip(cam, 0, 10) / 10.0; ci[cam > 0] = np.stack([40 + 30 * lv, 110 + 70 * lv, 200 + 55 * lv], -1)[cam > 0].astype(np.uint8)
        ci[(near > 0) & ~camd] = C_LIDAR_ONLY
        for im in (img, ci): self._draw_traj(im); self._draw_fov(im)
        return img[::-1], ci[::-1]                                                       # 북쪽(+y) 위

    def _draw_fov(self, im):
        """현재 카메라 수평 시야를 위에서 본 부채꼴로 (길이 --fov-range)"""
        if self.cam_pose is None or self.g.x0 is None: return
        g = self.g; T = self.cam_pose; c0 = T[:3, 3]; f, r = T[:3, 2], T[:3, 0]; hf = self.cam["half_fov"]; L = self.a.fov_range
        ang = np.linspace(-hf, hf, 16); rays = np.cos(ang)[:, None] * f + np.sin(ang)[:, None] * r
        xy = np.vstack([c0[:2], c0[:2] + L * rays[:, :2] / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-6)])
        px = np.stack([xy[:, 0] / g.res - g.x0, xy[:, 1] / g.res - g.y0], -1).astype(np.int32).reshape(-1, 1, 2)
        ov = im.copy(); cv2.fillPoly(ov, [px], C_FOV); cv2.addWeighted(ov, 0.30, im, 0.70, 0, dst=im); cv2.polylines(im, [px], True, C_FOV, 1, cv2.LINE_AA)

    def _draw_traj(self, im):
        if len(self.traj) < 2: return
        g = self.g; xy = np.array([o[:2] for _, o in self.traj[:: max(1, len(self.traj) // 4000)]] + [self.traj[-1][1][:2]])
        px = np.stack([(xy[:, 0] / g.res - g.x0), (xy[:, 1] / g.res - g.y0)], -1).astype(np.int32)
        cv2.polylines(im, [px.reshape(-1, 1, 2)], False, C_TRAJ, 1, cv2.LINE_AA); cv2.circle(im, tuple(int(v) for v in px[-1]), 4, C_POS, -1)

    def summary(self):
        g = self.g; s = {"scans": self.n["scans"], "depth_frames": self.n["depth_frames"], "duration_s": round((self.traj[-1][0] - self.traj[0][0]) / 1e9, 1) if self.traj else 0,
                         "path_m": round(float(np.sum(np.linalg.norm(np.diff(np.array([o for _, o in self.traj]), axis=0), axis=1))), 1) if len(self.traj) > 1 else 0,
                         "params": {k: getattr(self.a, k) for k in ("near_range", "far_range", "res", "ok_scans", "depth_stride", "depth_px", "mask_bottom", "camera")}}
        if g.x0 is not None:
            near, far, cam = g.L["near"], g.L["far"], g.L["cam"]; camd = cam > 0
            s["area_m2"] = {"near_ok": g.area(near >= self.a.ok_scans), "near_weak": g.area((near > 0) & (near < self.a.ok_scans)), "far_only": g.area((far > 0) & (near == 0)),
                            "camera": g.area(cam > 0), "near_without_camera": g.area((near > 0) & ~camd)}
            s["area_m2"] = {k: round(v, 1) for k, v in s["area_m2"].items()}
        s["vox3d_near"] = int(self.vox.count()); s["tiles"] = len(self.vox.T); s["skipped"] = {k: self.n[k] for k in ("scan_frame_skip", "scan_reloc_skip", "scan_nopose", "depth_nopose", "depth_norgb") if self.n[k]}
        s["relocalized"] = self.relocalized; s["topic_gaps"] = dict(self.gaps); s["slam_latency_ms"] = {"median": round(float(np.median(self.lat)), 1), "max": round(float(np.max(self.lat)), 1)} if self.lat else None
        s["recorder"] = {k: v for k, v in self.rec.items() if k not in ("dropped0", "last_sys")}; return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="mcap 폴더 (log_*.mcap) 또는 .mcap 파일"); ap.add_argument("--live", action="store_true", help="녹화 중인 폴더를 따라 읽기")
    ap.add_argument("--calib", help="기본: <폴더>/config_snapshot/raisin_root/calibration 또는 raisin_config/calibration")
    ap.add_argument("--near-range", type=float, default=5.0, help="보행용 맵에 쓰는 LiDAR 거리 (build_voxel3d max_range, 사용자 결정 5 m)")
    ap.add_argument("--far-range", type=float, default=40.0); ap.add_argument("--res", type=float, default=0.1); ap.add_argument("--ok-scans", type=int, default=3)
    ap.add_argument("--vox3d", type=float, default=0.1, help="판정 복셀 크기 (검증한 값 0.1 m)"); ap.add_argument("--view-vox", type=float, default=0.25, help="3D 화면에 보내는 굵은 복셀 (m)")
    ap.add_argument("--tile-m", type=float, default=10.0, help="갱신 단위 타일 (m)"); ap.add_argument("--cam-range", type=float, default=10.0, help="카메라 판정 거리 (빌드 실측: 라벨은 10 m 밖에선 0 %%)"); ap.add_argument("--lidar-extrinsic", default=os.environ.get("LIDAR_EXTRINSIC", "calib"))
    ap.add_argument("--camera", default="d455_front"); ap.add_argument("--depth-stride", type=int, default=1, help="depth 프레임 N 장에 1 장 처리 (1: 빌드와 일치율 최고, CPU 코어 하나의 ~8 %%)")
    ap.add_argument("--depth-px", type=int, default=4, help="depth 픽셀 간격"); ap.add_argument("--mask-bottom", type=int, default=0, help="화면 아래 로봇 몸체 px (세션 mask_bottom_px; 핸드헬드 0, 로봇 60)")
    ap.add_argument("--pair-ms", type=float, default=50.0, help="depth 와 RGB 최대 시차 (RGB 없는 depth 는 semantic 에 안 쓰임)")
    ap.add_argument("--max-latency-ms", type=float, default=300.0); ap.add_argument("--min-gap-s", type=float, default=0.3); ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--max-psi-io", type=float, default=20.0, help="레코더 IO 대기 압력(%%) 이 넘으면 경고 · 녹화 중엔 depth 처리 멈춤 (시뮬 PC 는 평소 5–7 %%)")
    ap.add_argument("--view", default="spawn", choices=["spawn", "connect", "save", "none"]); ap.add_argument("--connect", default="127.0.0.1:9876", help="--view connect 주소 (rerun --serve 의 SDK 포트)")
    ap.add_argument("--rrd", help="--view save 경로 (기본 <out>/coverage.rrd)"); ap.add_argument("--memory-limit", default="2GB", help="rerun 뷰어 메모리 상한 (넘으면 오래된 것부터 버림)")
    ap.add_argument("--update-s", type=float, default=0.5, help="3D 커버리지 · 경로 갱신 간격 (녹화 시간 기준 s)"); ap.add_argument("--bev-s", type=float, default=2.0, help="위에서 본 2D 지도 갱신 간격 (s)"); ap.add_argument("--out")
    ap.add_argument("--bev-max-px", type=int, default=1600, help="2D 지도 화면 최대 픽셀 (넘으면 줄여 보냄)")
    ap.add_argument("--fov-range", type=float, default=5.0, help="위에서 본 지도에 그리는 카메라 시야 부채꼴 길이 (m)")
    ap.add_argument("--cam-hz", type=float, default=4.0, help="3D 뷰 카메라 영상 갱신 (녹화 시간 기준 Hz)"); ap.add_argument("--cam-img-w", type=int, default=320, help="3D 뷰 카메라 영상 폭 (px, 축소)")
    ap.add_argument("--sim", action="store_true", help="시뮬 녹화: --lidar-extrinsic slam · --pair-ms 250 (RGB/depth 비동기). 시뮬 시계는 자동 감지")
    ap.add_argument("--no-nice", action="store_true", help="nice/ionice 를 걸지 않음 (재생 전용 PC)")
    ap.add_argument("--done-idle", type=float, default=0.0, help="--live: 마지막 로그를 다 읽고 이 초만큼 새 로그가 없으면 끝냄 (0: Ctrl+C 까지 계속)")
    a = ap.parse_args()
    if a.sim: a.lidar_extrinsic = "slam"; a.pair_ms = max(a.pair_ms, 250.0)
    if not a.no_nice:
        try: os.nice(19)
        except OSError: pass
        if shutil.which("ionice"): subprocess.run(["ionice", "-c3", "-p", str(os.getpid())], stderr=subprocess.DEVNULL)
    cv2.setNumThreads(1)
    import signal; signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # kill/systemd stop 도 Ctrl+C 처럼 결과 저장 후 종료
    src = os.path.abspath(a.src); base = src if os.path.isdir(src) else os.path.dirname(src)
    while a.live and not log_files(src): print("녹화 파일 기다리는 중:", src); time.sleep(2)
    calib = a.calib or next((c for c in (os.path.join(base, "config_snapshot", "raisin_root", "calibration"), os.path.join(base, "raisin_config", "calibration")) if os.path.isdir(c)), None)
    if not calib: sys.exit(f"캘리브레이션 폴더 없음 (--calib 로 지정): {base}")
    out = a.out or os.path.join(REPO, "output", "_coverage", os.path.basename(base.rstrip("/"))); os.makedirs(out, exist_ok=True)

    import rerun as rr, rerun.blueprint as rrb
    v3 = lambda layer, name: rrb.Spatial3DView(origin="world", contents=[f"+ world/{layer}/**", "+ world/camera/**", "+ world/traj"], name=name)
    bp = rrb.Blueprint(rrb.Vertical(rrb.Horizontal(v3("lidar", f"LiDAR coverage (green >= {a.ok_scans} scans within {a.near_range:g} m, yellow fewer)"), v3("camera_coverage", "Camera coverage on LiDAR map (blue seen, magenta not seen)")),
                                    rrb.Horizontal(rrb.Spatial2DView(origin="bev/locomotion", name="Top view - LiDAR (gray = far only)"), rrb.Spatial2DView(origin="bev/camera", name="Top view - camera"),
                                                   rrb.TimeSeriesView(origin="health", name="Status"), rrb.TextLogView(origin="log", name="Warnings")), row_shares=[3, 1.2]),
                       collapse_panels=True)
    if a.view != "none": rr.init("coverage_monitor", default_blueprint=bp)
    if a.view == "spawn": rr.spawn(memory_limit=a.memory_limit)
    elif a.view == "connect": rr.connect(a.connect); rr.send_blueprint(bp, make_active=True)   # 떠 있는 뷰어가 예전 배치를 기억해도 이 배치로
    elif a.view == "save": rr.save(a.rrd or os.path.join(out, "coverage.rrd"))
    log_on = a.view != "none"
    warns = []; ref = []
    def warn(msg, en=None):
        """터미널 · 요약은 한국어, rerun 뷰어는 영어 (뷰어 글꼴에 한글이 없음)"""
        m = ref[0] if ref else None; t = (m.t_now - m.t0) / 1e9 if m is not None and m.t0 and m.t_now else 0.0
        line = f"[{t:7.1f}s] {msg}"; warns.append(line); print("⚠", line, flush=True)
        if log_on: rr.log("log", rr.TextLog(en or msg, level="WARN"))
    M = Monitor(a, calib, warn); M.t_now = None; ref.append(M)
    topics = ["/Odometry/base", "/cloud_registered", f"{a.camera}/rgb", f"{a.camera}/depth", "/livox/imu", "/livox/lidar", "__recorder/diagnostics", "__recorder/system"]
    state = {"quit": False, "warn": warn, "tail": None}
    print(f"{'녹화 중 따라 읽기' if a.live else '재생'}: {src} · calib {calib} · 보행 {a.near_range} m · 출력 {out}", flush=True)
    files = log_files(src); source = live_source(src, topics, state, done_idle=a.done_idle) if a.live else replay_source(files, topics)
    next_ui = None; t_wall = time.time(); cur_file = None
    st = a.live                                                                    # 녹화 중: 타일 · 경로 · 카메라는 마지막 값만 남김(static) → 뷰어 메모리가 녹화 길이에 비례해 늘지 않음
    def ui(final=False, bev=False):
        M.flush()
        if not log_on: return
        rr.set_time_seconds("rec_time", (M.t_now - M.t0) / 1e9)
        for t in sorted(M.vox.dirty | M.camv.dirty):                                     # 바뀐 타일만 다시 보냄
            if t not in M.vox.T: continue
            P, cmax, frac = M.tile_view(t, a.view_vox); r = a.view_vox * 0.5
            rr.log(f"world/lidar/t{t}", rr.Points3D(P, colors=np.where((cmax >= a.ok_scans)[:, None], np.array(C_OK, np.uint8), np.array(C_WEAK, np.uint8)), radii=r), static=st)
            rr.log(f"world/camera_coverage/t{t}", rr.Points3D(P, colors=np.where((frac >= 0.5)[:, None], np.array(C_CAM, np.uint8), np.array(C_LIDAR_ONLY, np.uint8)), radii=r), static=st)
        M.vox.dirty.clear(); M.camv.dirty.clear()
        if M.traj: rr.log("world/traj", rr.LineStrips3D([np.array([o for _, o in M.traj[:: max(1, len(M.traj) // 4000)]] + [M.traj[-1][1]], np.float32)], colors=[C_TRAJ]), static=st)
        if bev or final:
            img, ci = M.bev()
            if img is not None:
                for name, im in (("locomotion", img), ("camera", ci)):
                    f = max(1, int(np.ceil(max(im.shape[:2]) / a.bev_max_px)))
                    if f > 1: im = cv2.resize(im, (im.shape[1] // f, im.shape[0] // f), interpolation=cv2.INTER_NEAREST)
                    ok, jpg = cv2.imencode(".jpg", im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    rr.log(f"bev/{name}", rr.ImageEncoded(contents=jpg.tobytes(), format=rr.ImageFormat.JPEG), static=st)
            s = M.summary(); ar = s.get("area_m2", {})
            for k, v in ar.items(): rr.log(f"health/area_m2/{k}", rr.Scalar(v))
        if M.lat: rr.log("health/slam_latency_ms", rr.Scalar(M.lat[-1]))
        rr.log("health/recorder/dropped", rr.Scalar(M.rec["dropped"]))
        if "last_sys" in M.rec: rr.log("health/recorder/psi_io", rr.Scalar(M.rec["last_sys"]["psi_io"])); rr.log("health/recorder/cpu_busy_pct", rr.Scalar(M.rec["last_sys"]["cpu_busy_pct"]))
        if state["tail"] is not None: rr.log("health/monitor_backlog_mb", rr.Scalar(state["tail"].backlog() / 1e6))
    def cam_view():
        """3D 뷰의 카메라: 자세 + 프러스텀(Pinhole) + 그 안의 RGB 영상 (축소)"""
        T = M.cam_pose
        try: rgb = RM.rgb_image(M.last_rgb[0], M.last_rgb[1], order="rgb")[1]
        except Exception: return
        c = M.cam; w = a.cam_img_w; h = int(round(w * c["H"] / c["W"])); s_ = w / c["W"]; K = c["K"].copy(); K[:2] *= s_
        rr.set_time_seconds("rec_time", (M.t_now - M.t0) / 1e9)
        rr.log("world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]), static=st)
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=K, resolution=[w, h], camera_xyz=rr.ViewCoordinates.RDF), static=st)
        ok, jpg = cv2.imencode(".jpg", cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        rr.log("world/camera/image", rr.ImageEncoded(contents=jpg.tobytes(), format=rr.ImageFormat.JPEG), static=st)
    try:
        for p, sch, topic, log_ns, d in source:
            if p != cur_file: cur_file = p; print("  ▶", os.path.basename(p), flush=True)
            M.t_now = log_ns; M.feed(sch, topic, log_ns, d)
            if a.live and state["tail"] is not None and state["tail"].backlog() > 300e6: M.shed_depth_until = max(M.shed_depth_until, time.time() + 2)   # 밀리면 depth 건너뜀
            if next_ui is None: next_ui = next_cam = next_bev = next_print = log_ns
            if log_on and log_ns >= next_cam and M.cam_pose is not None and M.last_rgb is not None:
                cam_view(); next_cam = log_ns + int(1e9 / a.cam_hz)
            if log_ns >= next_ui:
                bev = log_ns >= next_bev; ui(bev=bev); next_ui = log_ns + int(a.update_s * 1e9)
                if bev: next_bev = log_ns + int(a.bev_s * 1e9)
                if M.n["scans"] and log_ns >= next_print:
                    next_print = log_ns + int(30e9); s = M.summary(); ar = s.get("area_m2", {})
                    print(f"  t={(log_ns - M.t0)/1e9:6.0f}s 스캔 {s['scans']} depth {s['depth_frames']} · 보행(≥{a.ok_scans}) {ar.get('near_ok', 0)} m² 약함 {ar.get('near_weak', 0)} 멀리서만 {ar.get('far_only', 0)} · 카메라 {ar.get('camera', 0)} · LiDAR만 {ar.get('near_without_camera', 0)}", flush=True)
    except KeyboardInterrupt: print("중단 (Ctrl+C)")
    finally:
        state["quit"] = True
        if M.t_now is not None:
            ui(final=True); s = M.summary(); s.update(src=src, live=a.live, wall_s=round(time.time() - t_wall, 1), warnings=warns[-200:])
            json.dump(s, open(os.path.join(out, "coverage_summary.json"), "w"), ensure_ascii=False, indent=1)
            img, ci = M.bev()
            if img is not None: cv2.imwrite(os.path.join(out, "coverage_bev.png"), img[:, :, ::-1]); cv2.imwrite(os.path.join(out, "coverage_camera.png"), ci[:, :, ::-1])
            if M.g.x0 is not None:                                                          # 격자 원본 (dev/coverage_vs_build.py 검증 · 다른 세션과 합치기용)
                np.savez_compressed(os.path.join(out, "coverage_grid.npz"), res=a.res, x0=M.g.x0, y0=M.g.y0, near=M.g.L["near"], far=M.g.L["far"], cam=M.g.L["cam"],
                                    traj=np.array([o for _, o in M.traj], np.float32), vox_keys=M.vox.all()[0], vox_cnt=M.vox.all()[1], cam_keys=M.camv.all()[0], cam_cnt=M.camv.all()[1], vox3d=a.vox3d)
            print(json.dumps({k: s[k] for k in ("scans", "depth_frames", "duration_s", "path_m", "area_m2", "skipped", "slam_latency_ms", "recorder", "wall_s") if k in s}, ensure_ascii=False))
            print("저장:", out)


if __name__ == "__main__":
    main()
