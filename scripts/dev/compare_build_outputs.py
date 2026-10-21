#!/usr/bin/env python3
"""두 빌드 root 의 산출물이 같은지 비교한다 (리팩터링 회귀 검증용).
사용: python dev/compare_build_outputs.py <root_a> <root_b> [--quiet]
규칙
  .npz            키별 np.array_equal (params 같은 JSON 문자열은 파싱 후 비교, 시간 키 제외)
  .json           파싱 후 비교 (built · seconds · build_seconds_total · 경로 문자열의 root 차이는 무시)
  .png .jpg .pcd .bin .las .txt .yaml .ply  바이트 비교
  .gz (pkl.gz)    압축 해제 후 바이트 비교 (gzip 헤더의 시각 무시)
  .rrd .mp4 .log .out  존재만 확인 (기록 시각·ID 가 들어감)
출력: 파일별 결과 + 요약. 다르면 exit 1."""
import gzip, json, os, sys
import numpy as np

A, B = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); QUIET = "--quiet" in sys.argv
TIME_KEYS = {"built", "seconds", "build_seconds_total", "time", "elapsed", "date", "optimized_at"}
EXIST_ONLY = (".rrd", ".mp4", ".log", ".out")
BYTE_EXT = (".png", ".jpg", ".jpeg", ".pcd", ".bin", ".las", ".laz", ".txt", ".yaml", ".yml", ".ply", ".html", ".csv")

def norm(o):
    if isinstance(o, dict): return {k: norm(v) for k, v in o.items() if k not in TIME_KEYS}
    if isinstance(o, list): return [norm(v) for v in o]
    if isinstance(o, str): return o.replace(A, "<ROOT>").replace(B, "<ROOT>")
    return o

def cmp_npz(a, b):
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    if set(x.files) != set(y.files): return f"키 다름 {sorted(set(x.files) ^ set(y.files))}"
    bad = []
    for k in x.files:
        u, v = x[k], y[k]
        if u.dtype.kind in "US" and u.shape == () and v.shape == ():
            try:
                if norm(json.loads(str(u))) != norm(json.loads(str(v))): bad.append(k)
                continue
            except Exception: pass
        if u.dtype == object or v.dtype == object:
            if repr(u.tolist()) != repr(v.tolist()): bad.append(k)
        elif u.shape != v.shape or not np.array_equal(u, v, equal_nan=u.dtype.kind == "f"): bad.append(k)
    return f"배열 다름 {bad}" if bad else None

def cmp_file(rel):
    a, b = os.path.join(A, rel), os.path.join(B, rel)
    if rel.endswith(EXIST_ONLY): return None
    if rel.endswith(".npz"): return cmp_npz(a, b)
    if rel.endswith(".json"):
        try: return None if norm(json.load(open(a))) == norm(json.load(open(b))) else "JSON 내용 다름"
        except Exception as e: return f"JSON 읽기 실패 {e}"
    if rel.endswith(".gz"):
        return None if gzip.open(a).read() == gzip.open(b).read() else "압축 해제 내용 다름"
    return None if open(a, "rb").read() == open(b, "rb").read() else "바이트 다름"

def walk(root):
    s = set()
    for dp, dn, fn in os.walk(root):
        for f in fn: s.add(os.path.relpath(os.path.join(dp, f), root))
    return s

fa, fb = walk(A), walk(B)
only_a, only_b = sorted(fa - fb), sorted(fb - fa)
diff, same, skipped = [], 0, 0
by_dir = {}
for rel in sorted(fa & fb):
    if rel.endswith(EXIST_ONLY): skipped += 1; continue
    r = cmp_file(rel); top = rel.split(os.sep)[0] + ("/" + rel.split(os.sep)[1] if rel.count(os.sep) >= 2 and rel.startswith("dataset") else "")
    by_dir.setdefault(top, [0, 0]); by_dir[top][0 if r is None else 1] += 1
    if r is None: same += 1
    else: diff.append((rel, r))
print(f"A {A}\nB {B}")
print(f"공통 {len(fa & fb)} (비교 {same + len(diff)}, 존재만 확인 {skipped}) · 같음 {same} · 다름 {len(diff)} · A에만 {len(only_a)} · B에만 {len(only_b)}")
print("폴더별 (같음/다름):", ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by_dir.items())))
for rel, r in diff[:200]: print(f"  다름  {rel}: {r}")
for rel in only_a[:50]: print(f"  A에만 {rel}")
for rel in only_b[:50]: print(f"  B에만 {rel}")
sys.exit(1 if diff or only_a or only_b else 0)
