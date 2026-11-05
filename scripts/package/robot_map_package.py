#!/usr/bin/env python3
"""빌드 결과 → 로봇에 바로 넣는 맵 폴더 (위치추정 지도 + 보행 지도를 한 이름으로 묶음).

  <out>/<name>/
    <name>.pcd            위치추정 지도: geometry_all_scans(없으면 S 단계 스캔)를 --loc-voxel 격자 중심점으로 (x y z intensity)
    <name>.txt            키프레임 궤적 (graph/keyframes_tum.txt, "stamp x y z qx qy qz qw") — lidar_slam 전역 위치추정 후보
    <name>_manifest.txt   map_id(새로 뽑음)·saved_at·voxel_size — lidar_slam 이 다시 줄이지 않게, GUI 가 옛 캐시를 쓰지 않게
    voxel3d_tiles/        보행 지도 (prior map controller 가 log/map/<SLAM 맵 이름>/voxel3d_tiles 를 자동으로 찾음)
    map_info.json         어느 빌드·옵션으로 만들었는지

왜 한 이름이어야 하나: lidar_slam 은 log/map/<name>/<name>.pcd 를 읽고, 위치추정 중 /Odometry/base 의 frame_id 가 <name> 이 된다.
prior map controller 는 그 frame_id 로 타일 폴더를 찾는다 → 폴더 이름 = pcd 이름 = 맵 이름.

사용: python robot_map_package.py <빌드 폴더(output/<세션>)> --name 1103_daejeon_grating [--loc-voxel 0.25] [--tiles auto|<변형>] [--out DIR] [--zip]
로봇에 넣기: tools/deploy_robot_map.sh <out>/<name> raion@<로봇주소>
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))  # scripts/
import argparse, glob, json, os, re, secrets, shutil, sys, time, zipfile
import numpy as np
from package.export_pcd import write_pcd

NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
TILE_PRIORITY = ("stair_lidaronly", "stair", "lidaronly", "")   # auto: 계단 채움(LiDAR 만) → 계단 채움 → LiDAR 만 → 기본


def read_pcd_xyz(path):
    """binary / ascii PCD 의 x y z (binary_compressed 는 open3d 로)."""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            line = f.readline()
            if not line: raise ValueError(f"PCD 헤더가 끝나지 않음: {path}")
            s = line.decode("ascii", "replace").strip()
            if not s or s.startswith("#"): continue
            k, _, v = s.partition(" "); hdr[k.upper()] = v.split()
            if k.upper() == "DATA": break
        data = f.read()
    fields = hdr["FIELDS"]; n = int(hdr["POINTS"][0]); mode = hdr["DATA"][0]
    if mode == "binary_compressed":
        import open3d as o3d
        return np.asarray(o3d.io.read_point_cloud(path).points, np.float64)
    sizes = [int(x) for x in hdr["SIZE"]]; types = hdr["TYPE"]; counts = [int(x) for x in hdr.get("COUNT", ["1"] * len(fields))]
    kind = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4"}
    if mode == "ascii":
        A = np.loadtxt(data.decode().splitlines(), ndmin=2); cols = np.cumsum([0] + counts)
        return np.stack([A[:, cols[fields.index(a)]] for a in "xyz"], 1)
    dt = np.dtype([(f, kind[(t, s)], (c,)) if c > 1 else (f, kind[(t, s)]) for f, t, s, c in zip(fields, types, sizes, counts)])
    rec = np.frombuffer(data, dt, count=n)
    return np.stack([rec[a].astype(np.float64) for a in "xyz"], 1)


def voxel_centroids(P, v):
    """v 격자 칸마다 점들의 평균 하나 (open3d voxel_down_sample 과 같은 규칙). v<=0 이면 그대로."""
    if v <= 0: return P
    k = np.floor(P / v).astype(np.int64); _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True); inv = inv.ravel()
    out = np.zeros((len(cnt), 3))
    for a in range(3): out[:, a] = np.bincount(inv, P[:, a], len(cnt)) / cnt
    return out


def localization_points(root, v):
    """위치추정 지도 원천: compiled/geometry_all_scans.pcd (0.1 m) → 없으면 dataset/log_*/lidar_scans/*.npz (기하 전용 빌드)."""
    g = os.path.join(root, "compiled", "geometry_all_scans.pcd")
    if os.path.exists(g): return voxel_centroids(read_pcd_xyz(g), v), "compiled/geometry_all_scans.pcd"
    files = sorted(glob.glob(os.path.join(root, "dataset", "log_*", "lidar_scans", "*.npz")))
    if not files: raise FileNotFoundError(f"{root}: geometry_all_scans.pcd 도 dataset/log_*/lidar_scans 도 없음 — 위치추정 지도를 만들 점이 없습니다")
    vv = v if v > 0 else 0.1                                                     # 스캔 전체를 그대로 쓰면 수천만 점 → dense 도 0.1 m 로 (geometry_all_scans 와 같은 해상도)
    acc = []; chunk = []
    for i, f in enumerate(files):                                               # 스캔을 묶어 줄여 가며 쌓아 메모리를 일정하게
        chunk.append(np.load(f)["xyz"].astype(np.float64))
        if len(chunk) == 50 or i == len(files) - 1:
            acc.append(voxel_centroids(np.concatenate(chunk), min(vv, 0.05))); chunk = []
    return voxel_centroids(np.concatenate(acc), vv), f"dataset/log_*/lidar_scans ({len(files)} 스캔)"


def tile_variants(root):
    """compiled/session_voxel3d[_<변형>]_tiles/index.json 이 있는 변형 이름들 ('' = 기본)."""
    out = {}
    for idx in glob.glob(os.path.join(root, "compiled", "session_voxel3d*_tiles", "index.json")):
        d = os.path.dirname(idx); b = os.path.basename(d)[len("session_voxel3d"):-len("_tiles")].lstrip("_"); out[b] = d
    return out


def pick_tiles(root, want):
    V = tile_variants(root)
    if not V: raise FileNotFoundError(f"{root}: 보행 지도 타일(compiled/session_voxel3d*_tiles)이 없음 — options.voxel3d 로 빌드했는지 확인")
    if want != "auto":
        key = "" if want in ("default", "base") else want
        if key not in V: raise ValueError(f"타일 변형 '{want}' 없음. 있는 것: {', '.join(k or 'default' for k in sorted(V))}")
        return key, V[key]
    for k in TILE_PRIORITY:
        if k in V: return k, V[k]
    k = sorted(V)[0]; return k, V[k]


def build(root, name, loc_voxel=0.25, tiles="auto", out=None, make_zip=False, log=print):
    root = os.path.abspath(root)
    if not NAME_RE.fullmatch(name): raise ValueError(f"맵 이름 '{name}': 영문·숫자·_·- 만, 64자 이하 (lidar_slam 맵 이름 = 폴더 이름 = pcd 이름)")
    out = os.path.abspath(out or os.path.join(root, "compiled", "export", "robot"))
    tmp = os.path.join(out, f".{name}.tmp"); dst = os.path.join(out, name)
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    try:
        t0 = time.time()
        P, src = localization_points(root, loc_voxel)
        write_pcd(os.path.join(tmp, f"{name}.pcd"), P.astype(np.float32), fmt="robot")
        log(f"위치추정 지도 {name}.pcd: {len(P):,} 점 (격자 {loc_voxel if loc_voxel > 0 else 'dense'} m, 원천 {src})")
        kf = os.path.join(root, "graph", "keyframes_tum.txt")
        if os.path.exists(kf):
            shutil.copy(kf, os.path.join(tmp, f"{name}.txt")); log(f"키프레임 궤적 {name}.txt: {sum(1 for l in open(kf) if l.strip() and not l.startswith('#'))} 개")
        with open(os.path.join(tmp, f"{name}_manifest.txt"), "w") as f:
            f.write(f"map_id={secrets.token_hex(16)}\nsaved_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n" + (f"voxel_size={loc_voxel:g}\n" if loc_voxel > 0 else ""))
        tkey, tdir = pick_tiles(root, tiles)
        shutil.copytree(tdir, os.path.join(tmp, "voxel3d_tiles"))
        if os.path.exists(tdir + "_stair.json"): shutil.copy(tdir + "_stair.json", os.path.join(tmp, "voxel3d_tiles", "stair.json"))
        idx = json.load(open(os.path.join(tdir, "index.json"))); nt = len(glob.glob(os.path.join(tdir, "t_*.bin")))
        log(f"보행 지도 voxel3d_tiles: 변형 '{tkey or 'default'}', 타일 {nt} 개, {idx.get('format')}")
        if idx.get("depth") and "lidaronly" not in tkey:
            log("주의: 이 보행 지도에는 depth 카메라 채움 셀이 섞여 있습니다 (대전 11/03 은 LiDAR 만). LiDAR 만 쓰려면 --tiles lidaronly, "
                "계단 채움까지 LiDAR 만이면 docker/laptop/run.sh 가 만드는 stair_lidaronly")
        man = os.path.join(root, "compiled", "map_manifest.json"); M = json.load(open(man)) if os.path.exists(man) else {}
        info = {"name": name, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "build_dir": root, "build_map_id": M.get("map_id"), "build_time": M.get("built"),
                "frame": M.get("frame"), "source": M.get("source"),
                "localization": {"file": f"{name}.pcd", "points": int(len(P)), "voxel_m": loc_voxel if loc_voxel > 0 else None, "from": src},
                "locomotion": {"dir": "voxel3d_tiles", "variant": tkey or "default", "tiles": nt, "from": os.path.relpath(tdir, root),
                               "available_variants": sorted(k or "default" for k in tile_variants(root))},
                "robot": f"<raisin 워크스페이스>/log/map/{name}/ 에 이 폴더 그대로. GUI Load Map → {name}"}
        json.dump(info, open(os.path.join(tmp, "map_info.json"), "w"), ensure_ascii=False, indent=1)
        shutil.rmtree(dst, ignore_errors=True); os.replace(tmp, dst)
    finally: shutil.rmtree(tmp, ignore_errors=True)
    zp = None
    if make_zip:
        zp = os.path.join(out, f"{name}.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for base, _, files in os.walk(dst):
                for f in sorted(files): z.write(os.path.join(base, f), os.path.relpath(os.path.join(base, f), out))
        log(f"zip {zp} ({os.path.getsize(zp) / 1e6:.1f} MB)")
    log(f"완료 {dst} ({time.time() - t0:.1f} s)")
    return dst, zp


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir", help="빌드 폴더 (output/<세션>)"); ap.add_argument("--name", required=True, help="맵 이름 = 로봇 log/map 아래 폴더 이름 = pcd 이름 (영문·숫자·_·-)")
    ap.add_argument("--loc-voxel", type=float, default=0.25, help="위치추정 지도 격자 (m). 0 = dense (geometry_all_scans 0.1 m 그대로)")
    ap.add_argument("--tiles", default="auto", help="보행 지도 타일 변형: auto(계단 채움 LiDAR 만 → 계단 채움 → LiDAR 만 → 기본) | default | stair | lidaronly | ...")
    ap.add_argument("--out", help="출력 폴더 (기본 <빌드>/compiled/export/robot)"); ap.add_argument("--zip", action="store_true", help="<name>.zip 도 (USB·포털 다운로드용)")
    a = ap.parse_args()
    try: build(a.build_dir, a.name, a.loc_voxel, a.tiles, a.out, a.zip)
    except (ValueError, FileNotFoundError) as e: sys.exit(f"오류: {e}")
