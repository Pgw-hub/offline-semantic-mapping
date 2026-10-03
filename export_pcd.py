#!/usr/bin/env python3
"""Map Package → .pcd (binary). 필드 x y z rgb(packed float) label(uint32; 클래스 사전 순서, other=N, unknown=N+1)
source: voxels(0.1m 전체) | voxels_occ(정합용 정적 표면: occupancy free·동적 제거, person/car 제외) | occupancy(0.2m occupied 셀, rgb 없음) | voxel3d(보행용 3D prior 셀, z=mean_z)
사용: python export_pcd.py <manifest> <out.pcd> [--source voxels] [--exclude person,car]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_map import MapPackage
import classdict as CD


def write_pcd(path, xyz, rgb=None, label=None, fmt="rgb"):
    """fmt='rgb': x y z rgb(packed float) label(u4) — 시각화·분석용.
       fmt='robot': x y z intensity — lidar_slam::Point(x,y,z,intensity,t) 과 맞는 필드. intensity 에 클래스 id 를 실어 의미를 보존한다
                    (로봇 정합은 xyz 만 쓰므로 무해; 로봇 저장 지도와 동일한 헤더라 map load 가 통과)."""
    n = len(xyz); fields = ["x", "y", "z"]; types = ["F"] * 3; arrs = [xyz[:, 0].astype("<f4"), xyz[:, 1].astype("<f4"), xyz[:, 2].astype("<f4")]
    if fmt == "robot":
        fields.append("intensity"); types.append("F")
        arrs.append((label.astype(np.float32) if label is not None else np.zeros(n, np.float32)))
        rgb = label = None
    if rgb is not None:
        packed = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
        fields.append("rgb"); types.append("F"); arrs.append(packed.astype("<u4").view("<f4"))          # PCL 관례: uint32 를 float 비트로
    if label is not None:
        fields.append("label"); types.append("U"); arrs.append(label.astype("<u4"))
    hdr = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS " + " ".join(fields) + "\nSIZE " + " ".join(["4"] * len(fields)) +
           "\nTYPE " + " ".join(types) + "\nCOUNT " + " ".join(["1"] * len(fields)) + f"\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    rec = np.empty(n, dtype=[(f, "<f4" if t == "F" else "<u4") for f, t in zip(fields, types)])
    for f, a in zip(fields, arrs): rec[f] = a
    with open(path, "wb") as fh: fh.write(hdr.encode("ascii")); fh.write(rec.tobytes())
    return n


def export(manifest, out, source="voxels", exclude=(), min_nsem=0, occ_filter=None, fmt="rgb"):
    """exclude: 클래스 이름 + 'other' + 'unknown' 중 뺄 것. occ_filter: None(소스 기본) / True / False — occupancy free·동적 복셀 제거 여부"""
    mp = MapPackage(manifest); names = mp.classes; lab = mp.labels(); n_cls = len(names)
    exclude = tuple(exclude)
    if source in ("voxels", "voxels_occ"):
        keep = np.ones(len(lab), bool)
        ex = [names.index(c) for c in exclude if c in names] + ([n_cls] if "other" in exclude else []) + ([n_cls + 1] if "unknown" in exclude else [])
        keep &= ~np.isin(lab, ex)
        use_occ = (source == "voxels_occ") if occ_filter is None else bool(occ_filter)
        if use_occ and mp.occ is not None: keep &= (mp.surf_state != 0) & ~mp.surf_removed
        if min_nsem > 0: keep &= mp.n_sem >= min_nsem
        xyz = mp.voxel_centers()[keep]; rgb = mp.rgb[keep]; label = lab[keep]
    elif source == "occupancy":
        O = mp.occ; occm = O["state"] == 1; xyz = ((O["ijk"] + 0.5) * O["res"])[occm]; rgb = None; label = None
    elif source == "voxel3d":                                                    # 보행용 3D 확률 복셀 사전맵 → 셀마다 점 하나 (x, y, mean_z)
        v3p = os.path.join(os.path.dirname(os.path.dirname(manifest)), "compiled", "session_voxel3d.npz")
        if not os.path.exists(v3p): raise FileNotFoundError("session_voxel3d.npz 없음 (F 단계 voxel3d 미생성)")
        v = np.load(v3p); rx = float(v["res_xy"]); I = v["ijk"]; c3 = v["cls"].astype(np.int64)
        label = np.where(c3 == 254, n_cls, np.where(c3 == 255, n_cls + 1, c3))
        ex = [names.index(c) for c in exclude if c in names] + ([n_cls] if "other" in exclude else []) + ([n_cls + 1] if "unknown" in exclude else [])
        keep = ~np.isin(label, ex)
        if min_nsem > 0: keep &= (v["count"].astype(int) >= min_nsem) | (v["source"] == 3)      # min_nsem 을 '셀 최소 점 수' 로 재사용 (평탄 채움 셀은 count 0 이므로 유지)
        xyz = np.stack([(I[:, 0] + 0.5) * rx, (I[:, 1] + 0.5) * rx, v["mean_z"]], -1)[keep]; label = label[keep]
        pal = np.array([CD.hex2rgb(c["color"]) for c in mp.classdict["classes"]] + [CD.hex2rgb(CD.OTHER_COLOR), CD.hex2rgb(CD.UNKNOWN_COLOR)], np.uint8); rgb = pal[label]
    else:
        raise ValueError(source)
    n = write_pcd(out, xyz, rgb, label, fmt)
    legend = {"source": source, "points": int(n), "frame": mp.man.get("frame"), "label_ids": {**{i: c for i, c in enumerate(names)}, n_cls: "other", n_cls + 1: "unknown"},
              "classdict": mp.classdict.get("name"), "excluded": list(exclude), "min_nsem": min_nsem, "occupancy_filter": (source == "voxels_occ") if occ_filter is None else bool(occ_filter),
              "format": fmt, "rgb": "packed uint32 (r<<16|g<<8|b) stored as float bits (PCL convention)" if fmt == "rgb" else "없음 (robot 포맷: intensity = label id)"}
    if source == "voxel3d": legend["note"] = "z = 셀 안 점 평균(mean_z). 세부 격자 0.025×0.025×0.1 m 셀마다 점 1개; 원본 필드(max_z, logodds, count, source)는 voxel3d_tiles zip 참조"
    json.dump(legend, open(os.path.splitext(out)[0] + "_legend.json", "w"), ensure_ascii=False, indent=1)
    return n


def export_voxel3d_zip(manifest, out):
    """로봇 적재용 타일 묶음: session_voxel3d_tiles/(index.json + t_*.bin) + 요약 + 규격 문서 → zip"""
    import zipfile
    root = os.path.dirname(os.path.dirname(manifest)); td = os.path.join(root, "compiled", "session_voxel3d_tiles")
    if not os.path.exists(os.path.join(td, "index.json")): raise FileNotFoundError("session_voxel3d_tiles/index.json 없음")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(os.listdir(td)): z.write(os.path.join(td, f), f"voxel3d_tiles/{f}"); n += 1
        summ = os.path.join(root, "compiled", "session_voxel3d_summary.json")
        if os.path.exists(summ): z.write(summ, "voxel3d_tiles/summary.json")
        spec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "VOXEL3D_PRIOR.md")
        if os.path.exists(spec): z.write(spec, "voxel3d_tiles/SPEC.md")
        z.writestr("voxel3d_tiles/manifest_ref.json", json.dumps({"map_id": json.load(open(manifest)).get("map_id"), "frame": json.load(open(manifest)).get("frame"), "source_manifest": manifest}, ensure_ascii=False, indent=1))
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("out"); ap.add_argument("--source", default="voxels", choices=["voxels", "voxels_occ", "occupancy", "voxel3d"])
    ap.add_argument("--format", default="rgb", choices=["rgb", "robot"], help="robot = x y z intensity(=label id), lidar_slam 이 읽는 필드"); ap.add_argument("--exclude", default="", help="뺄 클래스 (쉼표; other, unknown 포함 가능)"); ap.add_argument("--min-nsem", type=int, default=0); ap.add_argument("--occ-filter", type=int, default=-1, help="1/0 강제, -1 = 소스 기본")
    a = ap.parse_args(); t = time.time(); n = export(a.manifest, a.out, a.source, tuple(x for x in a.exclude.split(",") if x), a.min_nsem, None if a.occ_filter < 0 else bool(a.occ_filter), a.format)
    print(f"{n:,} points → {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB, {time.time()-t:.1f}s)")


def export_raisin_map_zip(manifest, out, map_name, road=90):
    """raisin Load Map 용 묶음: <map_name>/<map_name>.pcd (x y z intensity=라벨 id, lidar_slam 이 읽는 형식) + <map_name>/semantic/ (nav2 KeepoutFilter 마스크) + README.
    받는 쪽: zip 을 <raisin 워크스페이스>/log/map/ 아래에 풀면 GUI Load Map 목록에 <map_name> 이 뜬다. 의미 격자는 이 PCD 와 같은 좌표계."""
    import subprocess, tempfile, shutil, zipfile, re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", map_name): raise ValueError("맵 이름은 영문/숫자/_/- (lidar_slam isValidFileName)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(manifest)))
    tmp = tempfile.mkdtemp(prefix="raisin_map_"); d = os.path.join(tmp, map_name); os.makedirs(d)
    try:
        export(manifest, os.path.join(d, map_name + ".pcd"), "voxels_occ", ("person", "unknown"), 0, None, "robot")
        for f in os.listdir(d):
            if f.endswith("_legend.json"): os.replace(os.path.join(d, f), os.path.join(d, "semantic_pcd_legend.json"))
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "export_nav_semantic.py"), root, d, "--road", str(road), "--map-name", map_name], check=True, stdout=subprocess.DEVNULL)
        open(os.path.join(d, "README_semantic.txt"), "w").write(
            f"{map_name} — offline semantic map export\n"
            f"Put this folder at <raisin workspace>/log/map/{map_name}/ and use raisin_gui Load Map.\n"
            f"{map_name}.pcd: x y z intensity (intensity = semantic label id, see semantic_pcd_legend.json). No _sc.bin: localize with an initial pose (robot start = map origin).\n"
            "semantic/semantic_mask.png+yaml: OccupancyGrid values for nav2 KeepoutFilter (sidewalk 4, crosswalk 8, grass 35, road %d, else -1), same frame as the pcd.\n" % road)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for base, _, files in os.walk(d):
                for f in files: z.write(os.path.join(base, f), os.path.relpath(os.path.join(base, f), tmp))
    finally: shutil.rmtree(tmp, ignore_errors=True)
    return out
