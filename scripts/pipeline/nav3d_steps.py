"""3D 내비 블록 (N): 메인 맵 위에 미리 계산한 통행성 그래프·토폴로지 nav3d_graph.bin (입력 navigation_map.pcd = 메인 맵)."""
import json, os, shutil

import yaml

from pipeline.env import CODE, REPO, PY


class Nav3dSteps:
    # ───────────────────────────────────────────────────────────────────────────────────────────
    # N (nav3d) — 로봇 raisin_navigation3d_plugin 이 시작할 때마다 계산하던 것을 빌드에서 미리 만든다. 메인 맵(M)의 일부.
    #   ① navigation_map.pcd : 메인 맵 기하 main_map.pcd 그대로 (같은 바이트 — 하드링크/복사). 로봇이 읽는 이름은 파라미터 graph_map_file.
    #   ② nav3d_graph.bin    : 통행성 그래프(다층 표면, 여유 거리, 연결) + 토폴로지(방·계단·포털). raisin_navigation3d_plugin 의
    #      navigation3d_build_graph 실행 파일이 만든다 — 로봇과 같은 C++ 코드라 결과가 같다. 파일에 원본 pcd 해시와 파라미터가
    #      들어 있어, 로봇은 둘이 맞을 때만 이 파일을 읽고 아니면 예전처럼 직접 계산한다. 층 이름(1F·B1)은 로봇 위치 기준이라 로봇이 정한다.
    #   ③ nav3d_graph_check.png : 층별 윗면도 + 측면도 (nav3d/nav3d_graph_view.py), nav3d_nodes.csv : 노드별 x y z 여유 장소 (Potree 레이어)
    #   산출: compiled/nav3d/{navigation_map.pcd, nav3d_graph.bin, nav3d_graph_summary.json, nav3d_nodes.csv, nav3d_graph_check.png}
    #   실행 파일·파라미터 (둘 다 비공개 raisin 쪽이라 저장소에 넣지 않음, tools/bin/ 은 gitignore):
    #     실행 파일 = options.nav3d.bin → $NAV3D_GRAPH_BIN → tools/bin/navigation3d_build_graph → PATH. 없으면 ②③ 만 건너뜀.
    #     파라미터 = options.nav3d.params → $NAV3D_PARAMS → tools/bin/nav3d_params.yaml (로봇의 raisin_navigation3d_plugin params.yaml 사본).
    #                없으면 컴파일 기본값으로 만들고, 로봇 파라미터가 다르면 로봇이 다시 계산한다(로그에 이유).
    #   options.nav3d: false 면 생략. 실패해도 빌드는 계속.
    # ───────────────────────────────────────────────────────────────────────────────────────────
    opt_default_nav3d = {"bin": None, "params": None}

    def nav3d_tool(self, n3, key, env_name, default_name):
        for p in (n3.get(key), os.environ.get(env_name), os.path.join(REPO, "tools", "bin", default_name)):
            if p and os.path.exists(p): return p
        return shutil.which(default_name) if key == "bin" else None

    def step_N(self):
        n3 = self.opt.get("nav3d", True)
        if not n3: return
        n3 = dict(self.opt_default_nav3d, **(n3 if isinstance(n3, dict) else {}))
        d = os.path.join(self.D["compiled"], "nav3d")
        if not self.dry: os.makedirs(d, exist_ok=True)
        try:
            params = self.nav3d_tool(n3, "params", "NAV3D_PARAMS", "nav3d_params.yaml")
            name = "navigation_map.pcd"
            if params:   # 로봇이 읽는 파일 이름도 파라미터에서 (graph_map_file)
                name = (yaml.safe_load(open(params)) or {}).get("graph_map_file") or name
            navpcd = os.path.join(d, name); main = self.main_pcd()
            if not os.path.exists(main) and not self.dry: raise RuntimeError(f"메인 맵이 없음 ({main}) — M 단계 먼저")
            if not self.dry and not (os.path.exists(navpcd) and os.path.samefile(navpcd, main)) and \
                    not (os.path.exists(navpcd) and os.path.getsize(navpcd) == os.path.getsize(main) and os.path.getmtime(navpcd) >= os.path.getmtime(main)):
                if os.path.exists(navpcd): os.remove(navpcd)
                try: os.link(main, navpcd)                                    # 같은 바이트 = 같은 해시 (디스크도 아낌)
                except OSError: shutil.copy2(main, navpcd)
                self.say(f"  nav3d: {name} ← main_map/main_map.pcd")
            tool = self.nav3d_tool(n3, "bin", "NAV3D_GRAPH_BIN", "navigation3d_build_graph")
            if not tool:
                self.say("  nav3d: navigation3d_build_graph 실행 파일 없음 (options.nav3d.bin · $NAV3D_GRAPH_BIN · tools/bin/) → 그래프 생략, 로봇이 시작할 때 계산")
                return
            if not params: self.say("  nav3d: 파라미터 파일 없음 → 컴파일 기본값 (로봇 params 가 다르면 로봇이 다시 계산)")
            gbin = os.path.join(d, "nav3d_graph.bin"); png = os.path.join(d, "nav3d_graph_check.png")
            # 지도·파라미터·실행 파일 중 하나라도 그래프보다 새로우면 다시 (로봇이 옛 파라미터 그래프를 조용히 버리지 않게)
            inputs = [f for f in (navpcd, params, tool) if f and os.path.exists(f)]
            fresh = lambda f: os.path.exists(navpcd) and all(os.path.getmtime(f) >= os.path.getmtime(i) for i in inputs)
            if not self.done("N", gbin, lambda: fresh(gbin)):
                self.run([tool, d, "--graph-map-file", name, *(["--params", params] if params else []), "--nodes-csv", os.path.join(d, "nav3d_nodes.csv"), "--verify"],
                         logname="N_graph.log")
            if not self.done("N", png, lambda: fresh(png) and os.path.getmtime(png) >= os.path.getmtime(gbin)):
                self.run([PY, f"{CODE}/nav3d/nav3d_graph_view.py", d, png, "--name", os.path.basename(self.root)], env={"MPLCONFIGDIR": "/tmp/mpl"}, logname="N_graph.log")
            if not self.dry:
                S = json.load(open(os.path.join(d, "nav3d_graph_summary.json")))
                rooms = sum(p["type"] == "room" for p in S["places"]); stairs = sum(p["type"] == "stair" for p in S["places"])
                self.say(f"  nav3d: {S['points']:,} 점 → 노드 {S['nodes']:,}, 방 {rooms}, 계단 {stairs}, 포털 {len(S['portals'])} "
                         f"(계산 {S['build_ms'] / 1000:.1f} s, 파일 읽기 {S.get('load_ms', 0) / 1000:.2f} s) → nav3d/")
        except Exception as ex:
            self.say(f"  nav3d 실패(빌드는 계속, 로그 logs/N_*.log): {ex}")
