# offline_semantic_mapping

`.mcap` 로봇 로그 → 서버(GPU)에서 **Offline 3D Semantic Map** (semantic 복셀 · 확률적 occupancy · 코스트맵 · 3D scene graph) → `map_manifest.json` 패키지 → **Map Compiler** 가 로봇(nav2/BT) 요청에 맞게 가공.

- 빌드: `cd sessions && python ../scripts/build_map_package.py session_<이름>.yaml` (컨테이너 `offline_3d_semantic_map` 안에서)
- 결과: `output/<이름>/{compiled,voxel,dataset,sem,logs}` — 패키지는 `output/<이름>/compiled/map_manifest.json`
- 로드: `python scripts/package/load_map.py output/<이름>/compiled/map_manifest.json [--rrd out.rrd --coarse 0.3 --cubes]`
- 코드: `scripts/` — `preprocess/` · `semantic/{voxel,scene_graph,occupancy}/` · `locomotion_map/` (보행용 맵) · `package/` · `apps/` · `pipeline/` (빌드 단계) · `common/` · `dev/` (실험·평가). 세션 yaml 은 `sessions/`, 포털은 `tools/`.
- 노트북 빌드 (기하 전용 · semantic 맵): `docker/laptop/README.md`
- 문서: `docs/HANDOFF.md` (전체 이력·결정·함정), 사용 가이드는 노션.
- 외부 설치물(컨테이너 안, 데이터 디스크): ConceptGraphs 패치판 `/data2/cg/concept-graphs`, 모델 캐시 `/data2/cg/hf_cache`, 기존 세션 산출물 `/data2/cg/{data,voxel,compiled}`.
