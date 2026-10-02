# 로봇플러스 실외 테스트베드 — 시뮬 환경 GT

출처: 사용자가 Claude Design 으로 만든 3D 모델 `Outdoor Testbed 3D.html`
(claude.ai/design 프로젝트 `Outdoor Testbed 3D Model`, 097ba07d-…), RaiSim 월드 `/raisim_config/robotplus_outdoor_testbed.xml` 의 원본.
좌표: 단위 m · Y-up · 원점 = 교차로 중심 · −Z = 북쪽. 시공 오차 ±0.25 m (도면 기준 근사).

`gt_eval.py` 가 HTML 의 기하(폴리곤·박스·원기둥)를 파이썬으로 옮겨 0.2 m 격자 GT 라벨을 만들고,
의미 라벨 없이 **지면 높이(차도가 인도보다 0.15 m 낮음)** 만으로 디자인 좌표계 ↔ SLAM 맵 좌표계를 정합한 뒤,
빌드 결과 코스트맵과 혼동행렬·IoU 를 낸다.

- `python gt_eval.py build` → `gt_design.npz`, `gt_design.png` (디자인 좌표 GT 미리보기)
- `python gt_eval.py eval <map_root> [<map_root> …]` → 첫 root 로 정합(같은 데이터셋이면 변형 root 들은 같은 프레임), root 별 `compiled/gt_eval.json` + `gt_overlay.png`
