#!/usr/bin/env bash
# Potree 1.8.2 웹 뷰어 정적 파일(build/ · libs/ · LICENSE)을 tools/potree/viewer/ 에 받는다 (git 에 넣지 않은 65 MB).
# 서버의 viewer/ 와 공식 릴리스 Potree_1.8.2.zip 이 파일 1,900개 모두 같음을 확인함 (2026-10-21). 포털이 /tools/potree/viewer/... 로 서빙한다.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)/viewer"
if [ -f "$D/build/potree/potree.js" ] && [ -d "$D/libs" ]; then echo "이미 있음: $D"; exit 0; fi
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
curl -fL -o "$T/p.zip" https://github.com/potree/potree/releases/download/1.8.2/Potree_1.8.2.zip
python3 - "$T/p.zip" "$D" <<'P'
import sys, zipfile, os
z = zipfile.ZipFile(sys.argv[1]); out = sys.argv[2]; pre = "Potree_1.8.2/"
for n in z.namelist():
    rel = n[len(pre):]
    if not (rel.startswith(("build/", "libs/")) or rel == "LICENSE") or n.endswith("/"): continue
    p = os.path.join(out, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    with z.open(n) as fi, open(p, "wb") as fo: fo.write(fi.read())
print("받음:", out)
P
