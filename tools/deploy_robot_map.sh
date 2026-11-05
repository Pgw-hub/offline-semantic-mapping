#!/usr/bin/env bash
# 로봇 맵 패키지(scripts/package/robot_map_package.py 산출: <이름>/ 폴더 또는 <이름>.zip)를 로봇 raisin 워크스페이스 log/map/<이름>/ 에 넣는다.
# 같은 이름이 있으면 log/map_backup/<이름>_<시각> 으로 옮기고 → 폴더를 스트림으로 보내 풀고 → 파일 목록·크기를 노트북 쪽과 비교한다.
# 로봇에 unzip·python 필요 없음 (bash, tar, find 만). 포털 'Robot map package' 창의 '로봇에 넣기'도 이 스크립트를 부른다.
#
# 사용:
#   tools/deploy_robot_map.sh <패키지 폴더|zip> <로봇> [--ws <워크스페이스>] [--name <새 이름>] [--dry-run] [--forget-host]
#     <로봇> = tools/robots.json 의 이름, 또는 [user@]주소[:포트] (빠진 값은 robots.json 의 defaults)
#     --dry-run     : 보내지 않고 접속·워크스페이스·같은 이름 맵 유무·여유 공간만 확인 (처음이면 열쇠 등록까지)
#     --forget-host : 로봇을 다시 설치해 지문(host key)이 바뀌었을 때 예전 지문을 지우고 접속
#   tools/deploy_robot_map.sh <패키지 폴더|zip> --local [--ws <워크스페이스>]     # 이 컴퓨터
# 접속: 이 저장소 전용 열쇠(.robot_ssh/id_ed25519, 처음 실행 때 생성)로 먼저, 안 되면 robots.json 의 비밀번호로 접속해
#   그 열쇠를 로봇 ~/.ssh/authorized_keys 에 등록한다 (다음부터는 비밀번호 없이). 비밀번호 우선순위: --password > $ROBOT_SSH_PASS > robots.json.
# 종료 코드: 10 로봇에 닿지 않음 · 11 인증 실패 · 12 지문 바뀜 · 3 워크스페이스 없음 · 5 공간 부족 · 4 목록 불일치 · 2 전송 끊김
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTS_FILE="${ROBOTS_FILE:-$REPO/tools/robots.json}"
SSHDIR="${ROBOT_SSH_DIR:-$REPO/.robot_ssh}"

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
[ $# -ge 2 ] || usage
PKG="$1"; TARGET="$2"; shift 2
WS="${ROBOT_WS:-}"; NEWNAME=""; DRY=0; FORGET=0; PASS="${ROBOT_SSH_PASS:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --ws) WS="$2"; shift 2;;
    --name) NEWNAME="$2"; shift 2;;
    --password) PASS="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --forget-host) FORGET=1; shift;;
    -h|--help) usage 0;;
    *) echo "알 수 없는 옵션: $1" >&2; usage;;
  esac
done

STAGE="$(mktemp -d)"
OWNER="$(stat -c %u:%g "$REPO")"
cleanup() { rm -rf "$STAGE"; [ "$(id -u)" = 0 ] && [ -d "$SSHDIR" ] && chown -R "$OWNER" "$SSHDIR" 2>/dev/null; true; }
trap cleanup EXIT

# 0) 로봇 찾기: robots.json 이름 또는 [user@]주소[:포트] → user host port ws password label
if [ "$TARGET" != "--local" ]; then
  mapfile -t R < <(python3 - "$ROBOTS_FILE" "$REPO/tools/robots.example.json" "$TARGET" <<'PY'
import json, os, re, sys
f, ex, t = sys.argv[1:4]
cfg = {}
for p in (f, ex):
    if os.path.exists(p):
        cfg = json.load(open(p, encoding="utf-8")); break
d = {"user": "raion", "password": "", "ws": "raisin_master", "port": 22, **(cfg.get("defaults") or {})}
r = next((x for x in cfg.get("robots") or [] if x.get("name") == t and x.get("host")), None)
if r is None:
    m = re.fullmatch(r"(?:([a-z_][a-z0-9_.-]{0,31})@)?([A-Za-z0-9._-]{1,253})(?::(\d{1,5}))?", t)
    if not m:
        sys.exit(f"로봇 '{t}': {f} 에 없는 이름이고 [user@]주소[:포트] 형식도 아닙니다")
    r = {"host": m[2], **({"user": m[1]} if m[1] else {}), **({"port": int(m[3])} if m[3] else {})}
r = {**d, **r}
label = (r["name"] + " " if r.get("name") else "") + f'{r["user"]}@{r["host"]}' + (f':{r["port"]}' if int(r["port"]) != 22 else "")
for k in (r["user"], r["host"], str(int(r["port"])), r["ws"], r["password"], label):
    print(str(k).replace("\n", " "))
PY
  ) || exit 1
  [ ${#R[@]} -eq 6 ] || exit 1
  RUSER="${R[0]}"; RHOST="${R[1]}"; RPORT="${R[2]}"; [ -n "$WS" ] || WS="${R[3]}"; [ -n "$PASS" ] || PASS="${R[4]}"; LABEL="${R[5]}"
  DEST="$RUSER@$RHOST"
  [ "$RPORT" = 22 ] && HOSTKEY="$RHOST" || HOSTKEY="[$RHOST]:$RPORT"
fi
WS="${WS:-raisin_master}"; WS="${WS#\~/}"   # "~/x" → 원격 홈 기준 "x"
[[ "$WS" != *"'"* ]] || { echo "워크스페이스 경로에 작은따옴표는 쓸 수 없습니다" >&2; exit 1; }

# 1) 패키지 폴더 준비 (zip 이면 풀기)
if [ -f "$PKG" ] && [[ "$PKG" == *.zip ]]; then
  python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$PKG" "$STAGE/unz"
  mapfile -t TOP < <(find "$STAGE/unz" -mindepth 1 -maxdepth 1 -type d)
  [ ${#TOP[@]} -eq 1 ] || { echo "zip 안 최상위 폴더가 하나여야 합니다 (<이름>/...)" >&2; exit 1; }
  SRC="${TOP[0]}"
elif [ -d "$PKG" ]; then SRC="$(cd "$PKG" && pwd)"
else echo "패키지가 없습니다: $PKG" >&2; exit 1; fi
NAME="$(basename "$SRC")"

# 2) 이름 바꾸기 (선택)
if [ -n "$NEWNAME" ] && [ "$NEWNAME" != "$NAME" ]; then
  [[ "$NEWNAME" =~ ^[A-Za-z0-9_-]{1,64}$ ]] || { echo "맵 이름 '$NEWNAME': 영문·숫자·_·- 만 (64자 이하)" >&2; exit 1; }
  cp -a "$SRC" "$STAGE/$NEWNAME"
  for suf in .pcd .txt _manifest.txt _sc.bin _georef.txt; do
    [ -e "$STAGE/$NEWNAME/$NAME$suf" ] && mv "$STAGE/$NEWNAME/$NAME$suf" "$STAGE/$NEWNAME/$NEWNAME$suf"
  done
  [ -f "$STAGE/$NEWNAME/map_info.json" ] && sed -i "s/\"name\": \"$NAME\"/\"name\": \"$NEWNAME\"/; s#log/map/$NAME/#log/map/$NEWNAME/#; s/Load Map → $NAME\"/Load Map → $NEWNAME\"/" "$STAGE/$NEWNAME/map_info.json"
  SRC="$STAGE/$NEWNAME"; NAME="$NEWNAME"
fi

# 3) 노트북 쪽 검사: 이름 규칙, 위치추정 pcd, 보행 타일
[[ "$NAME" =~ ^[A-Za-z0-9_-]{1,64}$ ]] || { echo "맵 이름 '$NAME': 영문·숫자·_·- 만 (64자 이하)" >&2; exit 1; }
[ -f "$SRC/$NAME.pcd" ] || { echo "$SRC/$NAME.pcd 가 없습니다 — 폴더 이름과 pcd 이름이 같아야 합니다" >&2; ls "$SRC" >&2; exit 1; }
POINTS="$(head -c 2000 "$SRC/$NAME.pcd" | grep -a '^POINTS' | awk '{print $2}')"
TILES="-"
if [ -d "$SRC/voxel3d_tiles" ]; then
  grep -q 'voxel3d_tiles v2' "$SRC/voxel3d_tiles/index.json" || { echo "voxel3d_tiles/index.json 이 voxel3d_tiles v2 형식이 아닙니다" >&2; exit 1; }
  for f in $(grep -o '"file": *"[^"]*"' "$SRC/voxel3d_tiles/index.json" | sed 's/.*"\([^"]*\)"$/\1/'); do
    [ -f "$SRC/voxel3d_tiles/$f" ] || { echo "타일 파일 없음: voxel3d_tiles/$f" >&2; exit 1; }
  done
  TILES="$(ls "$SRC"/voxel3d_tiles/t_*.bin | wc -l)"
else
  echo "경고: voxel3d_tiles 가 없습니다 — 위치추정 지도만 들어갑니다 (prior map controller 는 이 맵에서 사전맵 없음)" >&2
fi
LISTING="$(cd "$SRC" && find . -type f -printf '%P %s\n' | LC_ALL=C sort | md5sum | cut -c1-32)"
SIZE="$(du -sh "$SRC" | cut -f1)"; SIZE_KB="$(du -sk "$SRC" | cut -f1)"
echo "맵 '$NAME': 위치추정 $NAME.pcd ${POINTS:-?} 점 · 보행 타일 $TILES 개 · $SIZE"
[ "$TARGET" = "--local" ] && echo "대상: 이 컴퓨터 $WS/log/map/$NAME/" || echo "대상: $LABEL  $WS/log/map/$NAME/"

# 4) 접속 (열쇠 → 비밀번호 + 열쇠 등록)
KEY="$SSHDIR/id_ed25519"
SSHOPTS=(-p "${RPORT:-22}" -o "UserKnownHostsFile=$SSHDIR/known_hosts" -o GlobalKnownHostsFile=/dev/null -o StrictHostKeyChecking=accept-new
         -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR)
KEYOPTS=(-i "$KEY" -o BatchMode=yes)
PWOPTS=(-o BatchMode=no -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password -o NumberOfPasswordPrompts=1)
MODE=key
rssh() {
  if [ "$MODE" = key ]; then ssh "${SSHOPTS[@]}" "${KEYOPTS[@]}" "$DEST" "$@"
  else SSH_ASKPASS="$STAGE/askpass" SSH_ASKPASS_REQUIRE=force DISPLAY="${DISPLAY:-:0}" ROBOT_SSH_PASS="$PASS" ssh "${SSHOPTS[@]}" "${PWOPTS[@]}" "$DEST" "$@"; fi
}
fail_ssh() {   # ssh 자체 실패(255) 메시지 → 안내 + 종료 코드
  local err="$1"
  if grep -qiE "IDENTIFICATION HAS CHANGED|Host key verification failed|host key .* (changed|mismatch)" <<<"$err"; then
    echo "✗ 로봇 지문(host key)이 예전에 접속했을 때와 다릅니다 ($HOSTKEY)." >&2
    echo "  로봇을 다시 설치했거나 주소가 다른 로봇으로 바뀐 경우입니다. 같은 로봇이 맞으면 --forget-host 를 붙여 다시 실행하세요 (포털: '지문 지우고 다시 확인')." >&2; exit 12
  elif grep -qiE "Permission denied|Authentication failed|Too many authentication" <<<"$err"; then
    echo "✗ 로봇 접속 인증 실패: 열쇠도, robots.json 의 비밀번호도 안 됩니다 ($DEST)." >&2
    echo "  비밀번호가 다르면 tools/robots.json 에서 그 로봇의 \"password\" 를 고치거나, 터미널에서 직접 열쇠를 등록하세요:" >&2
    echo "    ssh-copy-id -i $KEY.pub -p ${RPORT:-22} $DEST" >&2; exit 11
  elif grep -qiE "resolve|timed out|No route|refused|unreachable|Connection closed|Connection reset|kex_exchange" <<<"$err"; then
    echo "✗ 이 컴퓨터($(hostname))에서 로봇에 닿지 않습니다 ($DEST:${RPORT:-22}): $(tail -1 <<<"$err")" >&2
    echo "  접속은 이 스크립트(포털이면 포털)가 돌고 있는 컴퓨터에서 합니다. 그 컴퓨터와 로봇이 같은 네트워크인지, 주소가 맞는지, 로봇이 켜져 있는지 확인하세요." >&2; exit 10
  fi
  echo "✗ ssh 실패: $err" >&2; exit 13
}
connect() {
  mkdir -p "$SSHDIR"; chmod 700 "$SSHDIR"; touch "$SSHDIR/known_hosts"
  [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "offline-map@$(hostname)" -f "$KEY"
  if [ "$FORGET" = 1 ]; then ssh-keygen -f "$SSHDIR/known_hosts" -R "$HOSTKEY" >/dev/null 2>&1 || true; echo "예전 지문 지움: $HOSTKEY"; fi
  local err rc
  if err="$(ssh "${SSHOPTS[@]}" "${KEYOPTS[@]}" "$DEST" true 2>&1 </dev/null)"; then echo "접속: 열쇠(비밀번호 없이)"; return; else rc=$?; fi
  [ "$rc" = 255 ] || fail_ssh "$err"
  grep -qiE "Permission denied|Authentication" <<<"$err" || fail_ssh "$err"
  if [ -z "$PASS" ]; then
    echo "✗ 열쇠로 접속이 안 되고, 비밀번호가 없습니다 — tools/robots.json 의 defaults.password(또는 그 로봇의 password)를 적거나 --password 로 주세요." >&2
    echo "  또는 터미널에서 열쇠를 직접 등록: ssh-copy-id -i $KEY.pub -p ${RPORT:-22} $DEST" >&2; exit 11
  fi
  echo "열쇠로는 아직 안 됨 → 비밀번호로 접속해 이 노트북 열쇠를 로봇에 등록합니다"
  printf '#!/bin/sh\nprintf "%%s\\n" "$ROBOT_SSH_PASS"\n' > "$STAGE/askpass"; chmod 700 "$STAGE/askpass"
  MODE=pw
  local pub inst
  pub="$(cat "$KEY.pub")"
  inst="umask 077; mkdir -p \"\$HOME/.ssh\"; touch \"\$HOME/.ssh/authorized_keys\"; grep -qxF '$pub' \"\$HOME/.ssh/authorized_keys\" || echo '$pub' >> \"\$HOME/.ssh/authorized_keys\""
  if err="$(rssh "sh -c \"\$(echo $(printf '%s' "$inst" | base64 -w0) | base64 -d)\"" 2>&1 </dev/null)"; then :; else
    rc=$?; [ "$rc" = 255 ] && fail_ssh "$err"
    echo "경고: 열쇠 등록 실패 ($err) — 이번에는 비밀번호로 계속합니다"; return
  fi
  if ssh "${SSHOPTS[@]}" "${KEYOPTS[@]}" "$DEST" true </dev/null >/dev/null 2>&1; then MODE=key; echo "열쇠 등록 완료 — 다음부터는 비밀번호 없이 접속"
  else echo "경고: 열쇠를 넣었지만 열쇠 접속이 여전히 안 됩니다 (로봇 홈·~/.ssh 권한 또는 sshd 설정) — 비밀번호로 계속합니다"; fi
}
run_remote() {   # run_remote <스크립트> <인자...> : 원격 로그인 셸이 sh 여도 깨지지 않게 스크립트는 base64, 인자는 작은따옴표 (값은 위에서 검사)
  local script="$1"; shift
  if [ "$TARGET" = "--local" ]; then bash -c "$script" _ "$@"; return; fi
  local b64 args="" a; b64="$(printf '%s' "$script" | base64 -w0)"
  for a in "$@"; do args+=" '$a'"; done
  rssh "bash -c \"\$(echo $b64 | base64 -d)\" _$args"
}
[ "$TARGET" = "--local" ] || connect

# 5) 로봇 상태 확인: 워크스페이스, 같은 이름 맵, 여유 공간. 인자: ws name size_kb
CHECK='set -euo pipefail
ws="$1"; name="$2"; need="$3"
case "$ws" in /*) ;; *) ws="$HOME/$ws";; esac
[ -d "$ws" ] || { echo "✗ 로봇에 워크스페이스가 없습니다: $ws — tools/robots.json 의 \"ws\" 나 포털의 워크스페이스 칸(--ws)을 고치세요" >&2; exit 3; }
if [ -e "$ws/log/map/$name" ]; then echo "로봇 상태: 같은 이름의 맵이 이미 있음 ($(du -sh "$ws/log/map/$name" | cut -f1), $(date -r "$ws/log/map/$name" "+%Y-%m-%d %H:%M")) → log/map_backup/ 으로 옮기고 교체"
else echo "로봇 상태: 새로 설치 → $ws/log/map/$name"; fi
d="$ws"; [ -d "$ws/log" ] && d="$ws/log"
avail="$(df -Pk "$d" | awk "NR==2{print \$4}")"
echo "로봇 여유 공간: $((avail / 1024)) MB (필요 $((need / 1024 + 1)) MB)"
[ "$avail" -gt $((need + need / 10 + 1024)) ] || { echo "✗ 로봇 디스크 공간이 부족합니다" >&2; exit 5; }'
set +e; OUT="$(run_remote "$CHECK" "$WS" "$NAME" "$SIZE_KB" 2>&1 </dev/null)"; RC=$?; set -e
[ "$RC" = 255 ] && [ "$TARGET" != "--local" ] && fail_ssh "$OUT"
echo "$OUT"; [ "$RC" = 0 ] || exit "$RC"
[ "$DRY" = 1 ] && { echo "(확인만: 보내지 않음)"; exit 0; }

# 6) 보내기: stdin 의 tar 풀기 → 목록 비교 → 백업 → 설치. 인자: ws name stamp listing
REMOTE='set -euo pipefail
ws="$1"; name="$2"; stamp="$3"; want="$4"
case "$ws" in /*) ;; *) ws="$HOME/$ws";; esac
map="$ws/log/map"; inc="$ws/log/map_incoming"; mkdir -p "$map" "$inc"
rm -rf "$inc/$name"; tar -C "$inc" -xf -
got="$(cd "$inc/$name" && find . -type f -printf "%P %s\n" | LC_ALL=C sort | md5sum | cut -c1-32)"
[ "$got" = "$want" ] || { echo "✗ 받은 파일 목록·크기가 보낸 것과 다릅니다 — 기존 맵은 그대로 둡니다 ($inc/$name)" >&2; exit 4; }
if [ -e "$map/$name" ]; then mkdir -p "$ws/log/map_backup"; mv "$map/$name" "$ws/log/map_backup/${name}_$stamp"; echo "기존 맵 → $ws/log/map_backup/${name}_$stamp"; fi
mv "$inc/$name" "$map/$name"; rmdir "$inc" 2>/dev/null || true
echo "설치됨: $map/$name"; ls -la "$map/$name" | tail -n +2'
STAMP="$(date +%Y%m%d_%H%M%S)"
echo "보내는 중… ($SIZE)"
set +e
tar -C "$(dirname "$SRC")" -cf - "$NAME" | run_remote "$REMOTE" "$WS" "$NAME" "$STAMP" "$LISTING"
PS=("${PIPESTATUS[@]}"); set -e
if [ "${PS[1]}" != 0 ]; then
  [ "${PS[1]}" = 255 ] && { echo "✗ 보내는 도중 연결이 끊겼습니다 — 로봇의 기존 맵은 그대로입니다. 다시 실행하세요." >&2; exit 2; }
  exit "${PS[1]}"
fi
[ "${PS[0]}" = 0 ] || { echo "✗ 노트북 쪽 tar 실패" >&2; exit 2; }
echo "✓ 완료 — raisin GUI Load Map 에서 '$NAME' 선택 (위치추정 지도와 보행 지도가 함께 연결됨)"
