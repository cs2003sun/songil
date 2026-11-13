#!/usr/bin/env bash
# 서버와 cloudflared 터널을 tmux 세션 두 개(server, tunnel)로 실행
#
#   ./run.sh          시작 (이미 실행 중이면 다시 시작)
#   ./run.sh stop     둘 다 종료
#   ./run.sh status   상태 확인
#
#   화면 보기: tmux attach -t server   (빠져나오기: Ctrl+b 누른 뒤 d)
#
# .env 의 TUNNEL_TOKEN 이 있으면 고정 주소 터널, 없으면 임시 주소(trycloudflare.com) 터널
set -euo pipefail

cd "$(dirname "$0")"
DIR="$(pwd)"
ENV_FILE="${ENV_FILE:-$DIR/.env}"

stop() {
  tmux kill-session -t tunnel 2>/dev/null && echo "tunnel 종료" || true
  tmux kill-session -t server 2>/dev/null && echo "server 종료" || true
}

health() {
  curl -s -m 3 -H "X-API-Key: ${API_KEY:-}" "http://127.0.0.1:${PORT:-8000}/health" || true
}

[ -f "$ENV_FILE" ] || { echo "$ENV_FILE 이 없습니다. cp .env.example .env 후 값을 채우세요."; exit 1; }
set -a; source "$ENV_FILE"; set +a
PORT="${PORT:-8000}"

case "${1:-start}" in
  stop)   stop; exit 0 ;;
  status)
    tmux ls 2>/dev/null || echo "실행 중인 tmux 세션 없음"
    echo; health; echo
    grep -ho 'https://[a-z0-9-]*\.trycloudflare\.com' "$DIR/logs/tunnel.log" 2>/dev/null | tail -1 || true
    exit 0 ;;
  start)  ;;
  *) echo "사용법: $0 [start|stop|status]"; exit 1 ;;
esac

command -v tmux >/dev/null || { echo "tmux 가 없습니다: sudo apt install -y tmux"; exit 1; }
command -v cloudflared >/dev/null || { echo "cloudflared 가 없습니다 (README 참고)"; exit 1; }
if systemctl is-active --quiet cloudflared 2>/dev/null; then
  echo "주의: cloudflared 시스템 서비스가 이미 실행 중입니다. 같은 터널을 두 번 띄우지 않도록 확인하세요."
fi

mkdir -p "$DIR/logs"
stop

# tmux 세션은 이 스크립트의 환경변수를 물려받지 않을 수 있어서 세션 안에서 .env 를 다시 읽는다
LOAD_ENV="cd '$DIR' && set -a && . '$ENV_FILE' && set +a"
KEEP_OPEN='echo; echo "[종료됨] Enter를 누르면 창이 닫힙니다"; read _'

tmux new-session -d -s server \
  "$LOAD_ENV && python3 -m uvicorn server:app --host 127.0.0.1 --port $PORT --workers 1 2>&1 | tee -a logs/server.log; $KEEP_OPEN"
echo "server 시작 (tmux attach -t server)"

if [ -n "${TUNNEL_TOKEN:-}" ]; then
  # cloudflared 는 환경변수 TUNNEL_TOKEN 을 직접 읽음 (명령줄에 토큰이 보이지 않게)
  TUNNEL_CMD="cloudflared tunnel --no-autoupdate run"
else
  TUNNEL_CMD="cloudflared tunnel --no-autoupdate --url http://127.0.0.1:$PORT"
fi
: > logs/tunnel.log
tmux new-session -d -s tunnel "$LOAD_ENV && $TUNNEL_CMD 2>&1 | tee -a logs/tunnel.log; $KEEP_OPEN"
echo "tunnel 시작 (tmux attach -t tunnel)"

# 임시 터널이면 주소를 찾아서 보여줌
if [ -z "${TUNNEL_TOKEN:-}" ]; then
  for _ in $(seq 1 30); do
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' logs/tunnel.log | tail -1 || true)
    [ -n "$URL" ] && break
    sleep 1
  done
  echo "임시 HTTPS 주소: ${URL:-(아직 없음: tmux attach -t tunnel 로 확인)}"
fi

# 모델 준비 기다리기 (최대 3분)
echo -n "모델 준비 중"
for _ in $(seq 1 90); do
  if health | grep -q '"model_loaded":true'; then
    echo; echo "준비 완료: $(health)"; exit 0
  fi
  if health | grep -q '"status":"error"'; then
    echo; echo "모델 로딩 실패: $(health)"; exit 1
  fi
  echo -n "."; sleep 2
done
echo; echo "아직 준비 중입니다. tmux attach -t server 로 로그를 확인하세요."
