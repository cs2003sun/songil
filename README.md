# YOLOE 물건 찾기 API 서버 (GPU · 다중 사용자)

```
yoloe-api/
├── server.py        # 서버 (python3 -m uvicorn server:app ...)
├── vocab.py         # 공유 어휘 231개 + 동의어 그룹 + 별칭
├── prompt.txt       # /describe 기본 Gemini 프롬프트
├── describe_test.html  # /describe-test 페이지 (프롬프트 고쳐가며 시험)
├── static/          # /app 으로 제공되는 웹 앱 (index.html 등, 직접 넣기)
├── run.sh           # tmux 세션 server / tunnel 실행
├── loadtest.py      # 부하 테스트
├── test_detect.py   # 단건 테스트 (/detect, /describe)
├── requirements.txt
└── .env.example
```

## 1. 설치 (GCP Deep Learning VM, 한 번만)

```bash
sudo apt install -y tmux git
cd ~/yoloe-api

# 이미 설치된 torch 버전을 고정해서, 다른 패키지 설치 중에 torch가 바뀌지 않게 함
python3 - > constraints.txt <<'PY'
import importlib.metadata as m
for p in ("torch", "torchvision", "torchaudio"):
    try: print(f"{p}=={m.version(p)}")
    except m.PackageNotFoundError: pass
PY
cat constraints.txt
python3 -m pip install -r requirements.txt -c constraints.txt

# 확인: torch 버전이 그대로이고 cuda가 True
python3 -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

cloudflared 설치:

```bash
curl -L -o cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

## 2. 설정

```bash
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # API_KEY 로 사용
nano .env        # API_KEY, GEMINI_API_KEY, (고정 주소면) TUNNEL_TOKEN
```

prompt.txt 의 `{현재시간}`은 요청 때마다 서버 시각(TIMEZONE, 기본 Asia/Seoul) 기준으로 바뀝니다: 5~11시 아침, 12~17시 오후, 그 밖은 저녁 (VLM 테스트 페이지와 같은 기준). `#`으로 시작하는 줄은 보내지 않고, 파일을 고치면 서버를 다시 켜지 않아도 다음 요청부터 반영됩니다.

프롬프트를 고쳐가며 시험하려면 브라우저에서 `https://주소/describe-test`를 여세요. 텍스트 상자에 prompt.txt 내용이 기본값으로 들어 있고, 고친 내용은 그 요청에만 쓰입니다 (prompt.txt는 바뀌지 않음). 사진 위 손잡이 점 표시, 문장 읽어주기, 빠진 항목·안심 표현 점검이 있습니다. 발표 때는 `.env`에서 `TEST_PAGE=0`으로 끌 수 있습니다.

고정 주소 터널: Cloudflare Zero Trust → Networks → Tunnels → Create a tunnel 에서 토큰을 받아 `.env`의 `TUNNEL_TOKEN`에 넣고, Public Hostname 을 `HTTP` / `127.0.0.1:8000`으로 지정합니다. 비워두면 실행할 때마다 바뀌는 `*.trycloudflare.com` 주소를 씁니다.

## 3. 실행

```bash
bash run.sh          # 서버 + 터널 시작, 임시 주소 출력, 모델 준비까지 대기
bash run.sh status   # 세션·/health·임시 주소
bash run.sh stop
tmux attach -t server   # 로그 보기 (나오기: Ctrl+b, d)
```

run.sh 는 내부에서 `python3 -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1`을 실행합니다. 워커는 반드시 1개여야 합니다.

## 4. 테스트

```bash
export API_KEY=...   # .env 와 같은 값
python3 test_detect.py 사진.jpg --targets "cup,remote control" --repeat 5
python3 test_detect.py 컵_잘라낸.jpg --describe
python3 loadtest.py 사진.jpg --users 30 --duration 60
python3 loadtest.py 사진.jpg --users 30 --url https://주소   # 터널까지 포함해서
curl -s -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/stats | python3 -m json.tool
```

요청별 기록은 `logs/requests_날짜_시각.csv`에 쌓입니다 (서버를 켤 때마다 새 파일). IP와 기기 ID는 해시로만 남습니다.

## API

`/app`, `/describe-test`를 뺀 모든 엔드포인트는 `X-API-Key` 헤더가 필요합니다. 폰 앱은 `X-Client-Id`(기기마다 다른 임의 문자열)도 보내는 것을 권장합니다.

**POST /detect** (multipart): `image`, `targets`("cup,key"), `conf`(0.25), `imgsz`(640)

```json
{
  "detections": [
    {"label": "mug", "target": "cup", "score": 0.83, "box": {"x": 0.11, "y": 0.21, "w": 0.39, "h": 0.39}}
  ],
  "timing_ms": {"decode": 3.1, "vocab_update": 0, "queue_wait": 12.4, "inference": 28.7, "total": 46.0},
  "batch_size": 6, "device": "cuda", "model": "yoloe-26l-seg.pt"
}
```

`label`은 모델이 인식한 어휘 이름, `target`은 그 결과가 어떤 요청 이름 때문에 포함됐는지입니다 (동의어 그룹: cup을 요청하면 mug·tumbler 등도 찾음). 같은 물건이 두 이름으로 겹쳐 잡히면 점수 높은 것 하나만 남깁니다. `phone`→`cell phone`, `keys`→`key` 같은 별칭과 간단한 복수형은 자동으로 바꿉니다.

**POST /describe** (multipart): `image`, `prompt`(선택, 비우면 prompt.txt) → `{"result": {Gemini가 만든 JSON}, "model", "attempts", "time_of_day", "prompt_source", "timing_ms"}`

**POST /parse** (JSON): `{"text": "내 검은 텀블러 어디 있어?"}` → 한국어 음성 명령을 Gemini로 해석

```json
{
  "intent": "where", "object_ko": "검은 텀블러", "targets": ["black tumbler", "tumbler"],
  "cached": false, "attempts": 1, "timing_ms": {"vlm": 412.3, "total": 413.0}
}
```

`intent`는 find / where / how / stop / unknown, `targets`는 0~3개의 영어 소문자 이름으로 `/detect`의 `targets`에 그대로 넣을 수 있습니다. 같은 문장(공백과 끝의 ?·! 차이는 무시)은 메모리에 저장한 결과를 다시 쓰고(`cached: true`), 기기(`X-Client-Id`, 없으면 IP)별로 분당 20회까지 받습니다. CSV에는 음성 문장 대신 해석된 targets만 남깁니다.

**GET /app/** : `static/` 폴더의 파일을 API 키 없이 제공합니다. `/app/`은 `static/index.html`, `/app`은 `/app/`으로 이동합니다. 폴더 위치는 환경변수 `STATIC_DIR`로 바꿀 수 있습니다.

**GET /health**: `status`, `model_loaded`, `model`, `device`, `half`, `vocab_size`, `queue_length`, `describe_enabled`

**GET /stats**: 최근 5분 `detect`/`describe`/`parse`별 요청 수·성공 수·avg/p50/p95(ms)·429·503 횟수, `batch.avg_size`, `unique_ips`, `unique_clients`, `queue_length`

| 상태 | code | 상황 |
|---|---|---|
| 400 | `invalid_image`, `invalid_targets` | 이미지를 못 읽음 / 이름 형식 오류 |
| 401 | `unauthorized` | API 키 불일치 |
| 429 | `rate_limited` | /detect 초당 5회, /describe 분당 10회, /parse 분당 20회 초과 (`Retry-After` 헤더) |
| 502/504 | `vlm_error`, `vlm_invalid_json`, `vlm_timeout` 등 | Gemini 쪽 문제 |
| 503 | `queue_full`, `timeout`, `model_loading` | 대기열이 가득 참 / 오래 기다린 사진 버림 / 준비 중 → 폰은 기기 내 인식으로 전환 |
