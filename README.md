# YOLOE 물건 찾기 API 서버 (GPU · 다중 사용자)

```
yoloe-api/
├── server.py        # 서버 (python3 -m uvicorn server:app ...)
├── vocab.py         # 공유 어휘 231개 + 동의어 그룹 + 별칭 + 전체 탐지 때 뺄 배경 물체
├── prompt.txt       # /describe 기본 Gemini 프롬프트
├── describe_test.html  # /describe-test 페이지 (프롬프트 고쳐가며 시험)
├── static/          # /app 으로 제공되는 웹 앱 (index.html 등, 직접 넣기)
│   └── audio/       # 미리 생성한 안내 음성 + manifest.json (cache/ 는 /tts 캐시, git 제외)
├── elevenlabs_tts.py  # ElevenLabs 호출 공용 코드
├── pregen_tts.py    # 안내 음성 미리 생성
├── run.sh           # tmux 세션 server / tunnel 실행
├── loadtest.py      # 부하 테스트
├── test_detect.py   # 단건 테스트 (/detect, /describe)
├── requirements.txt
├── .gitignore
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
nano .env        # API_KEY, GEMINI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, (고정 주소면) TUNNEL_TOKEN
```

prompt.txt 의 `{현재시간}`은 요청 때마다 서버 시각(TIMEZONE, 기본 Asia/Seoul) 기준으로 바뀝니다: 5~11시 아침, 12~17시 오후, 그 밖은 저녁 (VLM 테스트 페이지와 같은 기준). `#`으로 시작하는 줄은 보내지 않고, 파일을 고치면 서버를 다시 켜지 않아도 다음 요청부터 반영됩니다.

프롬프트를 고쳐가며 시험하려면 브라우저에서 `https://주소/describe-test`를 여세요. 텍스트 상자에 prompt.txt 내용이 기본값으로 들어 있고, 고친 내용은 그 요청에만 쓰입니다 (prompt.txt는 바뀌지 않음). 사진 위 손잡이 점 표시, 문장 읽어주기, 빠진 항목·안심 표현 점검이 있습니다. 발표 때는 `.env`에서 `TEST_PAGE=0`으로 끌 수 있습니다.

### ElevenLabs 음성 설정

`.env`에 아래 항목을 채웁니다. 키는 서버에만 있고 응답·로그에는 나오지 않습니다.

| 항목 | 설명 |
|---|---|
| `ELEVENLABS_API_KEY` | ElevenLabs 대시보드 → API Keys 에서 만든 키 |
| `ELEVENLABS_VOICE_ID` | 쓸 목소리 ID (Voices 화면에서 목소리 → ID 복사) |
| `ELEVENLABS_PREGEN_MODEL` | 미리 생성용, 기본 `eleven_v3` (가장 자연스럽고 감정 표현이 풍부, 느리고 비쌈) |
| `ELEVENLABS_LIVE_MODEL` | 실시간 생성용, 기본 `eleven_flash_v2_5` (약 75ms, 한국어 지원, 글자당 비용 낮음) |
| `ELEVENLABS_STABILITY` 등 4개 | 비우면 그 목소리의 기본값. `eleven_v3`는 안정성을 0.0/0.5/1.0 세 단계로만 받으므로 가장 가까운 값으로 맞춰 보냅니다 |
| `TTS_DAILY_CHAR_LIMIT` | 실시간 생성 하루 글자 수 상한 (기본 5000, TIMEZONE 기준 자정에 초기화) |

요청은 ElevenLabs 공식 API(`POST /v1/text-to-speech/{voice_id}`, `xi-api-key` 헤더, `mp3_44100_128`)로 보내고, 한국어로 읽도록 `language_code: "ko"`를 함께 보냅니다.

### 안내 음성 미리 생성 (pregen_tts.py)

`~/app/phrases.json`에 id별 문장 목록을 씁니다. id는 파일 이름이 되므로 영문·숫자·`_`·`-`만 씁니다.

```json
{
  "searching": ["찾아볼게요.", "주변을 살펴볼게요."],
  "found": ["찾았어요!", "여기 있어요."],
  "lost": ["화면에서 사라졌어요. 폰을 천천히 움직여 주세요."]
}
```

```bash
python3 pregen_tts.py --dry-run            # 만들 문장 수와 글자 수만 확인 (크레딧 쓰지 않음)
python3 pregen_tts.py                      # 없는 파일만 생성
python3 pregen_tts.py --only found,lost    # 이 id만
python3 pregen_tts.py --only found --force # 이 id를 다시 생성 (문장을 고쳤을 때)
python3 pregen_tts.py --phrases ./phrases.json   # 다른 위치의 목록
```

- 결과: `static/audio/found_1.mp3`, `found_2.mp3` … → 웹 주소 `/app/audio/found_1.mp3` (API 키 없이 열림)
- `static/audio/manifest.json`: `{"found": 2, ...}` → 앱은 1~N 중 하나를 골라 재생하면 됩니다
- 실패한 문장은 2번 더 시도하고, 그래도 안 되면 끝에 목록으로 보여줍니다. 다시 실행하면 없는 파일만 만듭니다.
- manifest에는 1번부터 빠짐없이 있는 파일 개수만 적습니다. 중간 번호가 실패하면 그 id의 개수가 줄어들어 앱이 없는 파일을 부르지 않습니다.
- 마지막에 생성한 문장 수와 글자 수를 출력합니다. 크레딧은 이 글자 수에 모델·요금제의 글자당 단가를 곱한 값입니다.
- 문장을 고친 id는 `--force`로 다시 만들어야 합니다 (이미 있는 파일은 건너뛰므로).

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
curl -s -D - -o test.mp3 -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
     -d '{"text":"보드마카 찾아볼게요"}' http://127.0.0.1:8000/tts | grep -i x-tts
python3 loadtest.py 사진.jpg --users 30 --duration 60
python3 loadtest.py 사진.jpg --users 30 --url https://주소   # 터널까지 포함해서
curl -s -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/stats | python3 -m json.tool
```

요청별 기록은 `logs/requests_날짜_시각.csv`에 쌓입니다 (서버를 켤 때마다 새 파일). IP와 기기 ID는 해시로만 남습니다.

## API

`/app`, `/describe-test`를 뺀 모든 엔드포인트는 `X-API-Key` 헤더가 필요합니다. 폰 앱은 `X-Client-Id`(기기마다 다른 임의 문자열)도 보내는 것을 권장합니다.

**POST /detect** (multipart): `image`, `targets`(선택, "cup,key"), `conf`(0.25), `imgsz`(640)

- `targets`가 있으면 `mode: "targets"`: 그 물체만 찾습니다 (최대 10개 이름).
- `targets`에 색 단어(black, white, gray/grey/silver, red, orange, yellow, green, blue, purple, pink, brown)가 있으면 색상 필터를 씁니다. 예: `targets="red cup,cup"`
  - YOLOE에는 색을 뺀 이름(`cup`)만 보내 형태로 찾고, 박스의 상하좌우 25%를 뺀 중앙 50% 영역에서 그 색 픽셀(HSV 기준) 비율을 잽니다.
  - 비율이 `COLOR_MIN_RATIO`(0.30) 이상인 박스만 남기고, `score`를 `score × (0.6 + 0.4 × 일치율)`로 바꿔 다시 정렬합니다. 각 박스에 `color_match`(0~1)가 붙습니다.
  - 색이 달라 빠진 박스는 `color_rejected`에 `label`, `box`, `dominant_color`(가장 많은 색, 20% 미만이면 unknown)와 함께 최대 5개 들어갑니다.
  - 분홍은 빨강·보라 기준과 겹치므로, 빨강·보라 일치율을 셀 때는 분홍 픽셀을 뺍니다 (`COLOR_EXCLUDE`). 그래서 `red cup`으로 찾으면 분홍 컵은 빠집니다.
  - 색 기준값은 server.py 맨 위 `COLOR_HSV_RULES`에서 고칩니다. 색이 없는 요청은 `color_filter: null`, `color_rejected: []`이고 결과는 전과 같습니다.
- `targets`가 없거나 비어 있으면 `mode: "all"`: 기본 어휘 전체에서 찾되 배경 물체(table, desk, chair, person, window, door 등, vocab.py의 `BACKGROUND`)는 빼고 신뢰도 높은 순으로 최대 20개(`MAX_RESULTS_ALL`)를 돌려줍니다. 이때 `target`은 `label`과 같고 색상 필터는 쓰지 않습니다.

```json
{
  "mode": "targets",
  "detections": [
    {"label": "mug", "target": "cup", "score": 0.79, "color_match": 0.82,
     "box": {"x": 0.11, "y": 0.21, "w": 0.39, "h": 0.39}}
  ],
  "color_filter": "red",
  "color_rejected": [{"label": "cup", "box": {"x": 0.6, "y": 0.3, "w": 0.2, "h": 0.3}, "dominant_color": "blue"}],
  "timing_ms": {"decode": 3.1, "vocab_update": 0, "queue_wait": 12.4, "inference": 28.7, "color": 0.6, "total": 46.6},
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

`intent`는 find / where / how / stop / unknown이고 서버가 바꾸지 않습니다. `where`라도 문장에 물건 이름이 있으면 `object_ko`와 `targets`를 둘 다 채워서 돌려주므로, 앱은 `object_ko`를 지금 찾는 물건과 비교해 새 물건인지 판단하면 됩니다. 둘 중 하나만 채워져 오면 Gemini에 한 번 더 묻고, 그래도 같으면 받은 결과를 그대로 돌려줍니다. "이거 어떻게 잡아?"처럼 물건 이름이 없으면 `object_ko: null`, `targets: []`입니다. `targets`는 0~3개의 영어 소문자 이름으로 `/detect`의 `targets`에 그대로 넣을 수 있습니다. 같은 문장(공백과 끝의 ?·! 차이는 무시)은 메모리에 저장한 결과를 다시 쓰고(`cached: true`), 기기(`X-Client-Id`, 없으면 IP)별로 분당 20회까지 받습니다. CSV에는 음성 문장 대신 해석된 targets만 남깁니다. 문장에 색 형용사(빨간, 까만, 은색, 분홍 등)가 있으면 targets 첫 번째가 `"red cup"`처럼 영어 색 단어 + 물건 이름이 되므로, 그대로 `/detect`에 넘기면 색상 필터가 적용됩니다. Gemini가 색을 빼먹으면 서버가 보충합니다.

**POST /tts** (JSON): `{"text": "보드마카 찾아볼게요"}` → `audio/mpeg` (MP3)

- 실시간 모델(`ELEVENLABS_LIVE_MODEL`)로 만들고, 같은 문장·목소리·모델·음성 설정이면 `static/audio/cache/{해시}.mp3`에 저장한 파일을 다시 씁니다 (앞뒤·연속 공백은 무시).
- 응답 헤더: `X-TTS-Source`(`cache` 또는 `generated`), `X-TTS-Ms`(생성에 걸린 ms, 캐시는 0)
- 기기(`X-Client-Id`, 없으면 IP)별 분당 30회, 200자 초과는 `400 text_too_long`
- 하루 생성 글자 수가 `TTS_DAILY_CHAR_LIMIT`를 넘으면 `429 tts_daily_limit` (`Retry-After`는 자정까지 남은 초). 캐시된 문장은 계속 나갑니다. 사용량은 `logs/tts_usage.json`에 남아 서버를 다시 켜도 유지됩니다.
- ElevenLabs 쪽 문제(키 오류, 설정 값 오류, 8초 시간 초과, 크레딧 부족 등)는 `502`와 이유를 JSON으로 돌려줍니다. 앱은 200이 아니면 브라우저 음성(speechSynthesis)으로 대체하면 됩니다.
- 같은 문장이 동시에 여러 번 오면 ElevenLabs는 한 번만 부릅니다. 호출은 비동기라 그동안 `/detect`는 그대로 처리됩니다.

```js
// 앱 예시: 실패하면 브라우저 음성으로
async function speak(text) {
  try {
    const r = await fetch('/tts', { method: 'POST',
      headers: { 'X-API-Key': KEY, 'X-Client-Id': DEVICE_ID, 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }) });
    if (!r.ok) throw new Error(r.status);
    const audio = new Audio(URL.createObjectURL(await r.blob()));
    await audio.play();
  } catch {
    const u = new SpeechSynthesisUtterance(text); u.lang = 'ko-KR'; speechSynthesis.speak(u);
  }
}
```

**GET /app/** : `static/` 폴더의 파일을 API 키 없이 제공합니다. `/app/`은 `static/index.html`, `/app`은 `/app/`으로 이동합니다. 미리 생성한 음성은 `/app/audio/{id}_{번호}.mp3`, 목록은 `/app/audio/manifest.json`입니다. 폴더 위치는 환경변수 `STATIC_DIR`로 바꿀 수 있습니다.

**GET /health**: `status`, `model_loaded`, `model`, `device`, `half`, `vocab_size`, `queue_length`, `describe_enabled`, `tts_enabled`(ElevenLabs 설정 여부), `tts_today_chars`(오늘 실시간 생성 글자 수), `tts_daily_limit`

**GET /stats**: 최근 5분 `detect`/`describe`/`parse`/`tts`별 요청 수·성공 수·avg/p50/p95(ms)·429·503 횟수, `batch.avg_size`, `unique_ips`, `unique_clients`, `queue_length`

| 상태 | code | 상황 |
|---|---|---|
| 400 | `invalid_image`, `invalid_targets`, `invalid_text`, `text_too_long` | 이미지를 못 읽음 / 이름 형식 오류 / 문장이 비었거나 너무 김 |
| 401 | `unauthorized` | API 키 불일치 |
| 429 | `rate_limited`, `tts_daily_limit` | /detect 초당 5회, /describe 분당 10회, /parse 분당 20회, /tts 분당 30회 초과 / 오늘 음성 생성 한도 초과 (`Retry-After` 헤더) |
| 502/504 | `vlm_error`, `vlm_invalid_json`, `vlm_timeout` 등 | Gemini 쪽 문제 |
| 502 | `tts_error`, `tts_timeout`, `tts_unreachable`, `tts_bad_audio` | ElevenLabs 쪽 문제 → 앱은 브라우저 음성으로 대체 |
| 503 | `queue_full`, `timeout`, `model_loading` | 대기열이 가득 참 / 오래 기다린 사진 버림 / 준비 중 → 폰은 기기 내 인식으로 전환 |
| 503 | `tts_disabled` | ElevenLabs 키·목소리 ID가 없거나 음성 설정 값이 숫자가 아님 |
