"""
ElevenLabs 음성 생성 공용 모듈 (server.py 의 /tts 와 pregen_tts.py 가 함께 사용)

요청 형식 (ElevenLabs 공식 문서: Create speech)
  POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128
  헤더  xi-api-key: <API 키>
  본문  {"text", "model_id", "language_code", "voice_settings": {stability, similarity_boost, style, speed}}
  응답  200 이면 오디오 바이너리, 실패하면 JSON 오류

환경변수
  ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
  ELEVENLABS_PREGEN_MODEL (기본 eleven_v3)          미리 생성용, 품질 우선
  ELEVENLABS_LIVE_MODEL   (기본 eleven_flash_v2_5)  실시간 생성용, 속도 우선
  ELEVENLABS_STABILITY / _SIMILARITY / _STYLE / _SPEED  비우면 ElevenLabs 기본값
  ELEVENLABS_OUTPUT_FORMAT (기본 mp3_44100_128)
  ELEVENLABS_LANGUAGE      (기본 ko) 한국어로 읽도록 지정. 지원하지 않는 모델은 무시함

API 키는 이 모듈 밖으로 내보내지 않는다. 오류 메시지에 키가 섞여 있으면 *** 로 가린다.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

API_BASE = "https://api.elevenlabs.io"


def load_dotenv(path: str | Path) -> None:
    """간단한 .env 읽기 (이미 설정된 환경변수는 덮어쓰지 않음). pregen_tts.py 처럼 run.sh 없이 실행할 때 사용"""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _optional_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"환경변수 {name} 는 숫자여야 합니다: {raw!r}") from None


@dataclass(frozen=True)
class TTSConfig:
    api_key: str
    voice_id: str
    pregen_model: str
    live_model: str
    output_format: str
    language: str
    stability: float | None
    similarity: float | None
    style: float | None
    speed: float | None

    @classmethod
    def from_env(cls) -> "TTSConfig":
        return cls(
            api_key=os.getenv("ELEVENLABS_API_KEY", "").strip(),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "").strip(),
            pregen_model=os.getenv("ELEVENLABS_PREGEN_MODEL", "").strip() or "eleven_v3",
            live_model=os.getenv("ELEVENLABS_LIVE_MODEL", "").strip() or "eleven_flash_v2_5",
            output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "").strip() or "mp3_44100_128",
            language=os.getenv("ELEVENLABS_LANGUAGE", "ko").strip(),
            stability=_optional_float("ELEVENLABS_STABILITY"),
            similarity=_optional_float("ELEVENLABS_SIMILARITY"),
            style=_optional_float("ELEVENLABS_STYLE"),
            speed=_optional_float("ELEVENLABS_SPEED"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.voice_id)

    def voice_settings(self, model: str) -> dict:
        """값을 넣은 항목만 보낸다 (나머지는 그 목소리에 저장된 ElevenLabs 기본값 사용)"""
        vs: dict[str, float] = {}
        if self.stability is not None:
            s = self.stability
            if model.startswith("eleven_v3"):
                # Eleven v3 의 안정성은 Creative(0.0) / Natural(0.5) / Robust(1.0) 세 단계라 가장 가까운 값으로 맞춤
                s = min((0.0, 0.5, 1.0), key=lambda v: abs(v - s))
            vs["stability"] = s
        if self.similarity is not None:
            vs["similarity_boost"] = self.similarity
        if self.style is not None:
            vs["style"] = self.style
        if self.speed is not None:
            vs["speed"] = self.speed
        return vs

    def request(self, text: str, model: str) -> tuple[str, dict, dict, dict]:
        """(url, params, headers, json 본문)"""
        url = f"{API_BASE}/v1/text-to-speech/{self.voice_id}"
        params = {"output_format": self.output_format}
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"}
        body: dict = {"text": text, "model_id": model}
        if self.language:
            body["language_code"] = self.language
        vs = self.voice_settings(model)
        if vs:
            body["voice_settings"] = vs
        return url, params, headers, body

    def cache_key(self, text: str, model: str) -> str:
        """같은 문장 + 목소리 + 모델 (+ 음성 설정·형식) 이면 같은 파일 이름"""
        ident = json.dumps(
            [text, self.voice_id, model, self.voice_settings(model), self.output_format, self.language],
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(ident.encode("utf-8")).hexdigest()[:32]

    def redact(self, message: str) -> str:
        return message.replace(self.api_key, "***") if self.api_key else message


class TTSError(Exception):
    """ElevenLabs 호출 실패. code 는 짧은 이유, message 는 사람이 읽을 설명 (API 키 없음)"""

    def __init__(self, code: str, message: str, upstream_status: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.upstream_status = upstream_status


def _error_message(r: httpx.Response) -> str:
    """ElevenLabs 오류 본문에서 설명만 꺼냄. 예: {"detail": {"status": "...", "message": "..."}}"""
    try:
        data = r.json()
    except ValueError:
        return r.text[:200].strip()
    detail = data.get("detail", data) if isinstance(data, dict) else data
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("status") or detail)[:300]
    if isinstance(detail, list) and detail:  # 422 검증 오류
        first = detail[0]
        if isinstance(first, dict):
            loc = ".".join(str(x) for x in first.get("loc", []))
            return f"{loc}: {first.get('msg', '')}"[:300]
    return str(detail)[:300]


def _check_response(cfg: TTSConfig, r: httpx.Response) -> bytes:
    if r.status_code != 200:
        raise TTSError("tts_error", cfg.redact(f"ElevenLabs 오류 {r.status_code}: {_error_message(r)}"), r.status_code)
    ctype = r.headers.get("content-type", "")
    if not r.content or not (ctype.startswith("audio/") or ctype.startswith("application/octet-stream")):
        raise TTSError("tts_bad_audio", f"ElevenLabs 응답이 오디오가 아닙니다 ({ctype or '형식 없음'}).")
    return r.content


async def synthesize_async(client: httpx.AsyncClient, cfg: TTSConfig, text: str, model: str, timeout: float) -> bytes:
    """비동기 생성 (서버용). 다른 요청을 막지 않는다."""
    url, params, headers, body = cfg.request(text, model)
    try:
        r = await client.post(url, params=params, headers=headers, json=body, timeout=timeout)
    except httpx.TimeoutException:
        raise TTSError("tts_timeout", f"ElevenLabs 응답이 {timeout:g}초 안에 오지 않았습니다.") from None
    except httpx.HTTPError as e:
        raise TTSError("tts_unreachable", f"ElevenLabs 연결 실패: {type(e).__name__}") from None
    return _check_response(cfg, r)


def synthesize_sync(client: httpx.Client, cfg: TTSConfig, text: str, model: str, timeout: float) -> bytes:
    """동기 생성 (pregen_tts.py 용)"""
    url, params, headers, body = cfg.request(text, model)
    try:
        r = client.post(url, params=params, headers=headers, json=body, timeout=timeout)
    except httpx.TimeoutException:
        raise TTSError("tts_timeout", f"ElevenLabs 응답이 {timeout:g}초 안에 오지 않았습니다.") from None
    except httpx.HTTPError as e:
        raise TTSError("tts_unreachable", f"ElevenLabs 연결 실패: {type(e).__name__}") from None
    return _check_response(cfg, r)


def write_atomic(path: Path, data: bytes) -> None:
    """임시 파일에 쓴 뒤 이름을 바꿔서, 쓰는 도중의 반쪽 파일이 제공되지 않게 함"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
