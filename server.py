"""
시각장애인용 물건 찾기 API 서버 (YOLOE-26 + FastAPI, 여러 사용자 동시 접속용)

실행 (워커는 반드시 1개: 모델·배치 대기열·lock을 한 프로세스에서 공유)
  python3 -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1

구조
  요청 → 인증 → 요청 제한(429) → 대기열 검사(503) → 이미지 디코딩 → 대기열
       → 배치 작업 스레드가 최대 BATCH_WAIT_MS 동안 모아서 한 번에 GPU 추론
       → 요청별로 targets와 일치하는 결과만 골라 응답

주요 환경변수 (.env.example 참고)
  API_KEY(필수), MODEL_NAME, ALLOWED_ORIGINS, DEFAULT_IMGSZ, HALF,
  BATCH_WAIT_MS, MAX_BATCH, MAX_QUEUE, REQUEST_TIMEOUT,
  DETECT_RATE_PER_SEC, DESCRIBE_RATE_PER_MIN, PARSE_RATE_PER_MIN, RATE_LIMIT_BY,
  GEMINI_API_KEY, GEMINI_MODEL, GEMINI_THINKING_LEVEL, PROMPT_FILE, TIMEZONE, LOG_DIR,
  ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_LIVE_MODEL, TTS_DAILY_CHAR_LIMIT
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import html
import json
import logging
import os
import queue
import re
import secrets
import threading
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import httpx
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from elevenlabs_tts import TTSConfig, TTSError, synthesize_async, write_atomic
from vocab import ALIASES, BACKGROUND, BASE_VOCAB, SYNONYM_GROUPS


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "yoloe-26l-seg.pt")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

DEFAULT_IMGSZ = env_int("DEFAULT_IMGSZ", 640)
DEFAULT_CONF = env_float("DEFAULT_CONF", 0.25)
HALF = env_bool("HALF", True)  # GPU일 때만 적용
MAX_DET = env_int("MAX_DET", 100)
WARMUP_RUNS = 3

BATCH_WAIT_MS = env_float("BATCH_WAIT_MS", 20)  # 첫 요청 도착 후 최대 몇 ms 동안 더 모을지
MAX_BATCH = env_int("MAX_BATCH", 16)  # 한 번에 추론할 최대 이미지 수
MAX_QUEUE = env_int("MAX_QUEUE", 48)  # 처리 대기 중인 요청이 이만큼이면 503
REQUEST_TIMEOUT = env_float("REQUEST_TIMEOUT", 1.5)  # 대기열에서 이보다 오래 기다린 사진은 버림(503)

DETECT_RATE_PER_SEC = env_int("DETECT_RATE_PER_SEC", 5)
DESCRIBE_RATE_PER_MIN = env_int("DESCRIBE_RATE_PER_MIN", 10)
PARSE_RATE_PER_MIN = env_int("PARSE_RATE_PER_MIN", 20)  # /parse: 기기(X-Client-Id)별, 없으면 IP별
PARSE_CACHE_SIZE = env_int("PARSE_CACHE_SIZE", 2000)  # 같은 문장 결과를 메모리에 저장하는 최대 개수
PARSE_MAX_CHARS = env_int("PARSE_MAX_CHARS", 200)
RATE_LIMIT_BY = os.getenv("RATE_LIMIT_BY", "ip+client")  # "ip" 또는 "ip+client" (X-Client-Id 헤더가 있으면 함께 사용)

MAX_TARGETS = env_int("MAX_TARGETS", 10)
MAX_RESULTS_ALL = env_int("MAX_RESULTS_ALL", 20)  # targets 없이 전체 탐지할 때 최대 결과 수
MAX_EXTRA_VOCAB = env_int("MAX_EXTRA_VOCAB", 100)  # 요청으로 추가되는 단어 최대 개수 (넘으면 오래 안 쓴 것부터 제거)
SYNONYMS = env_bool("SYNONYMS", True)
DEDUPE_IOU = env_float("DEDUPE_IOU", 0.6)  # 같은 물건이 cup/mug로 두 번 잡히면 하나만 남김
MAX_IMAGE_BYTES = int(env_float("MAX_IMAGE_MB", 8) * 1024 * 1024)

# ---- 색상 구분 (/detect 후처리) ---------------------------------------------
# 요청 이름에 색상 단어가 있으면 박스 중앙 50% 영역의 HSV 픽셀 비율로 색을 검증한다.
COLOR_MIN_RATIO = env_float("COLOR_MIN_RATIO", 0.30)  # 요청 색 픽셀이 이 비율 이상이어야 남김
COLOR_SHAPE_ONLY = env_bool("COLOR_SHAPE_ONLY", True)  # YOLOE에는 색을 뺀 이름(예: tumbler)만 보내 형태로 찾기
DOMINANT_MIN_RATIO = 0.20  # 가장 많은 색이 이보다 적으면 "unknown"
COLOR_KEYWORDS = {"black", "white", "gray", "grey", "silver", "red", "orange", "yellow",
                  "green", "blue", "purple", "pink", "brown"}
COLOR_ALIASES = {"grey": "gray", "silver": "gray"}
# OpenCV HSV 기준 (H 0~179, S·V 0~255). 색마다 (H최소, H최대, S최소, S최대, V최소, V최대) 범위 목록, 범위끼리는 OR
COLOR_HSV_RULES: dict[str, list[tuple[int, int, int, int, int, int]]] = {
    "black":  [(0, 179, 0, 255, 0, 59)],                                 # V < 60
    "white":  [(0, 179, 0, 54, 181, 255)],                               # S < 55, V > 180
    "gray":   [(0, 179, 0, 54, 60, 180)],                                # S < 55, 60 ≤ V ≤ 180
    "red":    [(0, 10, 50, 255, 50, 255), (165, 179, 50, 255, 50, 255)],  # H ≤ 10 또는 H ≥ 165
    "orange": [(11, 22, 50, 255, 50, 255)],
    "yellow": [(23, 35, 50, 255, 50, 255)],
    "green":  [(36, 85, 50, 255, 50, 255)],
    "blue":   [(86, 130, 50, 255, 50, 255)],
    "purple": [(131, 164, 50, 255, 50, 255)],
    "pink":   [(0, 10, 30, 150, 150, 255), (150, 179, 30, 150, 150, 255)],  # 보라와 별도로 판별
    "brown":  [(5, 25, 50, 255, 40, 160)],
}
# 분홍은 빨강·보라 기준과 겹치므로, 빨강·보라를 셀 때는 분홍 픽셀을 뺀다 (빨간 컵을 찾을 때 분홍 컵이 남지 않게)
COLOR_EXCLUDE = {"red": ("pink",), "purple": ("pink",)}
# 갈색↔주황·빨강처럼 겹치는 색은 비율이 거의 같을 때 더 구체적인 색을 고름 (dominant_color 에만 사용)
COLOR_SPECIFIC_BONUS = {"pink": 0.05, "brown": 0.05}
COLOR_SAMPLE_MAX_SIDE = 160  # 중앙 영역이 크면 이 크기로 줄여서 계산 (비율은 거의 같고 빠름)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "minimal")  # 가장 낮은 추론 수준
GEMINI_TIMEOUT = env_float("GEMINI_TIMEOUT", 20)
DESCRIBE_MAX_SIDE = env_int("DESCRIBE_MAX_SIDE", 1024)  # 큰 이미지는 줄여서 전송
PROMPT_FILE = os.getenv("PROMPT_FILE", "prompt.txt")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Seoul")

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", str(Path(__file__).with_name("static"))))  # /app 으로 제공할 폴더
STATS_WINDOW_SEC = env_int("STATS_WINDOW_SEC", 300)
ACCESS_LOG = env_bool("ACCESS_LOG", False)  # 초당 수십 건이면 uvicorn 접속 로그가 너무 많아 기본으로 끔

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("yoloe-api")


class ApiError(Exception):
    """이유(code, message)를 담아 JSON으로 돌려줄 에러"""

    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


def ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


# ---------------------------------------------------------------------------
# 클라이언트 식별 (cloudflared 뒤에서는 모든 접속이 127.0.0.1로 보임)
# ---------------------------------------------------------------------------
_HASH_SALT = secrets.token_hex(8)  # CSV에 IP를 그대로 남기지 않도록 실행마다 바뀌는 소금값


def client_ip(request: Request) -> str:
    """CF-Connecting-IP → X-Forwarded-For 첫 번째 → 접속 IP 순서.
    주의: 서버가 127.0.0.1에만 열려 있어 cloudflared만 접속할 수 있다는 전제에서 헤더를 믿는다."""
    h = request.headers
    ip = h.get("cf-connecting-ip", "").strip()
    if not ip:
        ip = h.get("x-forwarded-for", "").split(",")[0].strip()
    if not ip:
        ip = request.client.host if request.client else "unknown"
    return ip


def client_id(request: Request) -> str:
    cid = request.headers.get("x-client-id", "").strip()[:64]
    return cid


def anon(value: str) -> str:
    return hashlib.sha256((_HASH_SALT + value).encode()).hexdigest()[:10] if value else ""


# ---------------------------------------------------------------------------
# 요청 제한 (슬라이딩 윈도우)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, limit: int, window_sec: float):
        self.limit = limit
        self.window = window_sec
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self.lock:
            dq = self.hits[key]
            while dq and now - dq[0] >= self.window:
                dq.popleft()
            if len(dq) >= self.limit:
                return False, max(0.0, self.window - (now - dq[0]))
            dq.append(now)
            if now - self._last_cleanup > 60:  # 오래된 키 정리
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] >= self.window]:
                    del self.hits[k]
                self._last_cleanup = now
            return True, 0.0


detect_limiter = RateLimiter(DETECT_RATE_PER_SEC, 1.0)
describe_limiter = RateLimiter(DESCRIBE_RATE_PER_MIN, 60.0)
parse_limiter = RateLimiter(PARSE_RATE_PER_MIN, 60.0)


def limit_key(ip: str, cid: str) -> str:
    return f"{ip}|{cid}" if (RATE_LIMIT_BY == "ip+client" and cid) else ip


# ---------------------------------------------------------------------------
# 통계 (/stats) 와 CSV 기록
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self, window_sec: int):
        self.window = window_sec
        self.events: deque = deque()  # (시각, endpoint, status, total_ms, ip, client)
        self.batches: deque = deque()  # (시각, 배치 크기)
        self.lock = threading.Lock()

    def record(self, endpoint: str, status: int, total_ms: float, ip: str, cid: str) -> None:
        with self.lock:
            self.events.append((time.time(), endpoint, status, total_ms, ip, cid))

    def record_batch(self, size: int) -> None:
        with self.lock:
            self.batches.append((time.time(), size))

    def snapshot(self) -> dict:
        cutoff = time.time() - self.window
        with self.lock:
            while self.events and self.events[0][0] < cutoff:
                self.events.popleft()
            while self.batches and self.batches[0][0] < cutoff:
                self.batches.popleft()
            events = list(self.events)
            batches = [b[1] for b in self.batches]

        def summarize(endpoint: str) -> dict:
            ev = [e for e in events if e[1] == endpoint]
            ok = np.array([e[3] for e in ev if e[2] == 200], dtype=float)
            return {
                "requests": len(ev),
                "ok": int(len(ok)),
                "avg_ms": round(float(ok.mean()), 1) if len(ok) else None,
                "p50_ms": round(float(np.percentile(ok, 50)), 1) if len(ok) else None,
                "p95_ms": round(float(np.percentile(ok, 95)), 1) if len(ok) else None,
                "rate_limited_429": sum(1 for e in ev if e[2] == 429),
                "rejected_503": sum(1 for e in ev if e[2] == 503),
                "other_errors": sum(1 for e in ev if e[2] not in (200, 429, 503)),
            }

        return {
            "window_sec": self.window,
            "detect": summarize("detect"),
            "describe": summarize("describe"),
            "parse": summarize("parse"),
            "tts": summarize("tts"),
            "batch": {
                "count": len(batches),
                "avg_size": round(float(np.mean(batches)), 2) if batches else None,
                "max_size": max(batches) if batches else None,
            },
            "unique_ips": len({e[4] for e in events}),
            "unique_clients": len({(e[4], e[5]) for e in events}),
        }


class CsvLogger:
    """요청별 처리 시간을 CSV로 기록 (별도 스레드에서 써서 요청 처리를 막지 않음)"""

    COLUMNS = [
        "time", "endpoint", "status", "code", "ip_hash", "client_hash", "targets", "n_detections",
        "batch_size", "imgsz", "decode_ms", "vocab_ms", "queue_wait_ms", "inference_ms", "total_ms", "chars",
    ]

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="csv-writer", daemon=True).start()

    def write(self, **row: Any) -> None:
        row.setdefault("time", datetime.now().isoformat(timespec="milliseconds"))
        self.q.put(row)

    def _run(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
            w.writeheader()
            f.flush()
            while True:
                w.writerow(self.q.get())
                while True:  # 쌓인 것을 한꺼번에 쓰고 flush
                    try:
                        w.writerow(self.q.get_nowait())
                    except queue.Empty:
                        break
                f.flush()


stats = Stats(STATS_WINDOW_SEC)
csv_log: CsvLogger | None = None


def record(endpoint: str, status: int, code: str, t_start: float, ip: str, cid: str, **extra: Any) -> None:
    total = ms(time.perf_counter() - t_start)
    stats.record(endpoint, status, total, ip, cid)
    if csv_log:
        csv_log.write(
            endpoint=endpoint, status=status, code=code, ip_hash=anon(ip), client_hash=anon(cid), total_ms=total, **extra
        )


# ---------------------------------------------------------------------------
# 물체 이름 처리
# ---------------------------------------------------------------------------
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9 \-']{0,49}$")
_SYNONYM_INDEX: dict[str, list[str]] = defaultdict(list)
for _g in SYNONYM_GROUPS:
    for _w in _g:
        for _m in _g:
            if _m not in _SYNONYM_INDEX[_w]:
                _SYNONYM_INDEX[_w].append(_m)


def parse_targets(raw: str, known: set[str]) -> list[str]:
    """'Phone, keys' → ['cell phone', 'key'] : 소문자·공백 정리, 별칭/복수형 변환, 중복 제거"""
    out: list[str] = []
    for part in raw.split(","):
        name = " ".join(part.strip().lower().split())
        if not name:
            continue
        if not _NAME_RE.match(name):
            raise ApiError(400, "invalid_targets", f"물체 이름 '{name}' 은(는) 영문/숫자/공백/-/' 로 50자 이내여야 합니다.")
        if name in ALIASES:
            name = ALIASES[name]
        elif name not in known and name.endswith("s") and name[:-1] in known:
            name = name[:-1]  # 간단한 복수형 처리 (keys → key)
        if name not in out:
            out.append(name)
    if not out:
        raise ApiError(400, "invalid_targets", "targets 에 찾을 물체 이름을 하나 이상 넣어주세요. 예: mug,cup")
    if len(out) > MAX_TARGETS:
        raise ApiError(400, "invalid_targets", f"물체 이름은 최대 {MAX_TARGETS}개까지 가능합니다.")
    return out


# targets 없이 부르면 기본 어휘에서 배경 물체를 뺀 이름 전체를 찾는다 (이름 → 이름)
ALL_MODE_TARGETS: dict[str, str] = {w: w for w in BASE_VOCAB if w not in BACKGROUND}


def expand_targets(requested: list[str]) -> dict[str, str]:
    """찾을 어휘 이름 → 사용자가 요청한 이름. 동의어 그룹이 켜져 있으면 같은 그룹 이름도 포함"""
    mapping: dict[str, str] = {}
    for t in requested:
        mapping.setdefault(t, t)
    if SYNONYMS:
        for t in requested:
            for m in _SYNONYM_INDEX.get(t, []):
                mapping.setdefault(m, t)
    return mapping


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise ApiError(400, "invalid_image", "image 파일이 비어 있습니다.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ApiError(413, "image_too_large", f"이미지가 너무 큽니다. 최대 {MAX_IMAGE_BYTES // (1024 * 1024)}MB.")
    # IMREAD_COLOR 는 EXIF 회전을 반영 → 박스 좌표는 '똑바로 세운 사진' 기준
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError(400, "invalid_image", "이미지를 읽을 수 없습니다. JPEG 파일을 보내주세요.")
    return img


# ---------------------------------------------------------------------------
# 색상 검증 (YOLOE 형태 탐지 → 박스 중앙 50% HSV 픽셀 비율로 색 확인)
# ---------------------------------------------------------------------------
def normalize_color(word: str) -> str | None:
    w = word.lower()
    if w not in COLOR_KEYWORDS:
        return None
    return COLOR_ALIASES.get(w, w)  # grey, silver → gray


def extract_color_from_targets(targets: list[str]) -> str | None:
    """["black tumbler", "tumbler"] → "black". 색상 단어가 없으면 None"""
    for t in targets:
        for word in re.split(r"[\s\-]+", t.lower()):
            color = normalize_color(word)
            if color:
                return color
    return None


def strip_color_words(name: str) -> str:
    """'black tumbler' → 'tumbler' (색만 있는 이름이면 그대로)"""
    words = [w for w in name.split() if w.lower() not in COLOR_KEYWORDS]
    return " ".join(words) or name


def color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """HSV 이미지에서 해당 색 픽셀이면 255인 마스크"""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h1, h2, s1, s2, v1, v2 in COLOR_HSV_RULES[color]:
        mask |= cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
    return mask


def match_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """일치율 계산용 마스크: color_mask 에서 더 구체적인 색(COLOR_EXCLUDE) 픽셀을 뺀 것"""
    mask = color_mask(hsv, color)
    for other in COLOR_EXCLUDE.get(color, ()):
        mask &= cv2.bitwise_not(color_mask(hsv, other))
    return mask


def _center_hsv(bgr_img: np.ndarray, box: dict) -> np.ndarray | None:
    """정규화 박스(x, y, w, h)에서 상하좌우 25%씩 뺀 중앙 50% 영역을 HSV로. 비면 None"""
    H, W = bgr_img.shape[:2]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    x1, x2 = int(round((x + 0.25 * w) * W)), int(round((x + 0.75 * w) * W))
    y1, y2 = int(round((y + 0.25 * h) * H)), int(round((y + 0.75 * h) * H))
    x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
    y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = bgr_img[y1:y2, x1:x2]
    scale = COLOR_SAMPLE_MAX_SIDE / max(crop.shape[:2])
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)


def compute_color_match_ratio(bgr_img: np.ndarray, box: dict, target_color: str) -> float:
    """박스 중앙 50% 영역에서 target_color 픽셀 비율 (0.0~1.0). 영역이 비면 0.0"""
    hsv = _center_hsv(bgr_img, box)
    if hsv is None or hsv.size == 0:
        return 0.0
    return float(np.count_nonzero(match_mask(hsv, target_color))) / (hsv.shape[0] * hsv.shape[1])


def color_ratios(bgr_img: np.ndarray, box: dict) -> dict[str, float]:
    hsv = _center_hsv(bgr_img, box)
    if hsv is None or hsv.size == 0:
        return {}
    n = hsv.shape[0] * hsv.shape[1]
    return {c: float(np.count_nonzero(match_mask(hsv, c))) / n for c in COLOR_HSV_RULES}


def dominant_color(bgr_img: np.ndarray, box: dict) -> str:
    """중앙 50% 영역에서 비율이 가장 높은 색. 그 비율이 0.2 미만이면 "unknown" """
    ratios = color_ratios(bgr_img, box)
    if not ratios:
        return "unknown"
    best = max(ratios, key=lambda c: ratios[c] + COLOR_SPECIFIC_BONUS.get(c, 0.0))
    return best if ratios[best] >= DOMINANT_MIN_RATIO else "unknown"


def filter_detections_by_color(bgr_img: np.ndarray, detections: list[dict], targets: list[str]):
    """요청 색과 맞는 박스만 남김. 반환: (남은 detections, rejected, 요청 색 또는 None)
    색 요청이 없으면 detections 를 손대지 않고 그대로 돌려준다."""
    color = extract_color_from_targets(targets)
    if color is None:
        return detections, [], None
    kept: list[dict] = []
    rejected: list[dict] = []
    for d in detections:
        ratio = compute_color_match_ratio(bgr_img, d["box"], color)
        if ratio >= COLOR_MIN_RATIO:
            kept.append({**d, "score": round(d["score"] * (0.6 + 0.4 * ratio), 4), "color_match": round(ratio, 3)})
        elif len(rejected) < 5:
            rejected.append({"label": d["label"], "box": d["box"], "dominant_color": dominant_color(bgr_img, d["box"])})
    kept.sort(key=lambda d: d["score"], reverse=True)
    return kept, rejected, color


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 모델 + 배치 추론
# ---------------------------------------------------------------------------
@dataclass
class Job:
    img: np.ndarray
    label_to_target: dict[str, str]  # 찾을 어휘 이름 → 요청 이름
    conf: float
    imgsz: int
    t_enqueue: float
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    max_results: int | None = None  # 전체 탐지 모드에서만 개수 제한


class Detector:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = HALF and self.device == "cuda"
        self.model = None
        self.ready = False
        self.load_error: str | None = None
        self.precision_kw: dict = {}

        self.model_lock = threading.Lock()  # 모델 사용(추론·어휘 변경)은 한 번에 하나
        self.vocab: list[str] = []
        self.name_to_id: dict[str, int] = {}
        self.extra_vocab: OrderedDict[str, None] = OrderedDict()  # 요청으로 추가된 단어 (LRU)
        self.extra_lock = threading.Lock()
        self.pe_cache: dict[str, Any] = {}  # 단어 → 텍스트 임베딩 (1, 1, D)
        self._text_model = None

        self.q: queue.Queue[Job] = queue.Queue()
        self._pending = 0
        self._pending_lock = threading.Lock()

    @property
    def pending(self) -> int:
        return self._pending

    # ---- 시작 시 1회 -------------------------------------------------------
    def load(self) -> None:
        try:
            t0 = time.perf_counter()
            from ultralytics import YOLOE
            from ultralytics.utils import DEFAULT_CFG_DICT

            if self.half:  # 최신 ultralytics는 half 대신 quantize=16
                self.precision_kw = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}

            log.info("모델 로딩: %s device=%s half=%s", self.model_name, self.device, self.half)
            model = YOLOE(self.model_name)
            model.to(self.device)
            self.model = model
            t1 = time.perf_counter()

            with self.model_lock:
                self._apply_vocab(list(BASE_VOCAB))
                t2 = time.perf_counter()
                # 워밍업 3회: 배치 크기를 바꿔가며 (1장, 중간, 최대)
                sizes = [1, max(1, MAX_BATCH // 2), MAX_BATCH][:WARMUP_RUNS]
                for n in sizes:
                    imgs = [np.full((480, 640, 3), 114, dtype=np.uint8) for _ in range(n)]
                    self._predict(imgs, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ)
                t3 = time.perf_counter()

            threading.Thread(target=self._worker, name="batch-worker", daemon=True).start()
            self.ready = True
            log.info(
                "모델 준비 완료: 어휘 %d개 | load=%.0fms vocab=%.0fms warmup=%.0fms(%s)",
                len(self.vocab), ms(t1 - t0), ms(t2 - t1), ms(t3 - t2), sizes,
            )
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            log.exception("모델 로딩 실패")

    # ---- 텍스트 임베딩 (단어별 캐시) --------------------------------------
    def _encode_text(self, names: list[str]):
        inner = self.model.model
        with torch.no_grad():
            try:
                # 텍스트 인코더를 보관해 두고 필요할 때만 붙여서 사용 (predictor가 모델을 복사할 때 함께 복사되지 않도록 떼어냄)
                if self._text_model is not None:
                    inner.clip_model = self._text_model
                pe = inner.get_text_pe(names, cache_clip_model=True)
                self._text_model = getattr(inner, "clip_model", None)
            except TypeError:
                pe = self.model.get_text_pe(names)
            finally:
                if "clip_model" in getattr(inner, "_modules", {}) or "clip_model" in vars(inner):
                    delattr(inner, "clip_model")
        return pe.to(self.device)

    def _apply_vocab(self, names: list[str]) -> int:
        """어휘 전체를 set_classes() 로 설정. 반환: 새로 계산한 임베딩 수 (model_lock 안에서 호출)"""
        missing = [n for n in names if n not in self.pe_cache]
        if missing:
            pe = self._encode_text(missing)
            for i, n in enumerate(missing):
                self.pe_cache[n] = pe[:, i : i + 1, :]
        embeddings = torch.cat([self.pe_cache[n] for n in names], dim=1)
        self.model.set_classes(list(names), embeddings)
        self.vocab = list(names)
        self.name_to_id = {n: i for i, n in enumerate(names)}
        return len(missing)

    def _predict(self, imgs: list[np.ndarray], conf: float, imgsz: int, classes: list[int] | None = None):
        return self.model.predict(
            source=imgs, conf=conf, imgsz=imgsz, device=self.device, classes=classes,
            max_det=MAX_DET, verbose=False, **self.precision_kw,
        )

    # ---- 어휘 추가 --------------------------------------------------------
    def touch(self, words) -> None:
        with self.extra_lock:
            for w in words:
                if w in self.extra_vocab:
                    self.extra_vocab.move_to_end(w)

    def missing_words(self, words) -> list[str]:
        known = self.name_to_id
        return [w for w in words if w not in known]

    def ensure_words(self, words: list[str]) -> float:
        """어휘에 없는 단어를 추가하고 set_classes() 재설정. 이때만 추론을 잠시 멈춘다. 반환: 걸린 시간(ms)"""
        t0 = time.perf_counter()
        with self.model_lock:
            missing = self.missing_words(words)
            if not missing:
                return ms(time.perf_counter() - t0)
            with self.extra_lock:
                for w in missing:
                    self.extra_vocab[w] = None
                # 너무 많아지면 오래 안 쓴 단어부터 제거 (이번 요청 단어는 유지)
                while len(self.extra_vocab) > MAX_EXTRA_VOCAB:
                    old = next((k for k in self.extra_vocab if k not in words), None)
                    if old is None:
                        break
                    del self.extra_vocab[old]
                extras = [w for w in self.extra_vocab if w not in BASE_VOCAB]
            n_new = self._apply_vocab(list(BASE_VOCAB) + extras)
            t1 = time.perf_counter()
            # set_classes()가 predictor를 초기화하므로 여기서 미리 한 번 추론해 다시 준비시킴
            self._predict([np.full((480, 640, 3), 114, dtype=np.uint8)], conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ)
            t2 = time.perf_counter()
        log.info(
            "어휘 추가 %s → 총 %d개 (새 임베딩 %d) | set_classes=%.0fms predictor 재준비=%.0fms",
            missing, len(self.vocab), n_new, ms(t1 - t0), ms(t2 - t1),
        )
        return ms(time.perf_counter() - t0)

    # ---- 대기열 -----------------------------------------------------------
    def submit(self, job: Job) -> None:
        with self._pending_lock:
            if self._pending >= MAX_QUEUE:
                raise ApiError(503, "queue_full", "요청이 많아 지금은 처리할 수 없습니다. 기기 내 인식을 사용하세요.")
            self._pending += 1
        self.q.put(job)

    def _finish(self, job: Job, result: Any = None, error: Exception | None = None) -> None:
        with self._pending_lock:
            self._pending -= 1

        def _set() -> None:
            if job.future.done():  # 클라이언트 쪽에서 이미 시간 초과로 포기한 경우
                return
            if error is not None:
                job.future.set_exception(error)
            else:
                job.future.set_result(result)

        job.loop.call_soon_threadsafe(_set)

    def _worker(self) -> None:
        wait = BATCH_WAIT_MS / 1000
        while True:
            first = self.q.get()
            batch = [first]
            deadline = first.t_enqueue + wait  # 이미 오래 기다린 요청이면 바로 처리
            while len(batch) < MAX_BATCH:
                remaining = deadline - time.perf_counter()
                try:
                    batch.append(self.q.get(timeout=remaining) if remaining > 0 else self.q.get_nowait())
                except queue.Empty:
                    break
            try:
                self._run_batch(batch)
            except Exception as e:  # 작업 스레드는 절대 죽지 않게
                log.exception("배치 처리 오류")
                for j in batch:
                    self._finish(j, error=ApiError(500, "inference_error", f"추론 오류: {type(e).__name__}"))

    def _run_batch(self, batch: list[Job]) -> None:
        t_start = time.perf_counter()
        live: list[Job] = []
        for j in batch:
            if j.future.cancelled():
                self._finish(j)
            elif t_start - j.t_enqueue > REQUEST_TIMEOUT:
                self._finish(j, error=ApiError(503, "timeout", "대기 시간이 길어 오래된 사진을 버렸습니다. 기기 내 인식을 사용하세요."))
            else:
                live.append(j)
        if not live:
            return
        stats.record_batch(len(live))

        groups: dict[int, list[Job]] = defaultdict(list)  # imgsz가 같은 요청끼리 배치
        for j in live:
            groups[j.imgsz].append(j)

        for imgsz, jobs in groups.items():
            t0 = time.perf_counter()
            with self.model_lock:
                name_to_id = self.name_to_id
                ids_per_job = [{name_to_id[n] for n in j.label_to_target if n in name_to_id} for j in jobs]
                union = sorted(set().union(*ids_per_job))
                results = (
                    self._predict([j.img for j in jobs], conf=min(j.conf for j in jobs), imgsz=imgsz, classes=union)
                    if union else [None] * len(jobs)
                )
                # GPU 텐서를 lock 안에서 numpy로 꺼내 둔다
                extracted = []
                for r in results:
                    if r is None or r.boxes is None or len(r.boxes) == 0:
                        extracted.append(None)
                    else:
                        extracted.append((
                            r.boxes.xyxyn.float().cpu().numpy(),
                            r.boxes.conf.float().cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.names,
                        ))
            t1 = time.perf_counter()

            for j, ids, ex in zip(jobs, ids_per_job, extracted):
                dets = self._build_detections(j, ids, ex)
                self._finish(j, result={
                    "detections": dets,
                    "queue_wait": ms(t0 - j.t_enqueue),
                    "inference": ms(t1 - t0),
                    "batch_size": len(jobs),
                })
            log.debug("batch size=%d imgsz=%d infer=%.1fms", len(jobs), imgsz, ms(t1 - t0))

    @staticmethod
    def _build_detections(job: Job, ids: set[int], ex) -> list[dict]:
        if ex is None:
            return []
        xyxyn, scores, classes, names = ex
        order = np.argsort(-scores)  # 신뢰도 높은 순
        kept: list[np.ndarray] = []
        dets: list[dict] = []
        for k in order:
            c, s = int(classes[k]), float(scores[k])
            if c not in ids or s < job.conf:
                continue
            box = np.clip(xyxyn[k], 0.0, 1.0)
            if any(iou(box, kb) >= DEDUPE_IOU for kb in kept):  # 같은 물건이 다른 이름으로 또 잡힌 경우
                continue
            kept.append(box)
            if job.max_results is not None and len(dets) >= job.max_results:
                break
            label = names[c]
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append({
                "label": label,
                "target": job.label_to_target.get(label, label),
                "score": round(s, 4),
                "box": {"x": round(x1, 4), "y": round(y1, 4), "w": round(x2 - x1, 4), "h": round(y2 - y1, 4)},
            })
        return dets


detector = Detector(MODEL_NAME)


# ---------------------------------------------------------------------------
# Gemini 중계
# ---------------------------------------------------------------------------
_http: httpx.AsyncClient | None = None
_thinking_supported = True
_prompt_cache: tuple[float, str] = (-1.0, "")  # (파일 수정 시각, 내용)
MAX_PROMPT_CHARS = env_int("MAX_PROMPT_CHARS", 4000)
TEST_PAGE = env_bool("TEST_PAGE", True)  # GET /describe-test 페이지 제공 여부


def default_prompt() -> str:
    """prompt.txt 내용 ('#' 줄 제외). 파일이 바뀌면 서버 재시작 없이 다시 읽는다."""
    global _prompt_cache
    path = Path(PROMPT_FILE)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return ""
    if mtime != _prompt_cache[0]:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
        _prompt_cache = (mtime, "\n".join(lines).strip())
        log.info("프롬프트 파일을 읽었습니다: %s (%d자)", path, len(_prompt_cache[1]))
    return _prompt_cache[1]


def time_of_day(now: datetime) -> str:
    """호출 시점의 시간대: 05~11시 아침, 12~17시 오후, 18~04시 저녁"""
    h = now.hour
    return "아침" if 5 <= h < 12 else "오후" if 12 <= h < 18 else "저녁"


def render_prompt(template: str) -> tuple[str, str]:
    """자리표시자 치환. 반환: (완성된 프롬프트, 시간대)"""
    now = datetime.now(ZoneInfo(TIMEZONE))
    tod = time_of_day(now)
    weekday = "월화수목금토일"[now.weekday()]
    # str.format 대신 replace: 프롬프트 안의 JSON 중괄호와 충돌하지 않게
    text = (
        template.replace("{현재시간}", tod)
        .replace("{time_of_day}", tod)
        .replace("{now}", now.strftime(f"%Y-%m-%d %H:%M ({weekday})"))
    )
    return text, tod


def prepare_describe_image(data: bytes) -> bytes:
    """이미지 확인 후, 너무 크면 줄여서 JPEG로 다시 인코딩"""
    img = decode_image(data)
    h, w = img.shape[:2]
    if max(h, w) <= DESCRIBE_MAX_SIDE and data[:3] == b"\xff\xd8\xff":
        return data
    scale = min(1.0, DESCRIBE_MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ApiError(400, "invalid_image", "이미지를 JPEG로 변환하지 못했습니다.")
    return buf.tobytes()


def parse_json_text(text: str) -> Any | None:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    for candidate in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        if not candidate:
            continue
        try:
            v = json.loads(candidate)
            if isinstance(v, (dict, list)):
                return v
        except ValueError:
            pass
    return None


async def call_gemini(jpeg: bytes | None, prompt: str) -> str:
    """Gemini 호출. jpeg 가 None 이면 텍스트만 보낸다 (/parse)."""
    global _thinking_supported
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    gen_cfg: dict[str, Any] = {"responseMimeType": "application/json"}
    if _thinking_supported and GEMINI_THINKING_LEVEL:
        gen_cfg["thinkingConfig"] = {"thinkingLevel": GEMINI_THINKING_LEVEL}
    body = {
        "contents": [{
            "role": "user",
            "parts": (
                [{"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}] if jpeg else []
            ) + [{"text": prompt}],
        }],
        "generationConfig": gen_cfg,
    }
    try:
        r = await _http.post(url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=body)
    except httpx.TimeoutException:
        raise ApiError(504, "vlm_timeout", f"Gemini 응답이 {GEMINI_TIMEOUT:.0f}초 안에 오지 않았습니다.")
    except httpx.HTTPError as e:
        raise ApiError(502, "vlm_unreachable", f"Gemini 연결 실패: {type(e).__name__}")

    if r.status_code == 400 and "thinking" in r.text.lower() and "thinkingConfig" in gen_cfg:
        # 모델이 이 추론 수준을 지원하지 않으면 설정 없이 다시 (이후 요청도 생략)
        log.warning("Gemini가 thinkingLevel=%s 를 거부해 설정 없이 다시 요청합니다: %s", GEMINI_THINKING_LEVEL, r.text[:200])
        _thinking_supported = False
        return await call_gemini(jpeg, prompt)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", "")
        except ValueError:
            msg = r.text[:200]
        raise ApiError(502, "vlm_error", f"Gemini 오류 {r.status_code}: {msg}")

    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "응답 없음")
        raise ApiError(502, "vlm_blocked", f"Gemini가 응답하지 않았습니다: {reason}")
    parts = (cands[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


# ---------------------------------------------------------------------------
# 음성 명령 해석 (/parse)
# ---------------------------------------------------------------------------
PARSE_INTENTS = ("find", "where", "how", "stop", "unknown")

# {text} 자리에 사용자 문장이 JSON 문자열로 들어간다 (따옴표·줄바꿈이 섞여도 지시문과 섞이지 않게)
PARSE_PROMPT = """너는 시각장애인용 물건 찾기 앱의 음성 명령 해석기야.
사용자가 한국어로 말한 문장을 보고 아래 JSON만 출력해. 다른 말은 하지 마.

{"intent": "find" | "where" | "how" | "stop" | "unknown", "object_ko": "물건 이름" 또는 null, "targets": ["english name", ...]}

intent
- find: 물건을 찾아 달라는 요청 (예: "컵 찾아줘", "리모컨 좀 찾아 줄래")
- where: 물건이 어디 있는지 묻는 질문 (예: "내 열쇠 어디 있어?")
- how: 물건을 어떻게 잡거나 쓰는지 묻는 질문 (예: "이거 어떻게 잡아?", "손잡이 어느 쪽이야?")
- stop: 안내를 멈추라는 말 (예: "그만", "멈춰", "됐어", "찾았어")
- unknown: 위에 해당하지 않거나 알아들을 수 없는 말

object_ko와 targets는 intent와 상관없이 채워
- find, where, how, stop 어느 경우든 문장에 물건 이름이 나오면 object_ko와 targets를 반드시 둘 다 채워
- "어디 있어?"(where)도 물건 이름이 있으면 object_ko와 targets를 비우지 마
- 물건 이름 없이 "이거", "그거"만 말했으면 object_ko는 null, targets는 빈 배열

object_ko
- 사용자가 말한 물건을 색·재질 같은 겉모습 표현까지 포함해 한국어로 (예: "검은 텀블러")
- "내", "왼쪽에 있는", "아까 그" 같은 소유·위치·관계 표현은 빼
- 물건이 없으면 null

targets: 물체 인식 모델에 넣을 영어 물체 이름
- 1~3개. 사용자가 말한 표현을 그대로 옮긴 이름을 먼저, 그다음 더 일반적인 이름 순서
- 색, 재질 같은 겉모습은 넣어도 되지만 my, left, on the table 같은 소유·위치·관계 표현은 빼
- 영어 소문자, 단수형, 관사(a, the) 없이
- 물리적인 물건이 아니면 (예: 날씨, 시간, 기분) 빈 배열

색상 (서버가 사진 색으로 다시 확인하므로 색 단어를 절대 빼먹지 마)
- 문장에 색상 형용사(빨간, 검은, 까만, 하얀, 흰, 파란, 노란, 초록, 회색, 은색, 주황, 보라, 분홍, 갈색 등)가 있으면
  targets의 첫 번째는 반드시 "영어 색 단어 + 물건 이름", 두 번째는 색 없는 물건 이름
- 영어 색 단어는 이 중에서만: black, white, gray, grey, silver, red, orange, yellow, green, blue, purple, pink, brown
  (까만·검정 → black, 흰·하얀 → white, 은색 → silver, 분홍·핑크 → pink, 남색·하늘색 → blue, 베이지 → brown)
- object_ko에도 색을 그대로 넣어 (예: "빨간 컵")

예시
"내 검은 텀블러 어디 있어?" → {"intent": "where", "object_ko": "검은 텀블러", "targets": ["black tumbler", "tumbler"]}
"빨간 컵 찾아줘" → {"intent": "find", "object_ko": "빨간 컵", "targets": ["red cup", "cup"]}
"분홍색 지갑 어디 있어?" → {"intent": "where", "object_ko": "분홍색 지갑", "targets": ["pink wallet", "wallet"]}
"갈색 가방 찾아줘" → {"intent": "find", "object_ko": "갈색 가방", "targets": ["brown bag", "bag"]}
"까만 리모컨 어디야?" → {"intent": "where", "object_ko": "까만 리모컨", "targets": ["black remote control", "remote control"]}
"약통 좀 찾아줘" → {"intent": "find", "object_ko": "약통", "targets": ["pill bottle", "medicine bottle"]}
"왼쪽에 있는 리모컨 찾아줘" → {"intent": "find", "object_ko": "리모컨", "targets": ["remote control"]}
"이 컵 어떻게 잡아?" → {"intent": "how", "object_ko": "컵", "targets": ["cup"]}
"아까 그 안경 어디 있어?" → {"intent": "where", "object_ko": "안경", "targets": ["glasses", "eyeglasses"]}
"이거 어떻게 잡아?" → {"intent": "how", "object_ko": null, "targets": []}
"지갑 찾았어" → {"intent": "stop", "object_ko": "지갑", "targets": ["wallet"]}
"그만해" → {"intent": "stop", "object_ko": null, "targets": []}
"오늘 날씨 어때?" → {"intent": "unknown", "object_ko": null, "targets": []}

사용자 문장: {text}"""

# 결과에서 한 번 더 걸러낼 관계 표현 (모델이 규칙을 어겼을 때 대비)
_RELATION_WORDS = {"my", "mine", "your", "his", "her", "our", "their", "the", "a", "an", "this", "that",
                   "left", "right", "front", "back", "near", "far"}


class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1, description="사용자의 한국어 음성 명령")


def normalize_command(text: str) -> str:
    """캐시 키: 공백 정리, 끝의 문장부호 제거 ("어디 있어?" 와 "어디 있어" 를 같은 문장으로)"""
    t = " ".join(text.split())
    return t.rstrip("?!.~,。？！ ").strip()


def clean_parse_result(raw: Any) -> dict | None:
    """Gemini 결과를 정해진 형식으로 정리. 형식이 아예 틀리면 None (→ 재시도)"""
    if not isinstance(raw, dict) or "intent" not in raw or "targets" not in raw:
        return None
    intent = str(raw.get("intent") or "").strip().lower()
    if intent not in PARSE_INTENTS:
        intent = "unknown"

    obj = raw.get("object_ko")
    obj = obj.strip() if isinstance(obj, str) and obj.strip() and obj.strip().lower() != "null" else None

    targets: list[str] = []
    items = raw.get("targets") if isinstance(raw.get("targets"), list) else []
    for item in items:
        if not isinstance(item, str):
            continue
        words = [w for w in re.sub(r"[^a-z0-9 \-']", " ", item.lower()).split() if w not in _RELATION_WORDS]
        name = " ".join(words)[:50].strip()
        # /detect 의 targets 형식과 같은 규칙 → 그대로 넘겨 쓸 수 있음
        if name and _NAME_RE.match(name) and name not in targets:
            targets.append(name)
        if len(targets) == 3:
            break
    return {"intent": intent, "object_ko": obj, "targets": targets}


class ParseCache:
    """같은 문장의 해석 결과를 메모리에 저장 (오래 안 쓴 것부터 삭제).
    같은 문장이 동시에 들어오면 Gemini는 한 번만 부르고 결과를 나눠 쓴다."""

    def __init__(self, size: int):
        self.size = size
        self.data: OrderedDict[str, dict] = OrderedDict()
        self.inflight: dict[str, asyncio.Future] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        v = self.data.get(key)
        if v is not None:
            self.data.move_to_end(key)
            self.hits += 1
        return v

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.size:
            self.data.popitem(last=False)


parse_cache = ParseCache(PARSE_CACHE_SIZE)


# 한국어 색 형용사 → 영어 색 단어 (Gemini가 targets에 색을 빼먹었을 때 서버가 보충)
KO_COLOR_WORDS: list[tuple[str, str]] = [
    ("빨간", "red"), ("빨강", "red"), ("붉은", "red"), ("검은", "black"), ("검정", "black"), ("까만", "black"),
    ("까망", "black"), ("하얀", "white"), ("하양", "white"), ("흰", "white"), ("파란", "blue"), ("파랑", "blue"),
    ("남색", "blue"), ("하늘색", "blue"), ("노란", "yellow"), ("노랑", "yellow"), ("초록", "green"), ("녹색", "green"),
    ("연두", "green"), ("회색", "gray"), ("은색", "silver"), ("주황", "orange"), ("보라", "purple"),
    ("분홍", "pink"), ("핑크", "pink"), ("갈색", "brown"), ("베이지", "brown"),
]


def ensure_color_in_targets(text: str, result: dict) -> dict:
    """문장에 색 형용사가 있는데 targets에 색 단어가 없으면 첫 번째에 "색 + 물건"을 넣음.
    targets 안의 색 단어가 목록 밖이면(예: navy) 그 이름은 그대로 두고 손대지 않는다."""
    targets = result["targets"]
    if not targets or extract_color_from_targets(targets):
        return result
    en = next((e for k, e in KO_COLOR_WORDS if k in text), None)
    if en is None:
        return result
    base = strip_color_words(targets[0])
    fixed = [f"{en} {base}"] + [t for t in targets if t != f"{en} {base}"]
    if base not in fixed:
        fixed.append(base)
    log.info("/parse: 색 단어 보충 %s → %s", targets, fixed[:3])
    return {**result, "targets": fixed[:3]}


def parse_incomplete(result: dict) -> bool:
    """object_ko와 targets 중 하나만 채워진 경우 (물건이 있으면 둘 다 있어야 앱이 새 물건인지 판단 가능)"""
    return bool(result["object_ko"]) != bool(result["targets"])


async def gemini_parse(text: str) -> tuple[dict, int]:
    """Gemini로 해석. JSON이 아니거나 object_ko·targets 중 하나만 있으면 한 번만 다시 시도.
    intent는 바꾸지 않는다 (where 라도 그대로). 반환: (결과, 시도 횟수)"""
    base = PARSE_PROMPT.replace("{text}", json.dumps(text, ensure_ascii=False))
    prompt, fallback = base, None
    for attempt in (1, 2):
        reply = await call_gemini(None, prompt)
        result = clean_parse_result(parse_json_text(reply))
        if result is None:
            log.warning("/parse: Gemini 응답 형식이 맞지 않음 (시도 %d): %s", attempt, reply[:200])
            continue
        result = ensure_color_in_targets(text, result)
        if not parse_incomplete(result):
            return result, attempt
        # 하나만 비어 있으면 짚어서 다시 물어봄. 두 번째도 그러면 받은 결과를 그대로 씀
        log.warning("/parse: object_ko와 targets 중 하나가 비어 있음 (시도 %d): %s", attempt, result)
        fallback = result
        prompt = base + "\n\n주의: 물건 이름이 있으면 object_ko와 targets를 둘 다 채워. 물건이 없으면 둘 다 비워."
    if fallback is not None:
        return ensure_color_in_targets(text, fallback), 2
    raise ApiError(502, "vlm_invalid_json", "Gemini 응답이 두 번 모두 올바른 JSON 형식이 아니었습니다.")


# ---------------------------------------------------------------------------
# 음성 생성 (/tts, ElevenLabs)
# ---------------------------------------------------------------------------
TTS_TIMEOUT = env_float("TTS_TIMEOUT", 8)  # ElevenLabs 응답 제한 시간(초)
TTS_MAX_CHARS = env_int("TTS_MAX_CHARS", 200)
TTS_RATE_PER_MIN = env_int("TTS_RATE_PER_MIN", 30)  # 기기(X-Client-Id)별, 없으면 IP별
TTS_DAILY_CHAR_LIMIT = env_int("TTS_DAILY_CHAR_LIMIT", 5000)  # 실시간 생성 하루 글자 수 상한
TTS_CACHE_DIR = STATIC_DIR / "audio" / "cache"  # /app/audio/cache/{해시}.mp3 로도 열림

try:
    tts_cfg: TTSConfig | None = TTSConfig.from_env()
    tts_cfg_error = ""
except ValueError as _e:  # 음성 설정 숫자가 잘못돼도 서버의 다른 기능은 그대로 동작하게
    tts_cfg, tts_cfg_error = None, str(_e)

tts_limiter = RateLimiter(TTS_RATE_PER_MIN, 60.0)
_tts_http: httpx.AsyncClient | None = None
_tts_inflight: dict[str, asyncio.Future] = {}


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="읽어줄 한국어 문장")


class TTSUsage:
    """실시간 생성 글자 수를 날짜별로 세고 파일에 남김 (서버를 다시 켜도 그날 사용량 유지)"""

    def __init__(self, path: Path, limit: int, tz: str):
        self.path = path
        self.limit = limit
        self.tz = ZoneInfo(tz)
        self.lock = threading.Lock()
        self.date = self._today()
        self.chars = 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("date") == self.date:
                self.chars = int(data.get("chars", 0))
        except (OSError, ValueError, AttributeError):
            pass

    def _today(self) -> str:
        return datetime.now(self.tz).strftime("%Y-%m-%d")

    def _roll(self) -> None:
        today = self._today()
        if today != self.date:  # 날짜가 바뀌면 0부터
            self.date, self.chars = today, 0

    def _save(self) -> None:
        try:
            write_atomic(self.path, json.dumps({"date": self.date, "chars": self.chars}).encode())
        except OSError as e:
            log.warning("TTS 사용량 파일 저장 실패: %s", e)

    def used(self) -> int:
        with self.lock:
            self._roll()
            return self.chars

    def reserve(self, n: int) -> bool:
        """생성 전에 글자 수를 미리 잡아둠 (동시 요청이 함께 상한을 넘지 않게)"""
        with self.lock:
            self._roll()
            if self.chars + n > self.limit:
                return False
            self.chars += n
            self._save()
            return True

    def refund(self, n: int) -> None:
        """생성에 실패하면 잡아둔 글자 수를 돌려줌"""
        with self.lock:
            self._roll()
            self.chars = max(0, self.chars - n)
            self._save()

    def seconds_until_reset(self) -> int:
        now = datetime.now(self.tz)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(1, int((midnight - now).total_seconds()))


tts_usage = TTSUsage(LOG_DIR / "tts_usage.json", TTS_DAILY_CHAR_LIMIT, TIMEZONE)


def tts_audio_response(audio: bytes | Path, source: str, gen_ms: float) -> Response:
    headers = {"X-TTS-Source": source, "X-TTS-Ms": f"{gen_ms:.0f}", "Cache-Control": "no-store"}
    if isinstance(audio, Path):
        return FileResponse(audio, media_type="audio/mpeg", headers=headers)
    return Response(content=audio, media_type="audio/mpeg", headers=headers)


# ---------------------------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global csv_log, _http, _tts_http
    if not API_KEY:
        raise RuntimeError("환경변수 API_KEY 가 비어 있습니다.")
    if not ACCESS_LOG:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 바깥 API 호출을 한 줄씩 남기지 않음 (주소에 목소리 ID 포함)
    csv_log = CsvLogger(LOG_DIR)
    log.info("요청 기록 CSV: %s", csv_log.path)
    if not default_prompt():
        log.warning("프롬프트 파일 %s 이 없거나 비어 있습니다. /describe 는 prompt 를 직접 보낼 때만 동작합니다.", PROMPT_FILE)
    _http = httpx.AsyncClient(timeout=GEMINI_TIMEOUT)
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY 가 없어 /describe 는 503을 돌려줍니다.")
    _tts_http = httpx.AsyncClient(limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
    if tts_cfg is None:
        log.warning("음성 설정 오류로 /tts 를 끕니다: %s", tts_cfg_error)
    elif not tts_cfg.enabled:
        log.warning("ELEVENLABS_API_KEY 또는 ELEVENLABS_VOICE_ID 가 없어 /tts 는 503을 돌려줍니다.")
    else:
        log.info("TTS 사용: 실시간 모델 %s, 오늘 %d/%d자 사용", tts_cfg.live_model, tts_usage.used(), TTS_DAILY_CHAR_LIMIT)
    if not (STATIC_DIR / "index.html").is_file():
        log.warning("%s 가 없어 /app 은 404를 돌려줍니다.", STATIC_DIR / "index.html")
    threading.Thread(target=detector.load, name="model-loader", daemon=True).start()
    yield
    await _http.aclose()
    await _tts_http.aclose()


app = FastAPI(title="YOLOE Object Finder API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials="*" not in ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "X-Client-Id", "Content-Type"],
    expose_headers=["X-TTS-Source", "X-TTS-Ms", "Retry-After"],
)


# ---- 정적 파일 (/app) : API 키 없이 접근 가능 -------------------------------
# 인증은 각 API 라우트의 dependencies 로만 걸려 있으므로 여기에는 적용되지 않는다.
@app.get("/app", include_in_schema=False)
async def app_root_redirect(request: Request):
    """/app → /app/ 로 이동. 상대 경로 Location 을 써서 터널 뒤에서도 https 가 유지되고,
    index.html 안의 상대 경로(style.css 등)가 /app/ 아래로 올바르게 풀린다."""
    query = request.url.query
    return RedirectResponse(url="/app/" + (f"?{query}" if query else ""), status_code=307)


# html=True: /app/ 처럼 폴더 경로로 들어오면 index.html 을 보여줌
app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True, check_dir=False), name="app")


def error_json(status: int, code: str, message: str, detail: Any = None, headers: dict | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if detail is not None:
        body["error"]["detail"] = detail
    return JSONResponse(status_code=status, content=body, headers=headers)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return error_json(exc.status, exc.code, exc.message, headers=exc.headers)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    codes = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}
    return error_json(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    detail = [{"field": ".".join(str(p) for p in e.get("loc", [])[1:]), "reason": e.get("msg")} for e in exc.errors()]
    return error_json(422, "invalid_request", "요청 값이 올바르지 않습니다.", detail)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    log.exception("처리 중 알 수 없는 오류")
    return error_json(500, "internal_error", f"서버 내부 오류: {type(exc).__name__}")


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise ApiError(401, "unauthorized", "X-API-Key 헤더가 없거나 올바르지 않습니다.")


# ---- 엔드포인트 -----------------------------------------------------------
@app.get("/health", dependencies=[Depends(require_api_key)])
async def health():
    status = "ok" if detector.ready else ("error" if detector.load_error else "loading")
    body = {
        "status": status,
        "model_loaded": detector.ready,
        "model": detector.model_name,
        "device": detector.device,
        "half": detector.half,
        "vocab_size": len(detector.vocab),
        "queue_length": detector.pending,
        "max_queue": MAX_QUEUE,
        "describe_enabled": bool(GEMINI_API_KEY and default_prompt()),
        "tts_enabled": bool(tts_cfg and tts_cfg.enabled),
        "tts_today_chars": tts_usage.used(),
        "tts_daily_limit": TTS_DAILY_CHAR_LIMIT,
    }
    if detector.load_error:
        body["error"] = detector.load_error
    return body


@app.get("/stats", dependencies=[Depends(require_api_key)])
async def get_stats():
    s = stats.snapshot()
    s["queue_length"] = detector.pending
    s["vocab_size"] = len(detector.vocab)
    s["csv_file"] = str(csv_log.path) if csv_log else None
    return s


@app.post("/detect", dependencies=[Depends(require_api_key)])
async def detect(
    request: Request,
    image: UploadFile = File(..., description="JPEG 이미지"),
    targets: str = Form("", description="찾을 물체 영어 이름, 쉼표로 여러 개 (예: mug,cup). 비우면 전체 탐지"),
    conf: float = Form(DEFAULT_CONF, ge=0.0, le=1.0),
    imgsz: int = Form(DEFAULT_IMGSZ, ge=160, le=1280),
):
    t_start = time.perf_counter()
    ip, cid = client_ip(request), client_id(request)
    # targets 가 없거나 쉼표·공백뿐이면 전체 탐지 모드
    mode = "targets" if any(p.strip() for p in targets.split(",")) else "all"
    row: dict[str, Any] = {"targets": targets[:100] if mode == "targets" else "*"}
    try:
        if not detector.ready:
            if detector.load_error:
                raise ApiError(503, "model_error", f"모델을 불러오지 못했습니다: {detector.load_error}")
            raise ApiError(503, "model_loading", "모델을 준비 중입니다. 잠시 후 다시 시도하세요.")

        allowed, retry = detect_limiter.allow(limit_key(ip, cid))
        if not allowed:
            raise ApiError(429, "rate_limited", f"요청이 너무 잦습니다 (초당 최대 {DETECT_RATE_PER_SEC}회).",
                           headers={"Retry-After": str(max(1, round(retry)))})
        if detector.pending >= MAX_QUEUE:
            raise ApiError(503, "queue_full", "요청이 많아 지금은 처리할 수 없습니다. 기기 내 인식을 사용하세요.")

        vocab_ms = 0.0
        # 별칭·동의어로 바꾸기 전의 원래 요청 목록 (색상 단어를 잃지 않도록 색 검증에는 이것을 씀)
        raw_targets = [" ".join(p.strip().lower().split()) for p in targets.split(",") if p.strip()]
        req_color = extract_color_from_targets(raw_targets) if mode == "targets" else None
        if mode == "all":
            # 기본 어휘는 항상 모델에 설정돼 있으므로 어휘 변경 없음 (색 필터도 적용하지 않음)
            requested, label_to_target, max_results = ["*"], ALL_MODE_TARGETS, MAX_RESULTS_ALL
        else:
            detect_names = targets
            if req_color and COLOR_SHAPE_ONLY:
                # YOLOE는 형태로만 찾고 색은 HSV로 확인: "black tumbler" → "tumbler"
                # (색이 붙은 새 이름을 어휘에 추가하느라 모든 사용자의 추론이 멈추는 것도 막음)
                detect_names = ",".join(strip_color_words(t) for t in raw_targets)
            requested = parse_targets(detect_names, set(detector.name_to_id))
            label_to_target, max_results = expand_targets(requested), None
            if detector.missing_words(label_to_target):
                vocab_ms = await run_in_threadpool(detector.ensure_words, list(label_to_target))
            else:
                detector.touch(label_to_target)

        t0 = time.perf_counter()
        img = await run_in_threadpool(decode_image, await image.read())
        decode_ms = ms(time.perf_counter() - t0)

        imgsz = max(32, int(round(imgsz / 32)) * 32)
        loop = asyncio.get_running_loop()
        job = Job(img, label_to_target, conf, imgsz, time.perf_counter(), loop.create_future(), loop, max_results)
        detector.submit(job)
        try:
            out = await asyncio.wait_for(job.future, timeout=REQUEST_TIMEOUT + 2.0)
        except asyncio.TimeoutError:
            raise ApiError(503, "timeout", "처리 시간이 너무 길어졌습니다. 기기 내 인식을 사용하세요.")

        # 추론·동의어·중복 박스 정리가 끝난 결과에 색 필터 (색 요청이 없으면 그대로 통과)
        detections, color_rejected, color_filter, color_ms = out["detections"], [], None, 0.0
        if req_color:
            t1 = time.perf_counter()
            detections, color_rejected, color_filter = await run_in_threadpool(
                filter_detections_by_color, img, detections, raw_targets)  # img 는 cv2로 디코딩한 BGR
            color_ms = ms(time.perf_counter() - t1)

        total = ms(time.perf_counter() - t_start)
        timing = {
            "decode": decode_ms,
            "vocab_update": vocab_ms,
            "queue_wait": out["queue_wait"],
            "inference": out["inference"],
            "color": color_ms,
            "total": total,
        }
        row.update(n_detections=len(detections), batch_size=out["batch_size"], imgsz=imgsz,
                   decode_ms=decode_ms, vocab_ms=vocab_ms, queue_wait_ms=out["queue_wait"],
                   inference_ms=out["inference"])
        record("detect", 200, "ok", t_start, ip, cid, **row)
        log.debug("detect %s color=%s found=%d rejected=%d %s",
                  requested, color_filter, len(detections), len(color_rejected), timing)
        return {
            "mode": mode,
            "detections": detections,
            "color_filter": color_filter,
            "color_rejected": color_rejected,
            "timing_ms": timing,
            "batch_size": out["batch_size"],
            "device": detector.device,
            "model": detector.model_name,
        }
    except ApiError as e:
        record("detect", e.status, e.code, t_start, ip, cid, **row)
        raise


@app.post("/describe", dependencies=[Depends(require_api_key)])
async def describe(
    request: Request,
    image: UploadFile = File(..., description="사진 (책상을 내려다본 사진 또는 컵 부분)"),
    prompt: str | None = Form(None, description="프롬프트를 바꿔 보낼 때만. 비우면 prompt.txt 사용"),
):
    t_start = time.perf_counter()
    ip, cid = client_ip(request), client_id(request)
    try:
        if not GEMINI_API_KEY:
            raise ApiError(503, "describe_disabled", "서버에 GEMINI_API_KEY 가 설정되지 않았습니다.")
        template = (prompt or "").strip() or default_prompt()
        if not template:
            raise ApiError(503, "describe_disabled", f"프롬프트 파일({PROMPT_FILE})이 없거나 비어 있습니다.")
        if len(template) > MAX_PROMPT_CHARS:
            raise ApiError(400, "prompt_too_long", f"프롬프트는 {MAX_PROMPT_CHARS}자 이내여야 합니다.")
        allowed, retry = describe_limiter.allow(limit_key(ip, cid))
        if not allowed:
            raise ApiError(429, "rate_limited", f"설명 요청이 너무 잦습니다 (분당 최대 {DESCRIBE_RATE_PER_MIN}회).",
                           headers={"Retry-After": str(max(1, round(retry)))})

        t0 = time.perf_counter()
        jpeg = await run_in_threadpool(prepare_describe_image, await image.read())
        decode_ms = ms(time.perf_counter() - t0)

        rendered, tod = render_prompt(template)
        t1 = time.perf_counter()
        result, attempts = None, 0
        for attempts in (1, 2):  # JSON이 아니면 한 번만 다시 시도
            text = await call_gemini(jpeg, rendered)
            result = parse_json_text(text)
            if result is not None:
                break
            log.warning("Gemini 응답이 JSON이 아님 (시도 %d): %s", attempts, text[:200])
        vlm_ms = ms(time.perf_counter() - t1)
        if result is None:
            raise ApiError(502, "vlm_invalid_json", "Gemini 응답이 두 번 모두 JSON 형식이 아니었습니다.")

        record("describe", 200, "ok", t_start, ip, cid, decode_ms=decode_ms, inference_ms=vlm_ms)
        return {
            "result": result,
            "model": GEMINI_MODEL,
            "attempts": attempts,
            "time_of_day": tod,
            "prompt_source": "custom" if (prompt or "").strip() else "default",
            "timing_ms": {"decode": decode_ms, "vlm": vlm_ms, "total": ms(time.perf_counter() - t_start)},
        }
    except ApiError as e:
        record("describe", e.status, e.code, t_start, ip, cid)
        raise


@app.post("/parse", dependencies=[Depends(require_api_key)])
async def parse(request: Request, body: ParseRequest):
    t_start = time.perf_counter()
    ip, cid = client_ip(request), client_id(request)
    try:
        if not GEMINI_API_KEY:
            raise ApiError(503, "parse_disabled", "서버에 GEMINI_API_KEY 가 설정되지 않았습니다.")
        # 기기(X-Client-Id)별 제한. 헤더가 없으면 IP별
        allowed, retry = parse_limiter.allow(f"cid:{cid}" if cid else f"ip:{ip}")
        if not allowed:
            raise ApiError(429, "rate_limited", f"음성 명령 요청이 너무 잦습니다 (분당 최대 {PARSE_RATE_PER_MIN}회).",
                           headers={"Retry-After": str(max(1, round(retry)))})

        key = normalize_command(body.text)
        if not key:
            raise ApiError(400, "invalid_text", "text 가 비어 있습니다.")
        if len(key) > PARSE_MAX_CHARS:
            raise ApiError(400, "invalid_text", f"text 는 {PARSE_MAX_CHARS}자 이내여야 합니다.")

        result = parse_cache.get(key)
        cached, attempts, vlm_ms = result is not None, 0, 0.0
        if result is None:
            fut = parse_cache.inflight.get(key)
            if fut is not None:  # 같은 문장을 이미 해석 중이면 그 결과를 기다림
                result = await asyncio.shield(fut)
                cached = True
            else:
                parse_cache.misses += 1
                fut = asyncio.get_running_loop().create_future()
                parse_cache.inflight[key] = fut
                t1 = time.perf_counter()
                try:
                    result, attempts = await gemini_parse(key)
                    parse_cache.put(key, result)
                    fut.set_result(result)
                except Exception as e:
                    fut.set_exception(e)
                    fut.exception()  # 기다리는 쪽이 없어도 경고가 나지 않게
                    raise
                finally:
                    parse_cache.inflight.pop(key, None)
                vlm_ms = ms(time.perf_counter() - t1)

        total = ms(time.perf_counter() - t_start)
        # CSV에는 음성 문장 대신 해석 결과만 남김
        record("parse", 200, "cached" if cached else "ok", t_start, ip, cid,
               targets=",".join(result["targets"]), inference_ms=vlm_ms)
        return {**result, "cached": cached, "attempts": attempts, "timing_ms": {"vlm": vlm_ms, "total": total}}
    except ApiError as e:
        record("parse", e.status, e.code, t_start, ip, cid)
        raise


@app.post("/tts", dependencies=[Depends(require_api_key)])
async def tts(request: Request, body: TtsRequest):
    t_start = time.perf_counter()
    ip, cid = client_ip(request), client_id(request)
    text = " ".join(body.text.split())
    row: dict[str, Any] = {"chars": len(text)}  # CSV에는 문장 대신 글자 수만 남김
    try:
        if tts_cfg is None:
            raise ApiError(503, "tts_disabled", f"음성 설정이 잘못되었습니다: {tts_cfg_error}")
        if not tts_cfg.enabled:
            raise ApiError(503, "tts_disabled", "서버에 ELEVENLABS_API_KEY 또는 ELEVENLABS_VOICE_ID 가 설정되지 않았습니다.")
        allowed, retry = tts_limiter.allow(f"cid:{cid}" if cid else f"ip:{ip}")
        if not allowed:
            raise ApiError(429, "rate_limited", f"음성 요청이 너무 잦습니다 (분당 최대 {TTS_RATE_PER_MIN}회).",
                           headers={"Retry-After": str(max(1, round(retry)))})
        if not text:
            raise ApiError(400, "invalid_text", "text 가 비어 있습니다.")
        if len(text) > TTS_MAX_CHARS:
            raise ApiError(400, "text_too_long", f"text 는 {TTS_MAX_CHARS}자 이내여야 합니다 (지금 {len(text)}자).")

        model = tts_cfg.live_model
        key = tts_cfg.cache_key(text, model)
        path = TTS_CACHE_DIR / f"{key}.mp3"

        # 1) 저장된 파일이 있으면 그대로 (하루 상한과 무관)
        if path.is_file() and path.stat().st_size > 0:
            record("tts", 200, "cache", t_start, ip, cid, **row)
            return tts_audio_response(path, "cache", 0)

        # 2) 같은 문장을 지금 만드는 중이면 그 결과를 기다림 (글자 수를 두 번 쓰지 않음)
        fut = _tts_inflight.get(key)
        if fut is not None:
            audio = await asyncio.shield(fut)
            gen_ms = ms(time.perf_counter() - t_start)
            record("tts", 200, "generated", t_start, ip, cid, inference_ms=gen_ms, **row)
            return tts_audio_response(audio, "generated", gen_ms)

        # 3) 새로 생성: 하루 글자 수를 먼저 잡아둠
        if not tts_usage.reserve(len(text)):
            raise ApiError(
                429, "tts_daily_limit",
                f"오늘 음성 생성 한도({TTS_DAILY_CHAR_LIMIT}자)를 다 썼습니다. 저장된 문장은 계속 들을 수 있습니다.",
                headers={"Retry-After": str(tts_usage.seconds_until_reset())},
            )
        fut = asyncio.get_running_loop().create_future()
        _tts_inflight[key] = fut
        t1 = time.perf_counter()
        try:
            audio = await synthesize_async(_tts_http, tts_cfg, text, model, TTS_TIMEOUT)
            fut.set_result(audio)
        except TTSError as e:
            tts_usage.refund(len(text))
            err = ApiError(502, e.code, e.message)  # 앱은 502를 받으면 브라우저 음성으로 대체
            fut.set_exception(err)
            fut.exception()  # 기다리는 쪽이 없어도 경고가 나지 않게
            raise err from None
        except BaseException as e:
            tts_usage.refund(len(text))
            fut.set_exception(ApiError(502, "tts_error", "음성 생성이 중단되었습니다."))
            fut.exception()
            raise
        finally:
            _tts_inflight.pop(key, None)
        gen_ms = ms(time.perf_counter() - t1)

        try:  # 저장에 실패해도 이번 응답은 그대로 보냄
            await run_in_threadpool(write_atomic, path, audio)
        except OSError as e:
            log.warning("TTS 캐시 저장 실패: %s", e)

        log.info("tts 생성 %d자 %.0fms (오늘 %d/%d자)", len(text), gen_ms, tts_usage.used(), TTS_DAILY_CHAR_LIMIT)
        record("tts", 200, "generated", t_start, ip, cid, inference_ms=gen_ms, **row)
        return tts_audio_response(audio, "generated", gen_ms)
    except ApiError as e:
        record("tts", e.status, e.code, t_start, ip, cid, **row)
        raise


# ---- VLM 테스트 페이지 ----------------------------------------------------
@app.get("/describe-test", include_in_schema=False)
async def describe_test_page():
    """프롬프트를 고쳐가며 /describe 를 시험하는 페이지. API 키는 페이지에서 입력한다."""
    if not TEST_PAGE:
        raise ApiError(404, "not_found", "테스트 페이지가 꺼져 있습니다 (TEST_PAGE=0).")
    page = Path(__file__).with_name("describe_test.html")
    if not page.exists():
        raise ApiError(404, "not_found", "describe_test.html 파일이 없습니다.")
    # 기본 프롬프트를 페이지의 텍스트 상자 기본값으로 넣는다 (</script> 깨짐 방지)
    default_json = json.dumps(default_prompt(), ensure_ascii=False).replace("</", "<\\/")
    body = page.read_text(encoding="utf-8").replace("__DEFAULT_PROMPT_JSON__", default_json)
    body = body.replace("__MODEL_NAME__", html.escape(GEMINI_MODEL))
    return HTMLResponse(body, headers={"Cache-Control": "no-store"})
