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
  GEMINI_API_KEY, GEMINI_MODEL, GEMINI_THINKING_LEVEL, PROMPT_FILE, TIMEZONE, LOG_DIR
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
from datetime import datetime
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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from vocab import ALIASES, BASE_VOCAB, SYNONYM_GROUPS


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
MAX_EXTRA_VOCAB = env_int("MAX_EXTRA_VOCAB", 100)  # 요청으로 추가되는 단어 최대 개수 (넘으면 오래 안 쓴 것부터 제거)
SYNONYMS = env_bool("SYNONYMS", True)
DEDUPE_IOU = env_float("DEDUPE_IOU", 0.6)  # 같은 물건이 cup/mug로 두 번 잡히면 하나만 남김
MAX_IMAGE_BYTES = int(env_float("MAX_IMAGE_MB", 8) * 1024 * 1024)

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
        "batch_size", "imgsz", "decode_ms", "vocab_ms", "queue_wait_ms", "inference_ms", "total_ms",
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

object_ko
- 사용자가 말한 물건을 색·재질 같은 겉모습 표현까지 포함해 한국어로 (예: "검은 텀블러")
- "내", "왼쪽에 있는", "아까 그" 같은 소유·위치·관계 표현은 빼
- 물건이 없으면 null

targets: 물체 인식 모델에 넣을 영어 물체 이름
- 1~3개. 사용자가 말한 표현을 그대로 옮긴 이름을 먼저, 그다음 더 일반적인 이름 순서
- 색, 재질 같은 겉모습은 넣어도 되지만 my, left, on the table 같은 소유·위치·관계 표현은 빼
- 영어 소문자, 단수형, 관사(a, the) 없이
- 물리적인 물건이 아니면 (예: 날씨, 시간, 기분) 빈 배열

예시
"내 검은 텀블러 어디 있어?" → {"intent": "where", "object_ko": "검은 텀블러", "targets": ["black tumbler", "tumbler"]}
"약통 좀 찾아줘" → {"intent": "find", "object_ko": "약통", "targets": ["pill bottle", "medicine bottle"]}
"왼쪽에 있는 리모컨 찾아줘" → {"intent": "find", "object_ko": "리모컨", "targets": ["remote control"]}
"이 컵 어떻게 잡아?" → {"intent": "how", "object_ko": "컵", "targets": ["cup"]}
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


async def gemini_parse(text: str) -> tuple[dict, int]:
    """Gemini로 해석. JSON이 아니거나 형식이 틀리면 한 번만 다시 시도. 반환: (결과, 시도 횟수)"""
    prompt = PARSE_PROMPT.replace("{text}", json.dumps(text, ensure_ascii=False))
    for attempt in (1, 2):
        reply = await call_gemini(None, prompt)
        result = clean_parse_result(parse_json_text(reply))
        if result is not None:
            return result, attempt
        log.warning("/parse: Gemini 응답 형식이 맞지 않음 (시도 %d): %s", attempt, reply[:200])
    raise ApiError(502, "vlm_invalid_json", "Gemini 응답이 두 번 모두 올바른 JSON 형식이 아니었습니다.")


# ---------------------------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global csv_log, _http
    if not API_KEY:
        raise RuntimeError("환경변수 API_KEY 가 비어 있습니다.")
    if not ACCESS_LOG:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    csv_log = CsvLogger(LOG_DIR)
    log.info("요청 기록 CSV: %s", csv_log.path)
    if not default_prompt():
        log.warning("프롬프트 파일 %s 이 없거나 비어 있습니다. /describe 는 prompt 를 직접 보낼 때만 동작합니다.", PROMPT_FILE)
    _http = httpx.AsyncClient(timeout=GEMINI_TIMEOUT)
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY 가 없어 /describe 는 503을 돌려줍니다.")
    if not (STATIC_DIR / "index.html").is_file():
        log.warning("%s 가 없어 /app 은 404를 돌려줍니다.", STATIC_DIR / "index.html")
    threading.Thread(target=detector.load, name="model-loader", daemon=True).start()
    yield
    await _http.aclose()


app = FastAPI(title="YOLOE Object Finder API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials="*" not in ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "X-Client-Id", "Content-Type"],
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
    targets: str = Form(..., description="찾을 물체 영어 이름, 쉼표로 여러 개 (예: mug,cup)"),
    conf: float = Form(DEFAULT_CONF, ge=0.0, le=1.0),
    imgsz: int = Form(DEFAULT_IMGSZ, ge=160, le=1280),
):
    t_start = time.perf_counter()
    ip, cid = client_ip(request), client_id(request)
    row: dict[str, Any] = {"targets": targets[:100]}
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

        requested = parse_targets(targets, set(detector.name_to_id))
        label_to_target = expand_targets(requested)
        vocab_ms = 0.0
        if detector.missing_words(label_to_target):
            vocab_ms = await run_in_threadpool(detector.ensure_words, list(label_to_target))
        else:
            detector.touch(label_to_target)

        t0 = time.perf_counter()
        img = await run_in_threadpool(decode_image, await image.read())
        decode_ms = ms(time.perf_counter() - t0)

        imgsz = max(32, int(round(imgsz / 32)) * 32)
        loop = asyncio.get_running_loop()
        job = Job(img, label_to_target, conf, imgsz, time.perf_counter(), loop.create_future(), loop)
        detector.submit(job)
        try:
            out = await asyncio.wait_for(job.future, timeout=REQUEST_TIMEOUT + 2.0)
        except asyncio.TimeoutError:
            raise ApiError(503, "timeout", "처리 시간이 너무 길어졌습니다. 기기 내 인식을 사용하세요.")

        total = ms(time.perf_counter() - t_start)
        timing = {
            "decode": decode_ms,
            "vocab_update": vocab_ms,
            "queue_wait": out["queue_wait"],
            "inference": out["inference"],
            "total": total,
        }
        row.update(n_detections=len(out["detections"]), batch_size=out["batch_size"], imgsz=imgsz,
                   decode_ms=decode_ms, vocab_ms=vocab_ms, queue_wait_ms=out["queue_wait"],
                   inference_ms=out["inference"])
        record("detect", 200, "ok", t_start, ip, cid, **row)
        log.debug("detect %s found=%d %s", requested, len(out["detections"]), timing)
        return {
            "detections": out["detections"],
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
