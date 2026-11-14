"""
안내 음성 미리 생성 (ElevenLabs, 품질 우선 모델 ELEVENLABS_PREGEN_MODEL)

입력  ~/app/phrases.json   형식: {"id": ["표현1", "표현2", ...], ...}
출력  static/audio/{id}_{번호}.mp3   (번호는 1부터)  → 웹에서는 /app/audio/{id}_{번호}.mp3
      static/audio/manifest.json     {"id": 표현 개수, ...}

사용 예
  python3 pregen_tts.py                       # 없는 파일만 생성
  python3 pregen_tts.py --only found,lost     # 이 id만
  python3 pregen_tts.py --only found --force  # 이 id를 다시 생성
  python3 pregen_tts.py --dry-run             # 만들지 않고 생성할 문장 수와 글자 수만 계산

설정은 환경변수 또는 이 파일 옆의 .env 에서 읽는다 (ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID 등).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

from elevenlabs_tts import TTSConfig, TTSError, load_dotenv, synthesize_sync, write_atomic

HERE = Path(__file__).resolve().parent
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # 파일 이름에 쓰므로 영문·숫자·_·- 만 허용
MAX_RETRIES = 2  # 실패하면 최대 2번 더 시도
TIMEOUT = 60.0  # 품질 우선 모델은 느릴 수 있어서 넉넉하게


def load_phrases(path: Path) -> dict[str, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"{path} 파일이 없습니다. --phrases 로 위치를 지정하세요.")
    except json.JSONDecodeError as e:
        sys.exit(f"{path} 가 올바른 JSON이 아닙니다: {e}")
    if not isinstance(data, dict):
        sys.exit(f'{path} 는 {{"id": ["표현1", ...]}} 형식이어야 합니다.')

    phrases: dict[str, list[str]] = {}
    for pid, items in data.items():
        if not ID_RE.match(pid):
            sys.exit(f"id '{pid}' 는 영문·숫자·_·- 로만 64자 이내여야 합니다 (파일 이름에 쓰임).")
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list) or not all(isinstance(t, str) and t.strip() for t in items):
            sys.exit(f"'{pid}' 의 값은 비어 있지 않은 문장들의 목록이어야 합니다.")
        phrases[pid] = [" ".join(t.split()) for t in items]
    return phrases


def contiguous_count(audio_dir: Path, pid: str) -> int:
    """{id}_1.mp3 부터 빠짐없이 있는 파일 개수 (앱이 1~N 중에서 골라도 404가 나지 않게)"""
    n = 0
    while (audio_dir / f"{pid}_{n + 1}.mp3").is_file() and (audio_dir / f"{pid}_{n + 1}.mp3").stat().st_size > 0:
        n += 1
    return n


def main() -> None:
    load_dotenv(HERE / ".env")

    ap = argparse.ArgumentParser(description="ElevenLabs 안내 음성 미리 생성")
    ap.add_argument("--phrases", default="~/app/phrases.json", help="문장 목록 JSON (기본 ~/app/phrases.json)")
    ap.add_argument("--static", default=None, help="static 폴더 (기본: 환경변수 STATIC_DIR 또는 이 파일 옆의 static)")
    ap.add_argument("--only", default="", help="이 id들만 생성 (쉼표 구분)")
    ap.add_argument("--force", action="store_true", help="이미 있는 파일도 다시 생성")
    ap.add_argument("--model", default=None, help="모델 (기본: ELEVENLABS_PREGEN_MODEL, 없으면 eleven_v3)")
    ap.add_argument("--dry-run", action="store_true", help="생성하지 않고 대상 문장 수와 글자 수만 출력")
    args = ap.parse_args()

    static_dir = Path(args.static or os.getenv("STATIC_DIR") or HERE / "static").expanduser()
    audio_dir = static_dir / "audio"
    phrases = load_phrases(Path(args.phrases).expanduser())

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = [s for s in only if s not in phrases]
    if unknown:
        sys.exit(f"phrases.json 에 없는 id: {', '.join(unknown)}")
    selected = only or list(phrases)

    try:
        cfg = TTSConfig.from_env()
    except ValueError as e:
        sys.exit(str(e))
    model = args.model or cfg.pregen_model

    # 이번에 만들 문장 목록
    jobs: list[tuple[str, int, str, Path]] = []
    skipped = 0
    for pid in selected:
        for i, text in enumerate(phrases[pid], start=1):
            path = audio_dir / f"{pid}_{i}.mp3"
            if path.is_file() and path.stat().st_size > 0 and not args.force:
                skipped += 1
                continue
            jobs.append((pid, i, text, path))

    planned_chars = sum(len(t) for _, _, t, _ in jobs)
    print(f"모델 {model} | 대상 id {len(selected)}개 | 생성할 문장 {len(jobs)}개 ({planned_chars}자) | 건너뜀 {skipped}개")
    if args.dry_run:
        for pid, i, text, _ in jobs:
            print(f"  {pid}_{i}: {text}")
        return
    if jobs and not cfg.enabled:
        sys.exit("ELEVENLABS_API_KEY 와 ELEVENLABS_VOICE_ID 를 .env 에 설정하세요.")
    vs = cfg.voice_settings(model)
    if cfg.stability is not None and vs.get("stability") != cfg.stability:
        print(f"참고: {model} 은 안정성을 0.0/0.5/1.0 중 하나로 받아서 {cfg.stability} → {vs['stability']} 로 보냅니다.")

    generated, used_chars = 0, 0
    failed: list[tuple[str, str, str]] = []  # (파일 이름, 문장, 이유)
    with httpx.Client() as client:
        for n, (pid, i, text, path) in enumerate(jobs, start=1):
            label = f"{pid}_{i}"
            for attempt in range(1 + MAX_RETRIES):
                t0 = time.perf_counter()
                try:
                    audio = synthesize_sync(client, cfg, text, model, TIMEOUT)
                    write_atomic(path, audio)
                    generated += 1
                    used_chars += len(text)
                    print(f"[{n}/{len(jobs)}] {label}.mp3  {len(text)}자  {(time.perf_counter() - t0):.1f}s  {text}")
                    break
                except TTSError as e:
                    if attempt < MAX_RETRIES:
                        wait = 2 * (attempt + 1)
                        print(f"[{n}/{len(jobs)}] {label} 실패 ({e.message}) → {wait}초 뒤 다시 시도")
                        time.sleep(wait)
                    else:
                        print(f"[{n}/{len(jobs)}] {label} 실패 ({e.message})")
                        failed.append((label, text, e.message))
                except OSError as e:  # 파일 쓰기 실패는 다시 해도 같으므로 바로 기록
                    failed.append((label, text, f"파일 저장 실패: {e}"))
                    break

    # manifest.json: phrases.json 의 모든 id 기준 (--only 로 일부만 만들어도 전체를 다시 계산)
    manifest: dict[str, int] = {}
    incomplete: list[str] = []
    for pid, items in phrases.items():
        count = contiguous_count(audio_dir, pid)
        if count < len(items):
            incomplete.append(f"{pid} ({count}/{len(items)})")
        if count:
            manifest[pid] = count
    write_atomic(audio_dir / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    print("\n===== 결과 =====")
    print(f"생성한 문장 {generated}개, 사용한 글자 수 {used_chars}자 (모델 {model}, 크레딧은 모델·요금제의 글자당 단가를 곱해 계산)")
    print(f"manifest.json: id {len(manifest)}개 → {audio_dir / 'manifest.json'}")
    if incomplete:
        print("파일이 다 갖춰지지 않은 id (manifest에는 1번부터 빠짐없이 있는 개수만 기록): " + ", ".join(incomplete))
    if failed:
        print(f"\n실패한 문장 {len(failed)}개 (다시 실행하면 이것만 다시 시도합니다):")
        for label, text, reason in failed:
            print(f"  - {label}: {text}  ← {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
