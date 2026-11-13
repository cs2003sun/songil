"""
부하 테스트: 가상 사용자 N명이 0.3초마다 사진을 /detect 로 보낼 때의 응답 시간과 실패율

사용 예
  python3 loadtest.py photo.jpg                          # 30명, 30초, 로컬 서버
  python3 loadtest.py photo.jpg --users 50 --duration 60 --url https://api.내도메인.com
  python3 loadtest.py photo.jpg --same-client            # 모두 같은 사용자로 보내 429 동작 확인

동작
  - 기본(closed loop): 각 사용자가 0.3초 간격으로 보내되, 응답을 받은 뒤에 다음 사진을 보낸다 (권장 앱 동작과 같음)
  - --open-loop: 응답을 기다리지 않고 0.3초마다 무조건 보낸다 (최악의 경우)
  - 사용자마다 X-Client-Id 를 다르게 넣는다. 한 컴퓨터에서 보내면 IP가 모두 같아서,
    이게 없으면 IP당 초당 5회 제한에 바로 걸린다.
"""

import argparse
import asyncio
import os
import random
import statistics
import time
from collections import Counter

import httpx


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


async def main() -> None:
    ap = argparse.ArgumentParser(description="YOLOE /detect 부하 테스트")
    ap.add_argument("image", help="보낼 JPEG 파일")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default=os.getenv("API_KEY", ""), help="X-API-Key (기본: 환경변수 API_KEY)")
    ap.add_argument("--users", type=int, default=30, help="가상 사용자 수 (기본 30)")
    ap.add_argument("--interval", type=float, default=0.3, help="사용자별 전송 간격(초, 기본 0.3)")
    ap.add_argument("--duration", type=float, default=30, help="테스트 시간(초, 기본 30)")
    ap.add_argument("--targets", default="cup,mug,key,remote control,cell phone",
                    help="사용자마다 이 중 하나를 골라 찾음 (쉼표 구분)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--open-loop", action="store_true", help="응답을 기다리지 않고 간격마다 무조건 전송")
    ap.add_argument("--same-client", action="store_true", help="X-Client-Id 없이 전송 (IP 제한 확인용)")
    ap.add_argument("--timeout", type=float, default=10)
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        data = f.read()
    base = args.url.rstrip("/")
    choices = [t.strip() for t in args.targets.split(",") if t.strip()]

    results: list[dict] = []  # {status, latency_ms, server}
    stop_at = time.perf_counter() + args.duration

    limits = httpx.Limits(max_connections=args.users * 4, max_keepalive_connections=args.users * 2)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        h = await client.get(f"{base}/health", headers={"X-API-Key": args.key})
        print(f"[health] {h.status_code} {h.text}")
        if h.status_code != 200 or not h.json().get("model_loaded"):
            print("서버가 준비되지 않았습니다.")
            return

        async def send(uid: int, target: str) -> None:
            headers = {"X-API-Key": args.key}
            if not args.same_client:
                headers["X-Client-Id"] = f"loadtest-{uid}"
            t0 = time.perf_counter()
            try:
                r = await client.post(
                    f"{base}/detect",
                    headers=headers,
                    files={"image": ("frame.jpg", data, "image/jpeg")},
                    data={"targets": target, "imgsz": str(args.imgsz)},
                )
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                results.append({
                    "status": r.status_code,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "server": body.get("timing_ms"),
                    "batch": body.get("batch_size"),
                    "code": (body.get("error") or {}).get("code"),
                })
            except Exception as e:
                results.append({"status": type(e).__name__, "latency_ms": (time.perf_counter() - t0) * 1000})

        async def user(uid: int) -> None:
            target = random.choice(choices)
            await asyncio.sleep(random.uniform(0, args.interval))  # 시작 시점 분산
            pending: set[asyncio.Task] = set()
            while time.perf_counter() < stop_at:
                t0 = time.perf_counter()
                if args.open_loop:
                    task = asyncio.create_task(send(uid, target))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                else:
                    await send(uid, target)
                await asyncio.sleep(max(0.0, args.interval - (time.perf_counter() - t0)))
            if pending:
                await asyncio.gather(*pending)

        mode = "open-loop" if args.open_loop else "closed-loop"
        print(f"사용자 {args.users}명, {args.interval}s 간격, {args.duration:.0f}s 동안 ({mode}) ...")
        t_start = time.perf_counter()
        await asyncio.gather(*(user(i) for i in range(args.users)))
        elapsed = time.perf_counter() - t_start

        stats = await client.get(f"{base}/stats", headers={"X-API-Key": args.key})

    # ---- 결과 출력 -----------------------------------------------------------
    total = len(results)
    ok = [r for r in results if r["status"] == 200]
    lat = [r["latency_ms"] for r in ok]
    counts = Counter(str(r["status"]) + (f" {r['code']}" if r.get("code") else "") for r in results)

    print("\n===== 결과 =====")
    print(f"요청 {total}건, {total / elapsed:.1f} req/s (목표 {args.users / args.interval:.1f} req/s)")
    print(f"성공 {len(ok)}건, 실패율 {(1 - len(ok) / total) * 100 if total else 0:.1f}%")
    print("상태별:", dict(counts))
    if lat:
        print(f"응답 시간(ms): 평균 {statistics.mean(lat):.1f} | p50 {pct(lat, 50):.1f} | "
              f"p95 {pct(lat, 95):.1f} | p99 {pct(lat, 99):.1f} | 최대 {max(lat):.1f}")
        srv = [r["server"] for r in ok if r.get("server")]
        if srv:
            def avg(k: str) -> float:
                return statistics.mean(s.get(k, 0) for s in srv)
            print(f"서버 내부 평균(ms): 디코딩 {avg('decode'):.1f} | 대기 {avg('queue_wait'):.1f} | "
                  f"추론 {avg('inference'):.1f} | 서버 전체 {avg('total'):.1f}")
            print(f"평균 배치 크기: {statistics.mean(r['batch'] for r in ok if r.get('batch')):.1f}")
    if stats.status_code == 200:
        print("\n[/stats]", stats.text)


if __name__ == "__main__":
    asyncio.run(main())
