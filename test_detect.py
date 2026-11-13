"""
/detect 테스트 스크립트: 사진 한 장을 보내고 결과와 걸린 시간을 출력한다.

사용 예
  python test_detect.py photo.jpg --targets "mug,cup"
  python test_detect.py photo.jpg --url https://xxxx.trycloudflare.com --key 내키 --repeat 5
  python test_detect.py cup_crop.jpg --describe       # /describe (Gemini 중계) 테스트
  (--key 를 생략하면 환경변수 API_KEY 사용)
"""

import argparse
import json
import os
import statistics
import sys
import time

import requests


def main() -> None:
    p = argparse.ArgumentParser(description="YOLOE /detect 테스트")
    p.add_argument("image", help="보낼 JPEG 파일 경로")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="서버 주소 (기본 http://127.0.0.1:8000)")
    p.add_argument("--key", default=os.getenv("API_KEY", ""), help="X-API-Key (기본: 환경변수 API_KEY)")
    p.add_argument("--targets", default="cup", help="찾을 물체, 쉼표로 여러 개 (기본 cup)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--describe", action="store_true", help="/detect 대신 /describe 로 보내기")
    p.add_argument("--repeat", type=int, default=1, help="같은 요청 반복 횟수 (두 번째부터가 실제 속도에 가까움)")
    args = p.parse_args()

    with open(args.image, "rb") as f:
        data = f.read()
    base = args.url.rstrip("/")
    headers = {"X-API-Key": args.key, "X-Client-Id": "test-script"}
    session = requests.Session()  # 연결 재사용 (폰 앱도 이렇게 하는 게 빠름)

    # 서버 상태 먼저 확인
    h = session.get(f"{base}/health", headers=headers, timeout=10)
    print(f"[health] {h.status_code} {h.text}")
    if h.status_code != 200:
        sys.exit(1)

    if args.describe:
        t0 = time.perf_counter()
        r = session.post(f"{base}/describe", headers=headers,
                         files={"image": (os.path.basename(args.image), data, "image/jpeg")}, timeout=60)
        print(f"\n[describe] HTTP {r.status_code}, 왕복 {(time.perf_counter() - t0) * 1000:.1f} ms")
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))
        return

    rtts = []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        r = session.post(
            f"{base}/detect",
            headers=headers,
            files={"image": (os.path.basename(args.image), data, "image/jpeg")},
            data={"targets": args.targets, "conf": str(args.conf), "imgsz": str(args.imgsz)},
            timeout=30,
        )
        rtt = (time.perf_counter() - t0) * 1000
        rtts.append(rtt)

        print(f"\n[{i + 1}/{args.repeat}] HTTP {r.status_code}, 왕복 {rtt:.1f} ms (이미지 {len(data) / 1024:.0f} KB)")
        try:
            body = r.json()
        except ValueError:
            print(r.text)
            continue
        if r.status_code != 200:
            print(json.dumps(body, ensure_ascii=False, indent=2))
            continue

        print(f"  device={body['device']} model={body['model']}")
        print(f"  서버 시간(ms): {body['timing_ms']}  배치 크기: {body.get('batch_size')}")
        print(f"  네트워크 등 서버 밖 시간: {rtt - body['timing_ms']['total']:.1f} ms")
        if not body["detections"]:
            print("  찾은 물체 없음")
        for d in body["detections"]:
            b = d["box"]
            cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
            print(
                f"  - {d['label']:<14} (요청: {d.get('target', d['label'])}) {d['score']:.2f}  "
                f"x={b['x']:.3f} y={b['y']:.3f} w={b['w']:.3f} h={b['h']:.3f}  (중심 {cx:.2f}, {cy:.2f})"
            )

    if len(rtts) > 1:
        warm = rtts[1:]
        print(f"\n왕복 시간: 첫 요청 {rtts[0]:.1f} ms, 이후 평균 {statistics.mean(warm):.1f} ms, 최대 {max(warm):.1f} ms")


if __name__ == "__main__":
    main()
