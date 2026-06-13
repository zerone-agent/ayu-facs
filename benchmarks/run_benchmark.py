#!/usr/bin/env python3
"""Run qualitative benchmark against the /analyze endpoint."""

import json
import os
import time
from pathlib import Path

import requests

IMAGES_DIR = Path(__file__).parent / "images"
DEFAULT_URL = "http://47.120.38.148:8003/analyze"
NEUTRAL_THRESHOLD = 0.1


def run_benchmark(url: str = DEFAULT_URL) -> dict:
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images not found at {IMAGES_DIR}. Run download_images.py first."
        )

    results = {}
    print("=" * 60)
    print("AYU FACS Benchmark")
    print(f"Endpoint: {url}")
    print("=" * 60)

    for path in sorted(IMAGES_DIR.glob("*.jpg")):
        expected = path.stem
        with open(path, "rb") as f:
            start = time.time()
            resp = requests.post(url, files={"file": f}, timeout=60)
            elapsed = time.time() - start

        data = resp.json()
        results[expected] = {"elapsed": elapsed, "response": data}

        print(f"\n[{expected}] {elapsed:.2f}s")
        if not data.get("success"):
            print(f"  ❌ FAIL: {data.get('error')}")
            continue

        emotions = data["emotions"]
        top_emotion, top_score = max(emotions.items(), key=lambda x: x[1])
        max_score = max(emotions.values())

        if expected == "neutral":
            passed = max_score < NEUTRAL_THRESHOLD
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} (neutral: max emotion={max_score:.3f})")
        else:
            passed = top_emotion.lower() == expected
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  predicted: {top_emotion}={top_score:.3f}")
            print(f"  expected:  {expected}")
            print(f"  {status}")

        print(f"  valence={data['valence']:.3f}, arousal={data['arousal']:.3f}")

    results_path = IMAGES_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    run_benchmark(args.url)
