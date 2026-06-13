#!/usr/bin/env python3
"""Download benchmark expression images from public sources.

FACES images: public preview collection, research-only license.
Surprise image: Wikimedia Commons CC BY-SA 4.0.
"""

import json
import os
import subprocess
import urllib.parse

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

USER_AGENT = "ayu-facs-benchmark/1.0 (research)"
FACES_COLLECTION = "IXTdg721TwZwyZ8e"

# Desired images from FACES public preview (emotion -> filename pattern)
FACES_TARGETS = {
    "anger": "116_m_m_a_a.jpg",
    "disgust": "066_y_m_d_a.jpg",
    "fear": "066_y_m_f_a.jpg",
    "happiness": "066_y_m_h_a.jpg",
    "neutral": "066_y_m_n_a.jpg",
    "sadness": "066_y_m_s_a.jpg",
}

SURPRISE = {
    "url": "https://upload.wikimedia.org/wikipedia/commons/8/8f/The_sincerity_of_amazement.jpg",
    "filename": "surprise.jpg",
    "attribution": "Alberto Castello, CC BY-SA 4.0",
}


def curl_download(url: str, out_path: str) -> None:
    """Download with curl using TLS 1.2 (required by some hosts)."""
    cmd = [
        "curl", "--tlsv1.2", "-L", "-A", USER_AGENT,
        "-o", out_path, url,
    ]
    subprocess.run(cmd, check=True)


def fetch_faces_items() -> dict:
    """Fetch all items from FACES public preview via REST API."""
    items = {}
    offset = 0
    size = 50
    while True:
        url = (
            f"https://faces.mpdl.mpg.de/imeji/rest/collections/{FACES_COLLECTION}/items"
            f"?offset={offset}&size={size}"
        )
        cmd = ["curl", "--tlsv1.2", "-s", "-A", USER_AGENT, url]
        resp = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(resp.stdout)
        for item in data.get("results", []):
            items[item["filename"]] = item
        total = data.get("totalNumberOfResults", 0)
        if len(items) >= total:
            break
        offset += size
    return items


def main():
    faces_items = fetch_faces_items()

    for emotion, filename in FACES_TARGETS.items():
        item = faces_items.get(filename)
        if not item:
            raise RuntimeError(f"FACES image not found: {filename}")
        out_path = os.path.join(IMAGES_DIR, f"{emotion}.jpg")
        curl_download(item["fileUrl"], out_path)
        print(f"Downloaded {emotion}: {filename}")

    surprise_path = os.path.join(IMAGES_DIR, SURPRISE["filename"])
    curl_download(SURPRISE["url"], surprise_path)
    print(f"Downloaded surprise: {SURPRISE['filename']} ({SURPRISE['attribution']})")

    print(f"\nAll images saved to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
