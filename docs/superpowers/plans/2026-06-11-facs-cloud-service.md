# FACS Cloud Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 FastAPI + py-feat 的 FACS 云服务，支持单帧图片分析和 WebSocket 实时帧流，输出 20 个 AU + EMFACS 6 类情绪。

**Architecture:** 单容器 Docker Compose 部署。FastAPI 暴露 HTTP + WebSocket 接口，facs_engine.py 封装 py-feat 检测器，config.py 集中管理配置（CPU/GPU 切换、帧率限制）。

**Tech Stack:** Python 3.11, FastAPI, uvicorn, py-feat, NumPy, PyTorch, Docker

---

## File Structure

```
ayu/
├── config.py                # 集中配置（DEVICE, MAX_FPS, AU 列表）
├── facs_engine.py           # py-feat 封装 + EMFACS 规则映射
├── server.py                # FastAPI 主服务
├── test_client.html         # 前端测试页（MediaPipe + WS）
├── requirements.txt         # Python 依赖
├── Dockerfile               # 镜像构建
├── docker-compose.yml       # 编排
├── tests/
│   ├── __init__.py
│   ├── test_emfacs.py       # EMFACS 规则映射测试
│   └── test_engine.py       # FACS engine 集成测试
└── docs/
    └── superpowers/
        ├── specs/2026-06-11-facs-cloud-service-design.md
        └── plans/2026-06-11-facs-cloud-service.md
```

---

### Task 1: 项目初始化 + config.py

**Files:**
- Create: `config.py`
- Create: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```txt
fastapi>=0.110
uvicorn[standard]>=0.29
py-feat>=0.7
numpy
python-multipart
websockets
pytest>=8.0
httpx
```

- [ ] **Step 2: 创建 tests/__init__.py**

空文件。

- [ ] **Step 3: 创建 config.py**

```python
import os
import torch

# === Device ===
# "auto" -> 自动检测 CUDA, "cpu" -> 强制 CPU, "cuda" -> 强制 GPU
DEVICE = os.getenv("DEVICE", "auto")


def get_device() -> str:
    """Resolve device string to actual pytorch device name."""
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


# === Frame Rate ===
MAX_FPS = int(os.getenv("MAX_FPS", "10"))

# === Server ===
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# === Face Detection ===
FACE_DETECTION_THRESHOLD = 0.5

# === 20 Core AUs (py-feat output column names) ===
CORE_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06",
    "AU07", "AU09", "AU10", "AU12", "AU14",
    "AU15", "AU17", "AU18", "AU20", "AU23",
    "AU24", "AU25", "AU26", "AU28", "AU45",
]

# === Emotion Fusion Weights ===
PYFEAT_EMOTION_WEIGHT = 0.6
EMFACS_EMOTION_WEIGHT = 0.4
```

- [ ] **Step 4: Commit**

```bash
git init
git add config.py requirements.txt tests/__init__.py
git commit -m "chore: project init with config and dependencies"
```

---

### Task 2: EMFACS 规则映射（TDD）

**Files:**
- Create: `tests/test_emfacs.py`
- Create: `facs_engine.py`（仅 EMFACS 部分）

这是纯逻辑模块，不依赖 py-feat，用 TDD 先写测试。

- [ ] **Step 1: 写 EMFACS 映射测试**

```python
# tests/test_emfacs.py
import pytest
from facs_engine import map_emfacs_emotions


def test_happiness_au06_au12_both_high():
    """AU06 + AU12 均 > 0.5 应触发 Happiness"""
    aus = {"AU06": 0.8, "AU12": 0.9}
    result = map_emfacs_emotions(aus)
    assert result["Happiness"] > 0.6


def test_happiness_one_low():
    """仅一个 AU > 0.5，Happiness 应较低"""
    aus = {"AU06": 0.8, "AU12": 0.2}
    result = map_emfacs_emotions(aus)
    assert result["Happiness"] < 0.5


def test_sadness_two_of_three():
    """AU01 + AU15 均 > 0.4 应触发 Sadness（AU15 权重 ×2）"""
    aus = {"AU01": 0.6, "AU04": 0.0, "AU15": 0.7}
    result = map_emfacs_emotions(aus)
    assert result["Sadness"] > 0.4


def test_anger_three_of_four():
    """AU04 + AU05 + AU23 均 > 0.4 应触发 Anger"""
    aus = {"AU04": 0.7, "AU05": 0.6, "AU07": 0.0, "AU23": 0.8}
    result = map_emfacs_emotions(aus)
    assert result["Anger"] > 0.5


def test_fear_four_of_six():
    """4/6 AU > 0.4 应触发 Fear"""
    aus = {"AU01": 0.5, "AU02": 0.6, "AU04": 0.5, "AU05": 0.5, "AU20": 0.1, "AU25": 0.1}
    result = map_emfacs_emotions(aus)
    assert result["Fear"] > 0.4


def test_surprise_strong():
    """AU01 + AU02 + AU05 均 > 0.5 强触发 Surprise"""
    aus = {"AU01": 0.8, "AU02": 0.7, "AU05": 0.9, "AU25": 0.6, "AU26": 0.5}
    result = map_emfacs_emotions(aus)
    assert result["Surprise"] > 0.6


def test_disgust_au09_core():
    """AU09 > 0.5 核心触发 Disgust"""
    aus = {"AU09": 0.7, "AU10": 0.3, "AU17": 0.2}
    result = map_emfacs_emotions(aus)
    assert result["Disgust"] > 0.3


def test_no_au_active():
    """所有 AU = 0 时，所有情绪应接近 0"""
    aus = {au: 0.0 for au in [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
        "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
        "AU18", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU45",
    ]}
    result = map_emfacs_emotions(aus)
    for emotion, score in result.items():
        assert score < 0.1, f"{emotion} should be ~0 but got {score}"


def test_emotions_sum_to_range():
    """输出值都应在 [0, 1] 范围内"""
    aus = {"AU06": 0.9, "AU12": 0.8, "AU04": 0.5}
    result = map_emfacs_emotions(aus)
    for emotion, score in result.items():
        assert 0.0 <= score <= 1.0, f"{emotion}={score} out of [0,1]"


def test_fuse_emotions_weights():
    """融合时 py-feat 权重 0.6, EMFACS 权重 0.4"""
    from facs_engine import fuse_emotions
    pyfeat = {"Happiness": 0.8, "Sadness": 0.1, "Anger": 0.0, "Fear": 0.0, "Surprise": 0.1, "Disgust": 0.0}
    emfacs = {"Happiness": 0.6, "Sadness": 0.2, "Anger": 0.0, "Fear": 0.0, "Surprise": 0.2, "Disgust": 0.0}
    result = fuse_emotions(pyfeat, emfacs)
    assert abs(result["Happiness"] - (0.8 * 0.6 + 0.6 * 0.4)) < 0.01
    assert abs(result["Sadness"] - (0.1 * 0.6 + 0.2 * 0.4)) < 0.01
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_emfacs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'facs_engine'`

- [ ] **Step 3: 实现 EMFACS 映射函数**

```python
# facs_engine.py
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import (
    CORE_AUS,
    EMFACS_EMOTION_WEIGHT,
    PYFEAT_EMOTION_WEIGHT,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# EMFACS Rule Definitions
# ──────────────────────────────────────────────

EMFACS_RULES: dict[str, dict[str, Any]] = {
    "Happiness": {
        "aus": ["AU06", "AU12"],
        "logic": "all",
        "threshold": 0.5,
        "weights": [1.0, 1.0],
    },
    "Sadness": {
        "aus": ["AU01", "AU04", "AU15"],
        "logic": "any_n",
        "n": 2,
        "threshold": 0.4,
        "weights": [1.0, 1.0, 2.0],
    },
    "Anger": {
        "aus": ["AU04", "AU05", "AU07", "AU23"],
        "logic": "any_n",
        "n": 3,
        "threshold": 0.4,
        "weights": [1.0, 1.0, 1.0, 1.0],
    },
    "Fear": {
        "aus": ["AU01", "AU02", "AU04", "AU05", "AU20", "AU25"],
        "logic": "any_n",
        "n": 4,
        "threshold": 0.4,
        "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "Surprise": {
        "aus": ["AU01", "AU02", "AU05", "AU25", "AU26"],
        "logic": "all_strong",
        "strong_aus": ["AU01", "AU02", "AU05"],
        "strong_threshold": 0.5,
        "threshold": 0.3,
        "weights": [1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "Disgust": {
        "aus": ["AU09", "AU10", "AU17"],
        "logic": "core",
        "core_au": "AU09",
        "core_threshold": 0.5,
        "threshold": 0.3,
        "weights": [2.0, 1.0, 1.0],
    },
}

EMOTION_NAMES = ["Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust"]


def _compute_rule_score(rule: dict[str, Any], aus: dict[str, float]) -> float:
    """Compute emotion score from a single EMFACS rule given AU presence values."""
    au_values = [aus.get(au, 0.0) for au in rule["aus"]]
    weights = rule["weights"]
    threshold = rule["threshold"]
    logic = rule["logic"]

    if logic == "all":
        if all(v >= threshold for v in au_values):
            weighted = sum(v * w for v, w in zip(au_values, weights))
            return min(weighted / sum(weights), 1.0)
        return 0.0

    if logic == "any_n":
        n = rule["n"]
        above = sum(1 for v in au_values if v >= threshold)
        if above >= n:
            active = [(v, w) for v, w in zip(au_values, weights) if v >= threshold]
            weighted = sum(v * w for v, w in active)
            return min(weighted / sum(w for _, w in active), 1.0)
        return 0.0

    if logic == "all_strong":
        strong_aus = rule["strong_aus"]
        strong_threshold = rule["strong_threshold"]
        strong_values = [aus.get(au, 0.0) for au in strong_aus]
        if all(v >= strong_threshold for v in strong_values):
            weighted = sum(v * w for v, w in zip(au_values, weights))
            return min(weighted / sum(weights), 1.0)
        return 0.0

    if logic == "core":
        core_au = rule["core_au"]
        core_threshold = rule["core_threshold"]
        if aus.get(core_au, 0.0) >= core_threshold:
            weighted = sum(v * w for v, w in zip(au_values, weights))
            return min(weighted / sum(weights), 1.0)
        return 0.0

    return 0.0


def map_emfacs_emotions(aus: dict[str, float]) -> dict[str, float]:
    """Map AU presence values to 6 basic emotions using EMFACS rules."""
    result = {}
    for emotion in EMOTION_NAMES:
        rule = EMFACS_RULES[emotion]
        result[emotion] = round(_compute_rule_score(rule, aus), 4)
    return result


def fuse_emotions(
    pyfeat_emotions: dict[str, float],
    emfacs_emotions: dict[str, float],
) -> dict[str, float]:
    """Fuse py-feat native emotions with EMFACS rule results."""
    fused = {}
    for emotion in EMOTION_NAMES:
        p = pyfeat_emotions.get(emotion, 0.0)
        e = emfacs_emotions.get(emotion, 0.0)
        fused[emotion] = round(
            p * PYFEAT_EMOTION_WEIGHT + e * EMFACS_EMOTION_WEIGHT, 4
        )
    return fused
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_emfacs.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_emfacs.py facs_engine.py
git commit -m "feat: add EMFACS rule mapping with TDD tests"
```

---

### Task 3: FACS Engine（py-feat 封装）

**Files:**
- Create: `tests/test_engine.py`
- Modify: `facs_engine.py`（添加 FACSengine 类）

- [ ] **Step 1: 写 engine 测试**

```python
# tests/test_engine.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from facs_engine import FACSengine


@pytest.fixture
def mock_detector():
    """Mock py-feat Detector"""
    with patch("facs_engine._create_detector") as mock_create:
        detector = MagicMock()
        # Build a fake Fex DataFrame with AU + emotion columns
        columns = (
            ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
             "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
             "AU18", "AU20", "AU23", "AU24", "AU25", "AU26",
             "AU28", "AU45"]
            + ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
            + ["valence", "arousal"]
        )
        values = [0.5] * 20 + [0.1] * 7 + [0.3, 0.4]
        fex = pd.DataFrame([values], columns=columns)
        detector.detect.return_value = fex
        mock_create.return_value = detector
        yield detector


def test_engine_analyze_returns_aus(mock_detector):
    """analyze_image 应返回 20 个 AU"""
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert result["success"] is True
    assert len(result["aus"]) == 20


def test_engine_analyze_returns_emotions(mock_detector):
    """analyze_image 应返回 6 类情绪"""
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert "emotions" in result
    emotions = result["emotions"]
    for name in ["Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust"]:
        assert name in emotions


def test_engine_analyze_returns_valence_arousal(mock_detector):
    """analyze_image 应返回 valence 和 arousal"""
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert "valence" in result
    assert "arousal" in result


def test_engine_no_face(mock_detector):
    """无人脸时应返回 success=False"""
    mock_detector.detect.return_value = pd.DataFrame()
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert result["success"] is False
    assert result["error"] == "no_face_detected"
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL — `ImportError: cannot import name 'FACSengine'`

- [ ] **Step 3: 实现 FACSengine 类**

在 `facs_engine.py` 末尾追加：

```python
# ── 以下追加到 facs_engine.py 末尾 ──

import torch
import uuid
from datetime import datetime, timezone

from config import get_device, CORE_AUS, FACE_DETECTION_THRESHOLD


def _create_detector():
    """Lazy import to avoid loading py-feat at module level."""
    from feat import Detector
    device = get_device()
    logger.info(f"Initializing py-feat Detector on device={device}")
    return Detector(device=device)


class FACSengine:
    """FACS analysis engine wrapping py-feat Detector."""

    def __init__(self):
        self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            self._detector = _create_detector()
        return self._detector

    def analyze_image(self, image_rgb: np.ndarray) -> dict:
        """
        Analyze a single RGB image.

        Args:
            image_rgb: numpy array (H, W, 3), uint8 RGB

        Returns:
            dict with success, aus, emotions, valence, arousal
        """
        try:
            tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.float() / 255.0

            fex = self.detector.detect(
                tensor,
                data_type="tensor",
                face_detection_threshold=FACE_DETECTION_THRESHOLD,
            )

            if fex.empty:
                return {
                    "success": False,
                    "error": "no_face_detected",
                    "frame_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            row = fex.iloc[0]

            # Extract AU presence values
            aus = {}
            for au_name in CORE_AUS:
                aus[au_name] = round(float(row.get(au_name, 0.0)), 4)

            # py-feat native emotions (column names are lowercase)
            pyfeat_emotions = {
                "Happiness": float(row.get("happiness", 0.0)),
                "Sadness": float(row.get("sadness", 0.0)),
                "Anger": float(row.get("anger", 0.0)),
                "Fear": float(row.get("fear", 0.0)),
                "Surprise": float(row.get("surprise", 0.0)),
                "Disgust": float(row.get("disgust", 0.0)),
            }

            # EMFACS rule-based emotions
            emfacs_emotions = map_emfacs_emotions(aus)

            # Fuse both sources
            emotions = fuse_emotions(pyfeat_emotions, emfacs_emotions)

            return {
                "success": True,
                "frame_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "aus": aus,
                "emotions": emotions,
                "valence": round(float(row.get("valence", 0.0)), 4),
                "arousal": round(float(row.get("arousal", 0.0)), 4),
            }

        except Exception as e:
            logger.error(f"FACS analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "frame_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_engine.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_engine.py facs_engine.py
git commit -m "feat: add FACSengine with py-feat wrapper and tests"
```

---

### Task 4: FastAPI Server

**Files:**
- Create: `server.py`

- [ ] **Step 1: 实现 server.py**

```python
# server.py
import asyncio
import logging
import time
import uuid

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

from config import MAX_FPS
from facs_engine import FACSengine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FACS Cloud Service", version="0.1.0")

engine = FACSengine()

FRAME_INTERVAL = 1.0 / MAX_FPS


@app.get("/health")
async def health():
    from config import get_device
    return {"status": "ok", "device": get_device()}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "invalid_image"},
        )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = engine.analyze_image(image_rgb)
    return result


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    last_time = 0.0

    try:
        while True:
            data = await websocket.receive_bytes()

            now = time.time()
            if now - last_time < FRAME_INTERVAL:
                continue  # throttle
            last_time = now

            if len(data) != 224 * 224 * 4:
                await websocket.send_json({"error": "invalid_frame", "expected_size": 224 * 224 * 4})
                continue

            start = time.time()

            rgba = np.frombuffer(data, dtype=np.uint8).reshape(224, 224, 4)
            rgb = rgba[:, :, :3].copy()

            result = engine.analyze_image(rgb)
            result["latency_ms"] = round((time.time() - start) * 1000, 1)

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"success": False, "error": str(e)})
        except Exception:
            pass


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "test_client.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>FACS Cloud Service</h1><p>test_client.html not found</p>")


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
```

- [ ] **Step 2: 手动验证 health 端点**

Run: `python -c "from server import app; print('server module OK')"`
Expected: `server module OK`（确认导入无错）

- [ ] **Step 3: Commit**

```bash
git add server.py
git commit -m "feat: add FastAPI server with /analyze, /ws/stream, /health"
```

---

### Task 5: 前端测试页 test_client.html

**Files:**
- Create: `test_client.html`

- [ ] **Step 1: 实现前端测试页**

单文件 HTML，包含：
- MediaPipe Face Detection 加载
- 摄像头打开 + 人脸裁剪到 224×224
- WebSocket 连接 + 帧推流
- 实时 AU 柱状图 + 情绪显示 + valence/arousal

```html
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>FACS Cloud Service - Test Client</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
h1 { text-align: center; margin-bottom: 16px; font-size: 20px; }
#status { text-align: center; margin-bottom: 12px; font-size: 14px; color: #888; }
.container { display: flex; gap: 20px; max-width: 1200px; margin: 0 auto; }
.panel { flex: 1; background: #16213e; border-radius: 12px; padding: 16px; }
.panel h2 { font-size: 14px; margin-bottom: 12px; color: #e94560; }
#video-wrap { position: relative; }
#video, #overlay { width: 100%; border-radius: 8px; display: block; }
#overlay { position: absolute; top: 0; left: 0; }
#face-crop { width: 224px; height: 224px; border: 2px solid #e94560; border-radius: 8px; margin-top: 8px; }
#au-bars { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
.au-row { display: flex; align-items: center; gap: 6px; font-size: 11px; }
.au-label { width: 42px; text-align: right; }
.au-bar-bg { flex: 1; height: 12px; background: #0f3460; border-radius: 6px; overflow: hidden; }
.au-bar { height: 100%; background: #e94560; border-radius: 6px; transition: width 0.15s; }
.au-val { width: 36px; font-size: 10px; color: #aaa; }
#emotions { margin-top: 16px; }
.emo-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }
.emo-label { width: 80px; }
.emo-bar-bg { flex: 1; height: 16px; background: #0f3460; border-radius: 8px; overflow: hidden; }
.emo-bar { height: 100%; border-radius: 8px; transition: width 0.15s; }
.emo-val { width: 40px; font-size: 11px; color: #aaa; text-align: right; }
#va-display { margin-top: 16px; display: flex; gap: 24px; font-size: 13px; }
.va-item span { font-weight: bold; }
.btn { padding: 8px 16px; border: none; border-radius: 8px; cursor: pointer; font-size: 13px; margin: 4px; }
.btn-start { background: #e94560; color: #fff; }
.btn-stop { background: #555; color: #fff; }
#controls { text-align: center; margin-bottom: 12px; }
</style>
</head>
<body>
<h1>FACS Cloud Service — Test Client</h1>
<div id="status">Disconnected</div>
<div id="controls">
  <button class="btn btn-start" id="btn-start" onclick="startSession()">Start</button>
  <button class="btn btn-stop" id="btn-stop" onclick="stopSession()" disabled>Stop</button>
</div>
<div class="container">
  <div class="panel">
    <h2>Camera</h2>
    <div id="video-wrap">
      <video id="video" autoplay playsinline></video>
      <canvas id="overlay"></canvas>
    </div>
    <canvas id="face-crop" width="224" height="224"></canvas>
  </div>
  <div class="panel">
    <h2>Action Units</h2>
    <div id="au-bars"></div>
    <h2 style="margin-top:16px">Emotions</h2>
    <div id="emotions"></div>
    <div id="va-display">
      <div class="va-item">Valence: <span id="valence">0.00</span></div>
      <div class="va-item">Arousal: <span id="arousal">0.00</span></div>
      <div class="va-item">Latency: <span id="latency">—</span> ms</div>
    </div>
  </div>
</div>

<script type="module">
import { FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

const AU_NAMES = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU12","AU14","AU15","AU17","AU18","AU20","AU23","AU24","AU25","AU26","AU28","AU45"];
const EMO_NAMES = ["Happiness","Sadness","Anger","Fear","Surprise","Disgust"];
const EMO_COLORS = {"Happiness":"#f9c74f","Sadness":"#577590","Anger":"#f94144","Fear":"#43aa8b","Surprise":"#f8961e","Disgust":"#90be6d"};

let ws = null, faceDetector = null, running = false, lastSend = 0;
const FPS = 10, INTERVAL = 1000 / FPS;

// Build AU bars
const auContainer = document.getElementById("au-bars");
AU_NAMES.forEach(au => {
  auContainer.innerHTML += `<div class="au-row"><span class="au-label">${au}</span><div class="au-bar-bg"><div class="au-bar" id="au-${au}"></div></div><span class="au-val" id="auv-${au}">0.00</span></div>`;
});
// Build emotion bars
const emoContainer = document.getElementById("emotions");
EMO_NAMES.forEach(e => {
  emoContainer.innerHTML += `<div class="emo-row"><span class="emo-label">${e}</span><div class="emo-bar-bg"><div class="emo-bar" id="emo-${e}" style="background:${EMO_COLORS[e]}"></div></div><span class="emo-val" id="emov-${e}">0.00</span></div>`;
});

async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm");
  faceDetector = await FaceDetector.createFromOptions(vision, { runningMode: "VIDEO", model: "short" });
}

window.startSession = async function() {
  if (!faceDetector) await initMediaPipe();
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
  const video = document.getElementById("video");
  video.srcObject = stream;
  await video.play();

  const overlay = document.getElementById("overlay");
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  ws = new WebSocket(`ws://${location.host}/ws/stream`);
  ws.onopen = () => { document.getElementById("status").textContent = "Connected"; running = true; detectLoop(); };
  ws.onmessage = (e) => { const d = JSON.parse(e.data); renderResult(d); };
  ws.onclose = () => { document.getElementById("status").textContent = "Disconnected"; running = false; };
  ws.onerror = () => { document.getElementById("status").textContent = "Error"; };

  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-stop").disabled = false;
};

window.stopSession = function() {
  running = false;
  if (ws) ws.close();
  const video = document.getElementById("video");
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  document.getElementById("btn-start").disabled = false;
  document.getElementById("btn-stop").disabled = true;
  document.getElementById("status").textContent = "Disconnected";
};

function detectLoop() {
  if (!running) return;
  const video = document.getElementById("video");
  const now = performance.now();
  if (now - lastSend >= INTERVAL) {
    lastSend = now;
    const result = faceDetector.detectForVideo(video, now);
    if (result.detections.length > 0) {
      const det = result.detections[0];
      const bb = det.boundingBox;
      // Draw bounding box
      const ctx = document.getElementById("overlay").getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.strokeStyle = "#e94560";
      ctx.lineWidth = 2;
      ctx.strokeRect(bb.x, bb.y, bb.width, bb.height);
      // Crop face to 224x224
      const cropCanvas = document.getElementById("face-crop");
      const cctx = cropCanvas.getContext("2d");
      cctx.drawImage(video, bb.x, bb.y, bb.width, bb.height, 0, 0, 224, 224);
      // Get RGBA pixels and send
      const imgData = cctx.getImageData(0, 0, 224, 224);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(imgData.data.buffer);
      }
    }
  }
  requestAnimationFrame(detectLoop);
}

function renderResult(data) {
  if (!data.success) return;
  // AU bars
  if (data.aus) {
    for (const [au, val] of Object.entries(data.aus)) {
      const bar = document.getElementById(`au-${au}`);
      const label = document.getElementById(`auv-${au}`);
      if (bar) { bar.style.width = `${Math.min(val * 100, 100)}%`; }
      if (label) { label.textContent = val.toFixed(2); }
    }
  }
  // Emotion bars
  if (data.emotions) {
    for (const [emo, val] of Object.entries(data.emotions)) {
      const bar = document.getElementById(`emo-${emo}`);
      const label = document.getElementById(`emov-${emo}`);
      if (bar) { bar.style.width = `${Math.min(val * 100, 100)}%`; }
      if (label) { label.textContent = val.toFixed(2); }
    }
  }
  // Valence / Arousal
  if (data.valence !== undefined) document.getElementById("valence").textContent = data.valence.toFixed(2);
  if (data.arousal !== undefined) document.getElementById("arousal").textContent = data.arousal.toFixed(2);
  if (data.latency_ms !== undefined) document.getElementById("latency").textContent = data.latency_ms;
}
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add test_client.html
git commit -m "feat: add test client with MediaPipe face crop and WebSocket streaming"
```

---

### Task 6: Docker 部署文件

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

- [ ] **Step 1: 创建 .dockerignore**

```
__pycache__
*.pyc
.git
docs/superpowers
tests
*.md
.DS_Store
```

- [ ] **Step 2: 创建 Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py facs_engine.py server.py test_client.html ./

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: 创建 docker-compose.yml**

```yaml
services:
  facs:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEVICE=auto
      - MAX_FPS=10
    # GPU support (uncomment to enable)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

- [ ] **Step 4: 验证 Docker 构建**

Run: `docker compose build`
Expected: 构建成功无报错

- [ ] **Step 5: Commit**

```bash
git add .dockerignore Dockerfile docker-compose.yml
git commit -m "feat: add Docker deployment with GPU reservation"
```

---

### Task 7: 集成测试

**Files:**
- 无新文件

- [ ] **Step 1: 启动服务**

Run: `docker compose up --build -d`
Expected: 容器启动，`docker compose logs` 显示 `Uvicorn running on 0.0.0.0:8000`

- [ ] **Step 2: 测试 health 端点**

Run: `curl http://localhost:8000/health`
Expected: `{"status":"ok","device":"cpu"}`

- [ ] **Step 3: 测试单帧分析**

```bash
# 使用 py-feat 自带测试图或任意含人脸的图片
curl -X POST http://localhost:8000/analyze \
  -F "file=@test_face.jpg" | python -m json.tool
```

Expected: JSON 包含 `aus`（20 个）、`emotions`（6 个）、`valence`、`arousal`

- [ ] **Step 4: 浏览器测试 WebSocket**

打开 `http://localhost:8000`，点击 Start，允许摄像头，观察：
- 左侧摄像头画面显示人脸框
- 224×224 裁剪区显示人脸
- 右侧 AU 柱状图实时变化
- 情绪条实时变化
- valence/arousal/latency 数值更新

Expected: 所有 UI 元素正常响应

- [ ] **Step 5: 运行单元测试**

Run: `docker compose exec facs python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 6: 停止服务**

Run: `docker compose down`
