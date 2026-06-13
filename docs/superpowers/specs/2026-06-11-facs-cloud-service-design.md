# FACS Cloud Service - Design Spec

## Overview

为心理评估场景构建 FACS（Facial Action Coding System）云服务。前端 MediaPipe 截取人脸，WebSocket 传输 224x224 帧到后端，后端通过 py-feat（后续可换 OpenFace 3.0）识别 20 个核心 AU，结合 EMFACS 规则映射为 6 类基础情绪，输出给咨询师参考。

## Architecture

```
┌──────────────┐    WebSocket     ┌──────────────────────────────────┐
│  前端         │ ── 224×224 RGBA ──→  FastAPI (:8000)               │
│  MediaPipe   │                   │  /ws/stream  - 实时帧流         │
│  Face Crop   │ ←── JSON ─────────│  /analyze     - 单帧 HTTP      │
│              │                   │  /health      - 健康检查        │
└──────────────┘                   │                                  │
                                   │  facs_engine.py (py-feat 封装)   │
                                   │    → 20 AU presence + intensity  │
                                   │    → EMFACS 6 类情绪映射        │
                                   │    → valence / arousal          │
                                   └──────────────────────────────────┘
```

## File Structure

```
ayu/
├── docker-compose.yml          # 一键启动
├── Dockerfile                   # 镜像构建
├── requirements.txt             # Python 依赖
├── server.py                    # FastAPI 主服务（WebSocket + HTTP）
├── facs_engine.py               # py-feat 封装（AU 检测 + EMFACS 映射）
├── config.py                    # 配置项（设备选择、帧率限制等）
├── test_client.html             # 前端测试页（MediaPipe 截脸 + WS 推帧）
└── docs/
    └── api.md                   # API 文档
```

## API Design

### POST /analyze

单帧图片分析。Request: `multipart/form-data`，字段 `file` 为图片文件。

Response:

```json
{
  "success": true,
  "frame_id": "uuid",
  "timestamp": "2026-06-11T14:30:00Z",
  "aus": {
    "AU01": {"presence": 0.92, "intensity": 2.1},
    "AU02": {"presence": 0.88, "intensity": 1.8}
  },
  "emotions": {
    "Happiness": 0.85,
    "Sadness": 0.03,
    "Anger": 0.01,
    "Fear": 0.02,
    "Surprise": 0.06,
    "Disgust": 0.03
  },
  "valence": 0.72,
  "arousal": 0.45
}
```

### WS /ws/stream

实时帧流。前端发送 224x224 RGBA 帧（binary），服务端返回 AU JSON。

Client → Server: binary frame (224x224x4 = 200704 bytes)
Server → Client: JSON (同 /analyze 格式，额外含 `latency_ms`)

### GET /health

`{"status": "ok", "device": "cpu"}`

## FACS Engine (facs_engine.py)

### 20 Core AUs

```
AU01(内眉上扬) AU02(外眉上扬) AU04(眉降)  AU05(上睑提升) AU06(颧颊提)
AU07(眼睑收紧) AU09(鼻皱)    AU10(上唇提) AU12(嘴角牵拉) AU14(酒窝)
AU15(嘴角下压) AU17(下巴上提) AU18(嘴唇撅) AU20(唇拉伸)   AU23(唇收紧)
AU24(唇按压)   AU25(唇分开)  AU26(下颌张) AU28(唇吸吮)   AU45(眨眼)
```

### EMFACS → 6 Basic Emotions Mapping

| Emotion   | Core AU Combination                   | Logic                              |
|-----------|---------------------------------------|------------------------------------|
| Happiness | AU06 + AU12                           | Both presence > 0.5, weight = mean |
| Sadness   | AU01 + AU04 + AU15                    | Any 2 > 0.4 triggers, AU15 x2     |
| Anger     | AU04 + AU05 + AU07 + AU23             | Any 3 > 0.4 triggers              |
| Fear      | AU01 + AU02 + AU04 + AU05 + AU20 + AU25 | Any 4 > 0.4 triggers           |
| Surprise  | AU01 + AU02 + AU05 + AU25 + AU26      | AU01+AU02+AU05 all > 0.5 = strong |
| Disgust   | AU09 + AU10 + AU17                    | AU09 core, > 0.5 triggers         |

Emotion Fusion: py-feat native emotion (0.6 weight) + EMFACS rule result (0.4 weight)。

### Pipeline

```
Input: 224x224 RGB numpy array
  → py-feat Detector.detect_image()
  → Extract 20 AU presence (0-1) + intensity
  → EMFACS rule mapping → 6 emotion scores
  → Fuse with py-feat native emotions (0.6 + 0.4)
  → Extract valence / arousal
  → Output: standardized JSON
```

### GPU/CPU Switch

- config.py: DEVICE = "auto" → auto-detect CUDA
- facs_engine.py: detector = Detector(device=config.DEVICE)
- docker-compose.yml: commented-out GPU reservation section

## Error Handling

| Scenario              | Response                                      |
|-----------------------|-----------------------------------------------|
| No face detected      | `{"success": false, "error": "no_face_detected"}` |
| py-feat inference error | Catch exception, return 500, don't crash    |
| Invalid WS frame      | Skip frame, send `{"error": "invalid_frame"}` |
| FPS exceeded          | Server-side throttle, drop frames over MAX_FPS |
| GPU OOM               | Auto fallback to CPU, log warning             |

## Docker Deployment

Dockerfile: python:3.11-slim + libgl1/libglib2.0 + pip install requirements
docker-compose.yml: single container, port 8000, DEVICE=auto, MAX_FPS=10, GPU section commented out

requirements.txt: fastapi, uvicorn[standard], py-feat>=0.7, numpy, python-multipart, websockets

## Test Client (test_client.html)

Single-file test page with MediaPipe Face Detection, 224x224 crop, WebSocket streaming, real-time AU bar chart + emotion radar chart + valence/arousal display.

## Quick Start

```bash
git clone <repo> && cd ayu
docker compose up --build
# Open http://localhost:8000 in browser
```

## Future (Out of Scope for MVP)

- Session mode: track visitor ID, multi-frame temporal trends
- OpenFace 3.0 replacement (drop-in engine swap)
- Redis for frame caching and session management
- SKILL wrapper for LLM psychological assessment
- Authentication and access control
