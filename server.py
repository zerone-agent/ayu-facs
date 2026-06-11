import asyncio
import logging
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
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
        return {"success": False, "error": "invalid_image", "frame_id": str(uuid.uuid4())}
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
                continue
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
