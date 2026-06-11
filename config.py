import os
import torch

# === Device ===
# "auto" -> auto-detect CUDA, "cpu" -> force CPU, "cuda" -> force GPU
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
