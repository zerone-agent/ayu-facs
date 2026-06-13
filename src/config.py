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

# === 8 Core AUs output by OpenFace 3.0 MTL model ===
# Source: CMU OpenFace-3.0 demo2.py au_labels
CORE_AUS = [
    "AU01",  # Inner Brow Raiser
    "AU02",  # Outer Brow Raiser
    "AU04",  # Brow Lowerer
    "AU06",  # Cheek Raiser
    "AU09",  # Nose Wrinkler
    "AU12",  # Lip Corner Puller
    "AU25",  # Lips Part
    "AU26",  # Jaw Drop
]

# === Emotion Fusion Weights ===
PYFEAT_EMOTION_WEIGHT = 0.6
EMFACS_EMOTION_WEIGHT = 0.4
