from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import os
import tempfile

import cv2
import numpy as np
import torch

from src.config import (
    CORE_AUS,
    EMFACS_EMOTION_WEIGHT,
    FACE_DETECTION_THRESHOLD,
    PYFEAT_EMOTION_WEIGHT,
    get_device,
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

# OpenFace 3.0 emotion index → label mapping (AffectNet 8 classes)
OPENFACE_EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

# OpenFace 3.0 AU output index mapping
# The multitask model outputs AU intensities in a fixed order
OPENFACE_AU_ORDER = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
    "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
    "AU18", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU45",
]


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
    openface_emotions: dict[str, float],
    emfacs_emotions: dict[str, float],
) -> dict[str, float]:
    """Fuse OpenFace 3.0 native emotions with EMFACS rule results."""
    fused = {}
    for emotion in EMOTION_NAMES:
        p = openface_emotions.get(emotion, 0.0)
        e = emfacs_emotions.get(emotion, 0.0)
        fused[emotion] = round(
            p * PYFEAT_EMOTION_WEIGHT + e * EMFACS_EMOTION_WEIGHT, 4
        )
    return fused


# ──────────────────────────────────────────────
# FACS Engine (OpenFace 3.0 wrapper)
# ──────────────────────────────────────────────

def _init_openface():
    """Lazy import and initialize OpenFace 3.0 models."""
    from openface.face_detection import FaceDetector
    from openface.multitask_model import MultitaskPredictor

    device = get_device()
    logger.info(f"Initializing OpenFace 3.0 on device={device}")

    weights_dir = os.environ.get("OPENFACE_WEIGHTS_DIR",
                                  os.path.expanduser("~/.openface/weights"))

    face_model_path = os.path.join(weights_dir, "Alignment_RetinaFace.pth")
    multitask_model_path = os.path.join(weights_dir, "MTL_backbone.pth")

    if not os.path.exists(face_model_path) or not os.path.exists(multitask_model_path):
        logger.info("Model weights not found, downloading via huggingface_hub...")
        from huggingface_hub import snapshot_download
        os.makedirs(weights_dir, exist_ok=True)
        snapshot_download(repo_id="nutPace/openface_weights",
                         local_dir=weights_dir, repo_type="model")

    face_detector = FaceDetector(model_path=face_model_path, device=device)
    multitask_predictor = MultitaskPredictor(model_path=multitask_model_path, device=device)

    return face_detector, multitask_predictor


class FACSengine:
    """FACS analysis engine wrapping OpenFace 3.0."""

    def __init__(self):
        self._face_detector = None
        self._multitask_predictor = None
        self._tmpdir = tempfile.mkdtemp(prefix="ayu_")

    def _ensure_initialized(self):
        if self._face_detector is None:
            self._face_detector, self._multitask_predictor = _init_openface()

    def analyze_image(self, image_rgb: np.ndarray) -> dict:
        """
        Analyze a single RGB image.

        Args:
            image_rgb: numpy array (H, W, 3), uint8 RGB

        Returns:
            dict with success, aus, emotions, valence, arousal
        """
        try:
            self._ensure_initialized()

            # FaceDetector.get_face() only accepts file path — write to temp
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            tmp_path = os.path.join(self._tmpdir, "frame.jpg")
            cv2.imwrite(tmp_path, image_bgr)

            # Detect face (returns BGR cropped face)
            face, dets = self._face_detector.get_face(tmp_path)

            if face is None:
                return {
                    "success": False,
                    "error": "no_face_detected",
                    "frame_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Run multitask prediction (takes BGR face)
            emotion_logits, gaze_output, au_output = self._multitask_predictor.predict(face)

            # Parse AU intensities → normalize to 0-1 range
            au_values = au_output.squeeze().detach().cpu().numpy()
            aus = {}
            for i, au_name in enumerate(OPENFACE_AU_ORDER):
                if i < len(au_values):
                    aus[au_name] = round(min(float(au_values[i]) / 5.0, 1.0), 4)

            # Parse emotion logits → probabilities (AffectNet 8 classes)
            emotion_probs = torch.softmax(emotion_logits, dim=1).squeeze().detach().cpu().numpy()
            openface_emotions = {
                "Happiness": float(emotion_probs[1]),   # Happy
                "Sadness": float(emotion_probs[2]),      # Sad
                "Surprise": float(emotion_probs[3]),     # Surprise
                "Fear": float(emotion_probs[4]),         # Fear
                "Disgust": float(emotion_probs[5]),      # Disgust
                "Anger": float(emotion_probs[6]),        # Anger
            }

            # EMFACS rule-based emotions
            emfacs_emotions = map_emfacs_emotions(aus)

            # Fuse both sources
            emotions = fuse_emotions(openface_emotions, emfacs_emotions)

            # Estimate valence/arousal from emotion distribution
            valence = (
                emotions["Happiness"] * 1.0
                + emotions["Surprise"] * 0.2
                - emotions["Sadness"] * 0.8
                - emotions["Anger"] * 0.7
                - emotions["Fear"] * 0.6
                - emotions["Disgust"] * 0.5
            )
            arousal = (
                emotions["Surprise"] * 0.8
                + emotions["Fear"] * 0.9
                + emotions["Anger"] * 0.7
                + emotions["Happiness"] * 0.4
                - emotions["Sadness"] * 0.3
            )

            return {
                "success": True,
                "frame_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "aus": aus,
                "emotions": emotions,
                "valence": round(float(valence), 4),
                "arousal": round(float(arousal), 4),
            }

        except Exception as e:
            logger.error(f"FACS analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "frame_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
