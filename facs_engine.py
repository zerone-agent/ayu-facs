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
