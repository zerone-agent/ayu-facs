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
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert result["success"] is True
    assert len(result["aus"]) == 20


def test_engine_analyze_returns_emotions(mock_detector):
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert "emotions" in result
    for name in ["Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust"]:
        assert name in result["emotions"]


def test_engine_analyze_returns_valence_arousal(mock_detector):
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert "valence" in result
    assert "arousal" in result


def test_engine_no_face(mock_detector):
    mock_detector.detect.return_value = pd.DataFrame()
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert result["success"] is False
    assert result["error"] == "no_face_detected"
