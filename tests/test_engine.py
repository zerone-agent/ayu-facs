import os
import tempfile

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from facs_engine import FACSengine


@pytest.fixture
def mock_init():
    """Mock OpenFace 3.0 detector and multitask predictor."""
    with patch("facs_engine._init_openface") as mock_create:
        detector = MagicMock()
        predictor = MagicMock()

        # Mock a detected BGR face (48, 48, 3)
        detector.get_face.return_value = (
            np.zeros((48, 48, 3), dtype=np.uint8),
            np.array([[0, 0, 48, 48, 0.95]]),
        )

        # Mock OpenFace 3.0 outputs: emotion logits, gaze, AU intensities
        emotion_logits = torch.zeros((1, 8))
        emotion_logits[0, 1] = 1.0  # Happiness highest
        gaze_output = torch.zeros((1, 2))
        au_output = torch.zeros((1, 20))
        au_output[0, 5] = 2.5   # AU06
        au_output[0, 8] = 3.0   # AU12
        predictor.predict.return_value = (emotion_logits, gaze_output, au_output)

        mock_create.return_value = (detector, predictor)
        yield detector, predictor


def test_engine_analyze_returns_aus(mock_init):
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert result["success"] is True
    assert len(result["aus"]) == 20


def test_engine_analyze_returns_emotions(mock_init):
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert "emotions" in result
    for name in ["Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust"]:
        assert name in result["emotions"]


def test_engine_analyze_returns_valence_arousal(mock_init):
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert "valence" in result
    assert "arousal" in result


def test_engine_no_face(mock_init):
    detector, _ = mock_init
    detector.get_face.return_value = (None, None)
    engine = FACSengine()
    result = engine.analyze_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert result["success"] is False
    assert result["error"] == "no_face_detected"
