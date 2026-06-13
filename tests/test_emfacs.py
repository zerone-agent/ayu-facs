import pytest
from facs_engine import map_emfacs_emotions, fuse_emotions


def test_happiness_au06_au12_both_high():
    aus = {"AU06": 0.8, "AU12": 0.9}
    result = map_emfacs_emotions(aus)
    assert result["Happiness"] > 0.6


def test_happiness_one_low():
    aus = {"AU06": 0.8, "AU12": 0.2}
    result = map_emfacs_emotions(aus)
    assert result["Happiness"] < 0.5


def test_sadness_both_active():
    aus = {"AU01": 0.6, "AU04": 0.7}
    result = map_emfacs_emotions(aus)
    assert result["Sadness"] > 0.4


def test_anger_both_active():
    aus = {"AU04": 0.7, "AU09": 0.6}
    result = map_emfacs_emotions(aus)
    assert result["Anger"] > 0.5


def test_fear_three_of_five():
    aus = {"AU01": 0.5, "AU02": 0.6, "AU04": 0.5, "AU25": 0.1, "AU26": 0.1}
    result = map_emfacs_emotions(aus)
    assert result["Fear"] > 0.4


def test_surprise_strong():
    aus = {"AU01": 0.8, "AU02": 0.7, "AU25": 0.6, "AU26": 0.5}
    result = map_emfacs_emotions(aus)
    assert result["Surprise"] > 0.6


def test_disgust_au09_core():
    aus = {"AU09": 0.7, "AU04": 0.3}
    result = map_emfacs_emotions(aus)
    assert result["Disgust"] > 0.3


def test_no_au_active():
    aus = {au: 0.0 for au in [
        "AU01", "AU02", "AU04", "AU06", "AU09", "AU12", "AU25", "AU26",
    ]}
    result = map_emfacs_emotions(aus)
    for emotion, score in result.items():
        assert score < 0.1, f"{emotion} should be ~0 but got {score}"


def test_emotions_in_range():
    aus = {"AU06": 0.9, "AU12": 0.8, "AU04": 0.5}
    result = map_emfacs_emotions(aus)
    for emotion, score in result.items():
        assert 0.0 <= score <= 1.0, f"{emotion}={score} out of [0,1]"


def test_fuse_emotions_weights():
    pyfeat = {"Happiness": 0.8, "Sadness": 0.1, "Anger": 0.0, "Fear": 0.0, "Surprise": 0.1, "Disgust": 0.0}
    emfacs = {"Happiness": 0.6, "Sadness": 0.2, "Anger": 0.0, "Fear": 0.0, "Surprise": 0.2, "Disgust": 0.0}
    result = fuse_emotions(pyfeat, emfacs)
    assert abs(result["Happiness"] - (0.8 * 0.6 + 0.6 * 0.4)) < 0.01
    assert abs(result["Sadness"] - (0.1 * 0.6 + 0.2 * 0.4)) < 0.01
