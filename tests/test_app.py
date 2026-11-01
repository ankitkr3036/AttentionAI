
import tempfile

import numpy as np
import pytest  # pyright: ignore[reportMissingImports]

from database import Database
from detector import (
    classify_attention,
    eye_aspect_ratio,
    calculate_attention_score
)


# --------------------------------
# EYE ASPECT RATIO TESTS
# --------------------------------

def test_eye_aspect_ratio_open_eye():

    points = np.array([
        [0, 0],
        [2, -2],
        [6, -2],
        [10, 0],
        [6, 2],
        [2, 2]
    ], dtype=float)

    assert eye_aspect_ratio(points) == pytest.approx(0.4)


def test_eye_aspect_ratio_rejects_wrong_shape():

    with pytest.raises(ValueError):

        eye_aspect_ratio(np.zeros((5, 2)))


def test_eye_aspect_ratio_closed_eye():

    points = np.array([
        [0, 0],
        [2, 0.1],
        [6, 0.1],
        [10, 0],
        [6, -0.1],
        [2, -0.1]
    ], dtype=float)

    assert eye_aspect_ratio(points) < 0.1


# --------------------------------
# ATTENTION CLASSIFICATION TESTS
# --------------------------------

def test_classification_states():

    assert classify_attention(
        False, None, 0, "Unknown", "Unknown"
    ) == "No face detected"

    assert classify_attention(
        True, 0.1, 2.0, "Center", "Forward"
    ) == "Drowsy"

    assert classify_attention(
        True, 0.3, 0, "Left", "Forward"
    ) == "Distracted"

    assert classify_attention(
        True, 0.3, 0, "Center", "Forward"
    ) == "Focused"


# --------------------------------
# ATTENTION SCORE TESTS
# --------------------------------

def test_attention_score_100_percent():

    score = calculate_attention_score(
        gaze="Center",
        head_pose="Forward",
        ear=0.3,
        closed_seconds=0
    )

    assert score == 100


def test_attention_score_gaze_left():

    score = calculate_attention_score(
        gaze="Left",
        head_pose="Forward",
        ear=0.3,
        closed_seconds=0
    )

    assert score == 80


def test_attention_score_head_turned():

    score = calculate_attention_score(
        gaze="Center",
        head_pose="Turned left",
        ear=0.3,
        closed_seconds=0
    )

    assert score == 80


def test_attention_score_head_lowered():

    score = calculate_attention_score(
        gaze="Center",
        head_pose="Lowered",
        ear=0.3,
        closed_seconds=0
    )

    assert score == 75


def test_attention_score_short_eye_closure():

    score = calculate_attention_score(
        gaze="Center",
        head_pose="Forward",
        ear=0.1,
        closed_seconds=0.5
    )

    assert score == 90


def test_attention_score_long_eye_closure():

    score = calculate_attention_score(
        gaze="Center",
        head_pose="Forward",
        ear=0.1,
        closed_seconds=2.0
    )

    assert score == 50


def test_attention_score_unknown_gaze():

    score = calculate_attention_score(
        gaze="Unknown",
        head_pose="Forward",
        ear=0.3,
        closed_seconds=0
    )

    assert score == 95


def test_attention_score_never_exceeds_100():

    score = calculate_attention_score(
        gaze="Center",
        head_pose="Forward",
        ear=0.3,
        closed_seconds=0
    )

    assert 0 <= score <= 100


# --------------------------------
# DATABASE TESTS
# --------------------------------

def test_database_persistence_and_export():

    with tempfile.TemporaryDirectory() as folder:

        db = Database(f"{folder}/test.db")

        session_id = db.create_session()

        db.add_sample(
            session_id,
            "Focused",
            100,
            0.3,
            4,
            10,
            "Center",
            "Forward"
        )

        db.finish_session(session_id, 10)

        assert len(db.recent_sessions()) == 1

        assert len(db.export_rows()) == 1

        db.clear()

        assert db.recent_sessions() == []


# --------------------------------
# FLASK CLIENT FIXTURE
# --------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):

    import app as app_module

    monkeypatch.setattr(
        app_module,
        "database",
        Database(tmp_path / "app.db")
    )

    app_module.app.config["TESTING"] = True

    return app_module.app.test_client()


# --------------------------------
# HOMEPAGE TEST
# --------------------------------

def test_homepage(client):

    response = client.get("/")

    assert response.status_code == 200

    assert b"AttentionAI" in response.data


# --------------------------------
# API STATUS AND HISTORY
# --------------------------------

def test_status_and_history_endpoints(client):

    status = client.get("/api/status")

    assert status.status_code == 200

    result = client.get("/api/history")

    assert result.status_code == 200

    assert result.get_json() == {
        "sessions": []
    }


# --------------------------------
# CSV EXPORT TEST
# --------------------------------

def test_csv_export(client):

    response = client.get("/export.csv")

    assert response.status_code == 200

    assert b"session_id,timestamp,status" in response.data
