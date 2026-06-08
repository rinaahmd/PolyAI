import os
import pytest
from fastapi.testclient import TestClient
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

from app import app, init_db, save_prediction_session, save_detection_object


# -----------------------------
# DB SETUP (isolated test DB)
# -----------------------------
@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_predictions.db")
    monkeypatch.setattr("app.DB_PATH", db_file)
    init_db()


# -----------------------------
# TEST CLIENT
# -----------------------------
@pytest.fixture
def client():
    return TestClient(app)


# -----------------------------
# TEST 1: label exists
# -----------------------------
def test_predictions_by_label_found(client):
    save_prediction_session(
        "session1",
        "original.jpg",
        "predicted.jpg"
    )

    save_detection_object(
        "session1",
        "person",
        0.95,
        [10, 20, 100, 200]
    )

    response = client.get("/predictions/label/person")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["uid"] == "session1"
    assert len(data[0]["detection_objects"]) == 1
    assert data[0]["detection_objects"][0]["label"] == "person"


# -----------------------------
# TEST 2: label not found
# -----------------------------
def test_predictions_by_label_not_found(client):
    response = client.get("/predictions/label/elephant")

    assert response.status_code == 200
    assert response.json() == []


# -----------------------------
# TEST 3: empty label (validation)
# -----------------------------
def test_predictions_by_label_empty(client):
    response = client.get("/predictions/label/ ")

    assert response.status_code == 400
    assert response.json()["detail"] == "Label cannot be empty"