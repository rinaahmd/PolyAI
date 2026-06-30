import pytest
from fastapi.testclient import TestClient
from app import app, init_db
import app as app_module


# --------------------
# FIXTURE: client
# --------------------
@pytest.fixture
def client():
    return TestClient(app)


# --------------------
# FIXTURE: test DB
# --------------------
@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(app_module, "DB_PATH", db_file)
    init_db()


# --------------------
# TESTS
# --------------------
def test_score_filter_returns_results(client):
    conn = app_module.sqlite3.connect(app_module.DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO prediction_sessions (uid, timestamp) VALUES (?, ?)",
        ("uid-1", "2024-01-01 12:00:00")
    )

    cur.execute("""
        INSERT INTO detection_objects (prediction_uid, label, score, box)
        VALUES (?, ?, ?, ?)
    """, ("uid-1", "person", 0.9, "[10,20,100,200]"))

    conn.commit()
    conn.close()

    response = client.get("/predictions/score/0.5")

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 1
    assert data[0]["label"] == "person"
    assert data[0]["score"] == 0.9


def test_score_filter_filters_low_scores(client):
    conn = app_module.sqlite3.connect(app_module.DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO prediction_sessions (uid, timestamp) VALUES (?, ?)",
        ("uid-2", "2024-01-01 12:00:00")
    )

    cur.execute("""
        INSERT INTO detection_objects (prediction_uid, label, score, box)
        VALUES (?, ?, ?, ?)
    """, ("uid-2", "car", 0.3, "[0,0,10,10]"))

    conn.commit()
    conn.close()

    response = client.get("/predictions/score/0.5")

    assert response.status_code == 200
    assert response.json() == []


def test_score_invalid_low(client):
    response = client.get("/predictions/score/-1")

    assert response.status_code == 400
    assert response.json()["detail"] == "min_score must be between 0.0 and 1.0"


def test_score_invalid_high(client):
    response = client.get("/predictions/score/2")

    assert response.status_code == 400
    assert response.json()["detail"] == "min_score must be between 0.0 and 1.0"


def test_score_empty_db_returns_empty_list(client):
    response = client.get("/predictions/score/0.5")

    assert response.status_code == 200
    assert response.json() == []