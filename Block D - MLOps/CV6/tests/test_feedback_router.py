import os
import io
import json
import shutil
import zipfile
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from fastapi import FastAPI
from src.app.routers.feedback_router import router

# Import the router from the file under test

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def temp_feedback_dir(tmp_path):
    # Setup temp_feedback/session_id with dummy files
    session_id = "testsession"
    session_dir = tmp_path / "temp_feedback" / session_id
    session_dir.mkdir(parents=True)
    # Create 12 dummy image/mask pairs (only 10 should be returned)
    for i in range(12):
        base = f"img{i}"
        (session_dir / f"{base}.png").write_bytes(b"imagecontent")
        (session_dir / f"{base}_mask.tif").write_bytes(b"maskcontent")
    # Patch the working directory so the router uses this temp dir
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield session_id, session_dir
    os.chdir(orig_cwd)

@pytest.fixture
def feedback_data_dir(tmp_path):
    feedback_dir = tmp_path / "feedback_data"
    feedback_dir.mkdir()
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield feedback_dir
    os.chdir(orig_cwd)

def test_get_feedback_images_success(client, temp_feedback_dir):
    session_id, session_dir = temp_feedback_dir
    response = client.get(f"/feedback/session/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 10  # Only 10 returned
    for item in data:
        assert "base" in item
        assert "image_url" in item
        assert "mask_url" in item
        assert item["image_url"].startswith("/feedback/image")
        assert item["mask_url"].startswith("/feedback/image")

def test_get_feedback_images_not_found(client):
    response = client.get("/feedback/session/nonexistent")
    assert response.status_code == 404

def test_serve_image_success(client, temp_feedback_dir):
    session_id, session_dir = temp_feedback_dir
    # Test serving image
    response = client.get(
        f"/feedback/image?type=image&session_id={session_id}&file=img0.png"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"imagecontent"
    # Test serving mask
    response = client.get(
        f"/feedback/image?type=mask&session_id={session_id}&file=img0_mask.tif"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff"
    assert response.content == b"maskcontent"

def test_serve_image_not_found(client, temp_feedback_dir):
    session_id, session_dir = temp_feedback_dir
    response = client.get(
        f"/feedback/image?type=image&session_id={session_id}&file=notfound.png"
    )
    assert response.status_code == 404

def test_submit_feedback_success(client, temp_feedback_dir, feedback_data_dir, monkeypatch):
    session_id, session_dir = temp_feedback_dir
    # Prepare feedback: 2 correct, 2 incorrect
    feedback = [
        {"base": "img0", "status": "correct"},
        {"base": "img1", "status": "correct"},
        {"base": "img2", "status": "incorrect"},
        {"base": "img3", "status": "incorrect"},
    ]
    # Patch subprocess.run to avoid actually running scripts
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
    response = client.post(
        "/feedback/submit",
        data={
            "feedback_raw": json.dumps(feedback),
            "session_id": session_id
        }
    )
    assert response.status_code == 200
    # Check correct files copied
    correct_dir = Path("./feedback_data")
    assert (correct_dir / "img0.png").exists()
    assert (correct_dir / "img0_mask.tif").exists()
    assert (correct_dir / "img1.png").exists()
    assert (correct_dir / "img1_mask.tif").exists()
    # Check ZIP content for incorrect
    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes) as zf:
        names = zf.namelist()
        assert "img2.png" in names
        assert "img2_mask.tif" in names
        assert "img3.png" in names
        assert "img3_mask.tif" in names

def test_submit_feedback_invalid_json(client, temp_feedback_dir):
    session_id, session_dir = temp_feedback_dir
    response = client.post(
        "/feedback/submit",
        data={
            "feedback_raw": "not a json",
            "session_id": session_id
        }
    )
    assert response.status_code == 400

def test_submit_feedback_triggers_retraining(client, temp_feedback_dir, feedback_data_dir, monkeypatch):
    session_id, session_dir = temp_feedback_dir
    # Pre-populate feedback_data with 19 pngs
    correct_dir = Path("./feedback_data")
    correct_dir.mkdir(exist_ok=True)
    for i in range(19):
        (correct_dir / f"old{i}.png").write_bytes(b"oldimage")
    # Patch subprocess.run to track calls
    calls = []
    def fake_run(cmd, check):
        calls.append(cmd)
    monkeypatch.setattr("subprocess.run", fake_run)
    # Feedback to add 1 more correct
    feedback = [{"base": "img0", "status": "correct"}]
    response = client.post(
        "/feedback/submit",
        data={
            "feedback_raw": json.dumps(feedback),
            "session_id": session_id
        }
    )
    assert response.status_code == 200
    # Should have triggered retraining scripts
    assert any("preprocess_feedback.py" in c for c in calls[0])
    assert any("train_model.py" in c for c in calls[1])