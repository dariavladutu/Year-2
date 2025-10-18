"""Test cases for segmentation utilities in the CV6 project."""

import io
import zipfile
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.app.routers.segmentation import router as segmentation_router
from fastapi import FastAPI
from src.utils import segmentation_utils as su
import tensorflow as tf


class DummyModel(tf.keras.Model):
    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.ones_like(x)


# ─── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def dummy_image_bytes():
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\xdac\xfc\xff\xff?"
        b"\x00\x05\xfe\x02\xfeA\x0b\x1e\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(segmentation_router)
    return TestClient(app)


# ─── Core Tests ───────────────────────────────────────────

def test_segment_image():
    dummy_bytes = np.zeros((256, 256), dtype=np.uint8)
    _, im_buf = cv2.imencode(".png", dummy_bytes)
    image_bytes = im_buf.tobytes()
    result = su.segment_image(DummyModel(), image_bytes, 128, 128)
    assert "mask" in result


def test_segment_plants_from_dish():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (20, 10), (30, 190), 255, -1)
    parts = su.segment_plants_from_dish(mask, num_plants=1)
    assert len(parts) == 1
    assert parts[0].shape == mask.shape


def test_merge_segmented_masks_and_reconstruct():
    base = np.zeros((100, 100), dtype=np.uint8)
    base[10:30, 10:30] = 255
    merged = su.merge_segmented_masks([base], {"orig_shape": (150, 150)})
    full = su.reconstruct_full_mask(
        merged, {"x_start": 10, "top_crop": 5, "orig_shape": (150, 150)}
    )
    assert full.shape == (150, 150)


# ─── Endpoint Tests ──────────────────────────────────────

@patch("src.app.routers.segmentation.load_model")
@patch("src.app.routers.segmentation.get_model_path_local")
@patch("src.app.routers.segmentation.segment_image")
@patch("src.app.routers.segmentation.encode_mask_to_tiff_base64")
@patch("src.app.routers.segmentation.morphological_closing")
@patch("src.app.routers.segmentation.threshold_mask")
def test_segment_endpoint_single_file(
    mock_thresh, mock_close, mock_encode, mock_segment,
    mock_get_path, mock_load, client, dummy_image_bytes
):
    mock_get_path.return_value = "dummy_path"
    mock_load.return_value = MagicMock()
    mock_segment.return_value = {"mask": b"mask", "crop_info": {}, "pad_info": {}}
    mock_thresh.return_value = b"thresh"
    mock_close.return_value = b"closed"
    mock_encode.return_value = (b"tiffbytes", None, None)

    files = {"files": ("test.png", dummy_image_bytes, "image/png")}
    data = {"model_id": "dummy_model"}

    response = client.post("/segment/", files=files, data=data)

    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith('test_mask.tif"')
    assert response.headers["x-session-id"]
    assert response.content == b"tiffbytes"


@patch("src.app.routers.segmentation.load_model")
@patch("src.app.routers.segmentation.get_model_path_local")
@patch("src.app.routers.segmentation.segment_image")
@patch("src.app.routers.segmentation.encode_mask_to_tiff_base64")
@patch("src.app.routers.segmentation.morphological_closing")
@patch("src.app.routers.segmentation.threshold_mask")
def test_segment_endpoint_multiple_files_zip(
    mock_thresh, mock_close, mock_encode, mock_segment,
    mock_get_path, mock_load, client, dummy_image_bytes
):
    mock_get_path.return_value = "dummy_path"
    mock_load.return_value = MagicMock()
    mock_segment.return_value = {"mask": b"mask", "crop_info": {}, "pad_info": {}}
    mock_thresh.return_value = b"thresh"
    mock_close.return_value = b"closed"
    mock_encode.return_value = (b"tiffbytes", None, None)

    files = [
        ("files", ("a.png", dummy_image_bytes, "image/png")),
        ("files", ("b.png", dummy_image_bytes, "image/png")),
    ]
    data = {"model_id": "dummy_model"}

    response = client.post("/segment/", files=files, data=data)

    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith('masks.zip"')
    assert response.headers["x-session-id"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert "a_mask.tif" in zf.namelist()
        assert "b_mask.tif" in zf.namelist()


def test_segment_endpoint_no_files(client):
    response = client.post("/segment/", data={"model_id": "dummy"})
    assert response.status_code == 422  # FastAPI file field validation