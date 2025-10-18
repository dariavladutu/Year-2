import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.app.routers import model_router

client = TestClient(model_router.router)

@patch("src.app.routers.model_router.list_local_models")
def test_get_models_returns_list(mock_list_local_models):
    mock_list_local_models.return_value = ["model1.h5", "model2.h5"]
    response = client.get("/model_management/models")
    assert response.status_code == 200
    assert response.json() == ["model1.h5", "model2.h5"]
    mock_list_local_models.assert_called_once()

@patch("src.app.routers.model_router.list_local_models")
def test_get_models_empty_list(mock_list_local_models):
    mock_list_local_models.return_value = []
    response = client.get("/model_management/models")
    assert response.status_code == 200
    assert response.json() == []
    mock_list_local_models.assert_called_once()