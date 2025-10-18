"""Auto-generated module docstring."""

# TODO: Temporarily commented out until the full pipeline is integrated.
"""
Smoke tests for verifying high-level pipeline imports and integration.
These tests ensure that the main components of the pipeline can be imported
and run without errors, serving as a basic sanity check for the project setup."""
# flake8: noqa: D100


# import pytest
# import subprocess
# import logging
# import sys
# import os

# # Ensure 'src' is in the Python path for module resolution
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)

# @pytest.fixture(scope="module")
# def setup_module():
#     """Optional setup and teardown for the smoke test module."""
#     logger.info("Setting up smoke tests...")
#     yield
#     logger.info("Tearing down smoke tests...")

# @pytest.mark.usefixtures("setup_module")
# def test_imports():
#     """Test that all key modules can be imported."""
#     try:
#         from src.app import main
#         from src.data import loader
#         import src.models.model_definition
#         import src.models.model_evaluation
#         import src.models.model_loader
#         from src.models.inference import run_pipeline
#         logger.info("All imports succeeded.")
#     except ImportError as e:
#         logger.error(f"Import failed: {e}")
#         pytest.fail(f"Import failed: {e}")

# def test_basic_pipeline():
#     """Build the model and ensure it was created."""
#     from src.models import simple_unet_model
#     try:
#         model = simple_unet_model()
#     except AttributeError:
#         pytest.fail("Function 'build_model' not found in model_definition.")
#     assert model is not None, "Model was not built"
#     logger.info("Model built successfully.")

# def test_data_loading():
#     """Load image/mask data and check non-emptiness."""
#     from src.data import loader
#     try:
#         data = loader.load_image_mask_pair("path/to/image.png", "path/to/mask.png")  # <-- Replace with test paths
#     except TypeError:
#         pytest.skip("Function 'load_image_mask_pair' requires file paths as arguments.")
#     assert data is not None, "Data loading returned None"
#     logger.info("Data loaded successfully.")

# def test_model_training():
#     """Train the model with sample data."""
#     from src.models import train_model
#     from src.data import loader
#     try:
#         data = loader.load_image_mask_pair("path/to/image.png", "path/to/mask.png")
#         model = train_model(data)
#     except Exception as e:
#         pytest.fail(f"Model training failed: {e}")
#     assert model is not None, "Model training returned None"
#     assert hasattr(model, 'predict'), "Model lacks 'predict' method"
#     logger.info("Model trained successfully.")

# def test_model_evaluation():
#     """Evaluate the model on test data."""
#     from src.models import model_evaluation
#     from src.data import loader
#     try:
#         data = loader.load_image_mask_pair("path/to/image.png", "path/to/mask.png")
#         result = model_evaluation(data)
#     except Exception as e:
#         pytest.fail(f"Model evaluation failed: {e}")
#     assert result is not None, "Evaluation returned None"
#     logger.info("Model evaluation ran successfully.")

# def test_model_loading():
#     """Load a pre-trained model file and verify functionality."""
#     from src.models.model_loader import load_unet_model
#     model_path = os.path.abspath("CV6/models/12_viktoria_231781_unet_model_256px.h5")
#     if not os.path.exists(model_path):
#         pytest.skip("Model file not found for loading test.")
#     model = load_unet_model(model_path)
#     assert model is not None, "Model loading returned None"
#     assert hasattr(model, 'predict'), "Loaded model lacks 'predict' method"
#     logger.info("Model loaded successfully.")

# def test_cli_runs():
#     """Run CLI app and verify no crash."""
#     cli_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'cli', 'main.py'))
#     result = subprocess.run(["python", cli_path], capture_output=True)
#     assert result.returncode == 0, (
#         f"CLI exited with code {result.returncode}\n"
#         f"Output:\n{result.stdout.decode()}\nError:\n{result.stderr.decode()}"
#     )
#     logger.info("CLI ran successfully.")

# def test_model_inference():
#     """Test full inference pipeline with mock data."""
#     from src.models.inference import run_pipeline
#     from src.data import loader
#     try:
#         data = loader.load_image_mask_pair("path/to/image.png", "path/to/mask.png")
#         model = run_pipeline(data)
#     except Exception as e:
#         pytest.fail(f"Inference pipeline failed: {e}")
#     assert model is not None, "Inference pipeline returned None"
#     assert hasattr(model, 'predict'), "Inferred model lacks 'predict' method"
#     logger.info("Inference pipeline ran successfully.")
