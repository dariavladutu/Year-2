"""Test cases for the CLI commands in the application."""
from unittest.mock import MagicMock, mock_open, patch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cli.cli_main import (
    analyze_image,
    analyze_mask,
    full_analysis_step_four,
    full_analysis_step_one,
    full_analysis_step_three,
    full_analysis_step_two,
    list_models,
    segment_image
)


@patch("src.cli.cli_main.requests.get")
def test_list_models_success(mock_get: MagicMock) -> None:
    """Test the list_models function with a mock GET request."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = [{
        "id": "model1",
        "description": "Test model"
    }]

    list_models()
    mock_get.assert_called_once()


@patch("src.cli.cli_main.requests.post")
@patch("builtins.open", new_callable=mock_open, read_data=b"fake image")
@patch("src.cli.cli_main.Path.exists", return_value=True)
def test_segment_image_single(
    mock_exists: MagicMock,
    mock_file: MagicMock,
    mock_post: MagicMock
) -> None:
    """Test the segment_image function with a mock POST request."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.content = b"output content"
    mock_post.return_value.headers = {
        "Content-Disposition": 
            'attachment;filename="output.zip"'
    }

    segment_image("model1", image_path="test.jpg")
    mock_post.assert_called_once()


@patch("src.cli.cli_main.requests.post")
@patch("builtins.open", new_callable=mock_open, read_data=b"mask data")
@patch("src.cli.cli_main.Path.exists", return_value=True)
def test_analyze_mask_image(
    mock_exists: MagicMock,
    mock_file: MagicMock,
    mock_post: MagicMock
) -> None:
    """Test the analyze_mask function with a mock POST request."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.content = b"zip data"
    mock_post.return_value.headers = {}

    analyze_mask(image_path="mask.tif")
    mock_post.assert_called_once()


@patch("src.cli.cli_main.requests.post")
@patch("builtins.open", new_callable=mock_open, read_data=b"image data")
@patch("src.cli.cli_main.Path.exists", return_value=True)
def test_analyze_image(
    mock_exists: MagicMock,
    mock_file: MagicMock,
    mock_post: MagicMock
) -> None:
    """Test the analyze_image function with a mock POST request."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.content = b"zip content"
    mock_post.return_value.headers = {
        "Content-Disposition": 
            'attachment; filename="out.zip"'
    }

    analyze_image("modelX", image_path="image.tif")
    mock_post.assert_called_once()


@patch("src.cli.cli_main.requests.post")
@patch("builtins.open", new_callable=mock_open, read_data=b"img data")
@patch("src.cli.cli_main.Path.exists", return_value=True)
def test_full_analysis_step_one(
    mock_exists: MagicMock,
    mock_file: MagicMock,
    mock_post: MagicMock
) -> None:
    """Test the first step of the full analysis pipeline."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.content = b"zip data"
    mock_post.return_value.headers = {
        "Content-Disposition":
        'attachment; filename="out1.zip"'
    }

    full_analysis_step_one("modelX", image_path="image.tif")
    mock_post.assert_called_once()


@patch("src.cli.cli_main.requests.post")
@patch("builtins.open", new_callable=mock_open, read_data=b"tiff data")
@patch("src.cli.cli_main.Path.exists", return_value=True)
def test_full_analysis_step_two(
    mock_exists: MagicMock,
    mock_file: MagicMock,
    mock_post: MagicMock
) -> None:
    """Test the second step of the full analysis pipeline."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.content = b"zip"
    mock_post.return_value.headers = {
        "Content-Disposition": 
            'attachment; filename="step2.zip"'
    }

    full_analysis_step_two(image_path="mask.tif")
    mock_post.assert_called_once()


@patch("src.cli.cli_main.requests.post")
@patch("builtins.open", new_callable=mock_open, read_data=b"overlay data")
@patch("src.cli.cli_main.Path.exists", return_value=True)
def test_full_analysis_step_three(
    mock_exists: MagicMock,
    mock_file: MagicMock,
    mock_post: MagicMock
) -> None:
    """Test the third step of the full analysis pipeline."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.content = b"overlay zip"
    mock_post.return_value.headers = {
        "Content-Disposition": 
            'attachment; filename="step3.zip"'
    }

    full_analysis_step_three("modelX", image_path="image.tif")
    mock_post.assert_called_once()


@patch("src.cli.cli_main.requests.post")
@patch("builtins.open", new_callable=mock_open, read_data="id,length\n1,10")
@patch("src.cli.cli_main.Path.exists", return_value=True)
def test_full_analysis_step_four(
    mock_exists: MagicMock,
    mock_file: MagicMock,
    mock_post: MagicMock
) -> None:
    """Test the final step of the full analysis pipeline."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.content = b"%PDF-1.4"
    mock_post.return_value.headers = {
        "Content-Disposition": 
            'attachment; filename="report.pdf"'
    }

    full_analysis_step_four("results.csv")
    mock_post.assert_called_once()
# ------------------- Additional Tests for Coverage -------------------

@patch("src.cli.cli_main.requests.get", side_effect=Exception("Connection failed"))
def test_list_models_failure(mock_get):
    list_models()
    mock_get.assert_called_once()


def test_segment_image_no_input():
    segment_image("model1", image_path=None, folder_path=None)


@patch("src.cli.cli_main.open", side_effect=OSError("Cannot open"))
def test_segment_image_open_fails(mock_open):
    segment_image("model1", image_path="bad.tif")


@patch("src.cli.cli_main.requests.post", side_effect=Exception("Request failed"))
@patch("builtins.open", new_callable=mock_open, read_data=b"img")
def test_segment_image_post_fails(mock_file, mock_post):
    segment_image("model1", image_path="test.tif")


def test_analyze_mask_no_input():
    analyze_mask(image_path=None, folder_path=None)


@patch("builtins.open", new_callable=mock_open, read_data=b"img")
def test_analyze_image_invalid_extension(mock_file):
    analyze_image("model1", image_path="image.bmp")


@patch("src.cli.cli_main.Path.exists", return_value=False)
def test_step_four_csv_missing(mock_exists):
    full_analysis_step_four("missing.csv")


@patch("builtins.open", new_callable=mock_open)
@patch("src.cli.cli_main.requests.post")
@patch("src.cli.cli_main.Path.exists", return_value=True)
def test_step_four_save_error(mock_exists, mock_post, mock_file):
    mock_post.return_value.status_code = 200
    mock_post.return_value.content = b"PDF"
    mock_post.return_value.headers = {
        "Content-Disposition": 'attachment; filename="report.pdf"'
    }
    mock_file.side_effect = OSError("write error")
    full_analysis_step_four("data.csv")
