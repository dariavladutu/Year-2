import pytest
import sys
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.cli import cli_main

@pytest.fixture(autouse=True)
def no_requests_real_calls(monkeypatch):
    monkeypatch.setattr("requests.get", MagicMock())
    monkeypatch.setattr("requests.post", MagicMock())

def test_list_models_success(capsys):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [
        {"id": "model1", "description": "desc1"},
        {"id": "model2"}
    ]
    with patch("requests.get", return_value=mock_response):
        cli_main.list_models()
    out = capsys.readouterr().out
    assert "Available local models:" in out
    assert "- model1 (desc1)" in out
    assert "- model2 (no description)" in out

def test_list_models_failure(capsys):
    with patch("requests.get", side_effect=Exception("fail")):
        cli_main.list_models()
    out = capsys.readouterr().out
    assert "[✗] Failed to fetch models:" in out

def test_segment_image_no_files(capsys):
    with patch("builtins.open", side_effect=FileNotFoundError()):
        cli_main.segment_image("model1", image_path="notfound.tif")
    out = capsys.readouterr().out
    assert "[✗] Failed to open image:" in out or "[✗] No valid image files found." in out

def test_segment_image_success(monkeypatch, tmp_path, capsys):
    img = tmp_path / "img.tif"
    img.write_bytes(b"fake")
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.headers = {"Content-Disposition": 'attachment; filename="output.zip"'}
    mock_post.content = b"zipdata"
    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_post)
    cli_main.segment_image("model1", image_path=str(img))
    out = capsys.readouterr().out
    assert "[✓] Saved output to output.zip" in out
    assert Path("output.zip").exists()
    Path("output.zip").unlink()

def test_segment_image_request_fail(monkeypatch, tmp_path, capsys):
    img = tmp_path / "img.tif"
    img.write_bytes(b"fake")
    mock_post = MagicMock(side_effect=Exception("fail"))
    monkeypatch.setattr("src.cli.cli_main.requests.post", mock_post)  # FIXED
    cli_main.segment_image("model1", image_path=str(img))
    out = capsys.readouterr().out
    assert "[✗] Request failed:" in out

def test_analyze_mask_no_files(capsys):
    cli_main.analyze_mask(image_path="notfound.tif")
    out = capsys.readouterr().out
    assert "[✗] Failed to open mask:" in out or "[✗] No valid .tif files found." in out

def test_analyze_mask_success(monkeypatch, tmp_path, capsys):
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"fake")
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.content = b"zipdata"
    mock_post.headers = {}
    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_post)
    cli_main.analyze_mask(image_path=str(mask))
    out = capsys.readouterr().out
    assert "[✓] Saved analysis results to analysis_output.zip" in out
    assert Path("analysis_output.zip").exists()
    Path("analysis_output.zip").unlink()

def test_analyze_mask_request_fail(monkeypatch, tmp_path, capsys):
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"fake")
    mock_post = MagicMock(side_effect=Exception("fail"))
    monkeypatch.setattr("src.cli.cli_main.requests.post", mock_post)  # FIXED
    cli_main.analyze_mask(image_path=str(mask))
    out = capsys.readouterr().out
    assert "[✗] Request failed:" in out

def test_analyze_image_success(monkeypatch, tmp_path, capsys):
    img = tmp_path / "img.tif"
    img.write_bytes(b"fake")
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.content = b"zipdata"
    mock_post.headers = {"Content-Disposition": 'attachment; filename="analyze_image_output.zip"'}
    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_post)
    cli_main.analyze_image("model1", image_path=str(img))
    out = capsys.readouterr().out
    assert "[✓] Saved output to analyze_image_output.zip" in out
    assert Path("analyze_image_output.zip").exists()
    Path("analyze_image_output.zip").unlink()

def test_full_analysis_step_one_success(monkeypatch, tmp_path, capsys):
    img = tmp_path / "img.tif"
    img.write_bytes(b"fake")
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.content = b"zipdata"
    mock_post.headers = {"Content-Disposition": 'attachment; filename="step_one_output.zip"'}
    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_post)
    cli_main.full_analysis_step_one("model1", image_path=str(img))
    out = capsys.readouterr().out
    assert "[✓] Saved output to step_one_output.zip" in out
    assert Path("step_one_output.zip").exists()
    Path("step_one_output.zip").unlink()

def test_full_analysis_step_two_success(monkeypatch, tmp_path, capsys):
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"fake")
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.content = b"zipdata"
    mock_post.headers = {"Content-Disposition": 'attachment; filename="step_two_results.zip"'}
    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_post)
    cli_main.full_analysis_step_two(image_path=str(mask))
    out = capsys.readouterr().out
    assert "[✓] Saved output to step_two_results.zip" in out
    assert Path("step_two_results.zip").exists()
    Path("step_two_results.zip").unlink()

def test_full_analysis_step_three_success(monkeypatch, tmp_path, capsys):
    img = tmp_path / "img.tif"
    img.write_bytes(b"fake")
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.content = b"zipdata"
    mock_post.headers = {"Content-Disposition": 'attachment; filename="step_three_overlays.zip"'}
    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_post)
    cli_main.full_analysis_step_three("model1", image_path=str(img))
    out = capsys.readouterr().out
    assert "[✓] Saved overlay results to step_three_overlays.zip" in out
    assert Path("step_three_overlays.zip").exists()
    Path("step_three_overlays.zip").unlink()

def test_full_analysis_step_four_csv_not_found(capsys):
    cli_main.full_analysis_step_four("notfound.csv")
    out = capsys.readouterr().out
    assert "[✗] CSV file not found:" in out

def test_full_analysis_step_four_success(monkeypatch, tmp_path, capsys):
    csv = tmp_path / "data.csv"
    csv.write_text("col1,col2\n1,2")
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.content = b"pdfdata"
    mock_post.headers = {"Content-Disposition": 'attachment; filename="root_analysis_report.pdf"'}

    # Patch correctly with a function that returns the mock object
    monkeypatch.setattr("src.cli.cli_main.requests.post", lambda *a, **kw: mock_post)

    cli_main.full_analysis_step_four(str(csv))
    out = capsys.readouterr().out
    assert "[✓] Saved report to root_analysis_report.pdf" in out
    assert Path("root_analysis_report.pdf").exists()
    Path("root_analysis_report.pdf").unlink()


def test_segment_image_server_error(monkeypatch, tmp_path, capsys):
    img = tmp_path / "bad_img.tif"
    img.write_bytes(b"fake")
    mock_post = MagicMock()
    mock_post.status_code = 500
    mock_post.text = "Internal Server Error"
    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_post)
    cli_main.segment_image("model1", image_path=str(img))
    out = capsys.readouterr().out
    assert "[✗] Request failed: 500" in out
    assert "Internal Server Error" in out


def test_fallback_output_filename(monkeypatch, tmp_path, capsys):
    csv = tmp_path / "data.csv"
    csv.write_text("col1,col2\n1,2")
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.content = b"pdf"
    mock_post.headers = {}  # No filename provided
    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_post)
    cli_main.full_analysis_step_four(str(csv))
    out = capsys.readouterr().out
    assert "[✓] Saved report to root_analysis_report.pdf" in out
    Path("root_analysis_report.pdf").unlink()


def run_cli(args):
    """Helper to run CLI main with args."""
    sys.argv = ["cli_main.py"] + args
    cli_main.main()

def test_cli_main_list(monkeypatch, capsys):
    monkeypatch.setattr("src.cli.cli_main.requests.get", lambda *a, **kw: MagicMock(
        status_code=200, json=lambda: [{"id": "m1", "description": "test"}],
        raise_for_status=lambda: None
    ))
    run_cli(["list"])
    out = capsys.readouterr().out
    assert "Available local models" in out

def test_cli_main_segment_image_missing(monkeypatch, capsys):
    run_cli(["segment", "--model_id", "m1"])
    out = capsys.readouterr().out
    assert "You must provide either --image or --folder" in out

def test_cli_main_segment_folder(monkeypatch, tmp_path, capsys):
    img = tmp_path / "img.tif"
    img.write_bytes(b"fake")
    monkeypatch.setattr("src.cli.cli_main.requests.post", lambda *a, **kw: MagicMock(
        status_code=200, content=b"zip", headers={"Content-Disposition": 'attachment; filename="segmented.zip"'}
    ))
    run_cli(["segment", "--model_id", "m1", "--folder", str(tmp_path)])
    out = capsys.readouterr().out
    assert "[✓] Saved output to segmented.zip" in out
    Path("segmented.zip").unlink()

def test_cli_main_analyze_mask_folder(monkeypatch, tmp_path, capsys):
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"123")
    monkeypatch.setattr("src.cli.cli_main.requests.post", lambda *a, **kw: MagicMock(
        status_code=200, content=b"zip", headers={}
    ))
    run_cli(["analyze_mask", "--folder", str(tmp_path)])
    out = capsys.readouterr().out
    assert "[✓] Saved analysis results to analysis_output.zip" in out
    Path("analysis_output.zip").unlink()

def test_cli_main_help(capsys):
    with pytest.raises(SystemExit):
        run_cli([])
    out = capsys.readouterr().out
    assert "usage:" in out