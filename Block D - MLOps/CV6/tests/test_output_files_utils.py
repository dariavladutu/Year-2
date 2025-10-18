"""Test cases for output file generation functions in the utils.

These tests cover CSV generation for root measurements 
and PDF report generation.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
import matplotlib

matplotlib.use("Agg")  # prevent Tk errors in headless test
import tempfile

from src.utils import output_files_utils as outf


def test_build_root_measurement_csv() -> None:
    """Test.

    Test the `build_root_measurement_csv` function from
    the output_files_utils module.
    """
    # Create a sample row for root measurement
    row = {
        "filename": "a.png",
        "plant": "p1",
        "length": 10.5,
        "bottom_tip": (100, 200),
        "top_tip": (50, 200),
        "smoothness": 0.95,
        "angle": 90.0,
        "depth": 50.0,
        "span": 5.0,
    }
    fname, content = outf.build_root_measurement_csv([row])
    assert fname == "root_summary.csv"
    assert isinstance(content, bytes)


def test_generate_root_analysis_report_creates_pdf() -> None:
    """Test.

    Test the `generate_root_analysis_report` function from
    the output_files_utils module.
    """
    # Create a sample row for root measurement
    row = {
        "filename": "a.png",
        "plant": "p1",
        "length": 10.5,
        "bottom_tip": (100, 200),
        "top_tip": (50, 200),
        "smoothness": 0.95,
        "angle": 90.0,
        "depth": 50.0,
        "span": 5.0,
    }

    # Save CSV temporarily
    fname, content = outf.build_root_measurement_csv([row])
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, fname)
        with open(csv_path, "wb") as f:
            f.write(content)

        output_pdf = os.path.join(temp_dir, "test_report.pdf")
        outf.generate_root_analysis_report(csv_path, output_path=output_pdf)

        assert os.path.exists(output_pdf)
        assert os.path.getsize(output_pdf) > 1000  # Not empty
