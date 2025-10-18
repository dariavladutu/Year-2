"""Auto-generated module docstring."""

import sys

# sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, r"S:\School\Year 2\2024-25d-fai2-adsai-group-CV6\CV6\src")

project = "CV6"
copyright = "2025, sally"
author = "sally"
release = "0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary"
    "sphinx.ext.napoleon",  # Needed for Google/NumPy docstring support
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
]

autodoc_mock_imports = [
    "tensorflow",
    "keras",
    "cv2",
    "patchify",
    "skimage",
    "networkx",
    "tifffile",
    "fastapi",
    "typer",
    "PIL",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
