[![PyPI version](https://badge.fury.io/py/popidd-spot.svg?icon=si%3Apython)](https://badge.fury.io/py/popidd-spot)
# POPIDD-SPOT: Spatial Profiling Overview Tool

POPIDD-SPOT let's you perform spot checks for spatial transcriptomics datasets and generate shareable reports.

## Requirements

### `uv` installation

- Open terminal
  - Windows Powershell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - WLS/macOS/Linux shell: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Close and open new terminal

### Data

CSV metadata file in the format of the metadata flat files exported by AtoMx.
Refer to the sample file for an example.

## Usage

Run POPIDD-SPOT

- Ensure *input* folder is empty before adding the data to it
- Serve a live app for exploration:
  - Run live with `uv run panel serve main.py`
  - The side toolbar can be used to, among others, load a file into POPIDD-SPOT.
- Export an HTML report with limited interactibility for sharing:
  - Create reports with `uv run python main.py`
  - A report for the first file in the input folder (if multiple are present) will
  be saved in the *report* folder.
