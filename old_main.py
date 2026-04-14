from __future__ import annotations

import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping

import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
import panel as pn
import param

import holoviews as hv
from holoviews import dim
from matplotlib import cm, colors as mcolors
from bokeh.models import Range1d


pn.extension("tabulator", "filedropper", design="bootstrap", theme="default")

def get_total_memory_gb() -> int:
    """Return the total physical memory in gigabytes.

    The function first attempts to use POSIX ``os.sysconf``, then falls back to
    the Windows ``GlobalMemoryStatusEx`` API. If both methods fail, a safe
    default of 4 GB is returned.
    """
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            return max(1, int((page_size * phys_pages) // (1024**3)))
        except (ValueError, OSError, AttributeError):
            pass

    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return max(1, int(stat.ullTotalPhys // (1024**3)))
    except Exception:
        return 4

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ACCENT = "teal"
DATA_DIR = Path("input")
DEFAULT_REPORT_PATH = Path("report/popidd-spot_report.html")

TABLE_PAGE_SIZE = 25

PLOT_MIN_HEIGHT = 360
PLOT_TALL_MIN_HEIGHT = 460
PLOT_SHORT_MIN_HEIGHT = 220

PLOT_MIN_WIDTH = 320
PLOT_PANEL_MIN_WIDTH = 420
PLOT_MAX_WIDTH = 1000

SPATIAL_X_COL = "CenterX_global_px"
SPATIAL_Y_COL = "CenterY_global_px"
DEFAULT_SPATIAL_COLOR_BY = "Mean.DAPI"
SPATIAL_PLOT_GRIDSIZE = 512
MAX_LIVE_SPATIAL_POINTS = get_total_memory_gb() * 100_000
MAX_SPATIAL_CATEGORIES = 20
SPATIAL_CATEGORY_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
    "#e6550d",
]

QC_CELL_THRESHOLD = 20
QC_FOV_THRESHOLD = 24
FOV_MEDIAN_RNA_CENTER = 48.0
FOV_MEDIAN_RNA_DISPLAY_MIN = FOV_MEDIAN_RNA_CENTER / 3
FOV_MEDIAN_RNA_DISPLAY_MAX = FOV_MEDIAN_RNA_CENTER * 3

CARD_STYLES = {
    "box-shadow": "rgba(50, 50, 93, 0.25) 0px 6px 12px -2px, rgba(0, 0, 0, 0.30) 0px 3px 7px -3px",
    "border-radius": "4px",
    "padding": "10px",
}

QC_STATUS_COLORS = {
    "passed": "#22c55e",
    "flagged": "#ef4444",
    "total": "#6b7280",
}

PLOT_TOOLS = ["hover", "box_select", "lasso_select", "reset"]

# Logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Generic Helpers & Setup
# ============================================================================

def empty_df() -> pd.DataFrame:
    """Return a new empty DataFrame."""
    return pd.DataFrame()

def has_cols(df: pd.DataFrame | None, *cols: str) -> bool:
    """Return whether the DataFrame exists, is non-empty, and contains all columns."""
    return df is not None and not df.empty and all(col in df.columns for col in cols)


def safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric values, replacing invalid entries with NaN."""
    return pd.to_numeric(series, errors="coerce")


def panel_message(text: str, alert_type: str | None = None) -> pn.pane.Pane:
    """Return a Panel message pane, optionally styled as an alert."""
    if alert_type:
        return pn.pane.Alert(text, alert_type=alert_type, sizing_mode="stretch_width")
    return pn.pane.Markdown(text, sizing_mode="stretch_width")

def plot_kwargs(**extra: dict) -> dict:
    """Return standard hvPlot keyword arguments with optional overrides."""
    base = {
        "responsive": True,
        "shared_axes": False,
    }
    base.update(extra)
    return base


# ============================================================================
# Data Sanitising
# ============================================================================

def normalize_colname(name: str) -> str:
    """Normalise a column name for case- and punctuation-insensitive matching."""
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def find_column(df: pd.DataFrame | None, *aliases: str) -> str | None:
    """Return the first DataFrame column matching any provided alias."""
    if df is None or df.empty:
        return None

    normalized = {normalize_colname(col): col for col in df.columns}
    for alias in aliases:
        key = normalize_colname(alias)
        if key in normalized:
            return normalized[key]
    return None

def is_supported_color_column(series: pd.Series) -> bool:
    """Return whether a Series can be used for spatial colour encoding."""
    return (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )

def classify_color_column(series: pd.Series) -> str:
    """Classify a colour column as numeric or categorical."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "numeric"
    return "categorical"


# ============================================================================
# State Model and Mutators
# ============================================================================

def get_first_csv_path(data_dir: Path = DATA_DIR) -> Path | None:
    """Return the first CSV file found in the input directory, if any."""
    if not data_dir.exists() or not data_dir.is_dir():
        return None
    csv_files = sorted(p for p in data_dir.glob("*.csv") if p.is_file())
    return csv_files[0] if csv_files else None

def get_initial_filename() -> str:
    """Return the initial dataset filename shown in the UI."""
    default_path = get_first_csv_path()
    return default_path.name if default_path is not None else "No dataset loaded"

class DataState(param.Parameterized):
    """Store the current dataset, file selection, and UI status."""

    data = param.Parameter(default=pd.DataFrame())
    filename = param.String(default=get_initial_filename())
    pending_filename = param.String(default="")
    data_revision = param.Integer(default=0)
    status_message = param.String(default="")
    status_level = param.ObjectSelector(
        default="info",
        objects=["info", "success", "warning", "danger"],
    )

def set_status(message: str = "", level: str = "info") -> None:
    """Update the global application status message and severity level."""
    state.status_message = message
    state.status_level = level

def set_data(df: pd.DataFrame | None, filename: str) -> None:
    """Update the global dataset state and increment its revision counter."""
    state.data = df if df is not None else empty_df()
    state.filename = filename
    state.data_revision += 1

def read_csv_from_source(source: bytes | bytearray | str | os.PathLike | None) -> pd.DataFrame:
    """Read CSV data from bytes, text, file-like objects, or filesystem paths."""
    if source is None:
        return empty_df()

    if isinstance(source, (bytes, bytearray)):
        if not source:
            return empty_df()
        return pd.read_csv(io.BytesIO(source))

    if isinstance(source, str):
        if not source.strip():
            return empty_df()
        return pd.read_csv(io.StringIO(source))

    if hasattr(source, "read"):
        return pd.read_csv(source)

    return pd.read_csv(source)

def add_qc_flags_to_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Add QC flag columns to DataFrame based on RNA thresholds.

    Adds two columns:
    - 'qc_flagged_cell': 1 if cell's nCount_RNA < 20, else 0
    - 'qc_flagged_fov': 1 if FOV's median_RNA < 24, else 0
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # -----------------------
    # Cell-level QC flag
    # -----------------------
    ncount_rna_col = find_column(df, "nCount_RNA", "ncountrna", "count_rna")
    if ncount_rna_col is not None:
        df["qc_flagged_cell"] = (safe_numeric(df[ncount_rna_col]) < 20).astype(int)
    else:
        df["qc_flagged_cell"] = 0

    # -----------------------
    # FOV-level QC flag
    # -----------------------
    fov_col = find_column(df, "fov", "FOV", "field_of_view", "fieldofview")
    median_rna_col = find_column(df, "median_RNA", "medianrna", "median_rna")

    if fov_col is not None and median_rna_col is not None:
        # Use precomputed median_RNA directly
        median_rna = safe_numeric(df[median_rna_col])
        df["qc_flagged_fov"] = (median_rna < 24).astype(int)

    elif fov_col is not None and ncount_rna_col is not None:
        # Fallback: compute median nCount_RNA per FOV
        fov_median = df.groupby(fov_col)[ncount_rna_col].transform(
            lambda x: safe_numeric(x).median()
        )
        df["qc_flagged_fov"] = (fov_median < 24).astype(int)

    else:
        df["qc_flagged_fov"] = 0

    return df

def load_default_data() -> tuple[pd.DataFrame, str]:
    """Load the first available CSV from the default data directory."""
    path = get_first_csv_path()

    if path is None:
        set_status(
            f"No CSV files found in `{DATA_DIR}`. Select a dataset from the sidebar to begin.",
            "warning",
        )
        return empty_df(), "No dataset loaded"

    try:
        df = read_csv_from_source(path)
        df = add_qc_flags_to_df(df)
        set_status(f"Loaded default dataset: `{path.name}`", "success")
        return df, path.name
    except Exception as exc:
        LOGGER.exception("Failed to load default dataset")
        set_status(f"Failed to load default dataset `{path}`: {exc}", "danger")
        return empty_df(), "No dataset loaded"


state = DataState()
_default_df, _default_name = load_default_data()
set_data(_default_df, _default_name)


# ============================================================================
# Data I/O & CSV Handling
# ============================================================================

def get_source_data() -> pd.DataFrame:
    """Return the current source dataset."""
    return state.data if state.data is not None else empty_df()


def get_colorby_options(df: pd.DataFrame | None) -> list[str]:
    """Return valid columns that can be used to colour the spatial plot."""
    if df is None or df.empty:
        return [DEFAULT_SPATIAL_COLOR_BY]

    excluded = {"CenterX_global_px", "CenterY_global_px"}
    cols = [c for c in df.columns if c not in excluded and is_supported_color_column(df[c])]

    ordered: list[str] = []
    if DEFAULT_SPATIAL_COLOR_BY in cols:
        ordered.append(DEFAULT_SPATIAL_COLOR_BY)
    ordered.extend(c for c in cols if c != DEFAULT_SPATIAL_COLOR_BY)

    # Add QC flags to the end if they exist
    for qc_col in ["qc_flagged_fov", "qc_flagged_cell"]:
        if qc_col in df.columns and qc_col not in ordered:
            ordered.append(qc_col)

    return ordered or [DEFAULT_SPATIAL_COLOR_BY]


def update_colorby_options(*_) -> None:
    """Refresh the spatial colour-by widget options from the current dataset."""
    options = get_colorby_options(state.data)
    color_by_select.options = options
    if color_by_select.value not in options:
        color_by_select.value = DEFAULT_SPATIAL_COLOR_BY


def majority_code(values) -> float:
    """Return the most frequent integer code in an array-like input.

    NaN values are ignored. If no valid values remain, NaN is returned.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return np.nan
    arr = arr[~pd.isna(arr)]
    if arr.size == 0:
        return np.nan
    arr = arr.astype(int)
    uniq, counts = np.unique(arr, return_counts=True)
    return float(uniq[np.argmax(counts)])


file_dropper = pn.widgets.FileDropper(
    multiple=False,
    max_files=1,
    chunk_size=10_000_000,
    layout="compact",
)

load_button = pn.widgets.Button(
    name="Load selected dataset",
    button_type="primary",
    icon="file-import",
    disabled=True,
)

fov_select = pn.widgets.Select(name="Field of View (FOV)", value="All", options=["All"])

color_by_select = pn.widgets.Select(
    name="Colour spatial plot by",
    value=DEFAULT_SPATIAL_COLOR_BY,
    options=[DEFAULT_SPATIAL_COLOR_BY],
)


# Export widgets for sidebar
export_html_button = pn.widgets.Button(
    name="Export HTML Report",
    button_type="success",
    icon="download",
)

export_csv_button = pn.widgets.Button(
    name="Export Data (CSV)",
    button_type="success",
    icon="download",
)


def _on_export_html(event) -> None:
    """Handle HTML export button click."""
    try:
        save_as_html()
        set_status("HTML report exported successfully.", "success")
    except Exception as exc:
        LOGGER.exception("Error exporting HTML")
        set_status(f"Error exporting HTML: {exc}", "danger")


def _on_export_csv(event) -> None:
    """Handle CSV export button click."""
    try:
        save_data_as_csv()
    except Exception as exc:
        LOGGER.exception("Error exporting CSV")
        set_status(f"Error exporting CSV: {exc}", "danger")


export_html_button.on_click(_on_export_html)
export_csv_button.on_click(_on_export_csv)


def _get_uploaded_csv() -> tuple[str | None, str | bytes | None]:
    """Return the uploaded CSV filename and content when exactly one valid file is present."""
    uploads = file_dropper.value or {}
    if len(uploads) != 1:
        return None, None

    filename, content = next(iter(uploads.items()))
    filename = str(filename)

    if not filename.lower().endswith(".csv"):
        return None, None

    return filename, content


def _on_file_selected(event) -> None:
    """Handle file-dropper changes and update pending file state."""
    filename, _ = _get_uploaded_csv()
    state.pending_filename = filename or ""

    has_valid_file = filename is not None
    load_button.disabled = not has_valid_file

    if has_valid_file:
        set_status(
            f"Uploaded `{filename}`. Click **Load selected dataset** to apply it.",
            "info",
        )
    else:
        uploads = file_dropper.value or {}
        if uploads:
            set_status("Please upload exactly one CSV file.", "warning")
        else:
            set_status("", "info")


def _load_selected_dataset(event=None) -> None:
    """Load the currently uploaded CSV into application state."""
    filename, content = _get_uploaded_csv()

    if filename is None or content is None:
        set_status("Please upload exactly one CSV file first.", "warning")
        return

    try:
        df = read_csv_from_source(content)
        df = add_qc_flags_to_df(df)
        set_data(df, filename)
        set_status(f"Loaded `{filename}` successfully.", "success")
    except Exception as exc:
        LOGGER.exception("Error loading uploaded CSV")
        set_status(f"Error loading `{filename}`: {exc}", "danger")


file_dropper.param.watch(_on_file_selected, "value")
load_button.on_click(_load_selected_dataset)


# ============================================================================
# Data Filtering & Caching
# ============================================================================


def compute_fov_summary(df: pd.DataFrame | None) -> pd.DataFrame:
    """Create a per-FOV summary table from summary-level or cell-level data."""
    if df is None or df.empty:
        return empty_df()

    fov_col = find_column(df, "fov", "FOV", "field_of_view", "fieldofview")
    if fov_col is None:
        return empty_df()

    summary_aliases = {
        "nCell": ("nCell", "ncell", "cell_count", "cells"),
        "nCount": ("nCount", "ncount", "total_counts", "totalcount"),
        "nCountPerCell": ("nCountPerCell", "ncountpercell", "count_per_cell"),
        "nFeaturePerCell": ("nFeaturePerCell", "nfeaturepercell", "feature_per_cell"),
        "qcCellsPassed": ("qcCellsPassed", "qccellspassed", "qc_passed"),
        "qcCellsFlagged": ("qcCellsFlagged", "qccellsflagged", "qc_flagged"),
        "qcFlagsFOV": ("qcFlagsFOV", "qcflagsfov", "qc_flags_fov"),
    }

    resolved = {"fov": fov_col}
    for canonical, aliases in summary_aliases.items():
        actual = find_column(df, *aliases)
        if actual is not None:
            resolved[canonical] = actual

    if "nCell" in resolved and "nCount" in resolved:
        cols = ["fov"] + [c for c in summary_aliases if c in resolved]
        out = df[[resolved[c] for c in cols]].copy()
        out.columns = cols
        return out.drop_duplicates().sort_values("fov")

    cell_id_col = find_column(df, "cell_id", "cellid")
    ncount_rna_col = find_column(df, "nCount_RNA", "ncountrna", "count_rna")
    nfeature_rna_col = find_column(df, "nFeature_RNA", "nfeaturerna", "feature_rna")
    area_col = find_column(df, "Area.um2", "area_um2", "areaum2", "area")

    required = [cell_id_col, ncount_rna_col, nfeature_rna_col]
    if all(col is not None for col in required):
        work = df[[fov_col, cell_id_col, ncount_rna_col, nfeature_rna_col]].copy()
        work.columns = ["fov", "cell_id", "nCount_RNA", "nFeature_RNA"]

        agg_kwargs = {
            "nCell": ("cell_id", "count"),
            "nCount": ("nCount_RNA", "sum"),
            "nCountPerCell": ("nCount_RNA", "mean"),
            "nFeaturePerCell": ("nFeature_RNA", "mean"),
        }

        if area_col is not None:
            work["Area.um2"] = safe_numeric(df[area_col])
            agg_kwargs["Area"] = ("Area.um2", "mean")

        return work.groupby("fov").agg(**agg_kwargs).reset_index().sort_values("fov")

    return empty_df()


@lru_cache(maxsize=64)
def cached_fov_summary(data_revision: int) -> pd.DataFrame:
    """Return a cached FOV summary keyed by dataset revision."""
    return compute_fov_summary(state.data)


def get_fov_options(summary: pd.DataFrame | None) -> list[str]:
    """Return the FOV selector options derived from a summary table."""
    if summary is None or summary.empty or "fov" not in summary.columns:
        return ["All"]

    values = summary["fov"].dropna().unique()
    try:
        ordered = sorted(int(v) for v in values)
        return ["All"] + [str(v) for v in ordered]
    except Exception:
        ordered = sorted(str(v) for v in values)
        return ["All"] + ordered


def update_fov_options(*_) -> None:
    """Refresh the FOV selector options for the current dataset."""
    options = get_fov_options(cached_fov_summary(state.data_revision))
    fov_select.options = options
    if fov_select.value not in options:
        fov_select.value = "All"


state.param.watch(update_fov_options, "data_revision")
update_fov_options()

state.param.watch(update_colorby_options, "data_revision")
update_colorby_options()


# ============================================================================
# UI Widgets & Controls
# ============================================================================


def filter_data(df: pd.DataFrame | None, fov: str) -> pd.DataFrame:
    """Filter a dataset to a selected field of view."""
    if df is None or df.empty:
        return empty_df()

    fov_col = find_column(df, "fov", "FOV", "field_of_view", "fieldofview")
    if fov == "All" or fov_col is None:
        return df

    fov_series = df[fov_col]

    try:
        numeric_fov = pd.to_numeric(fov_series, errors="coerce")
        target = int(fov)
        matched = df[numeric_fov == target]
        if not matched.empty:
            return matched
    except Exception:
        pass

    return df[fov_series.astype(str) == str(fov)]


@lru_cache(maxsize=128)
def cached_filtered_df(data_revision: int, fov: str) -> pd.DataFrame:
    """Return a cached FOV-filtered DataFrame."""
    return filter_data(state.data, fov)


@lru_cache(maxsize=128)
def cached_cell_stats(data_revision: int, fov: str) -> dict[str, float | int | None]:
    """Return cached summary statistics for the selected filtered dataset."""
    df = cached_filtered_df(data_revision, fov)
    if df is None or df.empty:
        return {}

    def col_stat(col: str, fn: Callable[[pd.Series], float], default=None):
        if col not in df.columns:
            return default
        s = safe_numeric(df[col]).dropna()
        return fn(s) if not s.empty else default

    return {
        "cell_count": len(df),
        "avg_rna": col_stat("nCount_RNA", pd.Series.mean),
        "median_rna": col_stat("nCount_RNA", pd.Series.median),
        "min_rna": col_stat("nCount_RNA", pd.Series.min),
        "max_rna": col_stat("nCount_RNA", pd.Series.max),
        "med_features": col_stat("nFeature_RNA", pd.Series.median),
        "avg_area": col_stat("Area.um2", pd.Series.mean),
    }


filtered_df = pn.bind(cached_filtered_df, state.param.data_revision, fov_select)
cell_stats = pn.bind(cached_cell_stats, state.param.data_revision, fov_select)


# ============================================================================
# Plot Rendering & Styling
# ============================================================================


def plot_box(
    plot,
    *,
    min_height: int = PLOT_MIN_HEIGHT,
    min_width: int = 320,
    max_width: int | None = None,
    title: str | None = None,
    square: bool = False,
    aspect_ratio: float | None = None,
    linked_axes: bool = False,
) -> pn.Column:
    """Responsive container for Bokeh/HoloViews plots.
    
    Parameters
    ----------
    plot : object
        The HoloViews or Bokeh plot to embed.
    min_height : int
        Minimum height in pixels for standard plots.
    min_width : int
        Minimum width in pixels.
    max_width : int, optional
        Maximum width in pixels. If None, no limit is applied.
    title : str, optional
        Title to display above the plot.
    square : bool
        If True, use square layout (1:1 aspect ratio) with scale_width sizing.
    aspect_ratio : float, optional
        Custom aspect ratio (e.g., 4/3 for 4:3). If provided, overrides square.
    linked_axes : bool
        Whether to enable linked axes in the HoloViews pane.
    """
    header = (
        pn.pane.Markdown(f"#### {title}", margin=(0, 0, 6, 0))
        if title
        else pn.Spacer(height=0)
    )

    styles_base = {
        "overflow": "visible",
        "box-sizing": "border-box",
        "width": "100%",
    }
    if max_width is not None:
        styles_base["max-width"] = f"{max_width}px"

    # Determine if we should use scale_width sizing (for square or aspect ratio)
    use_scale_width = square or aspect_ratio is not None

    if use_scale_width:
        # Square or custom aspect ratio plot with scale_width sizing
        return pn.Column(
            header,
            pn.pane.HoloViews(
                plot,
                sizing_mode="scale_width",
                min_width=min_width,
                margin=0,
                linked_axes=linked_axes
            ),
            min_width=min_width,
            sizing_mode="stretch_width",
            styles=styles_base,
        )

    # Standard rectangular plot
    return pn.Column(
        header,
        pn.pane.HoloViews(
            plot,
            sizing_mode="stretch_width",
            min_height=min_height,
            linked_axes=linked_axes
        ),
        min_height=min_height + (32 if title else 0),
        min_width=min_width,
        sizing_mode="stretch_width",
        styles=styles_base,
    )

def _square_responsive_bokeh_hook(plot, element):
    """Force the rendered Bokeh figure to stay square and resize with width."""
    fig = plot.state
    fig.sizing_mode = "scale_width"
    fig.aspect_ratio = 1
    fig.match_aspect = True


def _responsive_bokeh_hook_4_3(plot, element):
    """Force the rendered Bokeh figure to maintain 4:3 aspect ratio and resize with width."""
    fig = plot.state
    fig.sizing_mode = "scale_width"
    fig.aspect_ratio = 4 / 3
    fig.match_aspect = True


def _link_spatial_axes_hook(plot, element):
    """Mark figure as ready for linking through selection tools."""
    fig = plot.state
    fig._spatialplot_linked = True


def responsive_flexbox(*children, gap: str = "12px", justify: str = "flex-start") -> pn.FlexBox:
    """Return a wrapping flex container that behaves well in live apps and static HTML."""
    return pn.FlexBox(
        *children,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={
            "gap": gap,
            "align-items": "stretch",
            "justify-content": justify,
            "width": "100%",
        },
    )

def _compute_shared_spatial_ranges(df: pd.DataFrame | None, pad_frac: float = 0.02):
    """Build shared Bokeh ranges covering the global spatial extent."""
    if df is None or df.empty or not has_cols(df, "CenterX_global_px", "CenterY_global_px"):
        return None, None

    x = safe_numeric(df["CenterX_global_px"]).dropna()
    y = safe_numeric(df["CenterY_global_px"]).dropna()
    if x.empty or y.empty:
        return None, None

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    xpad = max((xmax - xmin) * pad_frac, 1.0)
    ypad = max((ymax - ymin) * pad_frac, 1.0)

    return (
        Range1d(start=xmin - xpad, end=xmax + xpad),
        Range1d(start=ymin - ypad, end=ymax + ypad),
    )

def _shared_range_hook(x_range, y_range):
    """Return a HoloViews hook that assigns shared Bokeh range objects."""
    def _hook(plot, element):
        fig = plot.state
        fig.x_range = x_range
        fig.y_range = y_range
    return _hook

def flex_item(
    *objects,
    min_width: int,
    grow: int = 1,
    height: int | None = None,
    allow_shrink_below_min: bool = False,
) -> pn.Column:
    """Wrap one or more Panel objects as a responsive flex item."""
    css_min_width = "0" if allow_shrink_below_min else f"{min_width}px"

    kwargs = {
        "sizing_mode": "stretch_width",
        "min_width": min_width,
        "styles": {
            "flex": f"{grow} 1 {min_width}px",
            "min-width": css_min_width,
        },
    }
    if height is not None:
        kwargs["height"] = height
    return pn.Column(*objects, **kwargs)



def indicator_card(
    value: float | int | None,
    label: str,
    fmt: str = "{:,.0f}",
    min_width: int = 220,
) -> pn.Column:
    """Render a single KPI indicator card as a responsive flex item."""
    if value is None:
        display_value = "—"
    elif isinstance(value, (int, float)):
        display_value = fmt.format(value)
    else:
        display_value = str(value)

    html = f"""
    <div style="
        text-align:center;
        height:100%;
        display:flex;
        flex-direction:column;
        justify-content:center;
    ">
        <div style="
            font-size:clamp(11px,1.5vw,14px);
            color:#999;
            margin-bottom:8px;
            font-weight:500;
        ">{label}</div>
        <div style="
            font-size:clamp(18px,4vw,32px);
            font-weight:bold;
            color:#333;
        ">{display_value}</div>
    </div>
    """

    return flex_item(
        pn.pane.HTML(
            html,
            sizing_mode="stretch_both",
            styles={
                **CARD_STYLES,
                "height": "100%",
                "min-height": "100px",
                "box-sizing": "border-box",
            },
        ),
        min_width=min_width,
        height=110,
    )

def status_card(
    value: int,
    label: str,
    color: str,
    min_width: int = 220,
) -> pn.Column:
    """Render a coloured QC status summary card as a responsive flex item."""
    html = f"""
    <div style="
        background-color:{color};
        border-radius:8px;
        padding:20px;
        text-align:center;
        color:white;
        min-height:100px;
        height:100%;
        display:flex;
        flex-direction:column;
        justify-content:center;
        box-shadow:0 2px 8px rgba(0,0,0,0.15);
        box-sizing:border-box;
    ">
        <div style="font-size:32px;font-weight:bold;">{value:,}</div>
        <div style="font-size:13px;margin-top:8px;opacity:0.95;">{label}</div>
    </div>
    """
    return flex_item(
        pn.pane.HTML(html, sizing_mode="stretch_both"),
        min_width=min_width,
        height=130,
    )


def qc_flag_status_card(
    value: int,
    label: str,
    percentage: float,
    color: str,
    min_width: int = 220,
) -> pn.Column:
    """Render a QC flag status card with label-number-percentage layout.
    
    Parameters
    ----------
    value : int
        Count of flagged items.
    label : str
        Card label (e.g., "Flagged FOVs").
    percentage : float
        Percentage flagged (0.0 to 100.0).
    color : str
        Hex color code for background.
    min_width : int
        Minimum width in pixels.
    """
    html = f"""
    <div style="
        background-color:{color};
        border-radius:8px;
        padding:20px;
        text-align:center;
        color:white;
        min-height:100px;
        height:100%;
        display:flex;
        flex-direction:column;
        justify-content:center;
        box-shadow:0 2px 8px rgba(0,0,0,0.15);
        box-sizing:border-box;
    ">
        <div style="font-size:13px;margin-bottom:8px;opacity:0.95;font-weight:500;">{label}</div>
        <div style="font-size:32px;font-weight:bold;">{value:,}</div>
        <div style="font-size:13px;margin-top:8px;opacity:0.90;">({percentage:.1f}%)</div>
    </div>
    """
    return flex_item(
        pn.pane.HTML(html, sizing_mode="stretch_both"),
        min_width=min_width,
        height=130,
    )


def maybe_sample_spatial(df: pd.DataFrame | None, enabled: bool) -> pd.DataFrame:
    """Downsample spatial data when enabled and above the live-plot threshold."""
    if df is None or df.empty or not enabled:
        return df if df is not None else empty_df()
    if len(df) <= MAX_LIVE_SPATIAL_POINTS:
        return df

    LOGGER.info(
        "Downsampling spatial data from %s to %s rows for live rendering.",
        len(df),
        MAX_LIVE_SPATIAL_POINTS,
    )
    return df.sample(n=MAX_LIVE_SPATIAL_POINTS, random_state=42)


# ============================================================================
# Raw Plot Generation (Primitives)
# ============================================================================


def hist_plot_raw(
    df: pd.DataFrame | None,
    column: str,
    title: str,
    bins: int = 50,
    xlim: tuple[float, float] | None = None,
) -> object | None:
    """Create a histogram plot for a single numeric column."""
    if not has_cols(df, column):
        return None

    data = safe_numeric(df[column]).dropna()
    if xlim is not None:
        data = data[(data >= xlim[0]) & (data <= xlim[1])]

    if data.empty:
        return None

    return data.hvplot.hist(
        bins=bins,
        xlabel=column,
        ylabel="Number of Cells",
        color=ACCENT,
        **plot_kwargs(),
    )


def scatter_plot_raw(
    df: pd.DataFrame | None,
    x: str,
    y: str,
) -> object | None:
    """Create a scatter plot for two numeric columns."""
    if not has_cols(df, x, y):
        return None

    plot_df = df[[x, y]].apply(safe_numeric).dropna()
    if plot_df.empty:
        return None

    return plot_df.hvplot.scatter(
        x=x,
        y=y,
        xlabel=x,
        ylabel=y,
        color=ACCENT,
        tools=PLOT_TOOLS,
        **plot_kwargs(),
    )


def metrics_jointplot_raw(
    df: pd.DataFrame | None,
    x: str,
    y: str,
    hue: str,
) -> object | None:
    """Create a hexbin plot showing cell metrics.
    
    Shows X (nCount_RNA) vs Y (nFeature_RNA) with hexes colored by mean hue (Area.um2).
    Hexagons are aggregated with mean area for better performance with large datasets.
    """
    if not has_cols(df, x, y, hue):
        return None

    plot_df = df[[x, y, hue]].copy()
    plot_df[x] = safe_numeric(plot_df[x])
    plot_df[y] = safe_numeric(plot_df[y])
    plot_df[hue] = safe_numeric(plot_df[hue])
    plot_df = plot_df.dropna()
    
    if plot_df.empty:
        return None

    # Create hexbin plot with mean color aggregation
    hexbin = plot_df.hvplot.hexbin(
        x=x,
        y=y,
        C=hue,
        reduce_function=np.mean,
        clabel=f"Mean {hue}",
        gridsize=SPATIAL_PLOT_GRIDSIZE,
        min_count=1,
        cmap="viridis",
        colorbar=True,
        xlabel=x,
        ylabel=y,
        tools=PLOT_TOOLS,
        **plot_kwargs(),
    )

    # Apply responsive options with 4:3 aspect ratio
    return hexbin.opts(
        toolbar="above",
        active_tools=["wheel_zoom"],
        responsive=True,
        data_aspect=4 / 3,
        hooks=[_responsive_bokeh_hook_4_3],
    ).clone()


def _matplotlib_palette(name: str = "RdYlGn", n: int = 256) -> list[str]:
    """Return a list of hex colours sampled from a Matplotlib colormap."""
    cmap = cm.get_cmap(name, n)
    return [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]


def _two_slope_normalize_series(
    values: pd.Series,
    *,
    vmin: float,
    vcenter: float,
    vmax: float,
) -> pd.Series:
    """Map values to [0,1] with vcenter fixed at 0.5, mimicking TwoSlopeNorm."""
    s = pd.to_numeric(values, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)

    if not np.isfinite(vmin) or not np.isfinite(vcenter) or not np.isfinite(vmax):
        return out

    if vmax <= vmin:
        out.loc[s.notna()] = 0.5
        return out

    lower = s <= vcenter
    upper = s > vcenter

    if vcenter > vmin:
        out.loc[lower] = 0.5 * (s.loc[lower] - vmin) / (vcenter - vmin)
    else:
        out.loc[lower] = 0.5

    if vmax > vcenter:
        out.loc[upper] = 0.5 + 0.5 * (s.loc[upper] - vcenter) / (vmax - vcenter)
    else:
        out.loc[upper] = 0.5

    return out.clip(0, 1)


def _fov_scatter_colorbar_relabel_hook(plot, element):
    """Relabel normalized [0,1] colorbar ticks back into median RNA units."""
    from bokeh.models import ColorBar, FixedTicker, CustomJSTickFormatter

    fig = plot.state
    data = element.data
    if data is None or len(data) == 0:
        return

    required = {"color_vmin", "color_vmax", "color_vcenter"}
    if not required.issubset(data):
        return

    try:
        vmin = float(pd.to_numeric(pd.Series(data["color_vmin"]), errors="coerce").dropna().iloc[0])
        vmax = float(pd.to_numeric(pd.Series(data["color_vmax"]), errors="coerce").dropna().iloc[0])
        vcenter = float(pd.to_numeric(pd.Series(data["color_vcenter"]), errors="coerce").dropna().iloc[0])
    except Exception:
        return

    colorbars = fig.select(dict(type=ColorBar))
    if not colorbars:
        return

    if vmax <= vmin:
        ticks = [0.5]
        formatter_code = f'return "{vmin:.1f}";'
    else:
        lower_mid = vmin + 0.5 * (vcenter - vmin)
        upper_mid = vcenter + 0.5 * (vmax - vcenter)
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        formatter_code = f"""
            const t = Math.round(tick * 100) / 100;
            if (t === 0.00) return "{vmin:.1f}";
            if (t === 0.25) return "{lower_mid:.1f}";
            if (t === 0.50) return "{vcenter:.1f}";
            if (t === 0.75) return "{upper_mid:.1f}";
            if (t === 1.00) return "{vmax:.1f}";
            return "";
        """

    for cb in colorbars:
        cb.ticker = FixedTicker(ticks=ticks)
        cb.formatter = CustomJSTickFormatter(code=formatter_code)
        cb.title = "Median RNA/FOV"


def fov_scatter_plot_raw(
    df: pd.DataFrame | None,
    extra_hooks: list | None = None,
) -> object | None:

    """Create a FOV-level scatter plot with Matplotlib-like diverging colour mapping centered at 48."""
    if df is None or df.empty:
        return None

    fov_col = find_column(df, "fov", "FOV", "field_of_view", "fieldofview")
    x_col = "CenterX_global_px"
    y_col = "CenterY_global_px"
    ncount_rna_col = find_column(df, "nCount_RNA", "ncountrna", "count_rna")
    ncount_negprobes_col = find_column(df, "nCount_negprobes", "ncountnegprobes", "count_negprobes")

    if not all([fov_col, x_col in df.columns, y_col in df.columns, ncount_rna_col]):
        return None

    work = df.copy()
    work[x_col] = safe_numeric(work[x_col])
    work[y_col] = safe_numeric(work[y_col])
    work[ncount_rna_col] = safe_numeric(work[ncount_rna_col])

    if ncount_negprobes_col:
        work[ncount_negprobes_col] = safe_numeric(work[ncount_negprobes_col])

    work = work.dropna(subset=[fov_col, x_col, y_col, ncount_rna_col])
    if work.empty:
        return None

    median_rna_col = find_column(work, "median_RNA", "medianrna", "median_rna")
    
    fov_data = work.groupby(fov_col).agg({
            x_col: "mean",
            y_col: "mean",
            median_rna_col: "first",
        }).reset_index()
    fov_data.columns = ["fov", "spatial_fov_x", "spatial_fov_y", "median_rna"]

    if ncount_negprobes_col and ncount_negprobes_col in work.columns:
        negprobe_stats = work.groupby(fov_col)[ncount_negprobes_col].median()
        rna_stats = work.groupby(fov_col)[ncount_rna_col].median()
        ratio = (negprobe_stats / rna_stats).replace([np.inf, -np.inf], np.nan).fillna(0.05)
        fov_data["ratio_negprobes"] = fov_data["fov"].map(ratio).fillna(0.05)
    else:
        fov_data["ratio_negprobes"] = 0.05

    fov_data = fov_data.dropna(subset=["spatial_fov_x", "spatial_fov_y", "median_rna"])
    if fov_data.empty:
        return None

    ratio_min = fov_data["ratio_negprobes"].min()
    ratio_max = fov_data["ratio_negprobes"].max()
    ratio_range = ratio_max - ratio_min if ratio_max > ratio_min else 1.0
    fov_data["bubble_size"] = 10 + 10 * (fov_data["ratio_negprobes"] - ratio_min) / ratio_range

    vcenter = 48.0
    # Recommended: robust upper cap for display
    # Change 0.99 to 1.0 if you want exact full-range behaviour
    # upper_q = 0.66
    # display_max = float(fov_data["median_rna"].quantile(upper_q))
    # display_max = max(display_max, vcenter + 1e-9)
    display_max = vcenter*3
    display_min = vcenter/3
    fov_data["color_vmin"] = display_min
    fov_data["color_vmax"] = display_max
    fov_data["color_vcenter"] = vcenter


    # Clip only for DISPLAY; keep true values in hover
    fov_data["median_rna_display"] = fov_data["median_rna"].clip(lower=display_min, upper=display_max)

    fov_data["median_rna_twoslope"] = _two_slope_normalize_series(
        fov_data["median_rna_display"],
        vmin=display_min,
        vcenter=vcenter,
        vmax=display_max,
    )

    palette = _matplotlib_palette("RdYlGn", 256)


    points = hv.Points(
        fov_data,
        kdims=["spatial_fov_x", "spatial_fov_y"],
        vdims=[
            "fov",
            "median_rna",
            "median_rna_display",
            "median_rna_twoslope",
            "ratio_negprobes",
            "bubble_size",
            "color_vmin",
            "color_vmax",
            "color_vcenter",
        ],
    )



    hooks = [
        _square_responsive_bokeh_hook,
        _fov_scatter_colorbar_relabel_hook,
    ]
    if extra_hooks:
        hooks.extend(extra_hooks)

    return points.opts(
        color="median_rna_twoslope",
        size=dim("bubble_size"),
        cmap=palette,
        clim=(0, 1),
        colorbar=True,
        colorbar_opts={"title": "Median RNA/FOV"},
        alpha=1.0,
        line_color=None,
        xlabel="Mean FOV X Position (pixels)",
        ylabel="Mean FOV Y Position (pixels)",
        tools=PLOT_TOOLS,
        toolbar="above",
        active_tools=["wheel_zoom"],
        responsive=True,
        data_aspect=1,
        framewise=True,
        hooks=hooks,
    )




def spatial_plot(
    df: pd.DataFrame | None,
    color_by: str = DEFAULT_SPATIAL_COLOR_BY,
    sample: bool = False,
    extra_hooks: list | None = None,
) -> pn.viewable.Viewable:

    """Create a spatial hexbin plot coloured by a selected column."""
    x_col = "CenterX_global_px"
    y_col = "CenterY_global_px"

    if not has_cols(df, x_col, y_col):
        return panel_message("No spatial data available.")

    if color_by not in df.columns:
        return panel_message(f"Column `{color_by}` is not available.")

    plot_df = maybe_sample_spatial(df, enabled=sample)

    keep_cols = [x_col, y_col, color_by]
    plot_df = plot_df[keep_cols].copy()
    plot_df[x_col] = safe_numeric(plot_df[x_col])
    plot_df[y_col] = safe_numeric(plot_df[y_col])
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        return panel_message("No usable spatial coordinates available.")

    sample_note = ""
    if sample and df is not None and len(plot_df) < len(df):
        sample_note = f" (sampled {len(plot_df):,} / {len(df):,} cells)"

    mode = classify_color_column(df[color_by])

    if mode == "numeric":
        plot_df[color_by] = safe_numeric(plot_df[color_by])
        plot_df = plot_df.dropna(subset=[color_by])

        if plot_df.empty:
            return panel_message(f"No usable numeric values in `{color_by}`.")

        title = f"Cell Spatial Distribution — mean {color_by}{sample_note}"

        main = plot_df.hvplot.hexbin(
            x=x_col,
            y=y_col,
            C=color_by,
            reduce_function=np.mean,
            clabel=f"Mean {color_by}",
            gridsize=SPATIAL_PLOT_GRIDSIZE,
            min_count=1,
            cmap="viridis",
            colorbar=True,
            xlabel="X Position (pixels)",
            ylabel="Y Position (pixels)",
            title=title,
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset"],
            **plot_kwargs(),
        )
    else:
        cat = plot_df[color_by].astype("string").fillna("Missing")

        if cat.nunique() > MAX_SPATIAL_CATEGORIES:
            top = set(cat.value_counts().nlargest(MAX_SPATIAL_CATEGORIES - 1).index)
            cat = cat.where(cat.isin(top), "Other")

        categories = list(cat.dropna().unique())
        if not categories:
            return panel_message(f"No usable categorical values in `{color_by}`.")

        code_map = {cat_name: idx for idx, cat_name in enumerate(categories)}
        plot_df["_color_code"] = cat.map(code_map).astype(float)

        palette = SPATIAL_CATEGORY_COLORS[: len(categories)]
        title = f"Cell Spatial Distribution — majority {color_by}{sample_note}"

        main = plot_df.hvplot.hexbin(
            x=x_col,
            y=y_col,
            C="_color_code",
            reduce_function=majority_code,
            clabel=f"Majority {color_by}",
            gridsize=SPATIAL_PLOT_GRIDSIZE,
            min_count=1,
            cmap=palette,
            colorbar=False,
            xlabel="X Position (pixels)",
            ylabel="Y Position (pixels)",
            title=title,
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset"],
            **plot_kwargs(),
        )



    hooks = [_square_responsive_bokeh_hook]
    if extra_hooks:
        hooks.extend(extra_hooks)

    return main.opts(
        toolbar="above",
        active_tools=["wheel_zoom"],
        shared_axes=True,
        responsive=True,
        data_aspect=1,
        hooks=hooks,
    ).clone()


def compute_fov_cell_qc_flags(df: pd.DataFrame | None) -> dict[str, int | float]:
    """Compute FOV and cell QC flags based on median_RNA and nCount_RNA thresholds.
    
    Flags FOVs with median_RNA < 24 and cells with nCount_RNA < 20.
    """
    if df is None or df.empty:
        return {
            "flagged_fovs": 0,
            "total_fovs": 0,
            "flagged_cells": 0,
            "total_cells": 0,
        }

    result = {
        "flagged_fovs": 0,
        "total_fovs": 0,
        "flagged_cells": 0,
        "total_cells": 0,
    }

    # FOV-level flags based on median_RNA
    fov_col = find_column(df, "fov", "FOV", "field_of_view", "fieldofview")
    median_rna_col = find_column(df, "median_RNA", "medianrna", "median_rna")

    if fov_col is not None and median_rna_col is not None:
        fov_stats = df.groupby(fov_col)[median_rna_col].median()
        result["total_fovs"] = len(fov_stats)
        result["flagged_fovs"] = int((fov_stats < 24).sum())

    # Cell-level flags based on nCount_RNA
    ncount_rna_col = find_column(df, "nCount_RNA", "ncountrna", "count_rna")
    if ncount_rna_col is not None:
        result["total_cells"] = len(df)
        result["flagged_cells"] = int((safe_numeric(df[ncount_rna_col]) < 20).sum())

    return result


def get_qc_flag_color(ratio: float, flag_type: str) -> str:
    """Determine QC flag card color based on ratio and thresholds.
    
    Parameters
    ----------
    ratio : float
        Fraction of flagged items (0.0 to 1.0).
    flag_type : str
        Either 'fov' or 'cell' to use appropriate thresholds.
    
    Returns
    -------
    str
        Hex color code.
    """
    if flag_type == "fov":
        if ratio < 0.2:
            return "#22c55e"  # green
        elif ratio < 0.5:
            return "#f97316"  # orange
        else:
            return "#ef4444"  # red
    elif flag_type == "cell":
        if ratio < 0.1:
            return "#22c55e"  # green
        elif ratio < 0.3:
            return "#f97316"  # orange
        else:
            return "#ef4444"  # red
    return "#6b7280"  # gray default


def create_qc_flag_cards(df: pd.DataFrame | None) -> pn.viewable.Viewable:
    """Create QC flagging status cards for FOVs and cells."""
    flags = compute_fov_cell_qc_flags(df)

    fov_ratio = flags["flagged_fovs"] / flags["total_fovs"] if flags["total_fovs"] > 0 else 0.0
    cell_ratio = flags["flagged_cells"] / flags["total_cells"] if flags["total_cells"] > 0 else 0.0

    fov_color = get_qc_flag_color(fov_ratio, "fov")
    cell_color = get_qc_flag_color(cell_ratio, "cell")

    return responsive_flexbox(
        qc_flag_status_card(
            flags["flagged_fovs"],
            "Flagged FOVs",
            fov_ratio * 100,
            fov_color,
        ),
        qc_flag_status_card(
            flags["flagged_cells"],
            "Flagged Cells",
            cell_ratio * 100,
            cell_color,
        ),
    )


def create_indicators(stats: dict[str, float | int | None]) -> pn.viewable.Viewable:
    """Create a responsive KPI layout for the selected dataset slice."""
    if not stats:
        return panel_message("No data available.")

    return responsive_flexbox(
        indicator_card(stats.get("cell_count"), "Total Cells"),
        indicator_card(stats.get("median_rna"), "Med. Transcripts/Cell"),
        indicator_card(stats.get("med_features"), "Med. Genes/Cell"),
        indicator_card(stats.get("avg_area"), "Avg. Cell Area (µm²)", "{:,.1f}"),
    )



def _empty_tabulator() -> pn.widgets.Tabulator:
    """Return an empty Tabulator with the standard dashboard configuration."""
    return pn.widgets.Tabulator(
        empty_df(),
        sizing_mode="stretch_width",
        pagination="local",
        page_size=TABLE_PAGE_SIZE,
    )


def create_qc_metrics_table(df: pd.DataFrame | None) -> pn.widgets.Tabulator:
    """Create a tabular FOV QC summary from precomputed or derived metrics."""
    if df is None or df.empty:
        return _empty_tabulator()

    fov_col = find_column(df, "fov", "FOV", "field_of_view", "fieldofview")
    if fov_col is None:
        return _empty_tabulator()

    candidate_cols = [
        "nCell",
        "nCount",
        "nCountPerCell",
        "nFeaturePerCell",
        "propNegativeCellAvg",
        "complexityCellAvg",
        "qcFlagsFOV",
    ]

    resolved = {"fov": fov_col}
    for c in candidate_cols:
        actual = find_column(df, c)
        if actual is not None:
            resolved[c] = actual

    cols = ["fov"] + [c for c in candidate_cols if c in resolved]
    if len(cols) > 1:
        out = df[[resolved[c] for c in cols]].copy()
        out.columns = cols
        fov_metrics = out.drop_duplicates().set_index("fov")
        return pn.widgets.Tabulator(
            fov_metrics.sort_index(),
            sizing_mode="stretch_width",
            pagination="local",
            page_size=TABLE_PAGE_SIZE,
        )

    summary = compute_fov_summary(df)
    if summary is None or summary.empty or "fov" not in summary.columns:
        return _empty_tabulator()

    display_cols = [
        c
        for c in ["fov", "nCell", "nCount", "nCountPerCell", "nFeaturePerCell", "Area"]
        if c in summary.columns
    ]

    fov_metrics = summary[display_cols].drop_duplicates().set_index("fov")

    return pn.widgets.Tabulator(
        fov_metrics.sort_index(),
        sizing_mode="stretch_width",
        # height=552,
        pagination="local",
        page_size=TABLE_PAGE_SIZE,
    )




def create_qc_flags_summary(df: pd.DataFrame | None) -> pn.Column:
    if df is None or df.empty or not has_cols(df, "qcCellsPassed", "qcCellsFlagged"):
        return pn.Column(
            pn.pane.Markdown("No QC summary data available."),
            min_height=PLOT_SHORT_MIN_HEIGHT,
            sizing_mode="stretch_width",
        )

    passed = int((df["qcCellsPassed"] == True).sum())
    flagged = int((df["qcCellsFlagged"] == True).sum())
    total = int(len(df))
    pass_pct = (passed / total * 100) if total else 0.0

    chart = pd.DataFrame(
        {"Status": ["Passed", "Flagged"], "Count": [passed, flagged]}
    ).hvplot.barh(
        x="Status",
        y="Count",
        color=[QC_STATUS_COLORS["passed"], QC_STATUS_COLORS["flagged"]],
        legend=False,
        **plot_kwargs(),
    )

    return pn.Column(
        responsive_flexbox(
            status_card(passed, f"Passed QC ({pass_pct:.1f}%)", QC_STATUS_COLORS["passed"]),
            status_card(flagged, f"Flagged ({100 - pass_pct:.1f}%)", QC_STATUS_COLORS["flagged"]),
            status_card(total, "Total Cells", QC_STATUS_COLORS["total"]),
        ),
        plot_box(chart, min_height=PLOT_SHORT_MIN_HEIGHT),
        sizing_mode="stretch_width",
    )

def create_status_pane(message: str, level: str) -> pn.pane.Pane:
    """Create the top-level status alert pane."""
    if not message:
        return pn.Spacer(height=0)
    return pn.pane.Alert(message, alert_type=level, sizing_mode="stretch_width")


# ============================================================================
# Component Assembly (Helpers for Dashboard View Binding)
# ============================================================================


def _boxed_hist(
    df: pd.DataFrame | None,
    column: str,
    title: str,
    *,
    bins: int = 50,
    xlim: tuple[float, float] | None = None,
    min_height: int = PLOT_MIN_HEIGHT,
) -> pn.viewable.Viewable:
    """Create a boxed histogram plot with title and fallback messaging."""
    plot = hist_plot_raw(df, column, title, bins=bins, xlim=xlim)
    if plot is None:
        return panel_message(f"No {title.lower()} data available.")
    return plot_box(plot, min_height=min_height, title=title)


def _boxed_scatter(
    df: pd.DataFrame | None,
    x: str,
    y: str,
    title: str,
    *,
    min_height: int = PLOT_MIN_HEIGHT,
) -> pn.viewable.Viewable:
    """Create a boxed scatter plot with title and fallback messaging."""
    plot = scatter_plot_raw(df, x, y)
    if plot is None:
        return panel_message("No comparison data available.")
    return plot_box(plot, min_height=min_height, title=title)


def _boxed_metrics_jointplot(
    df: pd.DataFrame | None,
    x: str,
    y: str,
    hue: str,
    title: str,
    *,
    min_height: int = PLOT_TALL_MIN_HEIGHT,
) -> pn.viewable.Viewable:
    """Create a boxed metrics scatter plot with fallback messaging."""
    plot = metrics_jointplot_raw(df, x, y, hue)
    if plot is None:
        return panel_message("No metrics data available.")
    return plot_box(plot, min_height=min_height, max_width=800, title=title)


def _boxed_linked_spatial_views(
    df: pd.DataFrame | None,
    color_by: str,
    *,
    report_mode: bool,
) -> pn.viewable.Viewable:
    """Create responsive side-by-side spatial views with truly shared Bokeh ranges and 4:3 aspect ratio."""
    if df is None or df.empty:
        return panel_message("No spatial data available.")

    x_range, y_range = _compute_shared_spatial_ranges(df)
    extra_hooks = []
    if x_range is not None and y_range is not None:
        extra_hooks.append(_shared_range_hook(x_range, y_range))

    fov_plot = fov_scatter_plot_raw(df, extra_hooks=extra_hooks)
    spatial = spatial_plot(
        df,
        color_by=color_by,
        sample=not report_mode,
        extra_hooks=extra_hooks,
    )

    if fov_plot is None or spatial is None:
        return panel_message("No spatial data available.")

    return responsive_flexbox(
        flex_item(
            plot_box(fov_plot, aspect_ratio=4/3, max_width=600, min_width=PLOT_PANEL_MIN_WIDTH),
            min_width=PLOT_PANEL_MIN_WIDTH,
            grow=1,
            allow_shrink_below_min=True,
        ),
        flex_item(
            plot_box(spatial, aspect_ratio=4/3, max_width=600, min_width=PLOT_PANEL_MIN_WIDTH),
            min_width=PLOT_PANEL_MIN_WIDTH,
            grow=1,
            allow_shrink_below_min=True,
        ),
        gap="16px",
    )


def make_component_bindings(report_mode: bool = False) -> dict[str, object]:
    """Create reactive view bindings for a template instance."""

    return {
        # Status / summary
        "status_pane": pn.bind(
            create_status_pane,
            state.param.status_message,
            state.param.status_level,
        ),
        "indicators": pn.bind(create_indicators, cell_stats),
        "qc_flag_cards": pn.bind(create_qc_flag_cards, filtered_df),

        # Summary tab - combined linked spatial views
        "linked_spatial_views": pn.bind(
            _boxed_linked_spatial_views,
            filtered_df,
            color_by_select,
            report_mode=report_mode,
        ),
        "qc_metrics_tbl": pn.bind(create_qc_metrics_table, filtered_df),

        # Sequencing tab – QC
        "qc_flags_plot": pn.bind(create_qc_flags_summary, filtered_df),

        # Sequencing tab – Negative probes
        "negprobes_hist": pn.bind(
            _boxed_hist,
            filtered_df,
            "nCount_negprobes",
            "Negative Probes Count Distribution",
        ),
        "rna_vs_negprobes": pn.bind(
            _boxed_scatter,
            filtered_df,
            "nCount_RNA",
            "nCount_negprobes",
            "RNA Count vs Negative Probes",
        ),
        "propnegative_hist": pn.bind(
            _boxed_hist,
            filtered_df,
            "propNegative",
            "Proportion Negative",
            bins=30,
            xlim=(0, 1),
        ),

        # Cell segmentation / sequencing metrics
        "metrics_jointplot": pn.bind(
            _boxed_metrics_jointplot,
            filtered_df,
            "nCount_RNA",
            "nFeature_RNA",
            "Area.um2",
            "Cell Metrics Summary",
        ),
        "rna_hist": pn.bind(
            _boxed_hist,
            filtered_df,
            "nCount_RNA",
            "RNA Count Distribution",
        ),
        "feature_hist": pn.bind(
            _boxed_hist,
            filtered_df,
            "nFeature_RNA",
            "Feature Count Distribution",
        ),
        "area_hist": pn.bind(
            _boxed_hist,
            filtered_df,
            "Area.um2",
            "Cell Area Distribution",
        ),
    }



def build_summary_tab(views: Mapping[str, object]) -> pn.Column:
    """Build the summary tab layout."""
    return pn.Column(
        pn.pane.Markdown("### Key QC Metrics"),
        views["indicators"],
        views["qc_flag_cards"],
        pn.pane.Markdown("### Sample Overview"),
        views["linked_spatial_views"],
        sizing_mode="stretch_both",
        styles={
            "overflow-y": "auto",
        },
    )




def build_sequencing_tab(views: Mapping[str, object]) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("### QC Metrics"),
        views["qc_flags_plot"],
        pn.pane.Markdown("### Negative Probes"),
        responsive_flexbox(
            flex_item(views["negprobes_hist"], min_width=420),
            flex_item(views["rna_vs_negprobes"], min_width=420),
            flex_item(views["propnegative_hist"], min_width=420),
            gap="16px",
        ),
        sizing_mode="stretch_both",
        styles={
            "overflow-y": "auto",   # contain vertical overflow
        },
    )


def build_segmentation_tab(views: Mapping[str, object]) -> pn.Column:
    """Build the cell segmentation metrics tab layout."""
    return pn.Column(
        pn.pane.Markdown("### Cell Segmentation Metrics"),
        pn.pane.Markdown("### Summary View"),
        views["metrics_jointplot"],
        pn.pane.Markdown("### Metric Distribution"),
        responsive_flexbox(
            flex_item(views["area_hist"], min_width=420),
            flex_item(views["rna_hist"], min_width=420),
            flex_item(views["feature_hist"], min_width=420),
            gap="16px",
        ),
        pn.pane.Markdown("### FOV QC Metrics"),
        views["qc_metrics_tbl"],
        sizing_mode="stretch_both",
        # scroll=True,
        styles={
            "overflow-y": "auto",   # contain vertical overflow
        },
    )


def build_analysis_tab() -> pn.Column:
    """Build the placeholder analysis tab."""
    return pn.Column(
        pn.pane.Markdown("### WIP"),
        sizing_mode="stretch_both",
        # scroll=True,
        styles={
            "overflow-y": "auto",   # contain vertical overflow
        },
    )


def build_image_qc_tab() -> pn.Column:
    """Build the image QC placeholder tab."""
    return pn.Column(
        pn.pane.Markdown(
            "### Image QC\nImage quality control analysis is not available for this dataset."
        ),
        sizing_mode="stretch_both",
        # scroll=True,
        styles={
            "overflow-y": "auto",   # contain vertical overflow
        },
    )


def create_tabs(views: Mapping[str, object], report_mode: bool = False) -> pn.Tabs:
    """Create the dashboard tab container.

    Live mode uses dynamic rendering for responsiveness. Report mode renders all
    tab contents eagerly so the exported static HTML contains every view.
    """
    return pn.Tabs(
        ("Summary", build_summary_tab(views)),
        ("Sequencing", build_sequencing_tab(views)),
        ("Cell Segmentation", build_segmentation_tab(views)),
        ("Analysis", build_analysis_tab()),
        ("Image QC", build_image_qc_tab()),
        dynamic=not report_mode,
        styles=CARD_STYLES,
        sizing_mode="stretch_both",
        margin=10,
    )


def loaded_filename_pane():
    """Create a reactive pane showing the currently loaded filename."""
    return pn.bind(
        lambda filename: pn.pane.Markdown(
            f"**Loaded:** {filename}",
            styles={"color": "gray", "font-size": "12px"},
        ),
        state.param.filename,
    )


def pending_filename_pane():
    """Create a reactive pane showing the currently selected upload filename."""
    return pn.bind(
        lambda filename: (
            pn.pane.Markdown(
                f"**Selected:** {filename}",
                styles={"color": "#555", "font-size": "12px"},
            )
            if filename
            else pn.Spacer(height=0)
        ),
        state.param.pending_filename,
    )


def title_pane():
    """Create a reactive title pane for the loaded object."""
    return pn.bind(
        lambda filename: pn.pane.Markdown(f"## Loaded Object: {filename}"),
        state.param.filename,
    )


def create_template(report_mode: bool = False) -> pn.template.BootstrapTemplate:
    """Create the main dashboard template for live or report rendering."""
    views = make_component_bindings(report_mode=report_mode)
    tabs = create_tabs(views, report_mode=report_mode)

    if not report_mode:
        sidebar_items = [
            pn.pane.Markdown("### Load Data"),
            pn.pane.Markdown("Upload one `.csv` file (large files supported):"),
            file_dropper,
            pending_filename_pane(),
            load_button,
            loaded_filename_pane(),
            views["status_pane"],
            pn.pane.Markdown("### Filters"),
            fov_select,
            color_by_select,
            pn.pane.Markdown("### Export"),
            export_html_button,
            export_csv_button,
        ]

        return pn.template.BootstrapTemplate(
            title="POPIDD-SPOT: Spatial Profiling Overview Tool [LIVE]",
            sidebar=sidebar_items,
            collapsed_sidebar=True,
            header_background="#d4a300",
            main=[
                pn.Column(
                    title_pane(),
                    tabs,
                    sizing_mode="stretch_both",
                )
            ],
        )

    return pn.template.BootstrapTemplate(
        title="POPIDD-SPOT: Spatial Profiling Overview Tool [STATIC]",
        header_background="#d4a300",
        main=[
            pn.Column(
                title_pane(),
                tabs,
                sizing_mode="stretch_both",
            )
        ],
    )


app = create_template(report_mode=False).servable()


def save_as_html(filename: str | os.PathLike = DEFAULT_REPORT_PATH) -> None:
    """Export the dashboard as a static HTML report."""
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    create_template(report_mode=True).save(str(output), resources="cdn")
    LOGGER.info("Dashboard saved to %s", output)


def save_data_as_csv(filename: str | os.PathLike | None = None) -> Path:
    """Export the current dataset (with QC flags) as a CSV file.
    
    If filename is None, saves to report/ directory with a timestamped name.
    """
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path("report") / f"popidd-spot_qc_flagged_{timestamp}.csv"
    
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    df = get_source_data()
    if df is not None and not df.empty:
        df.to_csv(output, index=False)
        LOGGER.info("Data saved to %s", output)
        set_status(f"Data exported to `{output.name}`", "success")
    else:
        LOGGER.warning("No data to export")
        set_status("No data to export", "warning")
    
    return output


def main() -> None:
    """Run the default export workflow."""
    save_as_html()


if __name__ == "__main__":
    main()
