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

pn.extension("tabulator", "filedropper", design="bootstrap", theme="default")

ACCENT = "teal"
DATA_DIR = Path("input")
DEFAULT_REPORT_PATH = Path("report/popidd-spot_report.html")
TABLE_PAGE_SIZE = 25


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
DEFAULT_SPATIAL_COLOR_BY = "Mean.DAPI"

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def empty_df() -> pd.DataFrame:
    """Return a new empty DataFrame."""
    return pd.DataFrame()


def set_status(message: str = "", level: str = "info") -> None:
    """Update the global application status message and severity level."""
    state.status_message = message
    state.status_level = level


def set_data(df: pd.DataFrame | None, filename: str) -> None:
    """Update the global dataset state and increment its revision counter."""
    state.data = df if df is not None else empty_df()
    state.filename = filename
    state.data_revision += 1


def has_cols(df: pd.DataFrame | None, *cols: str) -> bool:
    """Return whether the DataFrame exists, is non-empty, and contains all columns."""
    return df is not None and not df.empty and all(col in df.columns for col in cols)


def safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric values, replacing invalid entries with NaN."""
    return pd.to_numeric(series, errors="coerce")


def panel_message(text: str, alert_type: str | None = None):
    """Return a Panel message pane, optionally styled as an alert."""
    if alert_type:
        return pn.pane.Alert(text, alert_type=alert_type, sizing_mode="stretch_width")
    return pn.pane.Markdown(text, sizing_mode="stretch_width")


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


def read_csv_any(source) -> pd.DataFrame:
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


def plot_kwargs(**extra):
    """Return standard hvPlot keyword arguments with optional overrides."""
    base = {
        "responsive": True,
        "shared_axes": False,
    }
    base.update(extra)
    return base


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
        df = read_csv_any(path)
        set_status(f"Loaded default dataset: `{path.name}`", "success")
        return df, path.name
    except Exception as exc:
        LOGGER.exception("Failed to load default dataset")
        set_status(f"Failed to load default dataset `{path}`: {exc}", "danger")
        return empty_df(), "No dataset loaded"


def is_supported_color_column(series: pd.Series) -> bool:
    """Return whether a Series can be used for spatial colour encoding."""
    return (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


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

    return ordered or [DEFAULT_SPATIAL_COLOR_BY]


def update_colorby_options(*_) -> None:
    """Refresh the spatial colour-by widget options from the current dataset."""
    options = get_colorby_options(state.data)
    color_by_select.options = options
    if color_by_select.value not in options:
        color_by_select.value = DEFAULT_SPATIAL_COLOR_BY


def classify_color_column(series: pd.Series) -> str:
    """Classify a colour column as numeric or categorical."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "numeric"
    return "categorical"


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


state = DataState()

_default_df, _default_name = load_default_data()
set_data(_default_df, _default_name)

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
        df = read_csv_any(content)
        set_data(df, filename)
        set_status(f"Loaded `{filename}` successfully.", "success")
    except Exception as exc:
        LOGGER.exception("Error loading uploaded CSV")
        set_status(f"Error loading `{filename}`: {exc}", "danger")


file_dropper.param.watch(_on_file_selected, "value")
load_button.on_click(_load_selected_dataset)


def get_source_data() -> pd.DataFrame:
    """Return the current source dataset."""
    return state.data if state.data is not None else empty_df()


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


def indicator_card(value, label: str, fmt: str = "{:,.0f}") -> pn.pane.HTML:
    """Render a single KPI indicator card."""
    if value is None:
        display_value = "—"
    elif isinstance(value, (int, float)):
        display_value = fmt.format(value)
    else:
        display_value = str(value)

    html = f"""
    <div style="text-align:center;height:100%;display:flex;flex-direction:column;justify-content:center;">
        <div style="font-size:clamp(11px,1.5vw,14px);color:#999;margin-bottom:8px;font-weight:500;">{label}</div>
        <div style="font-size:clamp(18px,4vw,32px);font-weight:bold;color:#333;">{display_value}</div>
    </div>
    """
    return pn.pane.HTML(html, styles=CARD_STYLES, sizing_mode="stretch_both")


def status_card(value: int, label: str, color: str) -> pn.pane.HTML:
    """Render a coloured QC status summary card."""
    html = f"""
    <div style="
        background-color:{color};
        border-radius:8px;
        padding:20px;
        text-align:center;
        color:white;
        min-height:100px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        box-shadow:0 2px 8px rgba(0,0,0,0.15);
    ">
        <div style="font-size:32px;font-weight:bold;">{value:,}</div>
        <div style="font-size:13px;margin-top:8px;opacity:0.95;">{label}</div>
    </div>
    """
    return pn.pane.HTML(html, sizing_mode="stretch_both")


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


def hist_plot(
    df: pd.DataFrame | None,
    column: str,
    title: str,
    bins: int = 50,
    xlim: tuple[float, float] | None = None,
):
    """Create a histogram for a numeric column."""
    if not has_cols(df, column):
        return panel_message(f"No `{column}` data available.")

    data = safe_numeric(df[column]).dropna()
    if xlim is not None:
        data = data[(data >= xlim[0]) & (data <= xlim[1])]

    if data.empty:
        return panel_message(f"`{column}` contains no usable numeric values.")

    return data.hvplot.hist(
        bins=bins,
        title=title,
        xlabel=column,
        ylabel="Number of Cells",
        color=ACCENT,
        height=360,
        **plot_kwargs(),
    )


def scatter_plot(df: pd.DataFrame | None, x: str, y: str, title: str):
    """Create a scatter plot comparing two numeric columns."""
    if not has_cols(df, x, y):
        return panel_message("No comparison data available.")

    plot_df = df[[x, y]].apply(safe_numeric).dropna()
    if plot_df.empty:
        return panel_message("No valid comparison data.")

    return plot_df.hvplot.scatter(
        x=x,
        y=y,
        title=title,
        xlabel=x,
        ylabel=y,
        color=ACCENT,
        height=360,
        tools=PLOT_TOOLS,
        **plot_kwargs(),
    )


def spatial_plot(
    df: pd.DataFrame | None,
    color_by: str = DEFAULT_SPATIAL_COLOR_BY,
    sample: bool = False,
):
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
            gridsize=256,
            min_count=1,
            cmap="viridis",
            colorbar=True,
            xlabel="X Position (pixels)",
            ylabel="Y Position (pixels)",
            title=title,
            height=520,
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset"],
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
            gridsize=256,
            min_count=1,
            cmap=palette,
            colorbar=False,
            xlabel="X Position (pixels)",
            ylabel="Y Position (pixels)",
            title=title,
            height=520,
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset"],
        )

    return main.opts(
        toolbar="right",
        active_tools=["wheel_zoom"],
        shared_axes=True,
    )


def create_indicators(stats: dict[str, float | int | None]):
    """Create the summary KPI row for the currently selected dataset slice."""
    if not stats:
        return panel_message("No data available.")

    return pn.Row(
        indicator_card(stats.get("cell_count"), "Total Cells"),
        indicator_card(stats.get("median_rna"), "Med. Transcripts/Cell"),
        indicator_card(stats.get("med_features"), "Med. Genes/Cell"),
        indicator_card(stats.get("avg_area"), "Avg. Cell Area (µm²)", "{:,.1f}"),
        sizing_mode="stretch_width",
        height=100,
    )


def _empty_tabulator() -> pn.widgets.Tabulator:
    """Return an empty Tabulator with the standard dashboard configuration."""
    return pn.widgets.Tabulator(
        empty_df(),
        sizing_mode="stretch_both",
        pagination="local",
        page_size=TABLE_PAGE_SIZE,
    )


def create_qc_metrics_table(df: pd.DataFrame | None):
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
            sizing_mode="stretch_both",
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
        sizing_mode="stretch_both",
        pagination="local",
        page_size=TABLE_PAGE_SIZE,
    )


def create_qc_flags_summary(df: pd.DataFrame | None):
    """Create a QC pass/fail summary view with status cards and bar chart."""
    if df is None or df.empty:
        return panel_message("No data available.")

    if not has_cols(df, "qcCellsPassed", "qcCellsFlagged"):
        return panel_message("No QC summary data available.")

    passed = int((df["qcCellsPassed"] == True).sum())
    flagged = int((df["qcCellsFlagged"] == True).sum())
    total = int(len(df))
    pass_pct = (passed / total * 100) if total else 0.0

    chart_data = pd.DataFrame({"Status": ["Passed", "Flagged"], "Count": [passed, flagged]})

    bar_chart = chart_data.hvplot.barh(
        x="Status",
        y="Count",
        color=[QC_STATUS_COLORS["passed"], QC_STATUS_COLORS["flagged"]],
        legend=False,
        ylabel="",
        xlabel="Cell Count",
        height=140,
        **plot_kwargs(),
    )

    return pn.Column(
        pn.Row(
            status_card(passed, f"Passed QC ({pass_pct:.1f}%)", QC_STATUS_COLORS["passed"]),
            status_card(flagged, f"Flagged ({100 - pass_pct:.1f}%)", QC_STATUS_COLORS["flagged"]),
            status_card(total, "Total Cells", QC_STATUS_COLORS["total"]),
            sizing_mode="stretch_width",
            height=130,
        ),
        bar_chart,
        sizing_mode="stretch_both",
    )


def create_status_pane(message: str, level: str):
    """Create the top-level status alert pane."""
    if not message:
        return pn.Spacer(height=0)
    return pn.pane.Alert(message, alert_type=level, sizing_mode="stretch_width")


def make_component_bindings(report_mode: bool = False) -> dict[str, object]:
    """Create reactive view bindings for a template instance."""
    spatial_binding = pn.bind(
        lambda df, color_by: spatial_plot(df, color_by=color_by, sample=not report_mode),
        filtered_df,
        color_by_select,
    )

    return {
        "status_pane": pn.bind(create_status_pane, state.param.status_message, state.param.status_level),
        "indicators": pn.bind(create_indicators, cell_stats),
        "spatial_plot": spatial_binding,
        "qc_metrics_tbl": pn.bind(create_qc_metrics_table, filtered_df),
        "qc_flags_plot": pn.bind(create_qc_flags_summary, filtered_df),
        "negprobes_hist": pn.bind(
            hist_plot,
            filtered_df,
            "nCount_negprobes",
            "Negative Probes Count Distribution",
        ),
        "rna_vs_negprobes": pn.bind(
            scatter_plot,
            filtered_df,
            "nCount_RNA",
            "nCount_negprobes",
            "RNA Count vs Negative Probes",
        ),
        "propnegative_hist": pn.bind(
            hist_plot,
            filtered_df,
            "propNegative",
            "Proportion Negative",
            bins=30,
            xlim=(0, 1),
        ),
        "complexity_hist": pn.bind(
            hist_plot,
            filtered_df,
            "complexity",
            "Cell Complexity Distribution",
        ),
        "rna_hist": pn.bind(
            hist_plot,
            filtered_df,
            "nCount_RNA",
            "RNA Count Distribution",
        ),
        "feature_hist": pn.bind(
            hist_plot,
            filtered_df,
            "nFeature_RNA",
            "Feature Count Distribution",
        ),
        "area_hist": pn.bind(
            hist_plot,
            filtered_df,
            "Area.um2",
            "Cell Area Distribution",
        ),
    }


def build_summary_tab(views: Mapping[str, object]) -> pn.Column:
    """Build the summary tab layout."""
    return pn.Column(
        pn.pane.Markdown("### Key QC Metrics"),
        pn.Row(views["indicators"], sizing_mode="stretch_width"),
        pn.pane.Markdown("### Sample Overview"),
        pn.Row(
            pn.Column(
                views["spatial_plot"],
                sizing_mode="stretch_both",
                styles={"flex": "3"},
            ),
            pn.Column(
                views["qc_metrics_tbl"],
                sizing_mode="stretch_both",
                styles={"flex": "2"},
            ),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_both",
        scroll=True,
    )


def build_sequencing_tab(views: Mapping[str, object]) -> pn.Column:
    """Build the sequencing QC tab layout."""
    return pn.Column(
        pn.pane.Markdown("### QC Metrics"),
        pn.Row(views["qc_flags_plot"], sizing_mode="stretch_width"),
        pn.pane.Markdown("### Negative Probes"),
        pn.Row(
            pn.Column(views["negprobes_hist"], sizing_mode="stretch_both"),
            pn.Column(views["rna_vs_negprobes"], sizing_mode="stretch_both"),
            pn.Column(views["propnegative_hist"], sizing_mode="stretch_both"),
            sizing_mode="scale_width",
        ),
        sizing_mode="stretch_both",
        scroll=True,
    )


def build_segmentation_tab(views: Mapping[str, object]) -> pn.Column:
    """Build the cell segmentation metrics tab layout."""
    return pn.Column(
        pn.pane.Markdown("### WIP: Cell Segmentation Metrics"),
        pn.pane.Markdown("### Physical Cell Characteristics"),
        pn.Row(
            pn.Column(views["area_hist"], sizing_mode="stretch_both"),
            pn.Column(views["complexity_hist"], sizing_mode="stretch_both"),
            sizing_mode="scale_width",
        ),
        pn.pane.Markdown("### Sequencing Cell Metrics"),
        pn.Row(
            pn.Column(views["rna_hist"], sizing_mode="stretch_both"),
            pn.Column(views["feature_hist"], sizing_mode="stretch_both"),
            sizing_mode="scale_width",
        ),
        sizing_mode="stretch_both",
        scroll=True,
    )


def build_analysis_tab() -> pn.Column:
    """Build the placeholder analysis tab."""
    return pn.Column(
        pn.pane.Markdown("### WIP"),
        sizing_mode="stretch_both",
        scroll=True,
    )


def build_image_qc_tab() -> pn.Column:
    """Build the image QC placeholder tab."""
    return pn.Column(
        pn.pane.Markdown(
            "### Image QC\nImage quality control analysis is not available for this dataset."
        ),
        sizing_mode="stretch_both",
        scroll=True,
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
        sizing_mode="stretch_width",
        min_height=800,
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


def main() -> None:
    """Run the default export workflow."""
    save_as_html()


if __name__ == "__main__":
    main()
