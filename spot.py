from __future__ import annotations

import html
import io
import logging
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping

import holoviews as hv
import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import ColorBar, CustomJSTickFormatter, FixedTicker, Range1d
from holoviews import dim
from matplotlib import cm, colors as mcolors

pn.extension("tabulator", "filedropper", design="bootstrap", theme="default")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def get_total_memory_gb() -> int:
    if hasattr(os, "sysconf"):
        try:
            return max(1, int((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // 1024**3))
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
        return max(1, int(stat.ullTotalPhys // 1024**3))
    except Exception:
        return 4


ACCENT_COLOR = "teal"
DATA_DIR = Path("input")
DEFAULT_REPORT_PATH = Path("report/SPOT-qc.html")

PLOT_MIN_HEIGHT = 360
PLOT_TALL_MIN_HEIGHT = 460
PLOT_PANEL_MIN_WIDTH = 420
TABLE_PAGE_SIZE = 25
FLEX_GAP_DEFAULT = "12px"
FLEX_GAP_WIDE = "16px"

SPATIAL_X_COL = "CenterX_global_px"
SPATIAL_Y_COL = "CenterY_global_px"
DEFAULT_SPATIAL_COLOR_BY = "Mean.DAPI"
SPATIAL_PLOT_GRIDSIZE = 768
MAX_LIVE_SPATIAL_POINTS = get_total_memory_gb() * 100_000
MAX_SPATIAL_CATEGORIES = 20

QC_CELL_THRESHOLD = 20
QC_FOV_THRESHOLD = 42
FOV_MEDIAN_RNA_CENTER = 56
FOV_MEDIAN_RNA_DISPLAY_MIN = FOV_MEDIAN_RNA_CENTER / 2
FOV_MEDIAN_RNA_DISPLAY_MAX = FOV_MEDIAN_RNA_CENTER * 3

PLOT_TOOLS = ["hover", "box_select", "lasso_select", "reset"]

CARD_STYLES = {
    "box-shadow": "rgba(50,50,93,.25) 0 6px 12px -2px, rgba(0,0,0,.30) 0 3px 7px -3px",
    "border-radius": "4px",
    "padding": "10px",
}

QC_STATUS_COLORS = {"passed": "#22c55e", "flagged": "#ef4444", "total": "#6b7280"}

SPATIAL_CATEGORY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#31a354", "#756bb1", "#636363", "#e6550d",
]

ALIASES = {
    "fov": ("fov", "FOV", "field_of_view", "fieldofview"),
    "cell_id": ("cell_id", "cellid"),
    "ncount_rna": ("nCount_RNA", "ncountrna", "count_rna"),
    "nfeature_rna": ("nFeature_RNA", "nfeaturerna", "feature_rna"),
    "ncount_negprobes": ("nCount_negprobes", "ncountnegprobes", "count_negprobes"),
    "area": ("Area.um2", "area_um2", "areaum2", "area"),
}


def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_colname(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


@lru_cache(maxsize=64)
def _column_lookup(df_id: int, columns: tuple[str, ...]) -> dict[str, str]:
    return {normalize_colname(c): c for c in columns}


def find_column(df: pd.DataFrame | None, *aliases: str) -> str | None:
    if df is None or df.empty:
        return None
    lookup = _column_lookup(id(df), tuple(map(str, df.columns)))
    for alias in aliases:
        hit = lookup.get(normalize_colname(alias))
        if hit is not None:
            return hit
    return None


def col(df: pd.DataFrame | None, key: str) -> str | None:
    return find_column(df, *ALIASES[key])


def has_cols(df: pd.DataFrame | None, *cols: str) -> bool:
    return df is not None and not df.empty and all(c in df.columns for c in cols)


def panel_message(text: str, alert_type: str | None = None) -> pn.viewable.Viewable:
    if alert_type:
        return pn.pane.Alert(text, alert_type=alert_type, sizing_mode="stretch_width")
    return pn.pane.Markdown(text, sizing_mode="stretch_width")


def plot_kwargs(**extra) -> dict:
    return {"responsive": True, "shared_axes": False, **extra}


def is_supported_color_column(series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def classify_color_column(series: pd.Series) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) else "categorical"


def get_first_csv_path(data_dir: Path = DATA_DIR) -> Path | None:
    if not data_dir.exists() or not data_dir.is_dir():
        return None
    files = sorted(p for p in data_dir.glob("*.csv") if p.is_file())
    return files[0] if files else None


def get_initial_filename() -> str:
    path = get_first_csv_path()
    return path.name if path else "No dataset loaded"


class DataState(param.Parameterized):
    data = param.Parameter(default=pd.DataFrame())
    filename = param.String(default=get_initial_filename())
    pending_filename = param.String(default="")
    data_revision = param.Integer(default=0)
    status_message = param.String(default="")
    status_level = param.ObjectSelector(default="info", objects=["info", "success", "warning", "danger"])


state = DataState()


def clear_data_caches() -> None:
    cached_fov_summary.cache_clear()
    cached_filtered_df.cache_clear()
    cached_cell_stats.cache_clear()
    _column_lookup.cache_clear()


def set_status(message: str = "", level: str = "info") -> None:
    state.status_message = message
    state.status_level = level


def set_data(df: pd.DataFrame | None, filename: str) -> None:
    clear_data_caches()
    state.data = df if df is not None else empty_df()
    state.filename = filename
    state.data_revision += 1


def read_csv_from_source(source: bytes | bytearray | str | os.PathLike | io.IOBase | None) -> pd.DataFrame:
    if source is None:
        return empty_df()
    if isinstance(source, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(source)) if source else empty_df()
    if isinstance(source, str):
        if os.path.exists(source):
            return pd.read_csv(source)
        return pd.read_csv(io.StringIO(source)) if source.strip() else empty_df()
    if hasattr(source, "read"):
        return pd.read_csv(source)
    return pd.read_csv(source)


def add_median_rna_spot_to_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_df() if df is None else df
    fov_col, ncount_col = col(df, "fov"), col(df, "ncount_rna")
    if fov_col is None or ncount_col is None:
        return df
    out = df.copy()
    out["median_RNA_SPOT"] = safe_numeric(out[ncount_col]).groupby(out[fov_col]).transform("median")
    return out


def add_qc_flags_to_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_df() if df is None else df
    out = add_median_rna_spot_to_df(df).copy()
    ncount_col = col(out, "ncount_rna")
    out["qc_flagged_cell"] = (safe_numeric(out[ncount_col]) < QC_CELL_THRESHOLD).astype(int) if ncount_col else 0
    out["qc_flagged_fov"] = (
        (safe_numeric(out["median_RNA_SPOT"]) < QC_FOV_THRESHOLD).astype(int)
        if "median_RNA_SPOT" in out.columns
        else 0
    )
    return out


def load_default_data() -> tuple[pd.DataFrame, str]:
    path = get_first_csv_path()
    if path is None:
        set_status(f"No CSV files found in `{DATA_DIR}`. Select a dataset from the sidebar to begin.", "warning")
        return empty_df(), "No dataset loaded"
    try:
        df = add_qc_flags_to_df(read_csv_from_source(path))
        set_status(f"Loaded default dataset: `{path.name}`", "success")
        return df, path.name
    except Exception as exc:
        LOGGER.exception("Failed to load default dataset")
        set_status(f"Failed to load default dataset `{path}`: {exc}", "danger")
        return empty_df(), "No dataset loaded"


def get_source_data() -> pd.DataFrame:
    return state.data if state.data is not None else empty_df()


def get_colorby_options(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return [DEFAULT_SPATIAL_COLOR_BY]
    excluded = {SPATIAL_X_COL, SPATIAL_Y_COL}
    cols = [c for c in df.columns if c not in excluded and is_supported_color_column(df[c])]
    ordered = ([DEFAULT_SPATIAL_COLOR_BY] if DEFAULT_SPATIAL_COLOR_BY in cols else []) + [
        c for c in cols if c != DEFAULT_SPATIAL_COLOR_BY
    ]
    for qc_col in ("qc_flagged_fov", "qc_flagged_cell"):
        if qc_col in df.columns and qc_col not in ordered:
            ordered.append(qc_col)
    return ordered or [DEFAULT_SPATIAL_COLOR_BY]


def majority_code(values) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.nan
    arr = arr[~pd.isna(arr)]
    if arr.size == 0:
        return np.nan
    uniq, counts = np.unique(arr.astype(int), return_counts=True)
    return float(uniq[np.argmax(counts)])


file_dropper = pn.widgets.FileDropper(multiple=False, max_files=1, chunk_size=10_000_000, layout="compact")
load_button = pn.widgets.Button(name="Load selected dataset", button_type="primary", icon="file-import", disabled=True)
fov_select = pn.widgets.Select(name="Field of View (FOV)", value="All", options=["All"])
color_by_select = pn.widgets.Select(
    name="Colour spatial plot by",
    value=DEFAULT_SPATIAL_COLOR_BY,
    options=[DEFAULT_SPATIAL_COLOR_BY],
)
export_html_button = pn.widgets.Button(name="Export HTML Report", button_type="success", icon="download")
export_csv_button = pn.widgets.Button(name="Export Data (CSV)", button_type="success", icon="download")


def get_uploaded_csv() -> tuple[str | None, str | bytes | None]:
    uploads = file_dropper.value or {}
    if len(uploads) != 1:
        return None, None
    filename, content = next(iter(uploads.items()))
    filename = str(filename)
    return (filename, content) if filename.lower().endswith(".csv") else (None, None)


def on_file_selected(event) -> None:
    filename, _ = get_uploaded_csv()
    state.pending_filename = filename or ""
    load_button.disabled = filename is None
    if filename:
        set_status(f"Uploaded `{filename}`. Click **Load selected dataset** to apply it.", "info")
    elif file_dropper.value:
        set_status("Please upload exactly one CSV file.", "warning")
    else:
        set_status("", "info")


def load_selected_dataset(event=None) -> None:
    filename, content = get_uploaded_csv()
    if filename is None or content is None:
        set_status("Please upload exactly one CSV file first.", "warning")
        return
    try:
        set_data(add_qc_flags_to_df(read_csv_from_source(content)), filename)
        set_status(f"Loaded `{filename}` successfully.", "success")
    except Exception as exc:
        LOGGER.exception("Error loading uploaded CSV")
        set_status(f"Error loading `{filename}`: {exc}", "danger")


file_dropper.param.watch(on_file_selected, "value")
load_button.on_click(load_selected_dataset)


def compute_fov_summary(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_df()

    fov_col = col(df, "fov")
    cell_id_col = col(df, "cell_id")
    ncount_col = col(df, "ncount_rna")
    nfeature_col = col(df, "nfeature_rna")
    neg_col = col(df, "ncount_negprobes")
    area_col = col(df, "area")

    if not all([fov_col, cell_id_col, ncount_col, nfeature_col]):
        return empty_df()

    cols = [fov_col, cell_id_col, ncount_col, nfeature_col]
    rename = {fov_col: "fov", cell_id_col: "cell_id", ncount_col: "nCount_RNA", nfeature_col: "nFeature_RNA"}
    if area_col:
        cols.append(area_col)
        rename[area_col] = "Area.um2"
    if neg_col:
        cols.append(neg_col)
        rename[neg_col] = "nCount_negprobes"
    for qc in ("qc_flagged_fov", "qc_flagged_cell"):
        if qc in df.columns:
            cols.append(qc)

    work = df.loc[:, list(dict.fromkeys(cols))].rename(columns=rename).copy()
    work["nCount_RNA"] = safe_numeric(work["nCount_RNA"])
    work["nFeature_RNA"] = safe_numeric(work["nFeature_RNA"])

    agg = {
        "nCell": ("cell_id", "count"),
        "Median nCount_RNA": ("nCount_RNA", "median"),
        "Median nFeature_RNA": ("nFeature_RNA", "median"),
    }

    if "Area.um2" in work:
        work["Area.um2"] = safe_numeric(work["Area.um2"])
        agg["Median Area.um2"] = ("Area.um2", "median")
    if "nCount_negprobes" in work:
        work["nCount_negprobes"] = safe_numeric(work["nCount_negprobes"])
        work["negprobes_ratio"] = (work["nCount_negprobes"] / work["nCount_RNA"]).replace([np.inf, -np.inf], np.nan)
        agg["Mean NegProbes/RNA"] = ("negprobes_ratio", "mean")
    if "qc_flagged_fov" in work:
        agg["qc_flagged_fov"] = ("qc_flagged_fov", "max")
    if "qc_flagged_cell" in work:
        agg["Total qc_flagged_cell"] = ("qc_flagged_cell", "sum")

    return work.groupby("fov", dropna=False).agg(**agg).reset_index().sort_values("fov")


@lru_cache(maxsize=64)
def cached_fov_summary(data_revision: int) -> pd.DataFrame:
    return compute_fov_summary(state.data)


def get_fov_options(summary: pd.DataFrame | None) -> list[str]:
    if summary is None or summary.empty or "fov" not in summary.columns:
        return ["All"]
    vals = summary["fov"].dropna().unique()
    nums = pd.to_numeric(pd.Series(vals), errors="coerce")
    if nums.notna().all():
        ordered = sorted(nums.astype(int).astype(str).tolist(), key=lambda x: int(x))
    else:
        ordered = sorted(map(str, vals))
    return ["All", *ordered]


def update_fov_options(*_) -> None:
    options = get_fov_options(cached_fov_summary(state.data_revision))
    fov_select.options = options
    if fov_select.value not in options:
        fov_select.value = "All"


def update_colorby_options(*_) -> None:
    options = get_colorby_options(state.data)
    color_by_select.options = options
    if color_by_select.value not in options:
        color_by_select.value = options[0]


def filter_data_by_fov(df: pd.DataFrame | None, fov: str) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_df()
    fov_col = col(df, "fov")
    if fov == "All" or fov_col is None:
        return df
    numeric = pd.to_numeric(df[fov_col], errors="coerce")
    try:
        matched = df[numeric == int(fov)]
        if not matched.empty:
            return matched
    except Exception:
        pass
    return df[df[fov_col].astype(str) == str(fov)]


@lru_cache(maxsize=128)
def cached_filtered_df(data_revision: int, fov: str) -> pd.DataFrame:
    return filter_data_by_fov(state.data, fov)


@lru_cache(maxsize=128)
def cached_cell_stats(data_revision: int, fov: str) -> dict[str, float | int | None]:
    df = cached_filtered_df(data_revision, fov)
    if df is None or df.empty:
        return {}

    def stat(column: str, fn: Callable[[pd.Series], float], default=None):
        if column not in df.columns:
            return default
        s = safe_numeric(df[column]).dropna()
        return fn(s) if not s.empty else default

    return {
        "cell_count": len(df),
        "avg_rna": stat("nCount_RNA", pd.Series.mean),
        "median_rna": stat("nCount_RNA", pd.Series.median),
        "min_rna": stat("nCount_RNA", pd.Series.min),
        "max_rna": stat("nCount_RNA", pd.Series.max),
        "med_features": stat("nFeature_RNA", pd.Series.median),
        "avg_area": stat("Area.um2", pd.Series.mean),
    }


state.param.watch(update_fov_options, "data_revision")
state.param.watch(update_colorby_options, "data_revision")

_default_df, _default_name = load_default_data()
set_data(_default_df, _default_name)
update_fov_options()
update_colorby_options()

filtered_df = pn.bind(cached_filtered_df, state.param.data_revision, fov_select)
cell_stats = pn.bind(cached_cell_stats, state.param.data_revision, fov_select)


def responsive_flexbox(*children, gap: str = FLEX_GAP_DEFAULT, justify: str = "flex-start") -> pn.FlexBox:
    return pn.FlexBox(
        *children,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": gap, "align-items": "stretch", "justify-content": justify, "width": "100%"},
    )


def flex_item(*objects, min_width: int, grow: int = 1, height: int | None = None, allow_shrink_below_min: bool = False) -> pn.Column:
    kwargs = {
        "sizing_mode": "stretch_width",
        "min_width": min_width,
        "styles": {"flex": f"{grow} 1 {min_width}px", "min-width": "0" if allow_shrink_below_min else f"{min_width}px"},
    }
    if height is not None:
        kwargs["height"] = height
    return pn.Column(*objects, **kwargs)


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
    header = pn.pane.Markdown(f"#### {title}", margin=(0, 0, 6, 0)) if title else pn.Spacer(height=0)
    styles = {"overflow": "visible", "box-sizing": "border-box", "width": "100%"}
    if max_width is not None:
        styles["max-width"] = f"{max_width}px"
    if square or aspect_ratio is not None:
        pane = pn.pane.HoloViews(plot, sizing_mode="scale_width", min_width=min_width, margin=0, linked_axes=linked_axes)
        return pn.Column(header, pane, min_width=min_width, sizing_mode="stretch_width", styles=styles)
    pane = pn.pane.HoloViews(plot, sizing_mode="stretch_width", min_height=min_height, linked_axes=linked_axes)
    return pn.Column(header, pane, min_height=min_height + (32 if title else 0), min_width=min_width, sizing_mode="stretch_width", styles=styles)


def apply_square_aspect_hook(plot, element):
    fig = plot.state
    fig.sizing_mode = "scale_width"
    fig.aspect_ratio = 1
    fig.match_aspect = True


def apply_4_3_aspect_hook(plot, element):
    fig = plot.state
    fig.sizing_mode = "scale_width"
    fig.aspect_ratio = 4 / 3
    fig.match_aspect = True


def create_shared_range_hook(x_range, y_range):
    def _hook(plot, element):
        fig = plot.state
        fig.x_range = x_range
        fig.y_range = y_range
    return _hook


def compute_shared_spatial_ranges(df: pd.DataFrame | None, pad_frac: float = 0.02):
    if df is None or df.empty or not has_cols(df, SPATIAL_X_COL, SPATIAL_Y_COL):
        return None, None
    x = safe_numeric(df[SPATIAL_X_COL]).dropna()
    y = safe_numeric(df[SPATIAL_Y_COL]).dropna()
    if x.empty or y.empty:
        return None, None
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xpad = max((xmax - xmin) * pad_frac, 1.0)
    ypad = max((ymax - ymin) * pad_frac, 1.0)
    return Range1d(xmin - xpad, xmax + xpad), Range1d(ymin - ypad, ymax + ypad)


def maybe_sample_spatial(df: pd.DataFrame | None, enabled: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_df()
    if not enabled or len(df) <= MAX_LIVE_SPATIAL_POINTS:
        return df
    LOGGER.info("Downsampling spatial data from %s to %s rows.", len(df), MAX_LIVE_SPATIAL_POINTS)
    return df.sample(n=MAX_LIVE_SPATIAL_POINTS, random_state=42)


def indicator_card(value, label: str, fmt: str = "{:,.0f}", min_width: int = 220) -> pn.Column:
    display = "—" if value is None else fmt.format(value) if isinstance(value, (int, float, np.number)) else str(value)
    html_body = f"""
    <div style="text-align:center;height:100%;display:flex;flex-direction:column;justify-content:center;">
      <div style="font-size:clamp(11px,1.5vw,14px);color:#999;margin-bottom:8px;font-weight:500;">{html.escape(label)}</div>
      <div style="font-size:clamp(18px,4vw,32px);font-weight:bold;color:#333;">{html.escape(display)}</div>
    </div>
    """
    return flex_item(
        pn.pane.HTML(html_body, sizing_mode="stretch_both", styles={**CARD_STYLES, "height": "100%", "min-height": "100px", "box-sizing": "border-box"}),
        min_width=min_width,
        height=110,
    )


def qc_flag_status_card(value: int, label: str, percentage: float, color: str, min_width: int = 220) -> pn.Column:
    html_body = f"""
    <div style="background-color:{color};border-radius:8px;padding:20px;text-align:center;color:white;min-height:100px;height:100%;
                display:flex;flex-direction:column;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,.15);box-sizing:border-box;">
      <div style="font-size:13px;margin-bottom:8px;opacity:.95;font-weight:500;">{html.escape(label)}</div>
      <div style="font-size:32px;font-weight:bold;">{value:,}</div>
      <div style="font-size:13px;margin-top:8px;opacity:.90;">({percentage:.1f}%)</div>
    </div>
    """
    return flex_item(pn.pane.HTML(html_body, sizing_mode="stretch_both"), min_width=min_width, height=130)


def hist_plot_raw(df: pd.DataFrame | None, column: str, title: str, bins: int = 50, xlim: tuple[float, float] | None = None):
    if not has_cols(df, column):
        return None
    data = safe_numeric(df[column]).dropna()
    if xlim is not None:
        data = data[(data >= xlim[0]) & (data <= xlim[1])]
    if data.empty:
        return None
    return data.hvplot.hist(bins=bins, xlabel=column, ylabel="Number of Cells", color=ACCENT_COLOR, **plot_kwargs())


def scatter_plot_raw(df: pd.DataFrame | None, x: str, y: str):
    if not has_cols(df, x, y):
        return None
    plot_df = df.loc[:, [x, y]].apply(safe_numeric).dropna()
    if plot_df.empty:
        return None
    return plot_df.hvplot.scatter(x=x, y=y, xlabel=x, ylabel=y, color=ACCENT_COLOR, tools=PLOT_TOOLS, **plot_kwargs())


def metrics_jointplot_raw(df: pd.DataFrame | None, x: str, y: str, hue: str):
    if not has_cols(df, x, y, hue):
        return None
    plot_df = df.loc[:, [x, y, hue]].copy()
    for c in (x, y, hue):
        plot_df[c] = safe_numeric(plot_df[c])
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None
    return plot_df.hvplot.hexbin(
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
    ).opts(toolbar="above", active_tools=["wheel_zoom"], responsive=True, data_aspect=4 / 3, hooks=[apply_4_3_aspect_hook]).clone()


@lru_cache(maxsize=16)
def sample_matplotlib_palette(name: str = "RdYlGn", n: int = 256) -> tuple[str, ...]:
    cmap = cm.get_cmap(name, n)
    return tuple(mcolors.to_hex(cmap(i)) for i in range(cmap.N))


def normalize_values_with_center(values: pd.Series, *, vmin: float, vcenter: float, vmax: float) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if not all(np.isfinite(v) for v in (vmin, vcenter, vmax)):
        return out
    if vmax <= vmin:
        out.loc[s.notna()] = 0.5
        return out
    lower, upper = s <= vcenter, s > vcenter
    out.loc[lower] = 0.5 * (s.loc[lower] - vmin) / (vcenter - vmin) if vcenter > vmin else 0.5
    out.loc[upper] = 0.5 + 0.5 * (s.loc[upper] - vcenter) / (vmax - vcenter) if vmax > vcenter else 0.5
    return out.clip(0, 1)


def update_fov_scatter_colorbar_labels(plot, element):
    fig, data = plot.state, element.data
    if data is None or len(data) == 0 or not {"color_vmin", "color_vmax", "color_vcenter"}.issubset(data):
        return
    try:
        vmin = float(pd.to_numeric(pd.Series(data["color_vmin"]), errors="coerce").dropna().iloc[0])
        vmax = float(pd.to_numeric(pd.Series(data["color_vmax"]), errors="coerce").dropna().iloc[0])
        vcenter = float(pd.to_numeric(pd.Series(data["color_vcenter"]), errors="coerce").dropna().iloc[0])
    except Exception:
        return
    colorbars = fig.select(dict(type=ColorBar))
    if vmax <= vmin:
        ticks, code = [0.5], f'return "{vmin:.1f}";'
    else:
        lower_mid, upper_mid = vmin + 0.5 * (vcenter - vmin), vcenter + 0.5 * (vmax - vcenter)
        ticks = [0, 0.25, 0.5, 0.75, 1]
        code = f"""
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
        cb.formatter = CustomJSTickFormatter(code=code)
        cb.title = "Median RNA/FOV"


def build_fov_scatter_plot(df: pd.DataFrame | None, extra_hooks: list | None = None):
    if df is None or df.empty:
        return None
    fov_col, ncount_col, neg_col = col(df, "fov"), col(df, "ncount_rna"), col(df, "ncount_negprobes")
    if not (fov_col and ncount_col and has_cols(df, SPATIAL_X_COL, SPATIAL_Y_COL)):
        return None

    cols = [fov_col, SPATIAL_X_COL, SPATIAL_Y_COL, ncount_col] + ([neg_col] if neg_col else [])
    work = df.loc[:, list(dict.fromkeys(cols))].copy()
    for c in (SPATIAL_X_COL, SPATIAL_Y_COL, ncount_col):
        work[c] = safe_numeric(work[c])
    if neg_col:
        work[neg_col] = safe_numeric(work[neg_col])
    work = work.dropna(subset=[fov_col, SPATIAL_X_COL, SPATIAL_Y_COL, ncount_col])
    if work.empty:
        return None

    med = work.groupby(fov_col)[ncount_col].median()
    fov_data = work.groupby(fov_col).agg({SPATIAL_X_COL: "mean", SPATIAL_Y_COL: "mean"}).reset_index()
    fov_data["median_RNA_SPOT"] = fov_data[fov_col].map(med)
    fov_data = fov_data.rename(columns={fov_col: "fov", SPATIAL_X_COL: "spatial_fov_x", SPATIAL_Y_COL: "spatial_fov_y"})

    if neg_col:
        ratio = (work.groupby(fov_col)[neg_col].mean() / work.groupby(fov_col)[ncount_col].mean()).replace([np.inf, -np.inf], np.nan)
        fov_data["ratio_negprobes"] = fov_data["fov"].map(ratio).fillna(0.05)
    else:
        fov_data["ratio_negprobes"] = 0.05

    fov_data = fov_data.dropna(subset=["spatial_fov_x", "spatial_fov_y", "median_RNA_SPOT"])
    if fov_data.empty:
        return None

    rmin, rmax = fov_data["ratio_negprobes"].min(), fov_data["ratio_negprobes"].max()
    rrng = rmax - rmin if rmax > rmin else 1.0
    fov_data["bubble_size"] = 10 + 10 * (fov_data["ratio_negprobes"] - rmin) / rrng
    fov_data["color_vmin"] = FOV_MEDIAN_RNA_DISPLAY_MIN
    fov_data["color_vmax"] = FOV_MEDIAN_RNA_DISPLAY_MAX
    fov_data["color_vcenter"] = FOV_MEDIAN_RNA_CENTER
    fov_data["median_rna_display"] = fov_data["median_RNA_SPOT"].clip(FOV_MEDIAN_RNA_DISPLAY_MIN, FOV_MEDIAN_RNA_DISPLAY_MAX)
    fov_data["median_rna_twoslope"] = normalize_values_with_center(
        fov_data["median_rna_display"],
        vmin=FOV_MEDIAN_RNA_DISPLAY_MIN,
        vcenter=FOV_MEDIAN_RNA_CENTER,
        vmax=FOV_MEDIAN_RNA_DISPLAY_MAX,
    )

    points = hv.Points(
        fov_data,
        kdims=["spatial_fov_x", "spatial_fov_y"],
        vdims=["fov", "median_RNA_SPOT", "median_rna_display", "median_rna_twoslope", "ratio_negprobes", "bubble_size", "color_vmin", "color_vmax", "color_vcenter"],
    )
    hooks = [apply_square_aspect_hook, update_fov_scatter_colorbar_labels, *(extra_hooks or [])]
    return points.opts(
        color="median_rna_twoslope",
        marker="square",
        size=dim("bubble_size"),
        cmap=list(sample_matplotlib_palette("RdYlGn", 256)),
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
        title="FOV Spatial Plot",
        hooks=hooks,
    )


def build_spatial_plot(df: pd.DataFrame | None, color_by: str = DEFAULT_SPATIAL_COLOR_BY, sample: bool = False, extra_hooks: list | None = None):
    if not has_cols(df, SPATIAL_X_COL, SPATIAL_Y_COL):
        return panel_message("No spatial data available.")
    if color_by not in df.columns:
        return panel_message(f"Column `{color_by}` is not available.")

    source_len = len(df)
    plot_df = maybe_sample_spatial(df, enabled=sample).loc[:, [SPATIAL_X_COL, SPATIAL_Y_COL, color_by]].copy()
    plot_df[SPATIAL_X_COL] = safe_numeric(plot_df[SPATIAL_X_COL])
    plot_df[SPATIAL_Y_COL] = safe_numeric(plot_df[SPATIAL_Y_COL])
    plot_df = plot_df.dropna(subset=[SPATIAL_X_COL, SPATIAL_Y_COL])
    if plot_df.empty:
        return panel_message("No usable spatial coordinates available.")

    sample_note = f" (sampled {len(plot_df):,} / {source_len:,} cells)" if sample and len(plot_df) < source_len else ""
    mode = classify_color_column(df[color_by])

    if mode == "numeric":
        plot_df[color_by] = safe_numeric(plot_df[color_by])
        plot_df = plot_df.dropna(subset=[color_by])
        if plot_df.empty:
            return panel_message(f"No usable numeric values in `{color_by}`.")
        main = plot_df.hvplot.hexbin(
            x=SPATIAL_X_COL,
            y=SPATIAL_Y_COL,
            C=color_by,
            reduce_function=np.mean,
            clabel=f"Mean {color_by}",
            gridsize=SPATIAL_PLOT_GRIDSIZE,
            min_count=1,
            cmap="viridis",
            colorbar=True,
            xlabel="X Position (pixels)",
            ylabel="Y Position (pixels)",
            title=f"Cell Spatial Plot — mean {color_by}{sample_note}",
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
        plot_df["_color_code"] = cat.map({name: i for i, name in enumerate(categories)}).astype(float)
        main = plot_df.hvplot.hexbin(
            x=SPATIAL_X_COL,
            y=SPATIAL_Y_COL,
            C="_color_code",
            reduce_function=majority_code,
            clabel=f"Majority {color_by}",
            gridsize=SPATIAL_PLOT_GRIDSIZE,
            min_count=1,
            cmap=SPATIAL_CATEGORY_COLORS[: len(categories)],
            colorbar=False,
            xlabel="X Position (pixels)",
            ylabel="Y Position (pixels)",
            title=f"Cell Spatial Plot — majority {color_by}{sample_note}",
            tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset"],
            **plot_kwargs(),
        )

    return main.opts(
        toolbar="above",
        active_tools=["wheel_zoom"],
        shared_axes=True,
        responsive=True,
        data_aspect=1,
        hooks=[apply_square_aspect_hook, *(extra_hooks or [])],
    ).clone()


def compute_fov_cell_qc_flags(df: pd.DataFrame | None) -> dict[str, int]:
    result = {"flagged_fovs": 0, "total_fovs": 0, "flagged_cells": 0, "total_cells": 0}
    if df is None or df.empty:
        return result
    fov_col = col(df, "fov")
    if fov_col and "qc_flagged_fov" in df.columns:
        flags = safe_numeric(df.groupby(fov_col)["qc_flagged_fov"].max()).fillna(0)
        result["total_fovs"] = len(flags)
        result["flagged_fovs"] = int(flags.sum())
    if "qc_flagged_cell" in df.columns:
        result["total_cells"] = len(df)
        result["flagged_cells"] = int(safe_numeric(df["qc_flagged_cell"]).fillna(0).sum())
    return result


def get_qc_flag_color(ratio: float, flag_type: str) -> str:
    limits = (0.2, 0.5) if flag_type == "fov" else (0.1, 0.3)
    return "#22c55e" if ratio < limits[0] else "#f97316" if ratio < limits[1] else "#ef4444"


def create_qc_flag_status_display(df: pd.DataFrame | None):
    flags = compute_fov_cell_qc_flags(df)
    fov_ratio = flags["flagged_fovs"] / flags["total_fovs"] if flags["total_fovs"] else 0
    cell_ratio = flags["flagged_cells"] / flags["total_cells"] if flags["total_cells"] else 0
    return responsive_flexbox(
        qc_flag_status_card(flags["flagged_fovs"], "Flagged FOVs", fov_ratio * 100, get_qc_flag_color(fov_ratio, "fov")),
        qc_flag_status_card(flags["flagged_cells"], "Flagged Cells", cell_ratio * 100, get_qc_flag_color(cell_ratio, "cell")),
    )


def create_qc_flag_cards(df: pd.DataFrame | None):
    return create_qc_flag_status_display(df)


def create_indicators(stats: dict[str, float | int | None]):
    if not stats:
        return panel_message("No data available.")
    return responsive_flexbox(
        indicator_card(stats.get("cell_count"), "Total Cells"),
        indicator_card(stats.get("median_rna"), "Med. Transcripts/Cell"),
        indicator_card(stats.get("med_features"), "Med. Genes/Cell"),
        indicator_card(stats.get("avg_area"), "Avg. Cell Area (µm²)", "{:,.1f}"),
    )


def build_empty_tabulator() -> pn.widgets.Tabulator:
    return pn.widgets.Tabulator(empty_df(), sizing_mode="stretch_width", pagination="local", page_size=TABLE_PAGE_SIZE)


def build_qc_metrics_table(df: pd.DataFrame | None) -> pn.widgets.Tabulator:
    summary = compute_fov_summary(df)
    if summary.empty or "fov" not in summary.columns:
        return build_empty_tabulator()
    return pn.widgets.Tabulator(
        summary.drop_duplicates().set_index("fov").sort_index(),
        sizing_mode="stretch_width",
        pagination="local",
        page_size=TABLE_PAGE_SIZE,
    )


def create_status_pane(message: str, level: str):
    return pn.Spacer(height=0) if not message else pn.pane.Alert(message, alert_type=level, sizing_mode="stretch_width")


def build_boxed_histogram(df, column, title, *, bins=50, xlim=None, min_height=PLOT_MIN_HEIGHT):
    plot = hist_plot_raw(df, column, title, bins=bins, xlim=xlim)
    return panel_message(f"No {title.lower() or column} data available.") if plot is None else plot_box(plot, min_height=min_height, title=title)


def build_boxed_scatter(df, x, y, title, *, min_height=PLOT_MIN_HEIGHT):
    plot = scatter_plot_raw(df, x, y)
    return panel_message("No comparison data available.") if plot is None else plot_box(plot, min_height=min_height, title=title)


def build_boxed_metrics_plot(df, x, y, hue, title, *, min_height=PLOT_TALL_MIN_HEIGHT):
    plot = metrics_jointplot_raw(df, x, y, hue)
    return panel_message("No metrics data available.") if plot is None else plot_box(plot, min_height=min_height, max_width=800, title=title)


def build_linked_spatial_views(df: pd.DataFrame | None, color_by: str, *, report_mode: bool):
    if df is None or df.empty:
        return panel_message("No spatial data available.")
    x_range, y_range = compute_shared_spatial_ranges(df)
    hooks = [create_shared_range_hook(x_range, y_range)] if x_range is not None and y_range is not None else []
    fov_plot = build_fov_scatter_plot(df, extra_hooks=hooks)
    spatial = build_spatial_plot(df, color_by=color_by, sample=not report_mode, extra_hooks=hooks)
    if fov_plot is None or spatial is None:
        return panel_message("No spatial data available.")
    return responsive_flexbox(
        flex_item(plot_box(fov_plot, aspect_ratio=4 / 3, max_width=600, min_width=PLOT_PANEL_MIN_WIDTH), min_width=PLOT_PANEL_MIN_WIDTH, grow=1, allow_shrink_below_min=True),
        flex_item(plot_box(spatial, aspect_ratio=4 / 3, max_width=600, min_width=PLOT_PANEL_MIN_WIDTH), min_width=PLOT_PANEL_MIN_WIDTH, grow=1, allow_shrink_below_min=True),
        gap=FLEX_GAP_WIDE,
    )


def make_component_bindings(report_mode: bool = False) -> dict[str, object]:
    return {
        "status_pane": pn.bind(create_status_pane, state.param.status_message, state.param.status_level),
        "indicators": pn.bind(create_indicators, cell_stats),
        "qc_flag_cards": pn.bind(create_qc_flag_cards, filtered_df),
        "linked_spatial_views": pn.bind(build_linked_spatial_views, filtered_df, color_by_select, report_mode=report_mode),
        "qc_metrics_tbl": pn.bind(build_qc_metrics_table, filtered_df),
        "negprobes_hist": pn.bind(build_boxed_histogram, filtered_df, "nCount_negprobes", ""),
        "rna_vs_negprobes": pn.bind(build_boxed_scatter, filtered_df, "nCount_RNA", "nCount_negprobes", ""),
        "metrics_jointplot": pn.bind(build_boxed_metrics_plot, filtered_df, "nCount_RNA", "nFeature_RNA", "Area.um2", ""),
        "rna_hist": pn.bind(build_boxed_histogram, filtered_df, "nCount_RNA", ""),
        "feature_hist": pn.bind(build_boxed_histogram, filtered_df, "nFeature_RNA", ""),
        "area_hist": pn.bind(build_boxed_histogram, filtered_df, "Area.um2", ""),
    }


def build_summary_tab(views: Mapping[str, object]) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("### Key QC Metrics"),
        views["indicators"],
        views["qc_flag_cards"],
        pn.pane.Markdown("### Spatial Sample Plots"),
        views["linked_spatial_views"],
        sizing_mode="stretch_both",
        styles={"overflow-y": "auto"},
    )


def build_details_tab(views: Mapping[str, object]) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("### Run Details"),
        pn.pane.Markdown("### Cell Metrics"),
        responsive_flexbox(
            flex_item(views["metrics_jointplot"], min_width=600),
            flex_item(views["area_hist"], min_width=200),
            flex_item(views["rna_hist"], min_width=200),
            flex_item(views["feature_hist"], min_width=200),
            gap="16px",
        ),
        pn.pane.Markdown("### Negative Probes"),
        responsive_flexbox(
            flex_item(views["negprobes_hist"], min_width=300),
            flex_item(views["rna_vs_negprobes"], min_width=300),
            gap="16px",
        ),
        pn.pane.Markdown("### FOV-level Metrics Table"),
        responsive_flexbox(flex_item(views["qc_metrics_tbl"], min_width=800), gap="16px"),
        sizing_mode="stretch_both",
        styles={"overflow-y": "auto"},
    )


def build_image_qc_tab() -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("### Fluorescent Markers"),
        pn.pane.Markdown("Work in progress"),
        sizing_mode="stretch_both",
        styles={"overflow-y": "auto"},
    )


def build_analysis_tab() -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("### DR and clustering based on metadata information"),
        sizing_mode="stretch_both",
        styles={"overflow-y": "auto"},
    )


def create_tabs(views: Mapping[str, object], report_mode: bool = False) -> pn.Tabs:
    return pn.Tabs(
        ("Summary", build_summary_tab(views)),
        ("Run Details", build_details_tab(views)),
        ("Image QC", build_image_qc_tab()),
        ("Analysis", build_analysis_tab()),
        dynamic=not report_mode,
        styles=CARD_STYLES,
        sizing_mode="stretch_both",
        margin=10,
    )


def loaded_filename_pane():
    return pn.bind(lambda filename: pn.pane.Markdown(f"**Loaded:** {filename}", styles={"color": "gray", "font-size": "12px"}), state.param.filename)


def pending_filename_pane():
    return pn.bind(
        lambda filename: pn.pane.Markdown(f"**Selected:** {filename}", styles={"color": "#555", "font-size": "12px"}) if filename else pn.Spacer(height=0),
        state.param.pending_filename,
    )


def title_pane():
    return pn.bind(lambda filename: pn.pane.Markdown(f"## Loaded Object: {filename}"), state.param.filename)


def create_template(report_mode: bool = False) -> pn.template.BootstrapTemplate:
    views = make_component_bindings(report_mode=report_mode)
    tabs = create_tabs(views, report_mode=report_mode)

    main = [pn.Column(title_pane(), tabs, sizing_mode="stretch_both")]
    if report_mode:
        return pn.template.BootstrapTemplate(
            title="POPIDD-SPOT: Spatial Profiling Overview Tool [STATIC]",
            header_background="#d4a300",
            main=main,
        )

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
        main=main,
    )


def save_as_html(filename: str | os.PathLike = DEFAULT_REPORT_PATH) -> None:
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    create_template(report_mode=True).save(str(output), resources="cdn")
    LOGGER.info("Dashboard saved to %s", output)


def _safe_stem(filename: str) -> str:
    stem = Path(str(filename)).stem or "dataset"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "dataset"


def save_data_as_csv(filename: str | os.PathLike | None = None) -> Path:
    if filename is None:
        filename = Path("report") / f"SPOT-qcData__{_safe_stem(state.filename)}__{datetime.now():%Y%m%d}.csv"
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


def on_export_html_clicked(event) -> None:
    try:
        save_as_html()
        set_status("HTML report exported successfully.", "success")
    except Exception as exc:
        LOGGER.exception("Error exporting HTML")
        set_status(f"Error exporting HTML: {exc}", "danger")


def on_export_csv_clicked(event) -> None:
    try:
        save_data_as_csv()
    except Exception as exc:
        LOGGER.exception("Error exporting CSV")
        set_status(f"Error exporting CSV: {exc}", "danger")


export_html_button.on_click(on_export_html_clicked)
export_csv_button.on_click(on_export_csv_clicked)

app = create_template(report_mode=False).servable()


def main() -> None:
    save_as_html()


if __name__ == "__main__":
    main()