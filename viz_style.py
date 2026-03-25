"""
Publication-quality visualization style for SODA primer figures.

Matches the style of Renz et al. (2026) LCI preprint:
- Arial (sans-serif) fonts
- Bold uppercase panel labels (A, B, C...)
- Blue/orange primary palette
- Clean spines (left + bottom only)
- Minimal gridlines
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parent.parent / "Figures"

# ---------------------------------------------------------------------------
# Core palette (matching Renz et al. 2026 / LCI preprint)
# ---------------------------------------------------------------------------
BLUE = "#2171B5"
ORANGE = "#D94801"
GREY = "#808080"
LIGHT_GREY = "#D9D9D9"
BLACK = "#1A1A1A"

# Forward / backward shading
FWD_COLOR = BLUE
BWD_COLOR = ORANGE
FWD_ALPHA = 0.12
BWD_ALPHA = 0.12

# ---------------------------------------------------------------------------
# 5-item (and up to 8-item) sequence palette
# Warm-to-cool gradient, colorblind-safe, matching Wittkuhn Fig 3a feel
# ---------------------------------------------------------------------------
_ITEM_PALETTE_8 = [
    "#D62728",  # 1 - red
    "#E6750A",  # 2 - orange
    "#DBAC00",  # 3 - gold/yellow
    "#2CA02C",  # 4 - green
    "#1F77B4",  # 5 - blue
    "#9467BD",  # 6 - purple
    "#8C564B",  # 7 - brown
    "#17BECF",  # 8 - cyan
]


def get_item_colors(n=5):
    """Return a list of n colors for sequence items 1..n."""
    return _ITEM_PALETTE_8[:n]


# ---------------------------------------------------------------------------
# Speed palette (ISI-keyed, matching Wittkuhn Fig 2e style)
# ---------------------------------------------------------------------------
_SPEED_COLORS = {
    0.032: "#1A1A1A",   # 32 ms  - near-black
    0.064: "#2171B5",   # 64 ms  - blue
    0.128: "#238B45",   # 128 ms - green
    0.512: "#D94801",   # 512 ms - orange
    2.048: "#CB181D",   # 2048 ms - red
}


def get_speed_colors():
    """Return dict mapping ISI (seconds) → color."""
    return dict(_SPEED_COLORS)


def speed_color(isi_seconds):
    """Get color for a specific ISI. Falls back to grey."""
    return _SPEED_COLORS.get(isi_seconds, GREY)


# ---------------------------------------------------------------------------
# Metric palette (7 aggregation metrics)
# ---------------------------------------------------------------------------
_METRIC_COLORS = {
    "mean_slope":          "#BDBDBD",  # light grey (worst metric)
    "abs_mean":            "#D94801",  # orange
    "slope_variance":      "#6A3D9A",  # purple
    "peak_to_trough":      "#238B45",  # green
    "spectral_power":      "#1F78B4",  # blue
    "cont_sin_amplitude":  "#E31A1C",  # red
    "win_sin_amplitude":   "#1A1A1A",  # black (best metric)
}

_METRIC_LABELS = {
    "mean_slope":          "Mean slope",
    "abs_mean":            "|Slope| mean",
    "slope_variance":      "Slope variance",
    "peak_to_trough":      "Peak-to-trough",
    "spectral_power":      "Spectral power",
    "cont_sin_amplitude":  "Cont. sin amp.",
    "win_sin_amplitude":   "Wind. sin amp.",
}


def get_metric_colors():
    """Return dict mapping metric name → color."""
    return dict(_METRIC_COLORS)


def metric_color(name):
    """Get color for a specific metric."""
    return _METRIC_COLORS.get(name, GREY)


def get_metric_labels():
    """Return dict mapping metric name → display label."""
    return dict(_METRIC_LABELS)


def metric_label(name):
    """Get display label for a metric."""
    return _METRIC_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# Figure size presets (in inches, converted from mm)
# ---------------------------------------------------------------------------
MM_TO_INCH = 1.0 / 25.4
SINGLE_COL = 85 * MM_TO_INCH    # ~3.35 in
ONE_HALF_COL = 114 * MM_TO_INCH  # ~4.49 in
FULL_WIDTH = 174 * MM_TO_INCH    # ~6.85 in


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
def setup_style():
    """Apply publication-quality rcParams globally.

    Call this once at the top of each script before creating figures.
    """
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,

        # Axes
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": BLACK,
        "axes.edgecolor": BLACK,

        # Ticks
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": BLACK,
        "ytick.color": BLACK,

        # Lines — thin default; use lw=1.8+ explicitly for emphasis
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,

        # Legend
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.5,

        # Grid
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.color": LIGHT_GREY,

        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "savefig.transparent": False,

        # Patch (bars, etc.)
        "patch.linewidth": 0.8,
    })


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def add_panel_label(ax, label, x=-0.15, y=1.05, fontsize=11):
    """Add a bold uppercase panel label (A, B, C...) to an axes.

    Positioned outside the plot area so it does not overlap with
    titles or data.

    Args:
        ax: Matplotlib axes.
        label: The label string (e.g., 'A', 'B').
        x, y: Position in axes fraction coordinates.
        fontsize: Font size for the label.
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
    )


def shade_periods(ax, forward_period, backward_period):
    """Add forward/backward period shading to an axes.

    Args:
        ax: Matplotlib axes.
        forward_period: (start, end) in TRs.
        backward_period: (start, end) in TRs.
    """
    ax.axvspan(forward_period[0], forward_period[1],
               color=FWD_COLOR, alpha=FWD_ALPHA, zorder=0, label="_nolegend_")
    ax.axvspan(backward_period[0], backward_period[1],
               color=BWD_COLOR, alpha=BWD_ALPHA, zorder=0, label="_nolegend_")


def add_zero_line(ax, **kwargs):
    """Add a horizontal dashed reference line at y=0."""
    defaults = dict(color=GREY, linewidth=0.75, linestyle="--", zorder=1)
    defaults.update(kwargs)
    ax.axhline(0, **defaults)


def add_chance_line(ax, chance=20.0, **kwargs):
    """Add a horizontal dashed reference line at chance level."""
    defaults = dict(color=GREY, linewidth=0.75, linestyle="--", zorder=1)
    defaults.update(kwargs)
    ax.axhline(chance, **defaults)


def despine(ax, keep_left=True, keep_bottom=True):
    """Remove spines from axes (beyond what rcParams does)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)


def save_figure(fig, name, formats=("png",), figures_dir=None):
    """Save figure to the Figures directory.

    Args:
        fig: Matplotlib figure.
        name: Filename without extension (e.g., 'sim1_ideal_case').
        formats: Tuple of format strings ('png', 'pdf', etc.).
        figures_dir: Override output directory.
    """
    out_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"  Saved: {path}")


def make_diverging_cmap(vmin=-1, vmax=1):
    """Create a diverging blue-white-red colormap centered at 0.

    Returns:
        (cmap, norm) tuple for use with imshow/pcolormesh.
    """
    from matplotlib.colors import TwoSlopeNorm
    cmap = mpl.colormaps.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    return cmap, norm


def annotated_heatmap(ax, data, row_labels, col_labels,
                      vmin=None, vmax=None, fmt=".2f",
                      cmap=None, norm=None,
                      text_color_threshold=None):
    """Draw an annotated heatmap with diverging colormap.

    Args:
        ax: Matplotlib axes.
        data: 2D numpy array.
        row_labels: List of row labels.
        col_labels: List of column labels.
        vmin, vmax: Colormap range. If None, auto from data.
        fmt: Format string for cell annotations.
        cmap, norm: Colormap and normalizer. If None, uses diverging.

    Returns:
        The AxesImage returned by imshow.
    """
    if vmin is None:
        vmin = -max(abs(data.min()), abs(data.max()))
    if vmax is None:
        vmax = max(abs(data.min()), abs(data.max()))

    if cmap is None or norm is None:
        cmap, norm = make_diverging_cmap(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    # Annotate cells
    threshold = (vmax + vmin) / 2 if text_color_threshold is None else text_color_threshold
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if abs(val) > abs(vmax) * 0.6 else BLACK
            ax.text(j, i, f"{val:{fmt}}",
                    ha="center", va="center", color=color, fontsize=7)

    # Remove default spines for heatmaps
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(top=False, bottom=True, left=True, right=False)

    return im


def format_isi_label(isi_seconds):
    """Format ISI in human-readable form."""
    ms = isi_seconds * 1000
    if ms >= 1000:
        return f"{ms/1000:.1f} s"
    if ms == int(ms):
        return f"{int(ms)} ms"
    return f"{ms:.0f} ms"
