"""
Simulation 9: Normalization x ISI interaction

Tests whether z-score normalization of classifier probabilities becomes
helpful at slower replay speeds where items are more temporally separated.

Key finding: normalization remains counterproductive across all ISIs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_model import ResponseParams, multi_item_response
from soda import compute_slope_timecourse
from sim6_normalization import normalize_zscore
from aggregation import peak_to_trough, abs_mean, slope_variance
from viz_style import (
    setup_style, add_panel_label, save_figure, add_zero_line,
    annotated_heatmap, make_diverging_cmap,
    get_item_colors,
    FULL_WIDTH, ONE_HALF_COL, BLUE, ORANGE, BLACK, GREY, LIGHT_GREY,
    FWD_COLOR, BWD_COLOR, FWD_ALPHA, BWD_ALPHA,
)

setup_style()


def _plot_slope_panel(ax, t_sec, probs, slopes, item_colors, n_items,
                      slope_lim, ylabel_prob, baseline_h=None):
    """Plot probability timecourses + slope overlay on one panel."""
    for i in range(n_items):
        ax.plot(t_sec, probs[i], "-", color=item_colors[i], lw=0.7, alpha=0.5)

    if baseline_h is not None:
        ax.axhline(baseline_h, color=GREY, ls=":", alpha=0.3, lw=0.5)

    ax2 = ax.twinx()
    ax2.plot(t_sec, slopes, color=BLACK, lw=1.3, alpha=0.8)
    ax2.fill_between(t_sec, slopes, 0, where=slopes > 0,
                     alpha=FWD_ALPHA * 1.5, color=FWD_COLOR)
    ax2.fill_between(t_sec, slopes, 0, where=slopes < 0,
                     alpha=BWD_ALPHA * 1.5, color=BWD_COLOR)
    ax2.set_ylim(slope_lim)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GREY)
    ax2.tick_params(axis="y", labelcolor=GREY, labelsize=6)
    ax2.set_ylabel("SODA slope", color=GREY, fontsize=6)

    ax.set_ylabel(ylabel_prob, fontsize=7)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.tick_params(labelsize=6)
    return ax2


def run_example_and_dprime():
    """
    Generate two-panel summary figure:
      Panel A: Example timecourses at ISI=0.5s (homo / hetero raw / hetero z-scored)
      Panel B: d' heatmap across ISIs x normalization strategies
    """
    n_items = 5
    n_trs = 25
    tr = 1.25
    noise_sd_example = 2.0
    noise_sd_dprime = 4.0
    n_trials = 80
    het_sd = 20.0
    mean_amp = 35.0
    params = ResponseParams(amplitude=mean_amp, baseline=20.0,
                            wavelength=5.26, onset_delay=0.56)
    item_colors = get_item_colors(n_items)
    t = np.arange(1, n_trs + 1, dtype=float)

    # ---- Panel A: Example at ISI=0.5s ----
    isi_example = 0.5
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_sd_example, (n_items, len(t)))

    homo_amps = np.full(n_items, mean_amp)
    hetero_amps = np.array([75, 55, 40, 30, 60], dtype=float)

    # Homogeneous
    probs_homo = multi_item_response(t, n_items, isi_example, 0.0, tr, params,
                                     class_amplitudes=homo_amps)
    probs_homo = np.clip(probs_homo + noise, 0, 100)
    slopes_homo = compute_slope_timecourse(probs_homo)

    # Heterogeneous raw
    probs_het = multi_item_response(t, n_items, isi_example, 0.0, tr, params,
                                    class_amplitudes=hetero_amps)
    probs_het = np.clip(probs_het + noise, 0, 100)
    slopes_het = compute_slope_timecourse(probs_het)

    # Heterogeneous z-scored
    probs_het_z = normalize_zscore(probs_het)
    slopes_het_z = compute_slope_timecourse(probs_het_z)

    # Global slope limits for example panels
    s_max = max(np.max(np.abs(slopes_homo)),
                np.max(np.abs(slopes_het)),
                np.max(np.abs(slopes_het_z))) * 1.15
    slope_lim = (-s_max, s_max)

    # ---- Panel B: d' across ISIs ----
    isi_values = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5]
    metrics = {
        "peak_to_trough": peak_to_trough,
        "abs_mean": abs_mean,
        "variance": slope_variance,
    }
    metric_display = {
        "peak_to_trough": "Peak-to-trough",
        "abs_mean": "|Slope| mean",
        "variance": "Slope variance",
    }
    norm_methods = ["raw", "zscore"]
    norm_display = {"raw": "Raw", "zscore": "Z-scored"}

    # results[metric][norm] = list of d' per ISI
    results = {m: {n: [] for n in norm_methods} for m in metrics}

    for isi in isi_values:
        rng_dp = np.random.default_rng(42)

        sig_vals = {m: {n: [] for n in norm_methods} for m in metrics}
        null_vals = {m: {n: [] for n in norm_methods} for m in metrics}

        for _ in range(n_trials):
            class_amps = np.clip(rng_dp.normal(mean_amp, het_sd, n_items), 10, 80)

            # Signal trial
            probs_s = multi_item_response(t, n_items, isi, 0.0, tr, params,
                                          class_amplitudes=class_amps)
            probs_s += rng_dp.normal(0, noise_sd_dprime, probs_s.shape)
            probs_s = np.clip(probs_s, 0, 100)

            slopes_raw = compute_slope_timecourse(probs_s)
            slopes_z = compute_slope_timecourse(normalize_zscore(probs_s))

            # Null trial
            probs_n = 20.0 + rng_dp.normal(0, noise_sd_dprime, (n_items, len(t)))
            probs_n = np.clip(probs_n, 0, 100)

            slopes_raw_n = compute_slope_timecourse(probs_n)
            slopes_z_n = compute_slope_timecourse(normalize_zscore(probs_n))

            for m, fn in metrics.items():
                sig_vals[m]["raw"].append(fn(slopes_raw))
                sig_vals[m]["zscore"].append(fn(slopes_z))
                null_vals[m]["raw"].append(fn(slopes_raw_n))
                null_vals[m]["zscore"].append(fn(slopes_z_n))

        for m in metrics:
            for norm in norm_methods:
                s = np.array(sig_vals[m][norm])
                n = np.array(null_vals[m][norm])
                ms_, mn_ = np.nanmean(s), np.nanmean(n)
                ss_, sn_ = np.nanstd(s, ddof=1), np.nanstd(n, ddof=1)
                pooled = np.sqrt((ss_**2 + sn_**2) / 2)
                dp = (ms_ - mn_) / pooled if pooled > 0 else 0.0
                results[m][norm].append(dp)

    # ========== Build the figure ==========
    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.38))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.35)

    # --- Panel A: 1x3 example sub-grid ---
    gs_a = gs[0, 0].subgridspec(1, 3, wspace=0.65)

    t_sec = t * tr

    ax_a1 = fig.add_subplot(gs_a[0, 0])
    _plot_slope_panel(ax_a1, t_sec, probs_homo, slopes_homo,
                      item_colors, n_items, slope_lim, "Prob. (%)", baseline_h=20)
    ax_a1.set_title("Homogeneous", fontsize=7, pad=6)
    add_panel_label(ax_a1, "A", x=-0.28)

    ax_a2 = fig.add_subplot(gs_a[0, 1])
    _plot_slope_panel(ax_a2, t_sec, probs_het, slopes_het,
                      item_colors, n_items, slope_lim, "Prob. (%)", baseline_h=20)
    ax_a2.set_title("Heterogeneous", fontsize=7, pad=6)

    ax_a3 = fig.add_subplot(gs_a[0, 2])
    _plot_slope_panel(ax_a3, t_sec, probs_het_z, slopes_het_z,
                      item_colors, n_items, slope_lim, "Z-scored", baseline_h=0)
    ax_a3.set_title("Hetero. + z-scored", fontsize=7, pad=6)

    # --- Panel B: d' line plot (ISI x metric x normalization) ---
    ax_b = fig.add_subplot(gs[0, 1])

    metric_styles = {
        "peak_to_trough": {"color": "#238B45", "marker": "o"},
        "abs_mean":       {"color": "#D94801", "marker": "s"},
        "variance":       {"color": "#6A3D9A", "marker": "^"},
    }

    for m in metrics:
        style = metric_styles[m]
        # Raw: solid line
        ax_b.plot(isi_values, results[m]["raw"], "-",
                  color=style["color"], marker=style["marker"],
                  lw=1.2, markersize=4,
                  label=f"{metric_display[m]} (raw)")
        # Z-scored: dashed line, thinner
        ax_b.plot(isi_values, results[m]["zscore"], "--",
                  color=style["color"], marker=style["marker"],
                  lw=0.8, markersize=3, alpha=0.5,
                  label=f"{metric_display[m]} (z-scored)")

    add_zero_line(ax_b)
    ax_b.set_xlabel("ISI (seconds)")
    ax_b.set_ylabel("d\u2032 (signal vs null)")
    ax_b.legend(fontsize=5.5, loc="upper left", ncol=1)
    add_panel_label(ax_b, "B", x=-0.18)

    save_figure(fig, "sim9_normalization_isi_summary")
    plt.close()
    print("  Sim 9 summary figure done.")


if __name__ == "__main__":
    print("=" * 60)
    print("Simulation 9: Normalization x ISI Interaction")
    print("=" * 60)
    run_example_and_dprime()
    print("\nDone!")
