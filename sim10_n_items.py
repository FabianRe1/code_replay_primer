"""
Simulation 10: Effect of Number of Sequence Items

How does SODA sensitivity change with sequence length (N items)?
More items = more data points for regression = potentially more stable slope,
but also more items that can introduce heterogeneity.

Tests N = 3, 4, 5, 6, 8 items.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_model import ResponseParams, multi_item_response
from soda import compute_slope_timecourse
from aggregation import (
    mean_slope, abs_mean, slope_variance, peak_to_trough,
    spectral_power, compute_all_metrics,
)
from viz_style import (
    setup_style, add_panel_label, save_figure, add_zero_line,
    get_item_colors, get_metric_colors, metric_label,
    FULL_WIDTH, BLUE, ORANGE, BLACK, GREY,
)

setup_style()


def run_n_items_sweep():
    """Sweep N = 3, 4, 5, 6, 8 and compute d' for each metric."""

    n_values = [3, 4, 5, 6, 8]
    isi = 0.128  # 128 ms
    tr = 1.25
    n_trs = 13
    noise_sd = 4.0
    signal_amp = 35.0
    n_trials = 200
    t = np.arange(1, n_trs + 1, dtype=float)

    params = ResponseParams(amplitude=signal_amp, baseline=20.0,
                            wavelength=5.26, onset_delay=0.56)

    metrics_to_test = {
        "abs_mean": abs_mean,
        "slope_variance": slope_variance,
        "peak_to_trough": peak_to_trough,
        "spectral_power": spectral_power,
    }

    metric_colors = get_metric_colors()

    # results[metric] = list of d' per N
    results = {m: [] for m in metrics_to_test}
    # Store example slopes for visualization
    example_slopes = {}

    for n_items in n_values:
        rng = np.random.default_rng(42)
        class_amps = np.full(n_items, signal_amp)

        sig_vals = {m: [] for m in metrics_to_test}
        null_vals = {m: [] for m in metrics_to_test}

        for trial in range(n_trials):
            # Signal trial
            probs_s = multi_item_response(t, n_items, isi, 0.1, tr, params,
                                          class_amplitudes=class_amps)
            probs_s += rng.normal(0, noise_sd, probs_s.shape)
            probs_s = np.clip(probs_s, 0, 100)
            slopes_s = compute_slope_timecourse(probs_s)

            # Null trial
            probs_n = 20.0 + rng.normal(0, noise_sd, (n_items, len(t)))
            probs_n = np.clip(probs_n, 0, 100)
            slopes_n = compute_slope_timecourse(probs_n)

            for m, fn in metrics_to_test.items():
                sig_vals[m].append(fn(slopes_s))
                null_vals[m].append(fn(slopes_n))

            # Save first trial's slopes for example
            if trial == 0:
                example_slopes[n_items] = slopes_s

        for m in metrics_to_test:
            s = np.array(sig_vals[m])
            n = np.array(null_vals[m])
            ms_, mn_ = np.nanmean(s), np.nanmean(n)
            ss_, sn_ = np.nanstd(s, ddof=1), np.nanstd(n, ddof=1)
            pooled = np.sqrt((ss_**2 + sn_**2) / 2)
            dp = (ms_ - mn_) / pooled if pooled > 0 else 0.0
            results[m].append(dp)

    # ========== Figure ==========
    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.35)

    # Panel A: d' vs N
    ax_a = fig.add_subplot(gs[0, 0])
    for m in metrics_to_test:
        ax_a.plot(n_values, results[m], "o-",
                  color=metric_colors.get(m, GREY),
                  lw=1.2, markersize=4,
                  label=metric_label(m))
    add_zero_line(ax_a)
    ax_a.set_xlabel("Number of sequence items (N)")
    ax_a.set_ylabel("d\u2032 (sensitivity)")
    ax_a.set_xticks(n_values)
    ax_a.legend(fontsize=7, loc="lower right")
    add_panel_label(ax_a, "A")

    # Panel B: Example slope timecourses at N=3 vs N=5 vs N=8
    ax_b = fig.add_subplot(gs[0, 1])
    show_ns = [3, 5, 8]
    n_colors = {3: ORANGE, 5: BLACK, 8: BLUE}
    for n_items in show_ns:
        if n_items in example_slopes:
            ax_b.plot(t, example_slopes[n_items], "o-",
                      color=n_colors[n_items], lw=1.0, markersize=3,
                      label=f"N = {n_items}")
    add_zero_line(ax_b)
    ax_b.set_xlabel("Time (TRs)")
    ax_b.set_ylabel("SODA slope")
    ax_b.legend(fontsize=7)
    add_panel_label(ax_b, "B")

    save_figure(fig, "sim10_n_items")
    plt.close()
    print("  Sim 10 done.")

    return n_values, results


if __name__ == "__main__":
    print("=" * 60)
    print("Simulation 10: Effect of Number of Sequence Items")
    print("=" * 60)
    run_n_items_sweep()
    print("\nDone!")
