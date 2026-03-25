"""
Simulation 11: TR Sensitivity

How does the repetition time affect SODA detection?
Modern multi-band sequences use TR = 0.7-1.0s.
Shorter TR = more samples per cycle but potentially noisier.

Tests TR = 0.7, 0.8, 1.0, 1.25, 1.5, 2.0 s.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_model import ResponseParams, multi_item_response
from soda import compute_slope_timecourse, slope_frequency_spectrum, predicted_frequency
from aggregation import abs_mean, slope_variance, peak_to_trough, spectral_power
from viz_style import (
    setup_style, add_panel_label, save_figure, add_zero_line,
    get_metric_colors, metric_label,
    FULL_WIDTH, BLUE, ORANGE, BLACK, GREY,
)

setup_style()


def run_tr_sweep():
    """Sweep TR = 0.7-2.0s and compute d' for each metric."""

    tr_values = [0.7, 0.8, 1.0, 1.25, 1.5, 2.0]
    isi = 0.128  # 128 ms
    n_items = 5
    signal_amp = 35.0
    noise_sd = 4.0
    n_trials = 200
    window_duration = 16.0  # seconds — keep constant

    params = ResponseParams(amplitude=signal_amp, baseline=20.0,
                            wavelength=5.26, onset_delay=0.56)

    metrics_to_test = {
        "abs_mean": abs_mean,
        "slope_variance": slope_variance,
        "peak_to_trough": peak_to_trough,
        "spectral_power": spectral_power,
    }
    metric_colors = get_metric_colors()

    results = {m: [] for m in metrics_to_test}
    freq_data = {}  # Store frequency spectra for example

    for tr in tr_values:
        n_trs = int(window_duration / tr)
        t = np.arange(1, n_trs + 1, dtype=float)
        rng = np.random.default_rng(42)
        class_amps = np.full(n_items, signal_amp)

        sig_vals = {m: [] for m in metrics_to_test}
        null_vals = {m: [] for m in metrics_to_test}

        for trial in range(n_trials):
            probs_s = multi_item_response(t, n_items, isi, 0.1, tr, params,
                                          class_amplitudes=class_amps)
            probs_s += rng.normal(0, noise_sd, probs_s.shape)
            probs_s = np.clip(probs_s, 0, 100)
            slopes_s = compute_slope_timecourse(probs_s)

            probs_n = 20.0 + rng.normal(0, noise_sd, (n_items, len(t)))
            probs_n = np.clip(probs_n, 0, 100)
            slopes_n = compute_slope_timecourse(probs_n)

            for m, fn in metrics_to_test.items():
                sig_vals[m].append(fn(slopes_s))
                null_vals[m].append(fn(slopes_n))

            # Save first trial for frequency spectrum
            if trial == 0:
                freq_data[tr] = (slopes_s, tr)

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

    # Panel A: d' vs TR
    ax_a = fig.add_subplot(gs[0, 0])
    for m in metrics_to_test:
        ax_a.plot(tr_values, results[m], "o-",
                  color=metric_colors.get(m, GREY),
                  lw=1.2, markersize=4,
                  label=metric_label(m))
    add_zero_line(ax_a)
    ax_a.set_xlabel("Repetition time TR (s)")
    ax_a.set_ylabel("d\u2032 (sensitivity)")
    ax_a.legend(fontsize=7, loc="best")
    add_panel_label(ax_a, "A")

    # Panel B: Frequency spectra at TR=0.7 vs 1.25 vs 2.0
    ax_b = fig.add_subplot(gs[0, 1])
    show_trs = [0.7, 1.25, 2.0]
    tr_colors = {0.7: BLUE, 1.25: BLACK, 2.0: ORANGE}

    for tr_show in show_trs:
        if tr_show in freq_data:
            slopes, tr_val = freq_data[tr_show]
            freqs, power = slope_frequency_spectrum(slopes, tr=tr_val)
            ax_b.plot(freqs, power, color=tr_colors[tr_show], lw=1.0,
                      label=f"TR = {tr_show} s")

    # Mark predicted frequency
    pred_f = predicted_frequency(isi, n_items=n_items, tr=1.25)
    ax_b.axvline(pred_f, color=GREY, ls="--", lw=0.8, alpha=0.6)
    ax_b.text(pred_f + 0.005, ax_b.get_ylim()[1] * 0.9,
              f"Predicted\n{pred_f:.2f} Hz", fontsize=6, color=GREY)

    ax_b.set_xlabel("Frequency (Hz)")
    ax_b.set_ylabel("Spectral power")
    ax_b.legend(fontsize=7)
    ax_b.set_xlim(0, 0.4)
    add_panel_label(ax_b, "B")

    save_figure(fig, "sim11_tr_sensitivity")
    plt.close()
    print("  Sim 11 done.")

    return tr_values, results


if __name__ == "__main__":
    print("=" * 60)
    print("Simulation 11: TR Sensitivity")
    print("=" * 60)
    run_tr_sweep()
    print("\nDone!")
