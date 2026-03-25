"""
Simulation 12: Realistic Noise Structure

Current simulations use i.i.d. Gaussian noise, but real fMRI classifier
probabilities have autocorrelated noise (HRF sluggishness, scanner drift).

Tests AR(1) noise with varying autocorrelation and 1/f (pink) noise.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_model import ResponseParams, multi_item_response
from soda import compute_slope_timecourse
from aggregation import abs_mean, slope_variance, peak_to_trough, spectral_power
from viz_style import (
    setup_style, add_panel_label, save_figure, add_zero_line,
    get_metric_colors, metric_label,
    FULL_WIDTH, BLUE, ORANGE, BLACK, GREY, LIGHT_GREY,
)

setup_style()


def generate_ar1_noise(n_items, n_trs, rho, sd, rng):
    """Generate AR(1) noise: x[t] = rho * x[t-1] + epsilon.

    The marginal variance is scaled to match `sd` regardless of rho.
    """
    innovation_sd = sd * np.sqrt(1 - rho**2) if rho < 1 else sd
    noise = np.zeros((n_items, n_trs))
    noise[:, 0] = rng.normal(0, sd, n_items)
    for t_idx in range(1, n_trs):
        noise[:, t_idx] = rho * noise[:, t_idx - 1] + \
                          rng.normal(0, innovation_sd, n_items)
    return noise


def generate_pink_noise(n_items, n_trs, sd, rng):
    """Generate 1/f (pink) noise via spectral shaping."""
    noise = np.zeros((n_items, n_trs))
    for i in range(n_items):
        white = rng.normal(0, 1, n_trs)
        freqs = np.fft.rfftfreq(n_trs)
        freqs[0] = 1  # avoid division by zero
        fft_white = np.fft.rfft(white)
        pink_fft = fft_white / np.sqrt(freqs)
        pink = np.fft.irfft(pink_fft, n=n_trs)
        # Scale to desired SD
        pink = pink / np.std(pink) * sd
        noise[i] = pink
    return noise


def run_noise_sweep():
    """Sweep AR(1) rho and test 1/f noise."""

    rho_values = [0.0, 0.1, 0.3, 0.5, 0.7]
    isi = 0.128
    n_items = 5
    n_trs = 13
    tr = 1.25
    signal_amp = 35.0
    noise_sd = 4.0
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

    # AR(1) results
    ar1_results = {m: [] for m in metrics_to_test}
    class_amps = np.full(n_items, signal_amp)

    for rho in rho_values:
        rng = np.random.default_rng(42)
        sig_vals = {m: [] for m in metrics_to_test}
        null_vals = {m: [] for m in metrics_to_test}

        for _ in range(n_trials):
            noise_s = generate_ar1_noise(n_items, len(t), rho, noise_sd, rng)
            probs_s = multi_item_response(t, n_items, isi, 0.1, tr, params,
                                          class_amplitudes=class_amps)
            probs_s = np.clip(probs_s + noise_s, 0, 100)
            slopes_s = compute_slope_timecourse(probs_s)

            noise_n = generate_ar1_noise(n_items, len(t), rho, noise_sd, rng)
            probs_n = 20.0 + noise_n
            probs_n = np.clip(probs_n, 0, 100)
            slopes_n = compute_slope_timecourse(probs_n)

            for m, fn in metrics_to_test.items():
                sig_vals[m].append(fn(slopes_s))
                null_vals[m].append(fn(slopes_n))

        for m in metrics_to_test:
            s, n = np.array(sig_vals[m]), np.array(null_vals[m])
            pooled = np.sqrt((np.nanstd(s, ddof=1)**2 +
                              np.nanstd(n, ddof=1)**2) / 2)
            dp = (np.nanmean(s) - np.nanmean(n)) / pooled if pooled > 0 else 0.0
            ar1_results[m].append(dp)

    # Pink noise results (single point)
    pink_results = {}
    rng_pink = np.random.default_rng(42)
    sig_vals_p = {m: [] for m in metrics_to_test}
    null_vals_p = {m: [] for m in metrics_to_test}

    for _ in range(n_trials):
        noise_s = generate_pink_noise(n_items, len(t), noise_sd, rng_pink)
        probs_s = multi_item_response(t, n_items, isi, 0.1, tr, params,
                                      class_amplitudes=class_amps)
        probs_s = np.clip(probs_s + noise_s, 0, 100)
        slopes_s = compute_slope_timecourse(probs_s)

        noise_n = generate_pink_noise(n_items, len(t), noise_sd, rng_pink)
        probs_n = 20.0 + noise_n
        probs_n = np.clip(probs_n, 0, 100)
        slopes_n = compute_slope_timecourse(probs_n)

        for m, fn in metrics_to_test.items():
            sig_vals_p[m].append(fn(slopes_s))
            null_vals_p[m].append(fn(slopes_n))

    for m in metrics_to_test:
        s, n = np.array(sig_vals_p[m]), np.array(null_vals_p[m])
        pooled = np.sqrt((np.nanstd(s, ddof=1)**2 +
                          np.nanstd(n, ddof=1)**2) / 2)
        pink_results[m] = (np.nanmean(s) - np.nanmean(n)) / pooled \
            if pooled > 0 else 0.0

    # ========== Figure ==========
    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

    # Panel A: d' vs AR(1) rho
    ax_a = fig.add_subplot(gs[0, 0])
    for m in metrics_to_test:
        ax_a.plot(rho_values, ar1_results[m], "o-",
                  color=metric_colors.get(m, GREY),
                  lw=1.2, markersize=4,
                  label=metric_label(m))
        # Mark pink noise result as star at rho=0.8 (visual offset)
        ax_a.plot(0.8, pink_results[m], "*",
                  color=metric_colors.get(m, GREY),
                  markersize=10, markeredgecolor=BLACK,
                  markeredgewidth=0.5)

    add_zero_line(ax_a)
    ax_a.set_xlabel("AR(1) autocorrelation (\u03C1)")
    ax_a.set_ylabel("d\u2032 (sensitivity)")
    ax_a.legend(fontsize=6, loc="best")
    # Add pink noise annotation
    ax_a.annotate("1/f", xy=(0.8, max(pink_results.values()) * 1.05),
                  fontsize=7, ha="center", color=GREY, fontweight="bold")
    add_panel_label(ax_a, "A")

    # Panel B: Example noise timecourses
    ax_b = fig.add_subplot(gs[0, 1])
    rng_ex = np.random.default_rng(99)
    t_ex = np.arange(1, 14, dtype=float)

    noise_types = [
        ("White (\u03C1=0)", 0.0, BLUE),
        ("AR(1) \u03C1=0.5", 0.5, ORANGE),
        ("AR(1) \u03C1=0.7", 0.7, BLACK),
    ]
    for label, rho, color in noise_types:
        noise_ex = generate_ar1_noise(1, len(t_ex), rho, 3.0, rng_ex)[0]
        ax_b.plot(t_ex, noise_ex, "o-", color=color, lw=1.2,
                  markersize=3, label=label, alpha=0.8)

    # Pink noise example
    noise_pink = generate_pink_noise(1, len(t_ex), 3.0, rng_ex)[0]
    ax_b.plot(t_ex, noise_pink, "s-", color="#6A3D9A", lw=1.2,
              markersize=3, label="1/f (pink)", alpha=0.8)

    add_zero_line(ax_b)
    ax_b.set_xlabel("Time (TRs)")
    ax_b.set_ylabel("Noise amplitude")
    ax_b.legend(fontsize=6, loc="best")
    add_panel_label(ax_b, "B")

    save_figure(fig, "sim12_realistic_noise")
    plt.close()
    print("  Sim 12 done.")

    return rho_values, ar1_results, pink_results


if __name__ == "__main__":
    print("=" * 60)
    print("Simulation 12: Realistic Noise Structure")
    print("=" * 60)
    run_noise_sweep()
    print("\nDone!")
