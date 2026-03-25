"""
Simulation 13: Reactivation vs Sequential Reactivation

Can SODA distinguish genuine sequential replay from non-sequential
co-activation of the same items?

The problem: If items A-E are reactivated with heterogeneous strengths
(some items decoded better than others), the SODA slope will be non-zero
even without any sequential timing structure. This slope will exceed that
of a different-items control (F-J at baseline), making it look like replay.
But it will NOT exceed the slope from a shuffled ordering of the same items,
because the slope is driven by amplitude differences, not timing.

This simulation models three conditions:
  1. TRUE: Items reactivated with sequential timing (genuine replay)
  2. COACTIVATION: Same items reactivated simultaneously (no timing)
  3. CONTROL: Different items (at baseline, no reactivation)

And tests whether SODA can separate TRUE from COACTIVATION as a function
of the ratio of sequential signal to amplitude heterogeneity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import permutations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_model import ResponseParams, multi_item_response, single_event_response
from soda import compute_slope_timecourse
from aggregation import peak_to_trough, abs_mean
from viz_style import (
    setup_style, add_panel_label, save_figure, add_zero_line,
    get_item_colors,
    FULL_WIDTH, BLUE, ORANGE, BLACK, GREY, LIGHT_GREY,
    FWD_COLOR, BWD_COLOR, FWD_ALPHA, BWD_ALPHA,
)

setup_style()


def simulate_coactivation_trial(n_items, n_trs, tr, class_amplitudes,
                                params, noise_sd, rng):
    """Simulate non-sequential co-activation: all items respond at
    the same time (onset=0), so there is no temporal ordering.

    The classifier probabilities reflect item-level reactivation strength
    but without the time-shifted hemodynamic overlap that produces
    sequential ordering.
    """
    t = np.arange(1, n_trs + 1, dtype=float)
    probs = np.zeros((n_items, len(t)))

    for i in range(n_items):
        item_params = ResponseParams(
            amplitude=class_amplitudes[i],
            wavelength=params.wavelength,
            onset_delay=params.onset_delay,
            baseline=params.baseline,
        )
        # All items onset at the same time — no sequential shift
        probs[i] = single_event_response(t, item_params)

    probs += rng.normal(0, noise_sd, probs.shape)
    probs = np.clip(probs, 0, 100)
    return probs


def simulate_sequential_trial(n_items, n_trs, tr, isi_seconds,
                              class_amplitudes, params, noise_sd, rng):
    """Simulate genuine sequential replay with heterogeneous amplitudes."""
    t = np.arange(1, n_trs + 1, dtype=float)
    probs = multi_item_response(t, n_items, isi_seconds, 0.0, tr, params,
                                class_amplitudes=class_amplitudes)
    probs += rng.normal(0, noise_sd, probs.shape)
    probs = np.clip(probs, 0, 100)
    return probs


def simulate_control_trial(n_items, n_trs, baseline, noise_sd, rng):
    """Simulate a control condition: different items not reactivated,
    probabilities fluctuate around baseline."""
    t_len = n_trs
    probs = baseline + rng.normal(0, noise_sd, (n_items, t_len))
    probs = np.clip(probs, 0, 100)
    return probs


def compute_permutation_pvalue(probs, true_order, metric_fn, n_perms=200,
                               rng=None):
    """Compare the true-order metric to a distribution of shuffled orderings.

    Returns: (true_metric, p_value, null_distribution)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_items = probs.shape[0]
    true_slopes = compute_slope_timecourse(probs, positions=true_order)
    true_metric = metric_fn(true_slopes)

    null_metrics = []
    for _ in range(n_perms):
        perm = rng.permutation(n_items) + 1  # 1-indexed positions
        perm_slopes = compute_slope_timecourse(probs, positions=perm.astype(float))
        null_metrics.append(metric_fn(perm_slopes))

    null_metrics = np.array(null_metrics)
    p_value = np.mean(null_metrics >= true_metric)
    return true_metric, p_value, null_metrics


def run_reactivation_vs_sequentiality():
    """Main simulation: three-panel figure.

    Panel A: Example timecourses and slopes for the three conditions
    Panel B: d' for TRUE vs CONTROL and TRUE vs SHUFFLED as a function
             of heterogeneity strength (at fixed sequential signal)
    Panel C: Same but varying sequential signal at fixed heterogeneity
    """
    n_items = 5
    n_trs = 13
    tr = 1.25
    isi = 0.128  # 128 ms replay speed
    noise_sd = 3.0
    n_trials = 200
    params = ResponseParams(amplitude=35.0, baseline=20.0)
    item_colors = get_item_colors(n_items)
    true_positions = np.arange(1, n_items + 1, dtype=float)

    # ================================================================
    # Panel A: Example timecourses
    # ================================================================
    t = np.arange(1, n_trs + 1, dtype=float)
    rng_ex = np.random.default_rng(42)
    het_amps = np.array([65, 50, 25, 35, 55], dtype=float)

    # Sequential replay
    probs_seq = simulate_sequential_trial(
        n_items, n_trs, tr, isi, het_amps, params, noise_sd=1.0, rng=rng_ex)
    slopes_seq = compute_slope_timecourse(probs_seq)

    # Co-activation (same items, no timing)
    probs_coact = simulate_coactivation_trial(
        n_items, n_trs, tr, het_amps, params, noise_sd=1.0, rng=rng_ex)
    slopes_coact = compute_slope_timecourse(probs_coact)

    # Control (different items, baseline)
    probs_ctrl = simulate_control_trial(
        n_items, n_trs, params.baseline, noise_sd=1.0, rng=rng_ex)
    slopes_ctrl = compute_slope_timecourse(probs_ctrl)

    # ================================================================
    # Panel B: d' as function of heterogeneity SD
    # ================================================================
    het_sds = [0, 3, 5, 8, 12, 16, 20, 25]
    mean_amp = 35.0
    fixed_isi = 0.128

    dprime_vs_ctrl_by_het = []
    dprime_vs_shuf_by_het = []

    for het_sd in het_sds:
        rng_b = np.random.default_rng(42)
        metrics_seq_vs_ctrl = {"seq": [], "ctrl": []}
        metrics_seq_vs_shuf = {"seq": [], "shuf": []}

        for _ in range(n_trials):
            amps = np.clip(rng_b.normal(mean_amp, het_sd, n_items), 10, 70)

            # Sequential trial
            probs_s = simulate_sequential_trial(
                n_items, n_trs, tr, fixed_isi, amps, params, noise_sd, rng_b)
            slopes_s = compute_slope_timecourse(probs_s)
            ptt_s = peak_to_trough(slopes_s)

            # Control trial (different items)
            probs_c = simulate_control_trial(
                n_items, n_trs, params.baseline, noise_sd, rng_b)
            slopes_c = compute_slope_timecourse(probs_c)
            ptt_c = peak_to_trough(slopes_c)

            # Shuffled trial (same items, same amplitudes, but random ordering)
            shuf_order = rng_b.permutation(n_items) + 1
            slopes_shuf = compute_slope_timecourse(probs_s,
                                                    positions=shuf_order.astype(float))
            ptt_shuf = peak_to_trough(slopes_shuf)

            metrics_seq_vs_ctrl["seq"].append(ptt_s)
            metrics_seq_vs_ctrl["ctrl"].append(ptt_c)
            metrics_seq_vs_shuf["seq"].append(ptt_s)
            metrics_seq_vs_shuf["shuf"].append(ptt_shuf)

        # d' vs control
        s1, s2 = np.array(metrics_seq_vs_ctrl["seq"]), np.array(metrics_seq_vs_ctrl["ctrl"])
        pooled = np.sqrt((np.std(s1, ddof=1)**2 + np.std(s2, ddof=1)**2) / 2)
        dp_ctrl = (np.mean(s1) - np.mean(s2)) / pooled if pooled > 0 else 0
        dprime_vs_ctrl_by_het.append(dp_ctrl)

        # d' vs shuffled
        s1, s2 = np.array(metrics_seq_vs_shuf["seq"]), np.array(metrics_seq_vs_shuf["shuf"])
        pooled = np.sqrt((np.std(s1, ddof=1)**2 + np.std(s2, ddof=1)**2) / 2)
        dp_shuf = (np.mean(s1) - np.mean(s2)) / pooled if pooled > 0 else 0
        dprime_vs_shuf_by_het.append(dp_shuf)

    # ================================================================
    # Panel C: d' as function of ISI (sequential signal strength)
    # ================================================================
    isi_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5]
    fixed_het_sd = 15.0

    dprime_vs_ctrl_by_isi = []
    dprime_vs_shuf_by_isi = []

    for isi_val in isi_values:
        rng_c = np.random.default_rng(42)
        metrics_c = {"seq": [], "ctrl": [], "shuf": []}

        for _ in range(n_trials):
            amps = np.clip(rng_c.normal(mean_amp, fixed_het_sd, n_items), 10, 70)

            probs_s = simulate_sequential_trial(
                n_items, n_trs, tr, isi_val, amps, params, noise_sd, rng_c)
            slopes_s = compute_slope_timecourse(probs_s)

            probs_c = simulate_control_trial(
                n_items, n_trs, params.baseline, noise_sd, rng_c)
            slopes_c = compute_slope_timecourse(probs_c)

            shuf_order = rng_c.permutation(n_items) + 1
            slopes_shuf = compute_slope_timecourse(probs_s,
                                                    positions=shuf_order.astype(float))

            metrics_c["seq"].append(peak_to_trough(slopes_s))
            metrics_c["ctrl"].append(peak_to_trough(slopes_c))
            metrics_c["shuf"].append(peak_to_trough(slopes_shuf))

        for key_pair, label in [
            (("seq", "ctrl"), "ctrl"),
            (("seq", "shuf"), "shuf"),
        ]:
            s1 = np.array(metrics_c[key_pair[0]])
            s2 = np.array(metrics_c[key_pair[1]])
            pooled = np.sqrt((np.std(s1, ddof=1)**2 + np.std(s2, ddof=1)**2) / 2)
            dp = (np.mean(s1) - np.mean(s2)) / pooled if pooled > 0 else 0
            if label == "ctrl":
                dprime_vs_ctrl_by_isi.append(dp)
            else:
                dprime_vs_shuf_by_isi.append(dp)

    # ================================================================
    # Build figure
    # ================================================================
    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.75))
    gs = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.45,
                           height_ratios=[1, 1])

    t_plot = t

    # --- Panel A1: Sequential replay ---
    ax_a1 = fig.add_subplot(gs[0, 0])
    for i in range(n_items):
        ax_a1.plot(t_plot, probs_seq[i], "-", color=item_colors[i],
                   lw=0.7, alpha=0.5)
    ax2 = ax_a1.twinx()
    ax2.plot(t_plot, slopes_seq, color=BLACK, lw=1.3)
    ax2.fill_between(t_plot, slopes_seq, 0, where=slopes_seq > 0,
                     alpha=FWD_ALPHA, color=FWD_COLOR)
    ax2.fill_between(t_plot, slopes_seq, 0, where=slopes_seq < 0,
                     alpha=BWD_ALPHA, color=BWD_COLOR)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(LIGHT_GREY)
    ax2.tick_params(axis="y", colors=GREY, labelsize=6)
    ax2.set_ylabel("Slope", fontsize=6, color=GREY)
    ax_a1.set_xlabel("Time (TRs)", fontsize=7)
    ax_a1.set_ylabel("Prob. (%)", fontsize=7)
    ax_a1.tick_params(labelsize=6)
    ax_a1.text(0.97, 0.97, "Sequential replay",
               transform=ax_a1.transAxes, fontsize=6, ha="right", va="top",
               color=GREY)
    add_panel_label(ax_a1, "A")

    # --- Panel A2: Co-activation ---
    ax_a2 = fig.add_subplot(gs[0, 1])
    for i in range(n_items):
        ax_a2.plot(t_plot, probs_coact[i], "-", color=item_colors[i],
                   lw=0.7, alpha=0.5)
    ax2b = ax_a2.twinx()
    ax2b.plot(t_plot, slopes_coact, color=BLACK, lw=1.3)
    ax2b.fill_between(t_plot, slopes_coact, 0, where=slopes_coact > 0,
                      alpha=FWD_ALPHA, color=FWD_COLOR)
    ax2b.fill_between(t_plot, slopes_coact, 0, where=slopes_coact < 0,
                      alpha=BWD_ALPHA, color=BWD_COLOR)
    ax2b.spines["right"].set_visible(True)
    ax2b.spines["right"].set_color(LIGHT_GREY)
    ax2b.tick_params(axis="y", colors=GREY, labelsize=6)
    ax2b.set_ylabel("Slope", fontsize=6, color=GREY)
    ax_a2.set_xlabel("Time (TRs)", fontsize=7)
    ax_a2.tick_params(labelsize=6)
    ax_a2.text(0.97, 0.97, "Co-activation\n(no timing)",
               transform=ax_a2.transAxes, fontsize=6, ha="right", va="top",
               color=GREY)
    add_panel_label(ax_a2, "B")

    # --- Panel A3: Control ---
    ax_a3 = fig.add_subplot(gs[0, 2])
    for i in range(n_items):
        ax_a3.plot(t_plot, probs_ctrl[i], "-", color=item_colors[i],
                   lw=0.7, alpha=0.5)
    ax2c = ax_a3.twinx()
    ax2c.plot(t_plot, slopes_ctrl, color=BLACK, lw=1.3)
    ax2c.spines["right"].set_visible(True)
    ax2c.spines["right"].set_color(LIGHT_GREY)
    ax2c.tick_params(axis="y", colors=GREY, labelsize=6)
    ax2c.set_ylabel("Slope", fontsize=6, color=GREY)
    ax_a3.set_xlabel("Time (TRs)", fontsize=7)
    ax_a3.tick_params(labelsize=6)
    ax_a3.text(0.97, 0.97, "Control\n(different items)",
               transform=ax_a3.transAxes, fontsize=6, ha="right", va="top",
               color=GREY)
    add_panel_label(ax_a3, "C")

    # --- Panel B: d' vs heterogeneity ---
    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_b.plot(het_sds, dprime_vs_ctrl_by_het, "o-", color=BLUE, lw=1.2,
              markersize=4, label="True vs control (reactivation)")
    ax_b.plot(het_sds, dprime_vs_shuf_by_het, "s-", color=ORANGE, lw=1.2,
              markersize=4, label="True vs shuffled (sequentiality)")
    add_zero_line(ax_b)
    ax_b.set_xlabel("Classifier amplitude heterogeneity (SD)")
    ax_b.set_ylabel("d\u2032 (sensitivity)")
    ax_b.legend(fontsize=6, loc="upper left")
    add_panel_label(ax_b, "D")

    # --- Panel C: d' vs ISI ---
    ax_c = fig.add_subplot(gs[1, 2])
    ax_c.plot(isi_values, dprime_vs_ctrl_by_isi, "o-", color=BLUE, lw=1.2,
              markersize=4, label="vs control")
    ax_c.plot(isi_values, dprime_vs_shuf_by_isi, "s-", color=ORANGE, lw=1.2,
              markersize=4, label="vs shuffled")
    add_zero_line(ax_c)
    ax_c.set_xlabel("ISI (seconds)")
    ax_c.set_ylabel("d\u2032")
    ax_c.legend(fontsize=6, loc="upper left")
    ax_c.text(0.97, 0.97, f"Het. SD = {fixed_het_sd:.0f}",
              transform=ax_c.transAxes, fontsize=6, ha="right", va="top",
              color=GREY)
    add_panel_label(ax_c, "E")

    save_figure(fig, "sim13_reactivation_vs_sequentiality")
    plt.close()
    print("  Sim 13 done.")


if __name__ == "__main__":
    print("=" * 60)
    print("Simulation 13: Reactivation vs Sequential Reactivation")
    print("=" * 60)
    run_reactivation_vs_sequentiality()
    print("\nDone!")
