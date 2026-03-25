"""
SODA Primer Simulations — Main Script

Demonstrates the SODA method step by step:
1. Ideal case: equal classifier performance, single sequence
2. Heterogeneous classifiers: different accuracy per class
3. Multiple replay events: overlapping responses
4. Trial averaging: phase misalignment across trials

Run from the code_replay_primer/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from response_model import (
    ResponseParams, single_event_response, two_event_difference,
    compute_periods, sequence_delta, multi_item_response,
)
from soda import (
    compute_slope_timecourse, compute_slope_for_trial,
    periods_to_tr_indices,
)
from classifier_sim import (
    simulate_ideal_sequence_trial, simulate_heterogeneous_trial,
    simulate_multiple_replay_events, simulate_trial_averaging,
)
from viz_style import (
    setup_style, add_panel_label, shade_periods, add_zero_line,
    add_chance_line, save_figure, despine, format_isi_label,
    get_item_colors, get_speed_colors, speed_color,
    BLUE, ORANGE, GREY, BLACK, LIGHT_GREY,
    FWD_COLOR, BWD_COLOR, FWD_ALPHA, BWD_ALPHA,
    FULL_WIDTH, ONE_HALF_COL,
)

setup_style()


# =============================================================================
# SIMULATION 1: Ideal case — reproduce Wittkuhn & Schuck core result
# =============================================================================

def sim1_ideal_case():
    """Reproduce the core SODA demonstration from the paper."""

    params = ResponseParams()
    n_items = 5
    n_trs = 13
    t = np.arange(1, n_trs + 1, dtype=float)
    t_fine = np.linspace(0, 14, 500)
    item_colors = get_item_colors(n_items)
    spd_colors = get_speed_colors()

    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.78))
    gs = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.45)

    # --- Panel A: Single event response ---
    ax_a = fig.add_subplot(gs[0, 0])
    response = single_event_response(t_fine, params)
    ax_a.plot(t_fine, response, color=BLACK, lw=1.5)
    add_chance_line(ax_a, chance=params.baseline)
    ax_a.set_xlabel("Time (TRs)")
    ax_a.set_ylabel("Probability (%)")
    ax_a.set_xlim(0, 10)
    add_panel_label(ax_a, "A")

    # --- Panel B: Two-event difference for different deltas ---
    ax_b = fig.add_subplot(gs[0, 1])
    for isi, color in spd_colors.items():
        delta = sequence_delta(n_items, isi)
        diff = two_event_difference(t_fine, delta, params)
        ax_b.plot(t_fine, diff, color=color, lw=0.9,
                  label=format_isi_label(isi))
    add_zero_line(ax_b)
    ax_b.set_xlabel("Time (TRs)")
    ax_b.set_ylabel("Prob. difference")
    ax_b.legend(fontsize=5, title="ISI", title_fontsize=6,
                loc="upper left", handlelength=1.2)
    ax_b.set_xlim(0, 14)
    add_panel_label(ax_b, "B")

    # --- Panel C: Probability timecourses for 32ms sequence ---
    ax_c = fig.add_subplot(gs[0, 2])
    probs_fast = simulate_ideal_sequence_trial(
        n_items=n_items, isi_seconds=0.032, n_trs=n_trs
    )
    for i in range(n_items):
        ax_c.plot(t, probs_fast[i], "o-", color=item_colors[i], lw=0.8,
                  markersize=2.5, label=f"Item {i+1}")
    ax_c.set_xlabel("Time (TRs)")
    ax_c.set_ylabel("Probability (%)")
    ax_c.legend(fontsize=5, ncol=1, loc="upper right", handlelength=1.0,
                borderpad=0.2, labelspacing=0.2)
    add_panel_label(ax_c, "C")

    # --- Panels D-F: Probability timecourses for 128, 512, 2048 ms ---
    panel_labels_d = ["D", "E", "F"]
    for idx, isi in enumerate([0.128, 0.512, 2.048]):
        ax = fig.add_subplot(gs[1, idx])
        probs = simulate_ideal_sequence_trial(
            n_items=n_items, isi_seconds=isi, n_trs=n_trs
        )
        for i in range(n_items):
            ax.plot(t, probs[i], "o-", color=item_colors[i], lw=0.8,
                    markersize=2.5)

        delta = sequence_delta(n_items, isi)
        fwd, bwd = compute_periods(delta, params)
        shade_periods(ax, fwd, bwd)

        ax.set_xlabel("Time (TRs)")
        if idx == 0:
            ax.set_ylabel("Probability (%)")
        ax.text(0.97, 0.97, format_isi_label(isi),
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                color=GREY)
        add_panel_label(ax, panel_labels_d[idx])

    # --- Panels G-I: SODA slope timecourses ---
    panel_labels_g = ["G", "H", "I"]
    for idx, isi in enumerate([0.032, 0.128, 2.048]):
        ax = fig.add_subplot(gs[2, idx])
        probs = simulate_ideal_sequence_trial(
            n_items=n_items, isi_seconds=isi, n_trs=n_trs
        )
        slopes = compute_slope_timecourse(probs)

        ax.plot(t, slopes, "o-", color=speed_color(isi), lw=1.5, markersize=3)
        add_zero_line(ax)
        ax.fill_between(t, slopes, 0, where=slopes > 0,
                        alpha=FWD_ALPHA * 2, color=FWD_COLOR)
        ax.fill_between(t, slopes, 0, where=slopes < 0,
                        alpha=BWD_ALPHA * 2, color=BWD_COLOR)

        ax.set_xlabel("Time (TRs)")
        if idx == 0:
            ax.set_ylabel("SODA slope")
        ax.text(0.97, 0.97, format_isi_label(isi),
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                color=GREY)
        add_panel_label(ax, panel_labels_g[idx])

    save_figure(fig, "sim1_ideal_case")
    plt.close()
    print("  Sim 1 done.")


# =============================================================================
# SIMULATION 2: Heterogeneous classifier performance
# =============================================================================

def sim2_heterogeneous_classifiers():
    """Show how unequal classifier accuracy distorts SODA slopes."""

    params = ResponseParams()
    n_items = 5
    n_trs = 13
    t = np.arange(1, n_trs + 1, dtype=float)
    isi = 0.128
    rng = np.random.default_rng(42)
    item_colors = get_item_colors(n_items)

    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.55))

    scenarios = [
        {"name": "Equal (ideal)",           "amplitudes": np.array([60, 60, 60, 60, 60], dtype=float)},
        {"name": "Gradient best\u2192worst", "amplitudes": np.array([80, 70, 60, 50, 40], dtype=float)},
        {"name": "Gradient worst\u2192best", "amplitudes": np.array([40, 50, 60, 70, 80], dtype=float)},
        {"name": "Outlier pos 1",           "amplitudes": np.array([90, 50, 50, 50, 50], dtype=float)},
        {"name": "Outlier pos 5",           "amplitudes": np.array([50, 50, 50, 50, 90], dtype=float)},
        {"name": "Random realistic",        "amplitudes": rng.uniform(30, 80, size=5)},
    ]

    panel_labels = ["A", "B", "C", "D", "E", "F"]

    for idx, scenario in enumerate(scenarios):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        probs = simulate_heterogeneous_trial(
            n_items=n_items, isi_seconds=isi, n_trs=n_trs,
            class_amplitudes=scenario["amplitudes"], params=params,
        )
        slopes = compute_slope_timecourse(probs)

        # Probability timecourses — thin
        for i in range(n_items):
            ax.plot(t, probs[i], "-", color=item_colors[i], lw=0.7, alpha=0.5)

        # Slope on right axis — moderate weight
        ax2 = ax.twinx()
        ax2.plot(t, slopes, "o-", color=BLACK, lw=1.3, markersize=2.5, zorder=10)
        add_zero_line(ax2)
        ax2.fill_between(t, slopes, 0, where=slopes > 0,
                         alpha=FWD_ALPHA, color=FWD_COLOR, zorder=5)
        ax2.fill_between(t, slopes, 0, where=slopes < 0,
                         alpha=BWD_ALPHA, color=BWD_COLOR, zorder=5)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(LIGHT_GREY)
        ax2.tick_params(axis="y", colors=GREY, labelsize=6)
        ax2.set_ylabel("Slope", fontsize=6, color=GREY)

        # Scenario name inside panel top-center to avoid overlap with panel label
        ax.text(0.5, 0.97, scenario["name"], transform=ax.transAxes,
                fontsize=6, ha="center", va="top")
        ax.set_xlabel("Time (TRs)", fontsize=7)
        if col == 0:
            ax.set_ylabel("Prob. (%)", fontsize=7)
        ax.tick_params(labelsize=6)
        add_panel_label(ax, panel_labels[idx])

    fig.subplots_adjust(hspace=0.45, wspace=0.50)
    save_figure(fig, "sim2_heterogeneous_classifiers")
    plt.close()
    print("  Sim 2 done.")


# =============================================================================
# SIMULATION 3: Multiple replay events — overlapping responses
# =============================================================================

def sim3_multiple_events():
    """Show how multiple replay events within one window interact."""

    params = ResponseParams(amplitude=40, baseline=20)
    n_items = 4
    n_trs = 20
    t = np.arange(1, n_trs + 1, dtype=float)
    isi = 0.05

    fig, axes = plt.subplots(3, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.78))

    row_labels = ["Same direction", "Opposing direction", "Close timing"]

    scenarios = [
        {"onsets": [3.0],           "directions": [1],      "name": "1 fwd event"},
        {"onsets": [3.0, 7.0],      "directions": [1, 1],   "name": "2 fwd, 4 TR"},
        {"onsets": [2.0, 6.0, 10.0], "directions": [1, 1, 1], "name": "3 fwd, 4 TR"},
        {"onsets": [3.0, 7.0],      "directions": [1, -1],  "name": "Fwd+bwd, 4 TR"},
        {"onsets": [3.0, 5.0],      "directions": [1, -1],  "name": "Fwd+bwd, 2 TR"},
        {"onsets": [3.0, 9.0],      "directions": [1, -1],  "name": "Fwd+bwd, 6 TR"},
        {"onsets": [3.0, 4.0],      "directions": [1, 1],   "name": "2 fwd, 1 TR"},
        {"onsets": [3.0, 5.0],      "directions": [1, 1],   "name": "2 fwd, 2 TR"},
        {"onsets": [3.0, 6.0],      "directions": [1, 1],   "name": "2 fwd, 3 TR"},
    ]

    all_slopes_list = []
    for scenario in scenarios:
        probs = simulate_multiple_replay_events(
            n_items=n_items, isi_seconds=isi, n_trs=n_trs,
            event_onsets_trs=scenario["onsets"],
            event_directions=scenario["directions"],
            params=params,
        )
        all_slopes_list.append(compute_slope_timecourse(probs))
    global_ymax = max(np.max(np.abs(s)) for s in all_slopes_list) * 1.15

    panel_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    for idx, (scenario, slopes) in enumerate(zip(scenarios, all_slopes_list)):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        ax.plot(t, slopes, color=BLACK, lw=1.2)
        ax.fill_between(t, slopes, 0, where=slopes > 0,
                        alpha=FWD_ALPHA * 1.5, color=FWD_COLOR)
        ax.fill_between(t, slopes, 0, where=slopes < 0,
                        alpha=BWD_ALPHA * 1.5, color=BWD_COLOR)
        add_zero_line(ax)

        # Event onset markers — smaller
        for onset, dirn in zip(scenario["onsets"], scenario["directions"]):
            marker = "^" if dirn == 1 else "v"
            color = BLUE if dirn == 1 else ORANGE
            ax.plot(onset, global_ymax * 0.82 * (1 if dirn == 1 else -1),
                    marker=marker, color=color, markersize=6, zorder=10,
                    markeredgecolor="none")

        ax.set_ylim(-global_ymax, global_ymax)
        ax.set_xlim(0, n_trs + 1)
        ax.text(0.97, 0.97, scenario["name"], transform=ax.transAxes,
                fontsize=5, ha="right", va="top", color=GREY)
        ax.set_xlabel("Time (TRs)", fontsize=7)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel("SODA slope", fontsize=7)
            ax.text(-0.42, 0.5, row_labels[row], transform=ax.transAxes,
                    fontsize=6, fontweight="bold", rotation=90,
                    va="center", ha="center", color=GREY)
        add_panel_label(ax, panel_labels[idx])

    fig.subplots_adjust(hspace=0.45, wspace=0.35, left=0.10)
    save_figure(fig, "sim3_multiple_events")
    plt.close()
    print("  Sim 3 done.")


# =============================================================================
# SIMULATION 4: Trial averaging with onset jitter
# =============================================================================

def sim4_trial_averaging():
    """Show how trial averaging with jittered onsets kills the signal."""

    params = ResponseParams(amplitude=40, baseline=20)
    n_items = 4
    n_trs = 15
    t = np.arange(1, n_trs + 1, dtype=float)

    jitter_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    n_trials = 30

    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.85))
    gs = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.35,
                           height_ratios=[1, 1, 0.8])

    panel_labels = ["A", "B", "C", "D", "E", "F"]

    for idx, jitter_sd in enumerate(jitter_levels):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        _, onsets, all_slopes = simulate_trial_averaging(
            n_trials=n_trials, n_items=n_items,
            isi_seconds=0.05, n_trs=n_trs,
            onset_jitter_sd=jitter_sd,
            params=params, noise_sd=1.0,
            rng=np.random.default_rng(42),
        )

        # Individual trial slopes — very light
        for trial in range(min(n_trials, 20)):
            ax.plot(t, all_slopes[trial], "-", color=LIGHT_GREY,
                    alpha=0.20, lw=0.4)

        mean_slopes = np.mean(all_slopes, axis=0)
        sem_slopes = np.std(all_slopes, axis=0) / np.sqrt(n_trials)

        ax.fill_between(t, mean_slopes - sem_slopes,
                        mean_slopes + sem_slopes,
                        alpha=0.25, color=BLUE)
        ax.plot(t, mean_slopes, color=BLACK, lw=1.5)
        add_zero_line(ax)

        ax.text(0.97, 0.97, f"Jitter SD = {jitter_sd:.1f}",
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                color=GREY)
        ax.set_xlabel("Time (TRs)", fontsize=7)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel("SODA slope", fontsize=7)
        add_panel_label(ax, panel_labels[idx])

    # --- Panel G: Degradation curve ---
    ax_g = fig.add_subplot(gs[2, :2])
    jitters = np.linspace(0, 6, 25)
    ptps = []
    for j in jitters:
        _, _, all_slopes = simulate_trial_averaging(
            n_trials=50, n_items=4, isi_seconds=0.05, n_trs=n_trs,
            onset_jitter_sd=j, params=params, noise_sd=1.0,
            rng=np.random.default_rng(42),
        )
        ptps.append(np.ptp(np.mean(all_slopes, axis=0)))

    ax_g.plot(jitters, ptps, "o-", color=BLACK, lw=1.5, markersize=3)
    add_zero_line(ax_g)
    ax_g.set_xlabel("Onset jitter SD (TRs)", fontsize=8)
    ax_g.set_ylabel("Peak-to-peak", fontsize=8)
    ax_g.tick_params(labelsize=6)
    add_panel_label(ax_g, "G", fontsize=10)

    save_figure(fig, "sim4_trial_averaging")
    plt.close()
    print("  Sim 4 done.")


# =============================================================================
# Sequentiality illustration figure (Figure 0)
# =============================================================================

def fig0_sequentiality_illustration():
    """Create the conceptual sequentiality illustration (Figure 0)."""
    params = ResponseParams(amplitude=40, baseline=5, wavelength=5.26,
                            onset_delay=0.56)
    n_items = 6
    isi = 0.5
    n_trs = 10
    t_fine = np.linspace(0.5, n_trs + 0.5, 300)
    item_colors = get_item_colors(n_items)

    probs = multi_item_response(t_fine, n_items, isi_seconds=isi,
                                stimulus_duration=0.1, tr=1.25, params=params)

    tr_fwd = 3.5
    tr_bwd = 7.5

    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.35))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1, 1], wspace=0.45)

    # --- Panel A ---
    ax_a = fig.add_subplot(gs[0, 0])
    for i in range(n_items):
        ax_a.plot(t_fine, probs[i], color=item_colors[i], lw=1.2,
                  label=f"Event {i+1}")

    for tr_sample in [tr_fwd, tr_bwd]:
        ax_a.axvline(tr_sample, color=GREY, lw=0.6, ls=":", alpha=0.5)
        ax_a.axvspan(tr_sample - 0.25, tr_sample + 0.25, alpha=0.06,
                     color=GREY, zorder=0)

    ax_a.set_xlabel("Time (in TRs; 1 TR = 1.25 s)", fontsize=7)
    ax_a.set_ylabel("Probability (%)", fontsize=7)
    ax_a.legend(fontsize=5, loc="upper right", ncol=3, handlelength=1.0,
                columnspacing=0.6, borderpad=0.3)
    add_panel_label(ax_a, "A")

    from scipy import stats as sp_stats

    # --- Panel B ---
    ax_b = fig.add_subplot(gs[0, 1])
    idx_fwd = np.argmin(np.abs(t_fine - tr_fwd))
    probs_fwd = probs[:, idx_fwd]
    positions = np.arange(n_items, 0, -1)

    for i in range(n_items):
        ax_b.scatter(positions[i], probs_fwd[i], color=item_colors[i],
                     s=30, zorder=5, edgecolors=BLACK, linewidth=0.3)

    slope_val, intercept, _, _, _ = sp_stats.linregress(positions, probs_fwd)
    x_line = np.array([positions.min(), positions.max()])
    ax_b.plot(x_line, slope_val * x_line + intercept, color=BLACK, lw=1.0)
    ax_b.set_xlabel("Serial position", fontsize=7)
    ax_b.set_ylabel("Prob. (%)", fontsize=7)
    ax_b.set_title("Forward", fontsize=8, pad=8)
    ax_b.tick_params(labelsize=6)
    add_panel_label(ax_b, "B", x=-0.25)

    # --- Panel C ---
    ax_c = fig.add_subplot(gs[0, 2])
    idx_bwd = np.argmin(np.abs(t_fine - tr_bwd))
    probs_bwd = probs[:, idx_bwd]

    for i in range(n_items):
        ax_c.scatter(positions[i], probs_bwd[i], color=item_colors[i],
                     s=30, zorder=5, edgecolors=BLACK, linewidth=0.3)

    slope_val2, intercept2, _, _, _ = sp_stats.linregress(positions, probs_bwd)
    ax_c.plot(x_line, slope_val2 * x_line + intercept2, color=BLACK, lw=1.0)
    ax_c.set_xlabel("Serial position", fontsize=7)
    ax_c.set_ylabel("Prob. (%)", fontsize=7)
    ax_c.set_title("Backward", fontsize=8, pad=8)
    ax_c.tick_params(labelsize=6)
    add_panel_label(ax_c, "C", x=-0.25)

    save_figure(fig, "sequentiality_illustration")
    plt.close()
    print("  Figure 0 (sequentiality illustration) done.")


# =============================================================================
# Run all simulations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SODA Primer Simulations")
    print("=" * 60)

    print("\n--- Figure 0: Sequentiality illustration ---")
    fig0_sequentiality_illustration()

    print("\n--- Simulation 1: Ideal case ---")
    sim1_ideal_case()

    print("\n--- Simulation 2: Heterogeneous classifiers ---")
    sim2_heterogeneous_classifiers()

    print("\n--- Simulation 3: Multiple replay events ---")
    sim3_multiple_events()

    print("\n--- Simulation 4: Trial averaging ---")
    sim4_trial_averaging()

    print("\n" + "=" * 60)
    print("All figures saved to Figures/")
    print("=" * 60)
