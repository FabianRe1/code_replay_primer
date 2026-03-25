"""
Simulation 5: Aggregation Metric Sensitivity Comparison

Systematically compares all 7 SODA aggregation metrics under controlled
conditions to determine which is most sensitive to true replay signal.

Design:
  - For each condition, simulate N_TRIALS trials with a true replay event
    embedded in noise, plus N_TRIALS null trials (noise only).
  - Compute all 7 metrics for each trial.
  - Measure sensitivity via d-prime (separation between signal and null).
  
Conditions tested:
  A. SNR sweep: varying signal amplitude relative to noise
  B. Onset jitter: fixed SNR, varying replay timing across trials
  C. Heterogeneous classifiers: fixed SNR, varying class amplitudes
  D. Multiple events per trial: 1 vs 2 events (same or opposing direction)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_model import ResponseParams, multi_item_response
from soda import compute_slope_timecourse
from aggregation import (
    compute_all_metrics, METRIC_NAMES, METRIC_LABELS,
)
from viz_style import (
    setup_style, add_panel_label, save_figure, get_metric_colors,
    metric_color, metric_label, add_zero_line, annotated_heatmap,
    make_diverging_cmap, FIGURES_DIR, FULL_WIDTH, BLUE, ORANGE, BLACK, GREY,
)

setup_style()

# Mapping from aggregation.py metric keys to viz_style metric keys
_AGG_TO_VIZ = {
    'mean_slope': 'mean_slope',
    'abs_mean': 'abs_mean',
    'variance': 'slope_variance',
    'peak_to_trough': 'peak_to_trough',
    'spectral_power': 'spectral_power',
    'continuous_sin_amplitude': 'cont_sin_amplitude',
    'windowed_sin_amplitude': 'win_sin_amplitude',
}


def _metric_color(agg_name):
    """Get viz_style color for an aggregation metric name."""
    return metric_color(_AGG_TO_VIZ.get(agg_name, agg_name))


def _metric_label(agg_name):
    """Get viz_style display label for an aggregation metric name."""
    return metric_label(_AGG_TO_VIZ.get(agg_name, agg_name))


# =====================================================================
# Helper: simulate one trial (signal or null)
# =====================================================================

def simulate_one_trial(
    has_signal: bool,
    n_items: int = 5,
    n_trs: int = 13,
    isi_seconds: float = 0.05,
    tr: float = 1.25,
    signal_amplitude: float = 40.0,
    noise_sd: float = 3.0,
    onset_tr: float = 2.0,
    class_amplitudes: np.ndarray = None,
    rng: np.random.Generator = None,
    # Multiple events
    n_events: int = 1,
    event_spacing_trs: float = 5.0,
    second_event_direction: int = 1,  # +1 same, -1 opposing
) -> np.ndarray:
    """
    Simulate one trial and return the SODA slope timecourse.
    
    If has_signal=False, returns pure noise slopes.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    params = ResponseParams(
        amplitude=signal_amplitude, baseline=20.0,
        wavelength=5.26, onset_delay=0.56
    )
    t = np.arange(1, n_trs + 1, dtype=float)
    
    if has_signal:
        # Build probability timecourses from overlapping HRFs
        total_probs = np.full((n_items, len(t)), params.baseline, dtype=float)
        
        for ev_idx in range(n_events):
            ev_onset = onset_tr + ev_idx * event_spacing_trs
            t_relative = t - ev_onset
            
            amps = class_amplitudes if class_amplitudes is not None else None
            ev_probs = multi_item_response(
                t_relative, n_items, isi_seconds, 0.0, tr, params,
                class_amplitudes=amps,
            )
            
            # Handle direction for 2nd+ events
            if ev_idx > 0 and second_event_direction == -1:
                ev_probs = ev_probs[::-1]
            
            total_probs += (ev_probs - params.baseline)
        
        # Add noise
        total_probs += rng.normal(0, noise_sd, total_probs.shape)
        total_probs = np.clip(total_probs, 0, 100)
    else:
        # Null trial: baseline + noise
        total_probs = params.baseline + rng.normal(0, noise_sd, (n_items, len(t)))
        total_probs = np.clip(total_probs, 0, 100)
    
    # Compute SODA slopes
    slopes = compute_slope_timecourse(total_probs)
    return slopes


def dprime(signal_vals, null_vals):
    """d-prime: (mean_signal - mean_null) / pooled_std"""
    ms, mn = np.nanmean(signal_vals), np.nanmean(null_vals)
    ss, sn = np.nanstd(signal_vals, ddof=1), np.nanstd(null_vals, ddof=1)
    pooled = np.sqrt((ss**2 + sn**2) / 2)
    return (ms - mn) / pooled if pooled > 0 else 0.0


# =====================================================================
# Condition A: SNR sweep
# =====================================================================

def condition_A_snr_sweep(n_trials=100, seed=42):
    """Vary signal amplitude while keeping noise fixed."""
    
    amplitudes = np.array([0, 5, 10, 15, 20, 30, 40, 50, 60])
    noise_sd = 5.0
    n_trs = 13
    
    results = {name: [] for name in METRIC_NAMES}
    
    for amp in amplitudes:
        rng = np.random.default_rng(seed)
        
        signal_metrics = {name: [] for name in METRIC_NAMES}
        null_metrics = {name: [] for name in METRIC_NAMES}
        
        for _ in range(n_trials):
            # Signal trial
            slopes_s = simulate_one_trial(True, signal_amplitude=amp,
                                         noise_sd=noise_sd, rng=rng)
            trs = np.arange(1, n_trs + 1, dtype=float)
            m_s = compute_all_metrics(trs, slopes_s)
            
            # Null trial
            slopes_n = simulate_one_trial(False, noise_sd=noise_sd, rng=rng)
            m_n = compute_all_metrics(trs, slopes_n)
            
            for name in METRIC_NAMES:
                signal_metrics[name].append(m_s[name])
                null_metrics[name].append(m_n[name])
        
        for name in METRIC_NAMES:
            dp = dprime(np.array(signal_metrics[name]),
                       np.array(null_metrics[name]))
            results[name].append(dp)
    
    return amplitudes, results


def condition_B_onset_jitter(n_trials=100, seed=42):
    """Fixed SNR, vary onset jitter across trials."""
    
    jitter_sds = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    signal_amplitude = 30.0
    noise_sd = 4.0
    n_trs = 13
    
    results = {name: [] for name in METRIC_NAMES}
    
    for jitter in jitter_sds:
        rng = np.random.default_rng(seed)
        
        signal_metrics = {name: [] for name in METRIC_NAMES}
        null_metrics = {name: [] for name in METRIC_NAMES}
        
        for _ in range(n_trials):
            onset = max(1.0, 2.0 + rng.normal(0, jitter))
            
            slopes_s = simulate_one_trial(True, signal_amplitude=signal_amplitude,
                                         noise_sd=noise_sd, onset_tr=onset, rng=rng)
            slopes_n = simulate_one_trial(False, noise_sd=noise_sd, rng=rng)
            
            trs = np.arange(1, n_trs + 1, dtype=float)
            m_s = compute_all_metrics(trs, slopes_s)
            m_n = compute_all_metrics(trs, slopes_n)
            
            for name in METRIC_NAMES:
                signal_metrics[name].append(m_s[name])
                null_metrics[name].append(m_n[name])
        
        for name in METRIC_NAMES:
            dp = dprime(np.array(signal_metrics[name]),
                       np.array(null_metrics[name]))
            results[name].append(dp)
    
    return jitter_sds, results


def condition_C_heterogeneous_classifiers(n_trials=100, seed=42):
    """Fixed SNR, vary degree of classifier heterogeneity."""
    
    # heterogeneity_levels: SD of class amplitudes around the mean
    het_sds = np.array([0, 3, 5, 8, 10, 15, 20, 25])
    mean_amp = 35.0
    noise_sd = 4.0
    n_items = 5
    n_trs = 13
    
    results = {name: [] for name in METRIC_NAMES}
    
    for het_sd in het_sds:
        rng = np.random.default_rng(seed)
        
        signal_metrics = {name: [] for name in METRIC_NAMES}
        null_metrics = {name: [] for name in METRIC_NAMES}
        
        for _ in range(n_trials):
            # Draw class amplitudes for this trial
            if het_sd == 0:
                class_amps = np.full(n_items, mean_amp)
            else:
                class_amps = rng.normal(mean_amp, het_sd, n_items)
                class_amps = np.clip(class_amps, 10, 80)
            
            slopes_s = simulate_one_trial(True, signal_amplitude=mean_amp,
                                         noise_sd=noise_sd,
                                         class_amplitudes=class_amps, rng=rng)
            slopes_n = simulate_one_trial(False, noise_sd=noise_sd, rng=rng)
            
            trs = np.arange(1, n_trs + 1, dtype=float)
            m_s = compute_all_metrics(trs, slopes_s)
            m_n = compute_all_metrics(trs, slopes_n)
            
            for name in METRIC_NAMES:
                signal_metrics[name].append(m_s[name])
                null_metrics[name].append(m_n[name])
        
        for name in METRIC_NAMES:
            dp = dprime(np.array(signal_metrics[name]),
                       np.array(null_metrics[name]))
            results[name].append(dp)
    
    return het_sds, results


def condition_D_multiple_events(n_trials=100, seed=42):
    """Compare 1 event vs 2 events (same/opposing direction)."""
    
    scenarios = [
        {'label': '1 event fwd', 'n_events': 1, 'direction': 1},
        {'label': '2 events fwd+fwd', 'n_events': 2, 'direction': 1},
        {'label': '2 events fwd+bwd', 'n_events': 2, 'direction': -1},
    ]
    
    signal_amplitude = 30.0
    noise_sd = 4.0
    n_trs = 15
    
    all_results = {}
    
    for scenario in scenarios:
        rng = np.random.default_rng(seed)
        
        signal_metrics = {name: [] for name in METRIC_NAMES}
        null_metrics = {name: [] for name in METRIC_NAMES}
        
        for _ in range(n_trials):
            slopes_s = simulate_one_trial(
                True, signal_amplitude=signal_amplitude,
                noise_sd=noise_sd, n_trs=n_trs,
                n_events=scenario['n_events'],
                event_spacing_trs=5.0,
                second_event_direction=scenario['direction'],
                rng=rng,
            )
            slopes_n = simulate_one_trial(False, noise_sd=noise_sd,
                                         n_trs=n_trs, rng=rng)
            
            trs = np.arange(1, n_trs + 1, dtype=float)
            m_s = compute_all_metrics(trs, slopes_s)
            m_n = compute_all_metrics(trs, slopes_n)
            
            for name in METRIC_NAMES:
                signal_metrics[name].append(m_s[name])
                null_metrics[name].append(m_n[name])
        
        dprimes = {}
        for name in METRIC_NAMES:
            dprimes[name] = dprime(np.array(signal_metrics[name]),
                                   np.array(null_metrics[name]))
        all_results[scenario['label']] = dprimes
    
    return scenarios, all_results


# =====================================================================
# Plotting
# =====================================================================

def plot_sensitivity_curves(x_values, results, xlabel, title, filename,
                            panel_label=None):
    """Plot d-prime curves for all metrics."""

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.55))

    for name in METRIC_NAMES:
        ax.plot(x_values, results[name], 'o-', markersize=4,
                color=_metric_color(name), label=_metric_label(name))

    add_zero_line(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("d\u2032 (sensitivity)")
    # Title as in-panel text to avoid overlap with panel label
    ax.text(0.97, 0.97, title, transform=ax.transAxes, fontsize=7,
            ha='right', va='top', color=GREY)
    ax.legend(loc='upper left', ncol=2, fontsize=5, handlelength=1.0,
              borderpad=0.3, labelspacing=0.2, columnspacing=0.8)

    if panel_label:
        add_panel_label(ax, panel_label)

    fig.tight_layout()
    save_figure(fig, filename.replace('.png', ''))
    plt.close()


def plot_bar_comparison(scenarios, all_results, filename, panel_label=None):
    """Bar chart comparing metrics across event scenarios."""

    labels = [s['label'] for s in scenarios]
    n_scenarios = len(labels)
    n_metrics = len(METRIC_NAMES)

    x = np.arange(n_scenarios)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.55))

    for i, name in enumerate(METRIC_NAMES):
        vals = [all_results[label][name] for label in labels]
        offset = (i - n_metrics/2 + 0.5) * width
        ax.bar(x + offset, vals, width, color=_metric_color(name),
               label=_metric_label(name), edgecolor='white', lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("d\u2032 (sensitivity)")
    ax.text(0.97, 0.97, "Single vs multiple replay events",
            transform=ax.transAxes, fontsize=7, ha='right', va='top', color=GREY)
    ax.legend(ncol=2, loc='upper left', fontsize=5.5, handlelength=1.2)
    add_zero_line(ax)

    if panel_label:
        add_panel_label(ax, panel_label)

    fig.tight_layout()
    save_figure(fig, filename.replace('.png', ''))
    plt.close()


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    N_TRIALS = 40  # per condition
    
    print("=" * 60)
    print("Simulation 5: Aggregation Metric Sensitivity Comparison")
    print("=" * 60)
    
    print(f"\nUsing {N_TRIALS} trials per condition")
    
    print("\n--- Condition A: SNR sweep ---")
    amps, res_A = condition_A_snr_sweep(n_trials=N_TRIALS)
    plot_sensitivity_curves(amps, res_A,
                           'Signal amplitude (%)',
                           "SNR sweep",
                           'sim5_A_snr_sweep.png',
                           panel_label='A')

    print("\n--- Condition B: Onset jitter ---")
    jitters, res_B = condition_B_onset_jitter(n_trials=N_TRIALS)
    plot_sensitivity_curves(jitters, res_B,
                           'Onset jitter SD (TRs)',
                           "Onset jitter",
                           'sim5_B_onset_jitter.png',
                           panel_label='B')

    print("\n--- Condition C: Classifier heterogeneity ---")
    hets, res_C = condition_C_heterogeneous_classifiers(n_trials=N_TRIALS)
    plot_sensitivity_curves(hets, res_C,
                           'Classifier amplitude SD',
                           "Classifier heterogeneity",
                           'sim5_C_heterogeneity.png',
                           panel_label='C')

    print("\n--- Condition D: Multiple events ---")
    scenarios, res_D = condition_D_multiple_events(n_trials=N_TRIALS)
    plot_bar_comparison(scenarios, res_D, 'sim5_D_multiple_events.png',
                        panel_label='D')

    # ---- Summary heatmap ----
    print("\n--- Summary heatmap ---")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.5))

    # Collect representative d' from each condition
    summary_data = np.zeros((len(METRIC_NAMES), 4))
    condition_labels = ['SNR', 'Jitter', 'Heterog.', 'Multi-event']
    row_labels = [_metric_label(n) for n in METRIC_NAMES]

    for i, name in enumerate(METRIC_NAMES):
        summary_data[i, 0] = res_A[name][5]   # amp=30
        summary_data[i, 1] = res_B[name][3]   # jitter=1.0
        summary_data[i, 2] = res_C[name][4]   # het_sd=10
        summary_data[i, 3] = res_D['2 events fwd+bwd'][name]

    # Diverging blue-white-red colormap centered at 0
    im = annotated_heatmap(ax, summary_data, row_labels, condition_labels)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("d' (sensitivity)")
    # No set_title — the heatmap content is self-explanatory

    fig.tight_layout()
    save_figure(fig, 'sim5_summary_heatmap')
    plt.close()
    
    print("\n" + "=" * 60)
    print("Done! All Simulation 5 figures saved.")
    print("=" * 60)
