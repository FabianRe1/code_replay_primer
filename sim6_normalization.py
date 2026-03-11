"""
Simulation 6: Classifier Normalization Strategies

When classifiers have unequal performance across classes, the SODA slope
is biased by amplitude differences rather than true sequential ordering.

Question: Can we normalize classifier probability timecourses *before*
computing the slope to remove this bias?

Normalization strategies tested:
  1. None (raw probabilities — baseline)
  2. Z-score per class: subtract mean, divide by SD across TRs within trial
  3. Min-max per class: scale each class to [0, 1] within trial
  4. Rank transform per TR: replace probabilities with ranks at each TR
  5. Divide by peak: scale each class by its maximum probability in trial

For each strategy, we compute SODA slopes and then all aggregation metrics,
comparing sensitivity (d') under heterogeneous vs homogeneous classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import rankdata

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_model import ResponseParams, multi_item_response
from soda import compute_slope_timecourse
from aggregation import (
    compute_all_metrics, METRIC_NAMES, METRIC_LABELS,
)

FIGDIR = Path(__file__).parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# =====================================================================
# Normalization strategies
# =====================================================================

def normalize_none(probs):
    """No normalization — return raw probabilities."""
    return probs.copy()


def normalize_zscore(probs):
    """Z-score each class across TRs within trial."""
    normed = np.zeros_like(probs, dtype=float)
    for i in range(probs.shape[0]):
        row = probs[i]
        mu, sd = np.mean(row), np.std(row)
        normed[i] = (row - mu) / sd if sd > 0 else row - mu
    return normed


def normalize_minmax(probs):
    """Min-max scale each class to [0, 1] within trial."""
    normed = np.zeros_like(probs, dtype=float)
    for i in range(probs.shape[0]):
        row = probs[i]
        mn, mx = np.min(row), np.max(row)
        normed[i] = (row - mn) / (mx - mn) if mx > mn else np.zeros_like(row)
    return normed


def normalize_rank_per_tr(probs):
    """At each TR, replace probabilities with their rank across classes."""
    normed = np.zeros_like(probs, dtype=float)
    for t in range(probs.shape[1]):
        normed[:, t] = rankdata(probs[:, t])
    return normed


def normalize_divide_by_peak(probs):
    """Divide each class by its peak probability in the trial."""
    normed = np.zeros_like(probs, dtype=float)
    for i in range(probs.shape[0]):
        peak = np.max(probs[i])
        normed[i] = probs[i] / peak if peak > 0 else probs[i]
    return normed


NORMALIZATIONS = {
    'None (raw)': normalize_none,
    'Z-score per class': normalize_zscore,
    'Min-max per class': normalize_minmax,
    'Rank per TR': normalize_rank_per_tr,
    'Divide by peak': normalize_divide_by_peak,
}


# =====================================================================
# Simulation helper
# =====================================================================

def simulate_trial(has_signal, n_items, n_trs, isi_seconds, tr,
                   signal_amplitude, noise_sd, class_amplitudes, rng):
    """Generate probability timecourses for one trial."""
    params = ResponseParams(amplitude=signal_amplitude, baseline=20.0,
                            wavelength=5.26, onset_delay=0.56)
    t = np.arange(1, n_trs + 1, dtype=float)

    if has_signal:
        probs = multi_item_response(
            t, n_items, isi_seconds, 0.0, tr, params,
            class_amplitudes=class_amplitudes,
        )
        probs += rng.normal(0, noise_sd, probs.shape)
        probs = np.clip(probs, 0, 100)
    else:
        probs = 20.0 + rng.normal(0, noise_sd, (n_items, len(t)))
        probs = np.clip(probs, 0, 100)

    return probs


def dprime(sig, null):
    ms, mn = np.nanmean(sig), np.nanmean(null)
    ss, sn = np.nanstd(sig, ddof=1), np.nanstd(null, ddof=1)
    pooled = np.sqrt((ss**2 + sn**2) / 2)
    return (ms - mn) / pooled if pooled > 0 else 0.0


# =====================================================================
# Main simulation
# =====================================================================

def run_normalization_comparison(n_trials=50, seed=42):
    """
    Compare normalization strategies across levels of classifier heterogeneity.
    
    For each heterogeneity level × normalization strategy:
      - Simulate signal and null trials
      - Apply normalization to probabilities
      - Compute SODA slopes on normalized probabilities
      - Compute all aggregation metrics
      - Calculate d' for each metric
    """
    het_sds = np.array([0, 5, 10, 15, 20, 25])
    mean_amp = 35.0
    noise_sd = 4.0
    n_items = 5
    n_trs = 13
    isi_seconds = 0.05
    tr = 1.25

    # Focus on the most informative metrics
    focus_metrics = ['mean_slope', 'abs_mean', 'variance',
                     'peak_to_trough']

    # results[norm_name][metric_name] = list of d' values over het_sds
    results = {
        norm_name: {m: [] for m in focus_metrics}
        for norm_name in NORMALIZATIONS
    }

    for het_sd in het_sds:
        print(f"  Heterogeneity SD = {het_sd}")

        for norm_name, norm_fn in NORMALIZATIONS.items():
            rng = np.random.default_rng(seed)

            signal_metrics = {m: [] for m in focus_metrics}
            null_metrics = {m: [] for m in focus_metrics}

            for _ in range(n_trials):
                # Draw class amplitudes
                if het_sd == 0:
                    class_amps = np.full(n_items, mean_amp)
                else:
                    class_amps = rng.normal(mean_amp, het_sd, n_items)
                    class_amps = np.clip(class_amps, 10, 80)

                # Signal trial
                probs_s = simulate_trial(
                    True, n_items, n_trs, isi_seconds, tr,
                    mean_amp, noise_sd, class_amps, rng)
                probs_s_norm = norm_fn(probs_s)
                slopes_s = compute_slope_timecourse(probs_s_norm)
                trs_arr = np.arange(1, n_trs + 1, dtype=float)
                m_s = {}
                for m in focus_metrics:
                    if m == 'mean_slope':
                        from aggregation import mean_slope as _ms
                        m_s[m] = _ms(slopes_s)
                    elif m == 'abs_mean':
                        from aggregation import abs_mean as _am
                        m_s[m] = _am(slopes_s)
                    elif m == 'variance':
                        from aggregation import slope_variance as _sv
                        m_s[m] = _sv(slopes_s)
                    elif m == 'peak_to_trough':
                        from aggregation import peak_to_trough as _ptt
                        m_s[m] = _ptt(slopes_s)

                # Null trial
                probs_n = simulate_trial(
                    False, n_items, n_trs, isi_seconds, tr,
                    mean_amp, noise_sd, class_amps, rng)
                probs_n_norm = norm_fn(probs_n)
                slopes_n = compute_slope_timecourse(probs_n_norm)
                m_n = {}
                for m in focus_metrics:
                    if m == 'mean_slope':
                        m_n[m] = _ms(slopes_n)
                    elif m == 'abs_mean':
                        m_n[m] = _am(slopes_n)
                    elif m == 'variance':
                        m_n[m] = _sv(slopes_n)
                    elif m == 'peak_to_trough':
                        m_n[m] = _ptt(slopes_n)

                for m in focus_metrics:
                    signal_metrics[m].append(m_s[m])
                    null_metrics[m].append(m_n[m])

            for m in focus_metrics:
                dp = dprime(np.array(signal_metrics[m]),
                           np.array(null_metrics[m]))
                results[norm_name][m].append(dp)

    return het_sds, results, focus_metrics


def plot_normalization_by_metric(het_sds, results, focus_metrics):
    """One panel per metric, lines = normalization strategies."""

    norm_colors = {
        'None (raw)': '#333333',
        'Z-score per class': '#e41a1c',
        'Min-max per class': '#377eb8',
        'Rank per TR': '#4daf4a',
        'Divide by peak': '#ff7f00',
    }

    n_metrics = len(focus_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=True)

    for ax, metric in zip(axes, focus_metrics):
        for norm_name in NORMALIZATIONS:
            ax.plot(het_sds, results[norm_name][metric], 'o-', lw=2,
                    markersize=5, color=norm_colors[norm_name], label=norm_name)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Classifier amplitude SD')
        ax.set_title(METRIC_LABELS.get(metric, metric).replace('\n', ' '),
                     fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel("d' (sensitivity)")
    axes[-1].legend(fontsize=7, loc='best', title='Normalization')

    fig.suptitle("Effect of Probability Normalization on SODA Sensitivity\n"
                 "across Classifier Heterogeneity Levels",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim6_normalization_by_metric.png', dpi=300)
    plt.close()
    print(f"✓ Saved sim6_normalization_by_metric.png")


def plot_normalization_summary_heatmap(het_sds, results, focus_metrics):
    """Heatmap: normalization × metric, at a representative heterogeneity level."""

    # Pick het_sd = 15 as representative high-heterogeneity case
    het_idx = 3  # index for SD=15

    norm_names = list(NORMALIZATIONS.keys())
    data = np.zeros((len(norm_names), len(focus_metrics)))

    for i, norm_name in enumerate(norm_names):
        for j, metric in enumerate(focus_metrics):
            data[i, j] = results[norm_name][metric][het_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=1.5)

    ax.set_yticks(range(len(norm_names)))
    ax.set_yticklabels(norm_names)
    ax.set_xticks(range(len(focus_metrics)))
    ax.set_xticklabels([METRIC_LABELS.get(m, m).replace('\n', ' ')
                        for m in focus_metrics], rotation=20, ha='right')

    for i in range(len(norm_names)):
        for j in range(len(focus_metrics)):
            color = 'white' if data[i, j] > 1.0 or data[i, j] < -0.2 else 'black'
            ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    plt.colorbar(im, label="d' (sensitivity)")
    ax.set_title(f"Normalization × Metric Sensitivity (classifier SD = {het_sds[het_idx]})")
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim6_normalization_heatmap.png', dpi=300)
    plt.close()
    print(f"✓ Saved sim6_normalization_heatmap.png")


def plot_example_normalization_effect():
    """
    Show what normalization does to probability timecourses from one trial.
    Side-by-side: raw vs each normalization strategy.
    """
    rng = np.random.default_rng(42)
    n_items, n_trs = 5, 13
    params = ResponseParams(amplitude=35.0, baseline=20.0)
    t = np.arange(1, n_trs + 1, dtype=float)

    # Strongly heterogeneous classifiers
    class_amps = np.array([75, 55, 40, 30, 60], dtype=float)

    probs = multi_item_response(t, n_items, 0.05, 0.0, 1.25, params,
                                 class_amplitudes=class_amps)
    probs += rng.normal(0, 2.0, probs.shape)
    probs = np.clip(probs, 0, 100)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    item_colors = plt.cm.Set2(np.linspace(0, 1, 8))[:n_items]

    for idx, (norm_name, norm_fn) in enumerate(NORMALIZATIONS.items()):
        ax = axes[idx]
        normed = norm_fn(probs)

        for i in range(n_items):
            ax.plot(t, normed[i], 'o-', color=item_colors[i], lw=1.5,
                    markersize=4, label=f'Item {i+1} (A={class_amps[i]:.0f})')

        slopes = compute_slope_timecourse(normed)

        ax2 = ax.twinx()
        ax2.plot(t, slopes, 'k-', lw=2.5, alpha=0.7)
        ax2.fill_between(t, slopes, 0, where=slopes > 0, alpha=0.1, color='blue')
        ax2.fill_between(t, slopes, 0, where=slopes < 0, alpha=0.1, color='red')
        ax2.set_ylabel('SODA slope', fontsize=8, color='gray')

        ax.set_title(norm_name, fontsize=10)
        ax.set_xlabel('Time (TRs)')
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    # Hide the 6th panel
    axes[5].axis('off')

    fig.suptitle("Effect of Normalization on Probability Timecourses\n"
                 "Heterogeneous classifiers: amplitudes = [75, 55, 40, 30, 60]",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim6_example_normalization.png', dpi=300)
    plt.close()
    print(f"✓ Saved sim6_example_normalization.png")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Simulation 6: Classifier Normalization Strategies")
    print("=" * 60)

    print("\n--- Example normalization effect ---")
    plot_example_normalization_effect()

    print("\n--- Running normalization comparison ---")
    het_sds, results, focus_metrics = run_normalization_comparison(n_trials=40)

    print("\n--- Plotting results ---")
    plot_normalization_by_metric(het_sds, results, focus_metrics)
    plot_normalization_summary_heatmap(het_sds, results, focus_metrics)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
