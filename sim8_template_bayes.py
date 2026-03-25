"""
Simulation 8: Template Matching & Bayesian Model Comparison

Two new approaches to replay detection that go beyond fitting the slope
timecourse post-hoc:

1. TEMPLATE MATCHING
   Generate a predicted slope timecourse from the Wittkuhn response model
   (given HRF parameters and assumed sequence speed), then correlate this
   template with the observed slope timecourse. The correlation (or its
   Fisher-z transform) serves as a single-trial replay metric.
   
   Advantage: exploits the full expected waveform shape (including
   asymmetry), not just sinusoidal amplitude.
   
   Extension: slide the template across onset times to find the best-
   matching onset → template correlation at optimal lag.

2. BAYESIAN MODEL COMPARISON
   For each trial, compute the marginal likelihood (or BIC approximation)
   under several models:
     M0: Null — slopes are i.i.d. Gaussian noise (no replay)
     M1: Forward replay — one-cycle sinusoid with constrained phase
     M2: Backward replay — one-cycle sinusoid with opposite phase
     M3: Any replay — one-cycle sinusoid with free phase
   
   The Bayes factor BF = p(data|M3) / p(data|M0) directly answers
   "how much more likely is replay than no replay?"

Both approaches are compared to the metrics from Simulation 5 using
the same d-prime framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, linregress
from scipy.optimize import minimize_scalar

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_model import (
    ResponseParams, single_event_response, multi_item_response,
    compute_periods, sequence_delta,
)
from soda import compute_slope_timecourse
from aggregation import (
    mean_slope, abs_mean, slope_variance, peak_to_trough,
    fit_windowed_sinusoid, METRIC_LABELS,
)
from viz_style import (
    setup_style, add_panel_label, save_figure, get_metric_colors, metric_color,
    FIGURES_DIR, FULL_WIDTH, BLUE, ORANGE, BLACK, GREY,
)

setup_style()


# =====================================================================
# 1. TEMPLATE MATCHING
# =====================================================================

def generate_soda_template(
    n_trs: int = 13,
    n_items: int = 5,
    isi_seconds: float = 0.05,
    onset_tr: float = 0.0,
    params: ResponseParams = None,
) -> np.ndarray:
    """
    Generate the expected SODA slope timecourse for a replay event.
    
    This is the "ideal" timecourse under the Wittkuhn model: simulate
    probabilities from overlapping HRFs, then compute slopes.
    """
    if params is None:
        params = ResponseParams(amplitude=1.0, baseline=0.0)  # unit amplitude
    
    t = np.arange(1, n_trs + 1, dtype=float) - onset_tr
    probs = multi_item_response(t, n_items, isi_seconds, 0.0, 1.25, params)
    slopes = compute_slope_timecourse(probs)
    
    # Normalize template to unit norm
    norm = np.linalg.norm(slopes)
    if norm > 0:
        slopes = slopes / norm
    
    return slopes


def template_correlation(
    observed_slopes: np.ndarray,
    template: np.ndarray,
) -> float:
    """Pearson correlation between observed slopes and template."""
    valid = ~(np.isnan(observed_slopes) | np.isnan(template))
    if valid.sum() < 4:
        return np.nan
    r, _ = pearsonr(observed_slopes[valid], template[valid])
    return r


def template_correlation_optimal_lag(
    observed_slopes: np.ndarray,
    n_items: int = 5,
    isi_seconds: float = 0.05,
    lag_range: tuple = (-2.0, 6.0),
    n_lags: int = 30,
) -> dict:
    """
    Find the onset lag that maximizes template-data correlation.
    
    Slides the template across different onset times and returns
    the best correlation and corresponding lag.
    """
    n_trs = len(observed_slopes)
    lags = np.linspace(lag_range[0], lag_range[1], n_lags)
    
    best_r = -np.inf
    best_lag = 0.0
    all_rs = []
    
    for lag in lags:
        template = generate_soda_template(
            n_trs=n_trs, n_items=n_items,
            isi_seconds=isi_seconds, onset_tr=lag,
        )
        r = template_correlation(observed_slopes, template)
        all_rs.append(r)
        if not np.isnan(r) and r > best_r:
            best_r = r
            best_lag = lag
    
    return {
        'best_correlation': best_r,
        'best_lag': best_lag,
        'all_correlations': np.array(all_rs),
        'lags': lags,
    }


# =====================================================================
# 2. BAYESIAN MODEL COMPARISON (BIC approximation)
# =====================================================================

def log_likelihood_null(slopes):
    """M0: slopes are i.i.d. Gaussian. MLE: mean and variance of data."""
    n = len(slopes)
    mu = np.mean(slopes)
    sigma2 = np.var(slopes)
    if sigma2 <= 0:
        return -np.inf
    ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * n
    return ll


def log_likelihood_sinusoid(slopes, trs, constrain_phase=None):
    """
    M1/M2/M3: one-cycle windowed sinusoid + Gaussian noise.
    
    constrain_phase:
        None → free phase (M3: any replay)
        'forward' → phase constrained to produce positive-then-negative (M1)
        'backward' → phase constrained to produce negative-then-positive (M2)
    """
    from aggregation import _windowed_for_curvefit
    from scipy.optimize import curve_fit
    
    n = len(slopes)
    t_min, t_max = trs.min(), trs.max()
    y_range = np.ptp(slopes)
    y_mean = np.mean(slopes)
    a0 = max(y_range / 2, 0.001)
    
    # Bounds depend on phase constraint
    if constrain_phase == 'forward':
        lower = [0.0, 3.0, t_min - 1.0, -np.inf]  # positive amplitude = forward first
        upper = [np.inf, 12.0, t_max, np.inf]
    elif constrain_phase == 'backward':
        lower = [-np.inf, 3.0, t_min - 1.0, -np.inf]  # negative amplitude = backward first
        upper = [0.0, 12.0, t_max, np.inf]
    else:
        lower = [-np.inf, 3.0, t_min - 1.0, -np.inf]
        upper = [np.inf, 12.0, t_max, np.inf]
    
    best_ll = -np.inf
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if restart == 0:
            p0 = [a0 if constrain_phase != 'backward' else -a0,
                  (t_max - t_min) * 0.7, t_min + 0.5, y_mean]
        else:
            amp_init = rng.uniform(lower[0] if np.isfinite(lower[0]) else -a0*2,
                                   upper[0] if np.isfinite(upper[0]) else a0*2)
            p0 = [amp_init,
                  rng.uniform(3.0, min(12.0, t_max - t_min + 2)),
                  rng.uniform(t_min - 0.5, t_min + (t_max - t_min) * 0.3),
                  y_mean + rng.normal(0, a0 * 0.1)]
        
        try:
            popt, _ = curve_fit(_windowed_for_curvefit, trs, slopes,
                               p0=p0, bounds=(lower, upper), maxfev=3000)
            y_hat = _windowed_for_curvefit(trs, *popt)
            residuals = slopes - y_hat
            sigma2 = np.var(residuals)
            if sigma2 <= 0:
                continue
            ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2
            if ll > best_ll:
                best_ll = ll
        except (RuntimeError, ValueError):
            continue
    
    return best_ll


def bic(ll, k, n):
    """BIC = -2*LL + k*log(n)"""
    return -2 * ll + k * np.log(n)


def bayesian_model_comparison(slopes, trs=None):
    """
    Compare null vs replay models for a single trial.
    
    Returns dict with BIC values and Bayes factor approximation.
    """
    if trs is None:
        trs = np.arange(1, len(slopes) + 1, dtype=float)
    
    n = len(slopes)
    
    # M0: Null (2 params: mean, variance)
    ll_null = log_likelihood_null(slopes)
    bic_null = bic(ll_null, 2, n)
    
    # M1: Forward replay (4 params + noise variance = 5)
    ll_fwd = log_likelihood_sinusoid(slopes, trs, constrain_phase='forward')
    bic_fwd = bic(ll_fwd, 5, n)
    
    # M2: Backward replay (5 params)
    ll_bwd = log_likelihood_sinusoid(slopes, trs, constrain_phase='backward')
    bic_bwd = bic(ll_bwd, 5, n)
    
    # M3: Any replay — free phase (5 params)
    ll_any = log_likelihood_sinusoid(slopes, trs, constrain_phase=None)
    bic_any = bic(ll_any, 5, n)
    
    # BIC-based Bayes factor approximation: BF ≈ exp(-0.5 * ΔBIC)
    delta_bic_any = bic_any - bic_null  # negative = replay preferred
    bf_any = np.exp(-0.5 * delta_bic_any)  # BF > 1 = replay preferred
    
    # Log10 BF for interpretability
    log10_bf = np.log10(bf_any) if bf_any > 0 else -np.inf
    
    return {
        'bic_null': bic_null,
        'bic_forward': bic_fwd,
        'bic_backward': bic_bwd,
        'bic_any_replay': bic_any,
        'delta_bic': delta_bic_any,
        'bayes_factor': bf_any,
        'log10_bf': log10_bf,
        'preferred': 'replay' if delta_bic_any < 0 else 'null',
    }


# =====================================================================
# Simulation: compare all methods
# =====================================================================

def simulate_trial(has_signal, n_items, n_trs, params, noise_sd,
                   onset_tr=2.0, rng=None):
    """Generate one trial's slope timecourse."""
    if rng is None:
        rng = np.random.default_rng()
    
    t = np.arange(1, n_trs + 1, dtype=float)
    
    if has_signal:
        t_shifted = t - onset_tr
        probs = multi_item_response(t_shifted, n_items, 0.05, 0.0, 1.25, params)
        probs += rng.normal(0, noise_sd, probs.shape)
        probs = np.clip(probs, 0, 100)
    else:
        probs = params.baseline + rng.normal(0, noise_sd, (n_items, len(t)))
        probs = np.clip(probs, 0, 100)
    
    return compute_slope_timecourse(probs), t


def dprime(sig, null):
    ms, mn = np.nanmean(sig), np.nanmean(null)
    ss, sn = np.nanstd(sig, ddof=1), np.nanstd(null, ddof=1)
    pooled = np.sqrt((ss**2 + sn**2) / 2)
    return (ms - mn) / pooled if pooled > 0 else 0.0


def run_comparison(n_trials=40, seed=42):
    """Compare template matching and Bayesian approach against existing metrics."""
    
    params = ResponseParams(amplitude=35.0, baseline=20.0)
    noise_sd = 4.0
    n_items = 5
    n_trs = 13
    
    # Sweep onset jitter (the most practically relevant condition)
    jitter_sds = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    
    metric_names = [
        'mean_slope', 'abs_mean', 'peak_to_trough', 'variance',
        'template_corr', 'template_corr_optlag', 'log10_bf',
    ]
    
    results = {m: [] for m in metric_names}
    
    for jitter_sd in jitter_sds:
        print(f"  Jitter SD = {jitter_sd}")
        rng = np.random.default_rng(seed)
        
        sig_vals = {m: [] for m in metric_names}
        null_vals = {m: [] for m in metric_names}
        
        for trial in range(n_trials):
            onset = max(0.5, 2.0 + rng.normal(0, jitter_sd))
            
            # Signal trial
            slopes_s, trs = simulate_trial(True, n_items, n_trs, params,
                                           noise_sd, onset_tr=onset, rng=rng)
            # Null trial
            slopes_n, _ = simulate_trial(False, n_items, n_trs, params,
                                        noise_sd, rng=rng)
            
            for slopes, storage in [(slopes_s, sig_vals), (slopes_n, null_vals)]:
                # Simple metrics
                storage['mean_slope'].append(mean_slope(slopes))
                storage['abs_mean'].append(abs_mean(slopes))
                storage['peak_to_trough'].append(peak_to_trough(slopes))
                storage['variance'].append(slope_variance(slopes))
                
                # Template matching (fixed onset)
                template_fixed = generate_soda_template(n_trs=n_trs, n_items=n_items)
                storage['template_corr'].append(
                    template_correlation(slopes, template_fixed))
                
                # Template matching (optimal lag)
                opt = template_correlation_optimal_lag(slopes, n_items=n_items)
                storage['template_corr_optlag'].append(opt['best_correlation'])
                
                # Bayesian model comparison
                bmc = bayesian_model_comparison(slopes, trs)
                storage['log10_bf'].append(bmc['log10_bf'])
        
        for m in metric_names:
            results[m].append(dprime(
                np.array(sig_vals[m]), np.array(null_vals[m])))
    
    return jitter_sds, results, metric_names


# =====================================================================
# Plotting
# =====================================================================

def plot_comparison(jitter_sds, results, metric_names):
    """Main comparison plot."""

    # Group metrics by type
    simple_metrics = ['mean_slope', 'abs_mean', 'peak_to_trough', 'variance']
    template_metrics = ['template_corr', 'template_corr_optlag']
    bayes_metrics = ['log10_bf']

    labels = {
        'mean_slope': 'Mean slope',
        'abs_mean': '|Slope| mean',
        'peak_to_trough': 'Peak-to-trough',
        'variance': 'Slope variance',
        'template_corr': 'Template corr (fixed)',
        'template_corr_optlag': 'Template corr (opt. lag)',
        'log10_bf': 'Bayes factor (log10 BF)',
    }

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 1.6, FULL_WIDTH * 1.0))

    # Simple metrics: thin dashed grey lines — label only the first one
    for i, m in enumerate(simple_metrics):
        if m in metric_names:
            lbl = 'Simple metrics' if i == 0 else '_nolegend_'
            ax.plot(jitter_sds, results[m], ls='--', lw=0.7, markersize=2.5,
                    marker='o', color=GREY, alpha=0.5, label=lbl)

    # Template methods: solid orange
    template_markers = ['s', 'D']
    for i, m in enumerate(template_metrics):
        if m in metric_names:
            ax.plot(jitter_sds, results[m], ls='-', lw=1.3,
                    marker=template_markers[i], markersize=3.5,
                    color=ORANGE, label=labels[m])

    # Bayes: solid blue
    for m in bayes_metrics:
        if m in metric_names:
            ax.plot(jitter_sds, results[m], ls='-', lw=1.3,
                    marker='^', markersize=3.5, color=BLUE, label=labels[m])

    ax.axhline(0, color=GREY, ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Onset jitter SD (TRs)')
    ax.set_ylabel("d\u2032 (sensitivity)")
    ax.legend(fontsize=6, loc='upper right', ncol=1, handlelength=1.5)

    add_panel_label(ax, 'A')
    fig.tight_layout()
    save_figure(fig, 'sim8_template_bayes_comparison')
    plt.close()


def plot_example_template_matching():
    """Show template matching on an example trial."""

    params = ResponseParams(amplitude=35.0, baseline=20.0)
    rng = np.random.default_rng(42)
    n_items, n_trs = 5, 13

    # Signal trial with onset at TR 2
    slopes_sig, trs = simulate_trial(True, n_items, n_trs, params,
                                     noise_sd=3.0, onset_tr=2.0, rng=rng)
    # Null trial
    slopes_null, _ = simulate_trial(False, n_items, n_trs, params,
                                    noise_sd=3.0, rng=rng)

    # Template at different lags
    opt_sig = template_correlation_optimal_lag(slopes_sig, n_items=n_items)
    opt_null = template_correlation_optimal_lag(slopes_null, n_items=n_items)

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH * 2, FULL_WIDTH * 1.5))

    # Panel A: Signal trial with best-matching template
    ax = axes[0, 0]
    ax.plot(trs, slopes_sig, 'o-', color=BLACK, lw=1.5, markersize=4,
            label='Observed slopes')
    best_template = generate_soda_template(n_trs=n_trs, n_items=n_items,
                                           onset_tr=opt_sig['best_lag'])
    scale = np.std(slopes_sig) / (np.std(best_template) + 1e-10)
    ax.plot(trs, best_template * scale + np.mean(slopes_sig), '-', color=ORANGE,
            lw=1.8, alpha=0.8,
            label=f'Best template (lag={opt_sig["best_lag"]:.1f})')
    ax.text(0.97, 0.97, f'Signal trial \u2014 r = {opt_sig["best_correlation"]:.3f}',
            transform=ax.transAxes, fontsize=7, ha='right', va='top')
    ax.set_xlabel('Time (TRs)')
    ax.set_ylabel('SODA slope')
    ax.legend(fontsize=6, loc='lower right')
    add_panel_label(ax, 'A')

    # Panel B: Null trial with best-matching template
    ax = axes[0, 1]
    ax.plot(trs, slopes_null, 'o-', color=BLACK, lw=1.0, markersize=3,
            label='Observed slopes')
    best_template_null = generate_soda_template(n_trs=n_trs, n_items=n_items,
                                                onset_tr=opt_null['best_lag'])
    scale_n = np.std(slopes_null) / (np.std(best_template_null) + 1e-10)
    ax.plot(trs, best_template_null * scale_n + np.mean(slopes_null), '-',
            color=ORANGE, lw=1.2, alpha=0.8,
            label=f'Best template (lag={opt_null["best_lag"]:.1f})')
    ax.text(0.97, 0.97, f'Null trial \u2014 r = {opt_null["best_correlation"]:.3f}',
            transform=ax.transAxes, fontsize=7, ha='right', va='top')
    ax.set_xlabel('Time (TRs)')
    ax.set_ylabel('SODA slope')
    ax.legend(fontsize=6, loc='lower right')
    add_panel_label(ax, 'B')

    # Panel C: Correlation as function of lag (signal)
    ax = axes[1, 0]
    ax.plot(opt_sig['lags'], opt_sig['all_correlations'], '-', color=ORANGE, lw=1.0)
    ax.axvline(opt_sig['best_lag'], color=ORANGE, ls='--', lw=0.6, alpha=0.5)
    ax.axhline(0, color=GREY, ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Template onset lag (TRs)')
    ax.set_ylabel('Correlation')
    ax.text(0.97, 0.97, 'Lags (signal)', transform=ax.transAxes,
            fontsize=7, ha='right', va='top', color=GREY)
    add_panel_label(ax, 'C')

    # Panel D: Correlation as function of lag (null)
    ax = axes[1, 1]
    ax.plot(opt_null['lags'], opt_null['all_correlations'], '-', color=BLUE, lw=1.0)
    ax.axvline(opt_null['best_lag'], color=BLUE, ls='--', lw=0.6, alpha=0.5)
    ax.axhline(0, color=GREY, ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Template onset lag (TRs)')
    ax.set_ylabel('Correlation')
    ax.text(0.97, 0.97, 'Lags (null)', transform=ax.transAxes,
            fontsize=7, ha='right', va='top', color=GREY)
    add_panel_label(ax, 'D')

    # Match y-axes for C and D
    ymin = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    ymax = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(ymin, ymax)
    axes[1, 1].set_ylim(ymin, ymax)
    fig.tight_layout()
    save_figure(fig, 'sim8_template_example')
    plt.close()


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Simulation 8: Template Matching & Bayesian Model Comparison")
    print("=" * 60)
    
    print("\n--- Example template matching ---")
    plot_example_template_matching()
    
    print("\n--- Running full comparison (onset jitter sweep) ---")
    jitter_sds, results, metric_names = run_comparison(n_trials=35)
    
    print("\n--- Plotting ---")
    plot_comparison(jitter_sds, results, metric_names)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
