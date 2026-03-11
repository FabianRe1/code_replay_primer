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

FIGDIR = Path(__file__).parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


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
    
    colors = {
        'mean_slope': '#999999',
        'abs_mean': '#fc8d62',
        'peak_to_trough': '#66c2a5',
        'variance': '#8da0cb',
        'template_corr': '#e7298a',
        'template_corr_optlag': '#d62728',
        'log10_bf': '#1f77b4',
    }
    
    labels = {
        'mean_slope': 'Mean slope (naive)',
        'abs_mean': '|Slope| mean',
        'peak_to_trough': 'Peak-to-trough',
        'variance': 'Slope variance',
        'template_corr': 'Template corr (fixed onset)',
        'template_corr_optlag': 'Template corr (optimal lag)',
        'log10_bf': 'Bayes factor (log₁₀ BF)',
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for m in metric_names:
        ls = '--' if m in ['mean_slope', 'abs_mean', 'peak_to_trough', 'variance'] else '-'
        lw = 1.5 if ls == '--' else 2.5
        ax.plot(jitter_sds, results[m], 'o-', lw=lw, markersize=6,
                color=colors[m], label=labels[m], linestyle=ls)
    
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Onset jitter SD (TRs)')
    ax.set_ylabel("d' (sensitivity)")
    ax.set_title("Replay Detection: Template Matching & Bayesian Model Comparison\n"
                 "vs existing metrics (amp=35, noise=4, 5 items)")
    ax.legend(fontsize=9, loc='best')
    ax.spines[['top', 'right']].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim8_template_bayes_comparison.png')
    plt.close()
    print(f"✓ Saved sim8_template_bayes_comparison.png")


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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Signal trial with best-matching template
    ax = axes[0, 0]
    ax.plot(trs, slopes_sig, 'ko-', lw=2, markersize=5, label='Observed slopes')
    best_template = generate_soda_template(n_trs=n_trs, n_items=n_items,
                                            onset_tr=opt_sig['best_lag'])
    # Scale template to data range for visualization
    scale = np.std(slopes_sig) / (np.std(best_template) + 1e-10)
    ax.plot(trs, best_template * scale + np.mean(slopes_sig), 'r-', lw=2,
            alpha=0.7, label=f'Best template (lag={opt_sig["best_lag"]:.1f})')
    ax.set_title(f'A) Signal trial — r={opt_sig["best_correlation"]:.3f}')
    ax.set_xlabel('Time (TRs)')
    ax.set_ylabel('SODA slope')
    ax.legend(fontsize=8)
    
    # Panel B: Null trial with best-matching template
    ax = axes[0, 1]
    ax.plot(trs, slopes_null, 'ko-', lw=2, markersize=5, label='Observed slopes')
    best_template_null = generate_soda_template(n_trs=n_trs, n_items=n_items,
                                                 onset_tr=opt_null['best_lag'])
    scale_n = np.std(slopes_null) / (np.std(best_template_null) + 1e-10)
    ax.plot(trs, best_template_null * scale_n + np.mean(slopes_null), 'r-', lw=2,
            alpha=0.7, label=f'Best template (lag={opt_null["best_lag"]:.1f})')
    ax.set_title(f'B) Null trial — r={opt_null["best_correlation"]:.3f}')
    ax.set_xlabel('Time (TRs)')
    ax.set_ylabel('SODA slope')
    ax.legend(fontsize=8)
    
    # Panel C: Correlation as function of lag (signal)
    ax = axes[1, 0]
    ax.plot(opt_sig['lags'], opt_sig['all_correlations'], 'r-', lw=2)
    ax.axvline(opt_sig['best_lag'], color='red', ls='--', alpha=0.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Template onset lag (TRs)')
    ax.set_ylabel('Correlation')
    ax.set_title('C) Template-data correlation across lags (signal)')
    
    # Panel D: Correlation as function of lag (null)
    ax = axes[1, 1]
    ax.plot(opt_null['lags'], opt_null['all_correlations'], 'b-', lw=2)
    ax.axvline(opt_null['best_lag'], color='blue', ls='--', alpha=0.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Template onset lag (TRs)')
    ax.set_ylabel('Correlation')
    ax.set_title('D) Template-data correlation across lags (null)')
    
    # Match y-axes for C and D
    ymin = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    ymax = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(ymin, ymax)
    axes[1, 1].set_ylim(ymin, ymax)
    
    fig.suptitle('Template Matching: Example Signal vs Null Trial',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim8_template_example.png')
    plt.close()
    print(f"✓ Saved sim8_template_example.png")


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
