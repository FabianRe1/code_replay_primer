"""
Aggregation metrics for SODA timecourses.

Implements multiple approaches to quantify sequential reactivation from
SODA slope timecourses, ranging from naive averaging to model-based fitting.

Metrics (ordered from simplest to most sophisticated):
1. mean_slope       — naive mean of slopes across TRs (cancels due to ±phases)
2. abs_mean         — mean of absolute slopes (sign-agnostic)
3. variance         — variance of slope timecourse (fluctuation = signal)
4. peak_to_trough   — range of slope timecourse
5. spectral_power   — power in expected frequency band (FFT-based)
6. continuous_sin   — full sinusoidal fit: A*sin(2πft + φ) + c → amplitude A
7. windowed_1cycle  — windowed single-cycle fit following Wittkuhn & Schuck → amplitude A

Each metric returns a single scalar per trial that indexes "replay strength."
"""

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Optional


# =====================================================================
# 1-5: Simple metrics (no model fitting)
# =====================================================================

def mean_slope(slopes: np.ndarray) -> float:
    """Naive mean of SODA slopes — cancels for balanced forward/backward."""
    v = slopes[~np.isnan(slopes)]
    return np.mean(v) if len(v) >= 1 else np.nan


def abs_mean(slopes: np.ndarray) -> float:
    """Mean of absolute slopes — ignores sign, captures magnitude."""
    v = slopes[~np.isnan(slopes)]
    return np.mean(np.abs(v)) if len(v) >= 1 else np.nan


def slope_variance(slopes: np.ndarray) -> float:
    """Variance of slope timecourse — high if sinusoidal pattern present."""
    v = slopes[~np.isnan(slopes)]
    return np.var(v) if len(v) >= 2 else np.nan


def peak_to_trough(slopes: np.ndarray) -> float:
    """Peak-to-trough range of slope timecourse."""
    v = slopes[~np.isnan(slopes)]
    return np.ptp(v) if len(v) >= 2 else np.nan


def spectral_power(slopes: np.ndarray, f_min: float = 0.05, f_max: float = 0.45) -> float:
    """Mean power in expected frequency band via FFT."""
    v = slopes[~np.isnan(slopes)]
    if len(v) < 4:
        return np.nan
    v = v - np.mean(v)
    freqs = np.fft.rfftfreq(len(v), d=1.0)  # cycles per TR
    power = np.abs(np.fft.rfft(v)) ** 2
    band = (freqs >= f_min) & (freqs <= f_max)
    return np.mean(power[band]) if band.any() else np.nan


# =====================================================================
# 6: Continuous sinusoidal fit
# =====================================================================

def _continuous_sinusoid(t, amplitude, frequency, phase, offset):
    """A * sin(2π*f*t + φ) + c"""
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset


def fit_continuous_sinusoid(
    trs: np.ndarray,
    slopes: np.ndarray,
    f_min: float = 0.03,
    f_max: float = 0.45,
    n_restarts: int = 3,
) -> dict:
    """
    Fit continuous sinusoidal model to SODA timecourse.
    
    Returns dict with 'amplitude', 'frequency', 'phase', 'offset',
    'r_squared', 'fit_success'.
    """
    trs = np.asarray(trs, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    valid = ~np.isnan(slopes)
    
    if valid.sum() < 4:
        return {'amplitude': np.nan, 'r_squared': np.nan, 'fit_success': False}
    
    t, y = trs[valid], slopes[valid]
    y_range = np.ptp(y)
    y_mean = np.mean(y)
    a0 = max(y_range / 2, 0.01)
    
    lower = [0.0, f_min, -2 * np.pi, -np.inf]
    upper = [np.inf, f_max, 2 * np.pi, np.inf]
    
    best_result, best_residual = None, np.inf
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        p0 = ([a0, 0.15, 0.0, y_mean] if restart == 0 else
              [rng.uniform(0, a0*2), rng.uniform(f_min, f_max),
               rng.uniform(-np.pi, np.pi), y_mean + rng.normal(0, a0*0.1)])
        try:
            popt, _ = curve_fit(_continuous_sinusoid, t, y, p0=p0,
                               bounds=(lower, upper), maxfev=5000)
            resid = np.sum((y - _continuous_sinusoid(t, *popt))**2)
            if resid < best_residual:
                best_residual = resid
                best_result = popt
        except (RuntimeError, ValueError):
            continue
    
    if best_result is None:
        return {'amplitude': np.nan, 'r_squared': np.nan, 'fit_success': False}
    
    amplitude, frequency, phase, offset = best_result
    y_hat = _continuous_sinusoid(t, *best_result)
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return {'amplitude': amplitude, 'frequency': frequency, 'phase': phase,
            'offset': offset, 'r_squared': r_squared, 'fit_success': True}


# =====================================================================
# 7: Windowed single-cycle sinusoidal fit (Wittkuhn & Schuck style)
# =====================================================================

def _windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15):
    """
    Single-cycle sinusoidal model with smooth Tukey-window edges.
    
    y(t) = A * sin(2π/λ * (t - d)) * W(t) + c
    """
    t = np.asarray(t, dtype=float)
    phase = 2 * np.pi / duration * (t - onset)
    sine_part = amplitude * np.sin(phase)
    
    active = (t >= onset) & (t <= onset + duration)
    taper_len = taper_frac * duration
    window = np.zeros_like(t)
    window[active] = 1.0
    
    if taper_len > 0:
        rise = active & (t < onset + taper_len)
        if rise.any():
            window[rise] = 0.5 * (1 - np.cos(np.pi * (t[rise] - onset) / taper_len))
        fall = active & (t > onset + duration - taper_len)
        if fall.any():
            window[fall] = 0.5 * (1 - np.cos(np.pi * (onset + duration - t[fall]) / taper_len))
    
    return sine_part * window + offset


def _windowed_for_curvefit(t, amplitude, duration, onset, offset):
    return _windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15)


def _full_cycle_visible(onset, duration, t_min, t_max, min_cycle_frac=0.90):
    """Check that at least *min_cycle_frac* of the cycle falls within the data range.

    This ensures both the positive and negative lobes of the sinusoid are
    captured by the data, preventing half-cycle fits that chase a single
    strong deflection.
    """
    cycle_start = onset
    cycle_end = onset + duration
    overlap_start = max(cycle_start, t_min)
    overlap_end = min(cycle_end, t_max)
    overlap = max(0.0, overlap_end - overlap_start)
    return (overlap / duration) >= min_cycle_frac


def fit_windowed_sinusoid(
    trs: np.ndarray,
    slopes: np.ndarray,
    duration_min: float = 3.0,
    duration_max: float = 12.0,
    n_restarts: int = 10,
    min_cycle_frac: float = 0.90,
) -> dict:
    """
    Fit windowed single-cycle sinusoidal model.

    Requires both positive and negative phases of the sinusoid to be visible
    within the data range (controlled by min_cycle_frac). Fits where the
    cycle extends too far outside the data are rejected, preventing
    half-cycle fits that chase a single strong deflection.

    Returns dict with 'amplitude', 'duration', 'onset', 'offset',
    'frequency' (=1/duration), 'r_squared', 'fit_success'.
    """
    trs = np.asarray(trs, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    valid = ~np.isnan(slopes)

    if valid.sum() < 4:
        return {'amplitude': np.nan, 'duration': np.nan, 'onset': np.nan,
                'r_squared': np.nan, 'fit_success': False}

    t, y = trs[valid], slopes[valid]
    t_min, t_max = t.min(), t.max()
    y_range = np.ptp(y)
    y_mean = np.mean(y)
    a0 = max(y_range / 2, 0.01)

    # Tighten onset bounds: cycle must start no earlier than data start
    # and must leave room for at least duration_min within the data range.
    onset_lower = t_min
    onset_upper = max(t_min, t_max - duration_min)

    lower = [-np.inf, duration_min, onset_lower, -np.inf]
    upper = [np.inf, duration_max, onset_upper, np.inf]

    best_result, best_residual = None, np.inf
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if restart == 0:
            p0 = [a0, (t_max - t_min) * 0.8, t_min, y_mean]
        elif restart == 1:
            p0 = [-a0, (t_max - t_min) * 0.8, t_min, y_mean]
        else:
            p0 = [rng.uniform(-a0*2, a0*2),
                  rng.uniform(duration_min, min(duration_max, t_max - t_min + 2)),
                  rng.uniform(onset_lower, onset_lower + (onset_upper - onset_lower) * 0.5),
                  y_mean + rng.normal(0, a0*0.1)]
        # Clip p0 to bounds
        p0 = [max(lo, min(hi, v)) for v, lo, hi in zip(p0, lower, upper)]
        try:
            popt, _ = curve_fit(_windowed_for_curvefit, t, y, p0=p0,
                               bounds=(lower, upper), maxfev=5000)
            # Reject fits where the full cycle is not visible in the data
            if not _full_cycle_visible(popt[2], popt[1], t_min, t_max, min_cycle_frac):
                continue
            resid = np.sum((y - _windowed_for_curvefit(t, *popt))**2)
            if resid < best_residual:
                best_residual = resid
                best_result = popt
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return {'amplitude': np.nan, 'duration': np.nan, 'onset': np.nan,
                'r_squared': np.nan, 'fit_success': False}

    amplitude, duration, onset, offset = best_result
    y_hat = _windowed_for_curvefit(t, *best_result)
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {'amplitude': amplitude, 'duration': duration, 'onset': onset,
            'offset': offset, 'frequency': 1.0 / duration,
            'r_squared': r_squared, 'fit_success': True}


# =====================================================================
# Master function: compute all metrics for one trial
# =====================================================================

def compute_all_metrics(
    trs: np.ndarray,
    slopes: np.ndarray,
) -> dict:
    """
    Compute all 7 aggregation metrics for a single trial's SODA timecourse.
    
    Args:
        trs: TR indices.
        slopes: SODA slope values at each TR.
        
    Returns:
        Dictionary with all metric values.
    """
    slopes = np.asarray(slopes, dtype=float)
    trs = np.asarray(trs, dtype=float)
    
    result = {
        'mean_slope': mean_slope(slopes),
        'abs_mean': abs_mean(slopes),
        'variance': slope_variance(slopes),
        'peak_to_trough': peak_to_trough(slopes),
        'spectral_power': spectral_power(slopes),
    }
    
    # Model-based fits
    cont_fit = fit_continuous_sinusoid(trs, slopes)
    result['continuous_sin_amplitude'] = cont_fit['amplitude']
    result['continuous_sin_r2'] = cont_fit.get('r_squared', np.nan)
    
    wind_fit = fit_windowed_sinusoid(trs, slopes)
    result['windowed_sin_amplitude'] = np.abs(wind_fit['amplitude']) if not np.isnan(wind_fit['amplitude']) else np.nan
    result['windowed_sin_r2'] = wind_fit.get('r_squared', np.nan)
    result['windowed_sin_duration'] = wind_fit.get('duration', np.nan)
    result['windowed_sin_onset'] = wind_fit.get('onset', np.nan)
    
    return result


# =====================================================================
# Metric names for iteration / plotting
# =====================================================================

METRIC_NAMES = [
    'mean_slope',
    'abs_mean',
    'variance',
    'peak_to_trough',
    'spectral_power',
    'continuous_sin_amplitude',
    'windowed_sin_amplitude',
]

METRIC_LABELS = {
    'mean_slope': 'Mean slope\n(naive average)',
    'abs_mean': '|Slope| mean\n(absolute)',
    'variance': 'Slope variance',
    'peak_to_trough': 'Peak-to-trough',
    'spectral_power': 'Spectral power\n(FFT band)',
    'continuous_sin_amplitude': 'Continuous sin\namplitude',
    'windowed_sin_amplitude': 'Windowed 1-cycle\namplitude',
}
