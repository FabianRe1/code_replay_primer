"""
SODA (Slope of Decoder Accuracy) computation.

Implements the core SODA metric: for each TR, regress sequence position
onto classifier probabilities. The slope indexes the degree of sequential
ordering at that time point.

Also includes frequency analysis of slope timecourses to distinguish
sub-second from supra-second sequences.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import stats


def compute_slope_timecourse(
    probabilities: np.ndarray,
    positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute SODA slope at each time point.
    
    For each TR, perform linear regression:
        position ~ probability
    
    The slope is positive if earlier items have higher probability (forward),
    negative if later items have higher probability (backward).
    
    Note: Following Wittkuhn & Schuck convention, we FLIP the sign so that
    positive = forward ordering, negative = backward ordering.
    
    Args:
        probabilities: Array of shape (n_items, n_timepoints).
            Classifier probability for each item at each TR.
        positions: Serial positions of items (1-indexed).
            If None, uses 1, 2, ..., n_items.
            
    Returns:
        Array of shape (n_timepoints,) with slope at each TR.
    """
    n_items, n_timepoints = probabilities.shape
    
    if positions is None:
        positions = np.arange(1, n_items + 1, dtype=float)
    
    slopes = np.zeros(n_timepoints)
    
    for t_idx in range(n_timepoints):
        probs_at_t = probabilities[:, t_idx]
        slope, _, _, _, _ = stats.linregress(positions, probs_at_t)
        # Flip sign: positive slope in regression means later items have
        # higher probability → backward. We flip so positive = forward.
        slopes[t_idx] = -slope
    
    return slopes


def compute_slope_for_trial(
    probabilities: np.ndarray,
    positions: Optional[np.ndarray] = None,
    forward_trs: Optional[Tuple[int, int]] = None,
    backward_trs: Optional[Tuple[int, int]] = None,
) -> dict:
    """Compute SODA slopes and summary statistics for a single trial.
    
    Args:
        probabilities: Array of shape (n_items, n_timepoints).
        positions: Serial positions of items.
        forward_trs: (start, end) TR indices for forward period (inclusive).
        backward_trs: (start, end) TR indices for backward period (inclusive).
        
    Returns:
        Dictionary with:
        - 'slope_timecourse': full slope timecourse
        - 'mean_forward_slope': average slope during forward period
        - 'mean_backward_slope': average slope during backward period
        - 'forward_trs': the forward period used
        - 'backward_trs': the backward period used
    """
    slopes = compute_slope_timecourse(probabilities, positions)
    
    result = {'slope_timecourse': slopes}
    
    if forward_trs is not None:
        fwd_start, fwd_end = forward_trs
        result['mean_forward_slope'] = np.mean(slopes[fwd_start:fwd_end + 1])
        result['forward_trs'] = forward_trs
    
    if backward_trs is not None:
        bwd_start, bwd_end = backward_trs
        result['mean_backward_slope'] = np.mean(slopes[bwd_start:bwd_end + 1])
        result['backward_trs'] = backward_trs
    
    return result


def periods_to_tr_indices(
    forward_period: Tuple[float, float],
    backward_period: Tuple[float, float],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Convert continuous time periods to discrete TR indices.
    
    Rounds to nearest TR, as done in Wittkuhn & Schuck (Table 1).
    TR indices are 1-based (TR 1 = first TR after sequence onset).
    
    Args:
        forward_period: (start, end) in TRs (continuous).
        backward_period: (start, end) in TRs (continuous).
        
    Returns:
        Tuple of ((fwd_start, fwd_end), (bwd_start, bwd_end)) as 0-based indices.
    """
    fwd_start = max(0, int(np.round(forward_period[0])))
    fwd_end = int(np.round(forward_period[1]))
    bwd_start = int(np.round(backward_period[0]))
    bwd_end = int(np.round(backward_period[1]))
    
    return (fwd_start, fwd_end), (bwd_start, bwd_end)


def aggregate_slopes_across_trials(
    trial_slopes: list[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Average slope timecourses across multiple trials.
    
    Args:
        trial_slopes: List of slope timecourse arrays, all same length.
        
    Returns:
        Tuple of (mean_slopes, sem_slopes).
    """
    stacked = np.stack(trial_slopes, axis=0)  # (n_trials, n_timepoints)
    mean_slopes = np.mean(stacked, axis=0)
    sem_slopes = np.std(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    
    return mean_slopes, sem_slopes


def slope_frequency_spectrum(
    slopes: np.ndarray,
    tr: float = 1.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frequency spectrum of slope timecourse.
    
    Uses Lomb-Scargle periodogram to handle potentially irregular sampling,
    following Wittkuhn & Schuck (2021).
    
    Args:
        slopes: Slope timecourse.
        tr: Repetition time in seconds.
        
    Returns:
        Tuple of (frequencies_hz, power).
    """
    from scipy.signal import lombscargle
    
    n = len(slopes)
    t_seconds = np.arange(n) * tr
    
    # Frequency range
    f_min = 1.0 / (n * tr)
    f_max = 0.5 / tr  # Nyquist
    frequencies = np.linspace(f_min, f_max, n * 2)
    angular_freqs = 2 * np.pi * frequencies
    
    # Normalize slopes
    slopes_centered = slopes - np.mean(slopes)
    
    power = lombscargle(t_seconds, slopes_centered, angular_freqs, normalize=True)
    
    return frequencies, power


def predicted_frequency(
    isi_seconds: float,
    n_items: int = 5,
    stimulus_duration: float = 0.1,
    tr: float = 1.25,
    wavelength: float = 5.26,
) -> float:
    """Compute predicted peak frequency for a given sequence speed.
    
    From Equation 5: f_delta = f / (1 + f * delta)
    Converted to Hz: f_delta_hz = f_delta / tr
    
    Args:
        isi_seconds: Inter-stimulus interval in seconds.
        n_items: Number of items in the sequence.
        stimulus_duration: Duration of each stimulus.
        tr: Repetition time.
        wavelength: Fitted wavelength in TRs.
        
    Returns:
        Predicted frequency in Hz.
    """
    f = 1.0 / wavelength  # frequency in 1/TR
    delta = ((n_items - 1) * isi_seconds + (n_items - 1) * stimulus_duration) / tr
    f_delta = f / (1.0 + f * delta)
    
    return f_delta / tr  # convert to Hz
