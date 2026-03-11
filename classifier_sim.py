"""
Classifier simulation module.

Generates simulated classifier probability timecourses under various
conditions: ideal (homogeneous), heterogeneous accuracy, with noise,
multiple replay events, etc.
"""

import numpy as np
from typing import Optional
from response_model import (
    ResponseParams, single_event_response, multi_item_response
)


def simulate_ideal_sequence_trial(
    n_items: int = 5,
    isi_seconds: float = 0.032,
    n_trs: int = 13,
    tr: float = 1.25,
    stimulus_duration: float = 0.1,
    params: Optional[ResponseParams] = None,
    noise_sd: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate one ideal sequence trial with equal classifier performance.
    
    Args:
        n_items: Number of items in the sequence.
        isi_seconds: Inter-stimulus interval in seconds.
        n_trs: Number of TRs to simulate.
        tr: Repetition time in seconds.
        stimulus_duration: Duration of each stimulus in seconds.
        params: Response parameters. 
        noise_sd: Standard deviation of Gaussian noise added to probabilities.
        rng: Random number generator for reproducibility.
        
    Returns:
        Array of shape (n_items, n_trs) with probability timecourses.
    """
    if params is None:
        params = ResponseParams()
    if rng is None:
        rng = np.random.default_rng()
    
    t = np.arange(1, n_trs + 1, dtype=float)  # TRs 1 to n_trs
    
    probs = multi_item_response(
        t, n_items, isi_seconds, stimulus_duration, tr, params
    )
    
    if noise_sd > 0:
        noise = rng.normal(0, noise_sd, probs.shape)
        probs = probs + noise
        # Clip to valid probability range
        probs = np.clip(probs, 0, 100)
    
    return probs


def simulate_heterogeneous_trial(
    n_items: int = 5,
    isi_seconds: float = 0.032,
    n_trs: int = 13,
    tr: float = 1.25,
    stimulus_duration: float = 0.1,
    class_amplitudes: Optional[np.ndarray] = None,
    params: Optional[ResponseParams] = None,
    noise_sd: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate a sequence trial with heterogeneous classifier accuracy.
    
    Different classes can have different peak amplitudes, modeling the
    real-world situation where some stimuli are decoded better than others.
    
    Args:
        n_items: Number of items.
        isi_seconds: ISI in seconds.
        n_trs: Number of TRs.
        tr: Repetition time.
        stimulus_duration: Stimulus duration.
        class_amplitudes: Peak amplitude per class (in % probability).
            If None, randomly samples from a realistic range.
        params: Base response parameters (amplitude will be overridden per class).
        noise_sd: Noise SD.
        rng: Random generator.
        
    Returns:
        Array of shape (n_items, n_trs).
    """
    if params is None:
        params = ResponseParams()
    if rng is None:
        rng = np.random.default_rng()
    
    if class_amplitudes is None:
        # Realistic range: some classes decoded much better than others
        class_amplitudes = rng.uniform(30, 70, size=n_items)
    
    t = np.arange(1, n_trs + 1, dtype=float)
    
    probs = multi_item_response(
        t, n_items, isi_seconds, stimulus_duration, tr, params,
        class_amplitudes=class_amplitudes,
    )
    
    if noise_sd > 0:
        noise = rng.normal(0, noise_sd, probs.shape)
        probs = probs + noise
        probs = np.clip(probs, 0, 100)
    
    return probs


def simulate_multiple_replay_events(
    n_items: int = 4,
    isi_seconds: float = 0.05,
    n_trs: int = 20,
    tr: float = 1.25,
    stimulus_duration: float = 0.0,
    event_onsets_trs: Optional[list[float]] = None,
    event_directions: Optional[list[int]] = None,
    params: Optional[ResponseParams] = None,
    noise_sd: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate multiple replay events within a single window.
    
    Each replay event generates its own set of overlapping response
    functions. Multiple events' responses are summed (linear superposition),
    which can cause reinforcement or cancellation.
    
    Args:
        n_items: Number of items per replay event.
        isi_seconds: ISI within each replay event.
        n_trs: Total number of TRs in the window.
        tr: Repetition time.
        stimulus_duration: Stimulus duration (0 for replay — no actual stimulus).
        event_onsets_trs: Onset time of each replay event (in TRs).
            If None, defaults to [2.0] (single event).
        event_directions: Direction of each event: +1 for forward, -1 for backward.
            If None, all forward.
        params: Response parameters.
        noise_sd: Noise SD.
        rng: Random generator.
        
    Returns:
        Array of shape (n_items, n_trs) with summed probability timecourses.
    """
    if params is None:
        params = ResponseParams()
    if rng is None:
        rng = np.random.default_rng()
    if event_onsets_trs is None:
        event_onsets_trs = [2.0]
    if event_directions is None:
        event_directions = [1] * len(event_onsets_trs)
    
    t = np.arange(1, n_trs + 1, dtype=float)
    total_probs = np.full((n_items, len(t)), params.baseline, dtype=float)
    
    for onset, direction in zip(event_onsets_trs, event_directions):
        # Time relative to this event's onset
        t_relative = t - onset
        
        # Generate response for this event
        event_probs = multi_item_response(
            t_relative, n_items, isi_seconds, stimulus_duration, tr, params
        )
        
        if direction == -1:
            # Reverse the item order for backward replay
            event_probs = event_probs[::-1]
        
        # Add to total (subtract baseline to avoid double-counting)
        total_probs += (event_probs - params.baseline)
    
    if noise_sd > 0:
        noise = rng.normal(0, noise_sd, total_probs.shape)
        total_probs = total_probs + noise
    
    total_probs = np.clip(total_probs, 0, 100)
    
    return total_probs


def simulate_trial_averaging(
    n_trials: int = 20,
    n_items: int = 4,
    isi_seconds: float = 0.05,
    n_trs: int = 15,
    tr: float = 1.25,
    onset_jitter_sd: float = 0.0,
    params: Optional[ResponseParams] = None,
    noise_sd: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate averaging of SODA slopes across trials with onset jitter.
    
    This demonstrates the critical problem: if replay events occur at
    different times across trials, averaging slope timecourses can cancel
    the sinusoidal pattern.
    
    Args:
        n_trials: Number of trials to simulate.
        n_items: Items per sequence.
        isi_seconds: ISI within replay events.
        n_trs: TRs per trial.
        tr: Repetition time.
        onset_jitter_sd: SD of replay onset jitter across trials (in TRs).
            0 = all events at same time (ideal).
        params: Response parameters.
        noise_sd: Per-TR noise.
        rng: Random generator.
        
    Returns:
        Tuple of:
        - all_probs: (n_trials, n_items, n_trs)
        - actual_onsets: (n_trials,) — the onset of each trial's event
        - all_slopes: (n_trials, n_trs) — slope timecourses per trial
    """
    from soda import compute_slope_timecourse
    
    if params is None:
        params = ResponseParams()
    if rng is None:
        rng = np.random.default_rng(42)
    
    base_onset = 3.0  # Base replay onset (in TRs from trial start)
    
    all_probs = np.zeros((n_trials, n_items, n_trs))
    actual_onsets = np.zeros(n_trials)
    all_slopes = np.zeros((n_trials, n_trs))
    
    for trial in range(n_trials):
        # Jittered onset
        onset = base_onset + rng.normal(0, onset_jitter_sd)
        onset = max(1.0, onset)  # Ensure onset is after trial start
        actual_onsets[trial] = onset
        
        # Simulate this trial
        probs = simulate_multiple_replay_events(
            n_items=n_items,
            isi_seconds=isi_seconds,
            n_trs=n_trs,
            tr=tr,
            event_onsets_trs=[onset],
            params=params,
            noise_sd=noise_sd,
            rng=rng,
        )
        
        all_probs[trial] = probs
        all_slopes[trial] = compute_slope_timecourse(probs)
    
    return all_probs, actual_onsets, all_slopes
