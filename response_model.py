"""
Response model for SODA simulations.

Implements the sinusoidal response function from Wittkuhn & Schuck (2021),
Equations 1-6. This models how probabilistic classifier evidence evolves
over time following a brief neural event, and how the difference between
two time-shifted events creates the forward/backward SODA pattern.

Key parameters (fitted in Wittkuhn & Schuck):
    - A (amplitude): peak deviation from baseline (~60-70% in their data)
    - f (frequency): 1/lambda where lambda ≈ 5.26 TRs
    - d (onset delay): ~0.56 TRs
    - b (baseline): ~20% for 5-class classification (chance level)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ResponseParams:
    """Parameters for the sinusoidal response function.
    
    Defaults are mean fitted values from Wittkuhn & Schuck (2021).
    """
    amplitude: float = 60.0       # A: peak probability deviation (%)
    wavelength: float = 5.26      # lambda = 1/f (in TRs)
    onset_delay: float = 0.56     # d: onset delay (in TRs)
    baseline: float = 20.0        # b: baseline probability (%)
    
    @property
    def frequency(self) -> float:
        """Angular frequency in 1/TR."""
        return 1.0 / self.wavelength


def single_event_response(
    t: np.ndarray,
    params: Optional[ResponseParams] = None,
) -> np.ndarray:
    """Sinusoidal response to a single brief neural event.
    
    Implements Equations 1-2 from Wittkuhn & Schuck (2021):
    h(t) = A/2 * sin(2*pi*f*t - 2*pi*f*d - pi/2) + b + A/2
    
    Flattened to baseline outside one cycle: [d, d + 1/f].
    
    Args:
        t: Time points (in TRs).
        params: Response parameters. Uses defaults if None.
        
    Returns:
        Response amplitude at each time point (in % probability).
    """
    if params is None:
        params = ResponseParams()
    
    A = params.amplitude
    f = params.frequency
    d = params.onset_delay
    b = params.baseline
    
    # Sinusoidal response (Eq. 1)
    h = (A / 2) * np.sin(2 * np.pi * f * t - 2 * np.pi * f * d - np.pi / 2) + b + (A / 2)
    
    # Flatten outside one cycle (Eq. 2)
    response = np.where(
        (t >= d) & (t <= d + 1.0 / f),
        h,
        b
    )
    
    return response


def two_event_difference(
    t: np.ndarray,
    delta: float,
    params: Optional[ResponseParams] = None,
) -> np.ndarray:
    """Difference between two time-shifted response functions.
    
    Implements Equation 6 from Wittkuhn & Schuck (2021):
    h_delta(t) = A * sin(2*pi*f*delta/2) * sin(2*pi*f_delta*t - 2*pi*f_delta*d)
    
    where f_delta = f / (1 + f*delta) accounts for the stretching
    due to flattening of the sine waves.
    
    Args:
        t: Time points (in TRs).
        delta: Time shift between events (in TRs).
        params: Response parameters.
        
    Returns:
        Difference in probability between first and second event.
        Positive = first event dominates (forward period).
        Negative = second event dominates (backward period).
    """
    if params is None:
        params = ResponseParams()
    
    A = params.amplitude
    f = params.frequency
    d = params.onset_delay
    
    # Adjusted frequency for flattened sine (Eq. 5)
    f_delta = f / (1.0 + f * delta)
    
    # Amplitude scaling
    amp_scale = A * np.sin(2 * np.pi * f * delta / 2.0)
    
    # Difference function (Eq. 6)
    diff = amp_scale * np.sin(2 * np.pi * f_delta * t - 2 * np.pi * f_delta * d)
    
    # Apply windowing: difference is non-zero only during the relevant period
    lambda_delta = 1.0 / f_delta
    response = np.where(
        (t >= d) & (t <= d + lambda_delta),
        diff,
        0.0
    )
    
    return response


def compute_periods(
    delta: float,
    params: Optional[ResponseParams] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute forward and backward period boundaries.
    
    Forward period: [d, 0.5 * lambda_delta + d]
    Backward period: [0.5 * lambda_delta + d, lambda_delta + d]
    
    Args:
        delta: Time between first and last sequence item (in TRs).
        params: Response parameters.
        
    Returns:
        Tuple of (forward_period, backward_period), each as (start, end) in TRs.
    """
    if params is None:
        params = ResponseParams()
    
    f = params.frequency
    d = params.onset_delay
    
    f_delta = f / (1.0 + f * delta)
    lambda_delta = 1.0 / f_delta
    
    forward = (d, 0.5 * lambda_delta + d)
    backward = (0.5 * lambda_delta + d, lambda_delta + d)
    
    return forward, backward


def sequence_delta(
    n_items: int,
    isi_seconds: float,
    stimulus_duration: float = 0.1,
    tr: float = 1.25,
) -> float:
    """Compute delta (time between first and last item onset) in TRs.
    
    For a sequence of n_items with given ISI:
    delta = ((n_items - 1) * isi + (n_items - 1) * stimulus_duration) / tr
    
    Note: In Wittkuhn & Schuck, delta = (ISI * (n-1) + stim_dur * (n-1)) / TR
    because the relevant interval is from first to last onset.
    
    Args:
        n_items: Number of items in the sequence.
        isi_seconds: Inter-stimulus interval in seconds.
        stimulus_duration: Duration of each stimulus in seconds.
        tr: Repetition time in seconds.
        
    Returns:
        Delta in TRs.
    """
    total_time = (n_items - 1) * isi_seconds + (n_items - 1) * stimulus_duration
    return total_time / tr


def multi_item_response(
    t: np.ndarray,
    n_items: int,
    isi_seconds: float,
    stimulus_duration: float = 0.1,
    tr: float = 1.25,
    params: Optional[ResponseParams] = None,
    class_amplitudes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulate classifier probability timecourses for a full sequence.
    
    Each item in the sequence triggers a response function, time-shifted
    by its onset relative to the first item.
    
    Args:
        t: Time points (in TRs).
        n_items: Number of items in the sequence.
        isi_seconds: Inter-stimulus interval in seconds.
        stimulus_duration: Duration of each stimulus in seconds.
        tr: Repetition time in seconds.
        params: Response parameters (shared baseline, wavelength, delay).
        class_amplitudes: Per-class amplitude array of shape (n_items,).
            If None, all classes use params.amplitude.
            
    Returns:
        Array of shape (n_items, len(t)) with probability timecourse per class.
    """
    if params is None:
        params = ResponseParams()
    
    if class_amplitudes is None:
        class_amplitudes = np.full(n_items, params.amplitude)
    
    probabilities = np.zeros((n_items, len(t)))
    
    for i in range(n_items):
        # Onset of item i relative to first item (in TRs)
        onset_shift = i * (isi_seconds + stimulus_duration) / tr
        
        # Create per-item params with potentially different amplitude
        item_params = ResponseParams(
            amplitude=class_amplitudes[i],
            wavelength=params.wavelength,
            onset_delay=params.onset_delay,
            baseline=params.baseline,
        )
        
        # Time-shifted response
        probabilities[i] = single_event_response(t - onset_shift, item_params)
    
    return probabilities
