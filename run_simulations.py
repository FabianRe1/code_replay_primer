"""
SODA Primer Simulations — Main Script

Demonstrates the SODA method step by step:
1. Ideal case: equal classifier performance, single sequence
2. Heterogeneous classifiers: different accuracy per class
3. Multiple replay events: overlapping responses
4. Trial averaging: phase misalignment across trials

Run from the src/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

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

# -- Setup --
FIGDIR = Path(__file__).parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = plt.cm.Set2(np.linspace(0, 1, 8))
SPEED_COLORS = {
    0.032: '#1b3a4b',
    0.064: '#4a7c9b',
    0.128: '#7fb3d3',
    0.512: '#d4a843',
    2.048: '#f0c75e',
}


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
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)
    
    # --- Panel A: Single event response ---
    ax_a = fig.add_subplot(gs[0, 0])
    response = single_event_response(t_fine, params)
    ax_a.plot(t_fine, response, 'k-', lw=2)
    ax_a.axhline(params.baseline, color='gray', ls='--', alpha=0.5)
    ax_a.set_xlabel('Time (TRs)')
    ax_a.set_ylabel('Probability (%)')
    ax_a.set_title('A) Single event response')
    ax_a.set_xlim(0, 10)
    
    # --- Panel B: Two-event difference for different deltas ---
    ax_b = fig.add_subplot(gs[0, 1])
    for isi, color in SPEED_COLORS.items():
        delta = sequence_delta(n_items, isi)
        diff = two_event_difference(t_fine, delta, params)
        ax_b.plot(t_fine, diff, color=color, lw=1.5, label=f'{int(isi*1000)} ms')
    ax_b.axhline(0, color='gray', ls='--', alpha=0.5)
    ax_b.set_xlabel('Time (TRs)')
    ax_b.set_ylabel('Probability difference')
    ax_b.set_title('B) Event difference by speed')
    ax_b.legend(fontsize=8, title='ISI')
    ax_b.set_xlim(0, 14)
    
    # --- Panel C: Probability timecourses for 32ms sequence ---
    ax_c = fig.add_subplot(gs[0, 2])
    probs_fast = simulate_ideal_sequence_trial(
        n_items=n_items, isi_seconds=0.032, n_trs=n_trs
    )
    for i in range(n_items):
        ax_c.plot(t, probs_fast[i], 'o-', color=COLORS[i], lw=1.5,
                  markersize=4, label=f'Item {i+1}')
    ax_c.set_xlabel('Time (TRs)')
    ax_c.set_ylabel('Probability (%)')
    ax_c.set_title('C) Classifier probs (32 ms ISI)')
    ax_c.legend(fontsize=7, ncol=2)
    
    # --- Panels D-F: Probability timecourses for 128, 512, 2048 ms ---
    for idx, isi in enumerate([0.128, 0.512, 2.048]):
        ax = fig.add_subplot(gs[1, idx])
        probs = simulate_ideal_sequence_trial(
            n_items=n_items, isi_seconds=isi, n_trs=n_trs
        )
        for i in range(n_items):
            ax.plot(t, probs[i], 'o-', color=COLORS[i], lw=1.5, markersize=4)
        
        # Shade forward/backward periods
        delta = sequence_delta(n_items, isi)
        fwd, bwd = compute_periods(delta, params)
        ax.axvspan(max(fwd[0], t[0]), min(fwd[1], t[-1]),
                   alpha=0.1, color='blue', label='Forward')
        ax.axvspan(max(bwd[0], t[0]), min(bwd[1], t[-1]),
                   alpha=0.1, color='red', label='Backward')
        
        ax.set_xlabel('Time (TRs)')
        ax.set_ylabel('Probability (%)')
        ax.set_title(f'D{idx+1}) Probs ({int(isi*1000)} ms ISI)')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    # --- Panels G-I: SODA slope timecourses ---
    for idx, isi in enumerate([0.032, 0.128, 2.048]):
        ax = fig.add_subplot(gs[2, idx])
        probs = simulate_ideal_sequence_trial(
            n_items=n_items, isi_seconds=isi, n_trs=n_trs
        )
        slopes = compute_slope_timecourse(probs)
        
        ax.plot(t, slopes, 'o-', color=SPEED_COLORS[isi], lw=2, markersize=5)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.fill_between(t, slopes, 0, where=slopes > 0,
                        alpha=0.2, color='blue', label='Forward')
        ax.fill_between(t, slopes, 0, where=slopes < 0,
                        alpha=0.2, color='red', label='Backward')
        
        ax.set_xlabel('Time (TRs)')
        ax.set_ylabel('SODA slope')
        ax.set_title(f'G{idx+1}) Slope ({int(isi*1000)} ms ISI)')
        ax.legend(fontsize=7)
    
    fig.suptitle('Simulation 1: SODA — Ideal Case', fontsize=15, fontweight='bold')
    fig.savefig(FIGDIR / 'sim1_ideal_case.png')
    plt.close()
    print(f"✓ Saved sim1_ideal_case.png")


# =============================================================================
# SIMULATION 2: Heterogeneous classifier performance
# =============================================================================

def sim2_heterogeneous_classifiers():
    """Show how unequal classifier accuracy distorts SODA slopes."""
    
    params = ResponseParams()
    n_items = 5
    n_trs = 13
    t = np.arange(1, n_trs + 1, dtype=float)
    isi = 0.128  # 128 ms — moderate speed where effects are visible
    rng = np.random.default_rng(42)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Define scenarios
    scenarios = [
        {
            'name': 'Equal (ideal)',
            'amplitudes': np.array([60, 60, 60, 60, 60], dtype=float),
        },
        {
            'name': 'Gradient: best→worst',
            'amplitudes': np.array([80, 70, 60, 50, 40], dtype=float),
        },
        {
            'name': 'Gradient: worst→best',
            'amplitudes': np.array([40, 50, 60, 70, 80], dtype=float),
        },
        {
            'name': 'One outlier high (pos 1)',
            'amplitudes': np.array([90, 50, 50, 50, 50], dtype=float),
        },
        {
            'name': 'One outlier high (pos 5)',
            'amplitudes': np.array([50, 50, 50, 50, 90], dtype=float),
        },
        {
            'name': 'Random realistic',
            'amplitudes': rng.uniform(30, 80, size=5),
        },
    ]
    
    all_slopes = {}
    
    for idx, scenario in enumerate(scenarios):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        
        probs = simulate_heterogeneous_trial(
            n_items=n_items, isi_seconds=isi, n_trs=n_trs,
            class_amplitudes=scenario['amplitudes'], params=params,
        )
        slopes = compute_slope_timecourse(probs)
        all_slopes[scenario['name']] = slopes
        
        # Plot probability timecourses (thin lines) and slope (thick)
        ax2 = ax.twinx()
        for i in range(n_items):
            ax.plot(t, probs[i], '-', color=COLORS[i], lw=1, alpha=0.5)
        
        ax2.plot(t, slopes, 'ko-', lw=2.5, markersize=5, zorder=10)
        ax2.axhline(0, color='gray', ls='--', alpha=0.5)
        ax2.fill_between(t, slopes, 0, where=slopes > 0,
                         alpha=0.15, color='blue')
        ax2.fill_between(t, slopes, 0, where=slopes < 0,
                         alpha=0.15, color='red')
        
        amp_str = ', '.join([f'{a:.0f}' for a in scenario['amplitudes']])
        ax.set_title(f'{scenario["name"]}\nAmplitudes: [{amp_str}]', fontsize=9)
        ax.set_xlabel('Time (TRs)')
        ax.set_ylabel('Probability (%)', fontsize=9)
        ax2.set_ylabel('SODA slope', fontsize=9)
    
    fig.suptitle(
        'Simulation 2: Heterogeneous Classifier Performance (128 ms ISI)',
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim2_heterogeneous_classifiers.png')
    plt.close()
    print(f"✓ Saved sim2_heterogeneous_classifiers.png")
    
    # Summary comparison plot
    fig2, ax = plt.subplots(figsize=(10, 5))
    for name, slopes in all_slopes.items():
        ax.plot(t, slopes, 'o-', lw=2, markersize=4, label=name)
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel('Time (TRs)')
    ax.set_ylabel('SODA slope')
    ax.set_title('Slope comparison across classifier accuracy scenarios')
    ax.legend(fontsize=8, loc='best')
    fig2.tight_layout()
    fig2.savefig(FIGDIR / 'sim2_slope_comparison.png')
    plt.close()
    print(f"✓ Saved sim2_slope_comparison.png")


# =============================================================================
# SIMULATION 3: Multiple replay events — overlapping responses
# =============================================================================

def sim3_multiple_events():
    """Show how multiple replay events within one window interact."""
    
    params = ResponseParams(amplitude=40, baseline=20)
    n_items = 4
    n_trs = 20
    t = np.arange(1, n_trs + 1, dtype=float)
    isi = 0.05  # 50 ms inter-item interval within replay
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    
    scenarios = [
        # Row 1: Varying number of events, same direction
        {
            'name': '1 event (forward)',
            'onsets': [3.0],
            'directions': [1],
        },
        {
            'name': '2 events, same dir, 4 TR apart',
            'onsets': [3.0, 7.0],
            'directions': [1, 1],
        },
        {
            'name': '3 events, same dir, 4 TR apart',
            'onsets': [2.0, 6.0, 10.0],
            'directions': [1, 1, 1],
        },
        # Row 2: Opposing directions
        {
            'name': '2 events, opposing, 4 TR apart',
            'onsets': [3.0, 7.0],
            'directions': [1, -1],
        },
        {
            'name': '2 events, opposing, 2 TR apart',
            'onsets': [3.0, 5.0],
            'directions': [1, -1],
        },
        {
            'name': '2 events, opposing, 6 TR apart',
            'onsets': [3.0, 9.0],
            'directions': [1, -1],
        },
        # Row 3: Close timing — cancellation
        {
            'name': '2 fwd events, 1 TR apart',
            'onsets': [3.0, 4.0],
            'directions': [1, 1],
        },
        {
            'name': '2 fwd events, 2 TR apart',
            'onsets': [3.0, 5.0],
            'directions': [1, 1],
        },
        {
            'name': '2 fwd events, 3 TR apart',
            'onsets': [3.0, 6.0],
            'directions': [1, 1],
        },
    ]
    
    for idx, scenario in enumerate(scenarios):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        
        probs = simulate_multiple_replay_events(
            n_items=n_items, isi_seconds=isi, n_trs=n_trs,
            event_onsets_trs=scenario['onsets'],
            event_directions=scenario['directions'],
            params=params,
        )
        slopes = compute_slope_timecourse(probs)
        
        # Plot slopes
        ax.plot(t, slopes, 'k-', lw=2)
        ax.fill_between(t, slopes, 0, where=slopes > 0,
                        alpha=0.2, color='blue')
        ax.fill_between(t, slopes, 0, where=slopes < 0,
                        alpha=0.2, color='red')
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        
        # Mark event onsets
        for onset, dirn in zip(scenario['onsets'], scenario['directions']):
            marker = '▲' if dirn == 1 else '▼'
            color = 'blue' if dirn == 1 else 'red'
            ax.annotate(marker, xy=(onset, ax.get_ylim()[1] * 0.9),
                       fontsize=14, color=color, ha='center')
        
        ax.set_title(scenario['name'], fontsize=9)
        ax.set_xlabel('Time (TRs)')
        ax.set_ylabel('SODA slope')
        ax.set_xlim(0, n_trs + 1)
    
    fig.suptitle(
        'Simulation 3: Multiple Replay Events — Overlapping Responses',
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim3_multiple_events.png')
    plt.close()
    print(f"✓ Saved sim3_multiple_events.png")


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
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for idx, jitter_sd in enumerate(jitter_levels):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        
        _, onsets, all_slopes = simulate_trial_averaging(
            n_trials=n_trials, n_items=n_items,
            isi_seconds=0.05, n_trs=n_trs,
            onset_jitter_sd=jitter_sd,
            params=params, noise_sd=1.0,
            rng=np.random.default_rng(42),
        )
        
        # Plot individual trial slopes (light)
        for trial in range(min(n_trials, 15)):
            ax.plot(t, all_slopes[trial], '-', color='gray', alpha=0.15, lw=0.5)
        
        # Plot mean slope (bold)
        mean_slopes = np.mean(all_slopes, axis=0)
        sem_slopes = np.std(all_slopes, axis=0) / np.sqrt(n_trials)
        
        ax.plot(t, mean_slopes, 'k-', lw=2.5)
        ax.fill_between(t, mean_slopes - sem_slopes, mean_slopes + sem_slopes,
                        alpha=0.3, color='steelblue')
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        
        # Peak-to-peak of average as metric
        ptp = np.ptp(mean_slopes)
        ax.set_title(f'Jitter SD = {jitter_sd:.1f} TRs\n'
                     f'Peak-to-peak: {ptp:.3f}', fontsize=10)
        ax.set_xlabel('Time (TRs)')
        ax.set_ylabel('SODA slope')
    
    fig.suptitle(
        f'Simulation 4: Trial Averaging with Onset Jitter ({n_trials} trials)',
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim4_trial_averaging.png')
    plt.close()
    print(f"✓ Saved sim4_trial_averaging.png")
    
    # Summary: peak-to-peak vs jitter
    fig2, ax = plt.subplots(figsize=(8, 5))
    ptps = []
    jitters = np.linspace(0, 6, 25)
    for j in jitters:
        _, _, all_slopes = simulate_trial_averaging(
            n_trials=50, n_items=4, isi_seconds=0.05, n_trs=n_trs,
            onset_jitter_sd=j, params=params, noise_sd=1.0,
            rng=np.random.default_rng(42),
        )
        ptps.append(np.ptp(np.mean(all_slopes, axis=0)))
    
    ax.plot(jitters, ptps, 'ko-', lw=2)
    ax.set_xlabel('Onset jitter SD (TRs)')
    ax.set_ylabel('Peak-to-peak of averaged slope')
    ax.set_title('Signal degradation with increasing onset jitter')
    ax.axhline(0, color='gray', ls='--')
    fig2.tight_layout()
    fig2.savefig(FIGDIR / 'sim4_jitter_degradation.png')
    plt.close()
    print(f"✓ Saved sim4_jitter_degradation.png")


# =============================================================================
# Run all simulations
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SODA Primer Simulations")
    print("=" * 60)
    
    print("\n--- Simulation 1: Ideal case ---")
    sim1_ideal_case()
    
    print("\n--- Simulation 2: Heterogeneous classifiers ---")
    sim2_heterogeneous_classifiers()
    
    print("\n--- Simulation 3: Multiple replay events ---")
    sim3_multiple_events()
    
    print("\n--- Simulation 4: Trial averaging ---")
    sim4_trial_averaging()
    
    print("\n" + "=" * 60)
    print(f"All figures saved to {FIGDIR.resolve()}")
    print("=" * 60)
