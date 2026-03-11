"""
Simulation 7: Permutation Null Distributions for SODA

Key questions:
  1. How does the expected SODA slope/amplitude vary across all possible
     permutations of a sequence?
  2. Are partial permutations (e.g., ABDC) biased toward the true signal?
  3. Are forward (ABCD) and backward (DCBA) equivalent for amplitude-based metrics?
  4. What is the effective null space when using amplitude vs signed slope?
  5. How should one construct a proper null distribution?

Approach:
  - For N items (4 and 5), enumerate ALL possible permutations
  - For each permutation, simulate the SODA slope timecourse as if that
    were the "true" ordering (i.e., compute slopes assuming that permutation
    is the hypothesis)
  - Compute: mean slope, absolute slope, amplitude of sinusoidal fit
  - Characterize the null distribution and identify problematic permutations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import permutations
from scipy.stats import kendalltau, spearmanr

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_model import ResponseParams, multi_item_response
from soda import compute_slope_timecourse
from aggregation import (
    mean_slope, abs_mean, slope_variance, peak_to_trough,
)

FIGDIR = Path(__file__).parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# =====================================================================
# Core: compute SODA slope for a given assumed ordering
# =====================================================================

def compute_soda_for_ordering(
    probs: np.ndarray,
    assumed_order: tuple,
) -> np.ndarray:
    """
    Compute SODA slope timecourse given classifier probabilities and an
    assumed sequential ordering.
    
    The slope at each TR regresses assumed position onto probabilities.
    
    Args:
        probs: (n_items, n_trs) — raw classifier probabilities, indexed by
               true item identity (item 0 = first in true sequence).
        assumed_order: tuple of item indices in the assumed order.
            E.g., (0,1,2,3) = true order, (3,2,1,0) = reversal.
            The slope uses positions 1,2,...,N assigned to these items.
    
    Returns:
        slopes: (n_trs,) — SODA slope at each TR under assumed ordering.
    """
    from scipy.stats import linregress
    
    n_items, n_trs = probs.shape
    slopes = np.zeros(n_trs)
    
    # Build position vector: item at assumed_order[0] gets position 1, etc.
    positions = np.zeros(n_items)
    for pos_idx, item_idx in enumerate(assumed_order):
        positions[item_idx] = pos_idx + 1  # 1-indexed
    
    for t in range(n_trs):
        probs_at_t = probs[:, t]
        slope, _, _, _, _ = linregress(positions, probs_at_t)
        slopes[t] = -slope  # flip: positive = forward
    
    return slopes


def ordinal_distance(perm, true_order):
    """
    Kendall's tau distance: number of pairwise disagreements.
    0 = identical, max = fully reversed.
    """
    tau, _ = kendalltau(true_order, perm)
    # Convert tau correlation to distance (0 to 1, where 1 = identical)
    return (1 - tau) / 2  # 0 = identical, 1 = fully reversed


def count_correct_adjacencies(perm, true_order):
    """Count how many adjacent pairs in perm match the true order."""
    count = 0
    true_pairs = set()
    for i in range(len(true_order) - 1):
        true_pairs.add((true_order[i], true_order[i+1]))
    for i in range(len(perm) - 1):
        if (perm[i], perm[i+1]) in true_pairs:
            count += 1
    return count


# =====================================================================
# Simulation: enumerate all permutations
# =====================================================================

def enumerate_permutation_space(n_items=4, isi_seconds=0.05, noise_sd=0.0):
    """
    For a given true sequence, simulate probabilities and compute
    SODA metrics under every possible assumed ordering.
    """
    params = ResponseParams(amplitude=40.0, baseline=20.0)
    n_trs = 13
    t = np.arange(1, n_trs + 1, dtype=float)
    
    # Simulate probabilities for the TRUE ordering
    true_order = tuple(range(n_items))
    probs = multi_item_response(t, n_items, isi_seconds, 0.0, 1.25, params)
    
    if noise_sd > 0:
        rng = np.random.default_rng(42)
        probs = probs + rng.normal(0, noise_sd, probs.shape)
        probs = np.clip(probs, 0, 100)
    
    all_perms = list(permutations(range(n_items)))
    
    results = []
    for perm in all_perms:
        slopes = compute_soda_for_ordering(probs, perm)
        
        # Metrics
        ms = mean_slope(slopes)
        am = abs_mean(slopes)
        var = slope_variance(slopes)
        ptt = peak_to_trough(slopes)
        
        # Ordinal relationship to true sequence
        tau, _ = kendalltau(true_order, perm)
        n_correct_adj = count_correct_adjacencies(perm, true_order)
        
        # Is this the reversal?
        is_reversal = (perm == tuple(reversed(true_order)))
        is_true = (perm == true_order)
        
        results.append({
            'perm': perm,
            'perm_str': ''.join(chr(65 + i) for i in perm),  # e.g., "ABCD"
            'mean_slope': ms,
            'abs_mean': am,
            'variance': var,
            'peak_to_trough': ptt,
            'kendall_tau': tau,
            'n_correct_adj': n_correct_adj,
            'is_reversal': is_reversal,
            'is_true': is_true,
            'slopes': slopes,
        })
    
    return results, probs, t


# =====================================================================
# Plotting
# =====================================================================

def plot_permutation_landscape(n_items=4):
    """
    Main figure: show how SODA metrics distribute across all permutations.
    """
    results, probs, t = enumerate_permutation_space(n_items=n_items)
    n_perms = len(results)
    
    # Sort by Kendall tau (similarity to true order)
    results_sorted = sorted(results, key=lambda r: r['kendall_tau'], reverse=True)
    
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)
    
    # --- Panel A: Mean slope vs Kendall tau ---
    ax_a = fig.add_subplot(gs[0, 0])
    taus = [r['kendall_tau'] for r in results]
    mean_slopes = [r['mean_slope'] for r in results]
    colors = ['red' if r['is_true'] else 'blue' if r['is_reversal'] else 'gray'
              for r in results]
    ax_a.scatter(taus, mean_slopes, c=colors, s=30, alpha=0.6, edgecolors='none')
    
    # Highlight true and reversal
    for r in results:
        if r['is_true']:
            ax_a.annotate(r['perm_str'], (r['kendall_tau'], r['mean_slope']),
                         fontsize=8, color='red', fontweight='bold',
                         xytext=(5, 5), textcoords='offset points')
        elif r['is_reversal']:
            ax_a.annotate(r['perm_str'], (r['kendall_tau'], r['mean_slope']),
                         fontsize=8, color='blue', fontweight='bold',
                         xytext=(5, 5), textcoords='offset points')
    
    ax_a.set_xlabel("Kendall's τ (similarity to true order)")
    ax_a.set_ylabel('Mean SODA slope')
    ax_a.set_title('A) Mean slope vs ordinal similarity')
    ax_a.axhline(0, color='gray', ls='--', alpha=0.5)
    
    # --- Panel B: Abs mean / Variance vs Kendall tau ---
    ax_b = fig.add_subplot(gs[0, 1])
    abs_means = [r['abs_mean'] for r in results]
    ax_b.scatter(taus, abs_means, c=colors, s=30, alpha=0.6, edgecolors='none')
    for r in results:
        if r['is_true'] or r['is_reversal']:
            ax_b.annotate(r['perm_str'], (r['kendall_tau'], r['abs_mean']),
                         fontsize=8, color='red' if r['is_true'] else 'blue',
                         fontweight='bold', xytext=(5, 5), textcoords='offset points')
    ax_b.set_xlabel("Kendall's τ")
    ax_b.set_ylabel('|Slope| mean')
    ax_b.set_title('B) Absolute slope mean vs ordinal similarity')
    
    # --- Panel C: Peak-to-trough vs Kendall tau ---
    ax_c = fig.add_subplot(gs[1, 0])
    ptts = [r['peak_to_trough'] for r in results]
    ax_c.scatter(taus, ptts, c=colors, s=30, alpha=0.6, edgecolors='none')
    for r in results:
        if r['is_true'] or r['is_reversal']:
            ax_c.annotate(r['perm_str'], (r['kendall_tau'], r['peak_to_trough']),
                         fontsize=8, color='red' if r['is_true'] else 'blue',
                         fontweight='bold', xytext=(5, 5), textcoords='offset points')
    ax_c.set_xlabel("Kendall's τ")
    ax_c.set_ylabel('Peak-to-trough')
    ax_c.set_title('C) Peak-to-trough vs ordinal similarity')
    
    # --- Panel D: Reversal equivalence demonstration ---
    ax_d = fig.add_subplot(gs[1, 1])
    true_result = [r for r in results if r['is_true']][0]
    rev_result = [r for r in results if r['is_reversal']][0]
    
    ax_d.plot(t, true_result['slopes'], 'r-o', lw=2, markersize=4,
              label=f"True ({true_result['perm_str']})")
    ax_d.plot(t, rev_result['slopes'], 'b-s', lw=2, markersize=4,
              label=f"Reversal ({rev_result['perm_str']})")
    ax_d.axhline(0, color='gray', ls='--', alpha=0.5)
    ax_d.set_xlabel('Time (TRs)')
    ax_d.set_ylabel('SODA slope')
    ax_d.set_title('D) True vs reversed: mirror-symmetric slopes')
    ax_d.legend(fontsize=9)
    
    # Add text with amplitude comparison
    ax_d.text(0.02, 0.98,
              f"True ptt={true_result['peak_to_trough']:.3f}\n"
              f"Rev  ptt={rev_result['peak_to_trough']:.3f}\n"
              f"→ identical for unsigned metrics",
              transform=ax_d.transAxes, fontsize=8, va='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # --- Panel E: Distribution of peak-to-trough across all permutations ---
    ax_e = fig.add_subplot(gs[2, 0])
    ptts_arr = np.array(ptts)
    ax_e.hist(ptts_arr, bins=20, color='gray', alpha=0.6, edgecolor='white')
    true_ptt = true_result['peak_to_trough']
    ax_e.axvline(true_ptt, color='red', lw=2, label=f'True: {true_ptt:.3f}')
    ax_e.axvline(np.mean(ptts_arr), color='black', ls='--', lw=1.5,
                 label=f'Null mean: {np.mean(ptts_arr):.3f}')
    
    # What percentile is the true value?
    percentile = np.mean(ptts_arr >= true_ptt) * 100
    ax_e.set_xlabel('Peak-to-trough')
    ax_e.set_ylabel('Count')
    ax_e.set_title(f'E) Null distribution of peak-to-trough ({n_items} items)\n'
                   f'{percentile:.0f}% of permutations ≥ true value')
    ax_e.legend(fontsize=8)
    
    # --- Panel F: Breakdown by number of correct adjacent pairs ---
    ax_f = fig.add_subplot(gs[2, 1])
    max_adj = n_items - 1
    adj_groups = {i: [] for i in range(max_adj + 1)}
    for r in results:
        adj_groups[r['n_correct_adj']].append(r['peak_to_trough'])
    
    positions = sorted(adj_groups.keys())
    bp_data = [adj_groups[p] for p in positions]
    bp = ax_f.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True)
    
    cmap = plt.cm.RdYlGn
    for i, (patch, pos) in enumerate(zip(bp['boxes'], positions)):
        patch.set_facecolor(cmap(pos / max_adj))
        patch.set_alpha(0.7)
    
    ax_f.axhline(true_ptt, color='red', ls='--', lw=1.5, label='True sequence')
    ax_f.set_xlabel('# correct adjacent pairs preserved')
    ax_f.set_ylabel('Peak-to-trough')
    ax_f.set_title(f'F) Peak-to-trough by partial order preservation')
    ax_f.legend(fontsize=8)
    
    fig.suptitle(f'Simulation 7: Permutation Null Space for SODA ({n_items} items, '
                 f'{n_perms} permutations)',
                 fontsize=14, fontweight='bold')
    fig.savefig(FIGDIR / f'sim7_permutation_landscape_{n_items}items.png')
    plt.close()
    print(f"✓ Saved sim7_permutation_landscape_{n_items}items.png")


def plot_reversal_equivalence_summary():
    """
    Show that for ALL unsigned metrics, forward and backward sequences
    produce identical values — across both 4 and 5 items.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, n_items in zip(axes, [4, 5]):
        results, _, _ = enumerate_permutation_space(n_items=n_items)
        
        # Find true and reversal
        true_r = [r for r in results if r['is_true']][0]
        rev_r = [r for r in results if r['is_reversal']][0]
        
        metrics = ['mean_slope', 'abs_mean', 'variance', 'peak_to_trough']
        labels = ['Mean slope\n(signed)', '|Slope| mean', 'Variance', 'Peak-to-trough']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        true_vals = [true_r[m] for m in metrics]
        rev_vals = [rev_r[m] for m in metrics]
        
        bars1 = ax.bar(x - width/2, true_vals, width, label=f'True ({true_r["perm_str"]})',
                       color='#e41a1c', alpha=0.8)
        bars2 = ax.bar(x + width/2, rev_vals, width, label=f'Reversal ({rev_r["perm_str"]})',
                       color='#377eb8', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Metric value')
        ax.set_title(f'{n_items} items')
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        
        # Annotate: signed metric differs, unsigned are identical
        for i, (tv, rv) in enumerate(zip(true_vals, rev_vals)):
            if abs(tv - rv) < 1e-10:
                ax.text(i, max(tv, rv) * 1.05, '=', ha='center', fontsize=12,
                       fontweight='bold', color='green')
            else:
                ax.text(i, max(abs(tv), abs(rv)) * 1.1, '≠', ha='center',
                       fontsize=12, fontweight='bold', color='red')
    
    fig.suptitle('Reversal Equivalence: Forward (ABCD) vs Backward (DCBA)\n'
                 'Unsigned metrics cannot distinguish true order from reversal',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim7_reversal_equivalence.png')
    plt.close()
    print(f"✓ Saved sim7_reversal_equivalence.png")


def plot_effective_null_size():
    """
    For 4 and 5 items, show how many permutations are "truly different"
    when using signed vs unsigned metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, n_items in zip(axes, [4, 5]):
        results, _, _ = enumerate_permutation_space(n_items=n_items)
        
        # Group permutations by peak-to-trough (unsigned)
        ptts = np.array([r['peak_to_trough'] for r in results])
        unique_ptts = np.unique(np.round(ptts, 6))
        
        # Group by mean slope (signed)
        mean_slopes = np.array([r['mean_slope'] for r in results])
        unique_slopes = np.unique(np.round(mean_slopes, 6))
        
        n_total = len(results)
        n_unique_unsigned = len(unique_ptts)
        n_unique_signed = len(unique_slopes)
        
        categories = ['Total\npermutations', 'Unique signed\n(mean slope)', 
                      'Unique unsigned\n(peak-to-trough)']
        values = [n_total, n_unique_signed, n_unique_unsigned]
        colors_bar = ['#999999', '#e41a1c', '#377eb8']
        
        bars = ax.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='white')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Count')
        ax.set_title(f'{n_items} items ({n_total}! = {n_total} permutations)')
        ax.spines[['top', 'right']].set_visible(False)
    
    fig.suptitle('Effective Null Space Size: Signed vs Unsigned Metrics',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sim7_effective_null_size.png')
    plt.close()
    print(f"✓ Saved sim7_effective_null_size.png")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Simulation 7: Permutation Null Distributions")
    print("=" * 60)
    
    print("\n--- 4 items ---")
    plot_permutation_landscape(n_items=4)
    
    print("\n--- 5 items ---")
    plot_permutation_landscape(n_items=5)
    
    print("\n--- Reversal equivalence ---")
    plot_reversal_equivalence_summary()
    
    print("\n--- Effective null size ---")
    plot_effective_null_size()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
