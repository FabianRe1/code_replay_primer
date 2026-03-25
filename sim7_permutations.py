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
from viz_style import (
    setup_style, add_panel_label, save_figure,
    FIGURES_DIR, FULL_WIDTH, BLUE, ORANGE, BLACK, GREY, LIGHT_GREY,
)

setup_style()


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

    fig = plt.figure(figsize=(FULL_WIDTH * 2.2, FULL_WIDTH * 2.2 * 0.78))
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # Extract data vectors
    taus = [r['kendall_tau'] for r in results]
    mean_slopes = [r['mean_slope'] for r in results]
    abs_means = [r['abs_mean'] for r in results]
    ptts = [r['peak_to_trough'] for r in results]

    true_result = [r for r in results if r['is_true']][0]
    rev_result = [r for r in results if r['is_reversal']][0]

    # Helper: scatter with highlighted true/reversal
    def _scatter_panel(ax, xs, ys, ylabel, label):
        # Background points (all non-special)
        bg_x, bg_y = [], []
        for r, x, y in zip(results, xs, ys):
            if not r['is_true'] and not r['is_reversal']:
                bg_x.append(x)
                bg_y.append(y)
        ax.scatter(bg_x, bg_y, c=LIGHT_GREY, s=12, alpha=0.3, edgecolors='none',
                   zorder=2)
        # True forward
        ax.scatter([true_result['kendall_tau']],
                   [true_result[ylabel] if isinstance(ylabel, str) else
                    ys[results.index(true_result)]],
                   c=BLUE, s=60, marker='D', edgecolors=BLACK, linewidths=0.5,
                   zorder=4, label=f"True ({true_result['perm_str']})")
        # True reversal
        ax.scatter([rev_result['kendall_tau']],
                   [rev_result[ylabel] if isinstance(ylabel, str) else
                    ys[results.index(rev_result)]],
                   c=ORANGE, s=60, marker='s', edgecolors=BLACK, linewidths=0.5,
                   zorder=4, label=f"Reversal ({rev_result['perm_str']})")
        add_panel_label(ax, label)

    # --- Panel A: Mean slope vs Kendall tau ---
    ax_a = fig.add_subplot(gs[0, 0])
    _scatter_panel(ax_a, taus, mean_slopes, 'mean_slope', 'A')
    ax_a.axhline(0, color=GREY, ls='--', lw=0.75, alpha=0.5)
    ax_a.set_xlabel("Kendall's \u03c4 (similarity to true order)")
    ax_a.set_ylabel('Mean SODA slope')
    ax_a.text(0.97, 0.97, 'Mean slope vs similarity', transform=ax_a.transAxes,
              fontsize=6, ha='right', va='top', color=GREY)
    ax_a.legend(fontsize=6, loc='upper left')

    # --- Panel B: Abs mean vs Kendall tau ---
    ax_b = fig.add_subplot(gs[0, 1])
    _scatter_panel(ax_b, taus, abs_means, 'abs_mean', 'B')
    ax_b.set_xlabel("Kendall's \u03c4")
    ax_b.set_ylabel('|Slope| mean')
    ax_b.text(0.97, 0.97, '|Slope| mean vs similarity', transform=ax_b.transAxes,
              fontsize=6, ha='right', va='top', color=GREY)

    # --- Panel C: Peak-to-trough vs Kendall tau ---
    ax_c = fig.add_subplot(gs[1, 0])
    _scatter_panel(ax_c, taus, ptts, 'peak_to_trough', 'C')
    ax_c.set_xlabel("Kendall's \u03c4")
    ax_c.set_ylabel('Peak-to-trough')
    ax_c.text(0.97, 0.97, 'Peak-to-trough vs similarity', transform=ax_c.transAxes,
              fontsize=6, ha='right', va='top', color=GREY)

    # --- Panel D: Reversal equivalence demonstration ---
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(t, true_result['slopes'], '-o', color=BLUE, lw=1.5, markersize=3,
              label=f"True ({true_result['perm_str']})")
    ax_d.plot(t, rev_result['slopes'], '-s', color=ORANGE, lw=1.5, markersize=3,
              label=f"Reversal ({rev_result['perm_str']})")
    ax_d.axhline(0, color=GREY, ls='--', lw=0.75, alpha=0.5)
    ax_d.set_xlabel('Time (TRs)')
    ax_d.set_ylabel('SODA slope')
    ax_d.text(0.97, 0.97, "Mirror-symmetric slopes",
              transform=ax_d.transAxes, fontsize=6, ha='right', va='top',
              color=GREY)
    ax_d.legend(fontsize=6, loc='lower right')
    ptt_txt = (f"ptt: {true_result['peak_to_trough']:.2f} (fwd) "
               f"= {rev_result['peak_to_trough']:.2f} (rev)")
    ax_d.text(0.02, 0.03, ptt_txt, transform=ax_d.transAxes,
              fontsize=5.5, va='bottom', color=GREY)
    add_panel_label(ax_d, 'D')

    # --- Panel E: Null distribution histogram ---
    ax_e = fig.add_subplot(gs[2, 0])
    ptts_arr = np.array(ptts)
    true_ptt = true_result['peak_to_trough']
    ax_e.hist(ptts_arr, bins=20, color=LIGHT_GREY, edgecolor='white', linewidth=0.5)
    ax_e.axvline(true_ptt, color=BLUE, lw=2, label=f'True: {true_ptt:.3f}')
    ax_e.axvline(np.mean(ptts_arr), color=BLACK, ls='--', lw=1,
                 label=f'Null mean: {np.mean(ptts_arr):.3f}')
    percentile = np.mean(ptts_arr >= true_ptt) * 100
    ax_e.set_xlabel('Peak-to-trough')
    ax_e.set_ylabel('Count')
    ax_e.text(0.97, 0.97, f'Null dist. \u2014 {percentile:.0f}% \u2265 true',
              transform=ax_e.transAxes, fontsize=6, ha='right', va='top', color=GREY)
    ax_e.legend(fontsize=6)
    add_panel_label(ax_e, 'E')

    # --- Panel F: Breakdown by correct adjacent pairs ---
    ax_f = fig.add_subplot(gs[2, 1])
    max_adj = n_items - 1
    adj_groups = {i: [] for i in range(max_adj + 1)}
    for r in results:
        adj_groups[r['n_correct_adj']].append(r['peak_to_trough'])

    positions = sorted(adj_groups.keys())
    bp_data = [adj_groups[p] for p in positions]
    bp = ax_f.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True,
                      medianprops=dict(color=BLACK, lw=1),
                      whiskerprops=dict(color=GREY), capprops=dict(color=GREY),
                      flierprops=dict(marker='.', markersize=3, color=GREY))

    # Gradient from ORANGE (0 correct) to BLUE (all correct)
    for i, (patch, pos) in enumerate(zip(bp['boxes'], positions)):
        frac = pos / max_adj
        # Interpolate ORANGE -> BLUE via hex
        r1, g1, b1 = int(ORANGE[1:3], 16), int(ORANGE[3:5], 16), int(ORANGE[5:7], 16)
        r2, g2, b2 = int(BLUE[1:3], 16), int(BLUE[3:5], 16), int(BLUE[5:7], 16)
        ri = int(r1 + frac * (r2 - r1))
        gi = int(g1 + frac * (g2 - g1))
        bi = int(b1 + frac * (b2 - b1))
        patch.set_facecolor(f'#{ri:02x}{gi:02x}{bi:02x}')
        patch.set_alpha(0.7)

    ax_f.axhline(true_ptt, color=BLUE, ls='--', lw=1, label='True sequence')
    ax_f.set_xlabel('# correct adjacent pairs preserved')
    ax_f.set_ylabel('Peak-to-trough')
    ax_f.text(0.97, 0.97, 'By adjacent pairs preserved',
              transform=ax_f.transAxes, fontsize=6, ha='right', va='top', color=GREY)
    ax_f.legend(fontsize=6)
    add_panel_label(ax_f, 'F')

    fig.suptitle(f'Simulation 7: Permutation null space ({n_items} items, '
                 f'{n_perms} permutations)',
                 fontsize=10, fontweight='bold')
    save_figure(fig, f'sim7_permutation_landscape_{n_items}items')
    plt.close()


def plot_reversal_equivalence_summary():
    """
    Show that for ALL unsigned metrics, forward and backward sequences
    produce identical values -- across both 4 and 5 items.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 2, FULL_WIDTH * 0.75))

    for idx, (ax, n_items) in enumerate(zip(axes, [4, 5])):
        results, _, _ = enumerate_permutation_space(n_items=n_items)

        true_r = [r for r in results if r['is_true']][0]
        rev_r = [r for r in results if r['is_reversal']][0]

        metrics = ['mean_slope', 'abs_mean', 'variance', 'peak_to_trough']
        labels = ['Mean slope\n(signed)', '|Slope|\nmean', 'Variance', 'Peak-to-\ntrough']

        x = np.arange(len(metrics))
        width = 0.35

        true_vals = [true_r[m] for m in metrics]
        rev_vals = [rev_r[m] for m in metrics]

        ax.bar(x - width / 2, true_vals, width,
               label=f'True ({true_r["perm_str"]})', color=BLUE, alpha=0.85)
        ax.bar(x + width / 2, rev_vals, width,
               label=f'Reversal ({rev_r["perm_str"]})', color=ORANGE, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('Metric value')
        ax.set_title(f'{n_items} items')
        ax.legend(fontsize=7)
        ax.axhline(0, color=GREY, ls='--', lw=0.75, alpha=0.5)

        for i, (tv, rv) in enumerate(zip(true_vals, rev_vals)):
            if abs(tv - rv) < 1e-10:
                ax.text(i, max(tv, rv) * 1.05, '=', ha='center', fontsize=10,
                        fontweight='bold', color=BLUE)
            else:
                ax.text(i, max(abs(tv), abs(rv)) * 1.1, '\u2260', ha='center',
                        fontsize=10, fontweight='bold', color=ORANGE)

        add_panel_label(ax, chr(65 + idx))

    fig.suptitle('Reversal equivalence: unsigned metrics cannot\n'
                 'distinguish true order from reversal',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'sim7_reversal_equivalence')
    plt.close()


def plot_effective_null_size():
    """
    For 4 and 5 items, show how many permutations are "truly different"
    when using signed vs unsigned metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 1.8, FULL_WIDTH * 0.75))

    for idx, (ax, n_items) in enumerate(zip(axes, [4, 5])):
        results, _, _ = enumerate_permutation_space(n_items=n_items)

        ptts = np.array([r['peak_to_trough'] for r in results])
        unique_ptts = np.unique(np.round(ptts, 6))

        mean_slopes_arr = np.array([r['mean_slope'] for r in results])
        unique_slopes = np.unique(np.round(mean_slopes_arr, 6))

        n_total = len(results)
        n_unique_unsigned = len(unique_ptts)
        n_unique_signed = len(unique_slopes)

        categories = ['Total\npermutations', 'Unique signed\n(mean slope)',
                       'Unique unsigned\n(peak-to-trough)']
        values = [n_total, n_unique_signed, n_unique_unsigned]
        colors_bar = [GREY, ORANGE, BLUE]

        bars = ax.bar(categories, values, color=colors_bar, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha='center', fontsize=9, fontweight='bold',
                    color=BLACK)

        ax.set_ylabel('Count')
        ax.set_title(f'{n_items} items ({n_total} permutations)')
        add_panel_label(ax, chr(65 + idx))

    fig.suptitle('Effective null space size: signed vs unsigned metrics',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'sim7_effective_null_size')
    plt.close()


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
