"""
Master runner: regenerate all SODA primer figures.

Usage:
    python run_all.py           # Run all simulations
    python run_all.py --sim 5   # Run only simulation 5
    python run_all.py --list     # List available simulations
"""

import sys
import argparse
import time
from pathlib import Path

# Ensure we can import from this directory
sys.path.insert(0, str(Path(__file__).parent))

from viz_style import setup_style, FIGURES_DIR

setup_style()


def list_sims():
    """Print available simulations."""
    sims = {
        0: ("Figure 0: Sequentiality illustration", "run_simulations.py"),
        1: ("Sim 1: Ideal SODA case", "run_simulations.py"),
        2: ("Sim 2: Heterogeneous classifiers", "run_simulations.py"),
        3: ("Sim 3: Multiple replay events", "run_simulations.py"),
        4: ("Sim 4: Trial averaging & jitter", "run_simulations.py"),
        5: ("Sim 5: Metric sensitivity comparison", "sim5_metric_sensitivity.py"),
        6: ("Sim 6: Normalization strategies", "sim6_normalization.py"),
        7: ("Sim 7: Permutation null distributions", "sim7_permutations.py"),
        8: ("Sim 8: Template matching & Bayes", "sim8_template_bayes.py"),
        9: ("Sim 9: Normalization x ISI interaction", "sim9_normalization_isi.py"),
        10: ("Sim 10: Number of sequence items", "sim10_n_items.py"),
        11: ("Sim 11: TR sensitivity", "sim11_tr_sensitivity.py"),
        12: ("Sim 12: Realistic noise structure", "sim12_realistic_noise.py"),
        13: ("Sim 13: Reactivation vs sequentiality", "sim13_reactivation_vs_sequentiality.py"),
    }
    print("\nAvailable simulations:")
    for num, (desc, script) in sims.items():
        print(f"  {num:2d}. {desc}  ({script})")
    print()


def run_sim(num):
    """Run a specific simulation by number."""
    t0 = time.time()

    if num == 0:
        from run_simulations import fig0_sequentiality_illustration
        fig0_sequentiality_illustration()
    elif num == 1:
        from run_simulations import sim1_ideal_case
        sim1_ideal_case()
    elif num == 2:
        from run_simulations import sim2_heterogeneous_classifiers
        sim2_heterogeneous_classifiers()
    elif num == 3:
        from run_simulations import sim3_multiple_events
        sim3_multiple_events()
    elif num == 4:
        from run_simulations import sim4_trial_averaging
        sim4_trial_averaging()
    elif num == 5:
        from sim5_metric_sensitivity import (
            condition_A_snr_sweep, condition_B_onset_jitter,
            condition_C_heterogeneous_classifiers, condition_D_multiple_events,
            plot_summary_heatmap,
        )
        n_trials = 40
        _, res_A = condition_A_snr_sweep(n_trials=n_trials)
        _, res_B = condition_B_onset_jitter(n_trials=n_trials)
        _, res_C = condition_C_heterogeneous_classifiers(n_trials=n_trials)
        res_D = condition_D_multiple_events(n_trials=n_trials)
        plot_summary_heatmap(res_A, res_B, res_C, res_D)
    elif num == 6:
        from sim6_normalization import (
            plot_example_normalization_effect, run_normalization_comparison,
        )
        plot_example_normalization_effect()
        run_normalization_comparison(n_trials=40)
    elif num == 7:
        from sim7_permutations import (
            plot_permutation_landscape, plot_reversal_equivalence_summary,
            plot_effective_null_size,
        )
        plot_permutation_landscape(n_items=4)
        plot_permutation_landscape(n_items=5)
        plot_reversal_equivalence_summary()
        plot_effective_null_size()
    elif num == 8:
        from sim8_template_bayes import (
            run_comparison, plot_comparison, plot_example_template_matching,
        )
        plot_example_template_matching()
        jitter_sds, results, metric_names = run_comparison(n_trials=35)
        plot_comparison(jitter_sds, results, metric_names)
    elif num == 9:
        from sim9_normalization_isi import run_example_and_dprime
        run_example_and_dprime()
    elif num == 10:
        from sim10_n_items import run_n_items_sweep
        run_n_items_sweep()
    elif num == 11:
        from sim11_tr_sensitivity import run_tr_sweep
        run_tr_sweep()
    elif num == 12:
        from sim12_realistic_noise import run_noise_sweep
        run_noise_sweep()
    elif num == 13:
        from sim13_reactivation_vs_sequentiality import run_reactivation_vs_sequentiality
        run_reactivation_vs_sequentiality()
    else:
        print(f"  Unknown simulation number: {num}")
        return

    elapsed = time.time() - t0
    print(f"  [{elapsed:.1f}s]")


def main():
    parser = argparse.ArgumentParser(description="Run SODA primer simulations")
    parser.add_argument("--sim", type=int, nargs="+",
                        help="Simulation number(s) to run")
    parser.add_argument("--list", action="store_true",
                        help="List available simulations")
    args = parser.parse_args()

    if args.list:
        list_sims()
        return

    sim_nums = args.sim if args.sim else list(range(0, 13))

    print("=" * 60)
    print("SODA Primer — Figure Generation")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)

    t_total = time.time()

    for num in sim_nums:
        print(f"\n--- Simulation {num} ---")
        try:
            run_sim(num)
        except Exception as e:
            print(f"  ERROR: {e}")

    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"All done in {elapsed_total:.1f}s")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
