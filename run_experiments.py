r"""Main multi-seed benchmark runner producing Table 1 and Table 2 in results/."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DEFAULT_SEEDS, SPARKOV_CONFIG, ULB_CONFIG
from src.controllers import ConformalFraudController
from src.data_loader import apply_stress_regime
from src.metrics import evaluate_trajectory_metrics
from src.scorers import build_and_cache_scored_stream


MAIN_POLICIES = [
    ("B1_static", "B1: Static LARC (Zi et al. 2026)"),
    ("B2_monthly", "B2: Monthly IPCW Recalibration"),
    ("B3_naive", "B3: Naive Early Feedback (Unweighted)"),
    ("B4_single_ipcw", "B4: Single-Stage IPCW (Delta_1 Only)"),
    ("B5_arora_prog", "B5: Arora et al. (Nov 2026) Prog-IPCW"),
    ("Ours1_unclipped_ht", "Ours-1: Dual-Channel (Uniform HT)"),
    ("Ours2_kkt_dr", "Ours-2: Dual-Channel (KKT-DR Proposed)"),
    ("Oracle", "Oracle: Instant-Label Conformal"),
]

ABLATION_POLICIES = [
    ("Ours1_unclipped_ht", "1. Uniform Audit + Unclipped HT (beta=0)"),
    ("Abl_unif_clipped_ht", "2. Uniform Audit + Clipped HT (beta=0)"),
    ("Abl_unif_dr", "3. Uniform Audit + Doubly Robust (beta=1)"),
    ("Abl_kkt_ht", "4. KKT Value-Stratified + Pure HT (beta=0)"),
    ("Ours2_kkt_dr", "5. KKT Value-Stratified + DR (beta=1, Ours-2)"),
]

REGIMES = [
    "R1_stationary",
    "R2_drift_shock",
    "R3_delay_shift",
    "R4_compound",
]


def run_benchmark_suite(n_seeds: int = 5) -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = DEFAULT_SEEDS[:n_seeds]

    table1_rows = []
    for cfg in (SPARKOV_CONFIG, ULB_CONFIG):
        base_df, splits = build_and_cache_scored_stream(cfg)

        for regime in REGIMES:
            print(f"\n--- Running {cfg.name} | Regime: {regime} ({n_seeds} seeds) ---")
            for policy_code, policy_label in MAIN_POLICIES:
                seed_metrics = []
                for s in seeds:
                    regime_df = apply_stress_regime(base_df, regime=regime, cfg=cfg, seed=s)
                    ctrl = ConformalFraudController(policy_mode=policy_code, cfg=cfg)
                    sim_out = ctrl.run_stream(regime_df, splits=splits, seed=s)
                    seed_metrics.append(evaluate_trajectory_metrics(sim_out))

                df_m = pd.DataFrame(seed_metrics)
                row = {
                    "dataset": cfg.name,
                    "regime": regime,
                    "policy_code": policy_code,
                    "policy_name": policy_label,
                    "rfr_long_mean": df_m["rfr_long"].mean(),
                    "rfr_long_std": df_m["rfr_long"].std(ddof=0),
                    "overshoot_30d_mean": df_m["overshoot_30d_max"].mean(),
                    "overshoot_30d_std": df_m["overshoot_30d_max"].std(ddof=0),
                    "cefl_shock_mean": df_m["cefl_shock"].mean(),
                    "workload_pct_mean": df_m["total_workload_pct"].mean(),
                    "net_value_mean": df_m["net_value_per_1k"].mean(),
                    "theta_chatter_mean": df_m["theta_chatter"].mean(),
                }
                table1_rows.append(row)
                print(
                    f"  {policy_label:<40} | RFR_long: {row['rfr_long_mean']:.2f}x | "
                    f"Peak30d: {row['overshoot_30d_mean']:.2f}x | "
                    f"CEFL: ${row['cefl_shock_mean']:6.1f} | "
                    f"Work: {row['workload_pct_mean']:4.2f}% | "
                    f"NetVal: ${row['net_value_mean']:7.1f} | "
                    f"Chatter: {row['theta_chatter_mean']:.4f}"
                )

    table1_df = pd.DataFrame(table1_rows)
    table1_path = results_dir / "table1_main_benchmark.csv"
    table1_df.to_csv(table1_path, index=False)
    print(f"\n[Saved Table 1] -> {table1_path}")

    # Run Table 2: Stage-0 Estimator & Allocation Ablation on Sparkov R4 Compound Shock
    print("\n--- Running Table 2: Stage-0 Estimator & KKT Allocation Ablation (Sparkov R4) ---")
    sparkov_df, sparkov_splits = build_and_cache_scored_stream(SPARKOV_CONFIG)
    table2_rows = []
    for policy_code, abl_label in ABLATION_POLICIES:
        seed_metrics = []
        for s in seeds:
            regime_df = apply_stress_regime(
                sparkov_df, regime="R4_compound", cfg=SPARKOV_CONFIG, seed=s
            )
            ctrl = ConformalFraudController(policy_mode=policy_code, cfg=SPARKOV_CONFIG)
            sim_out = ctrl.run_stream(regime_df, splits=sparkov_splits, seed=s)
            seed_metrics.append(evaluate_trajectory_metrics(sim_out))

        df_m = pd.DataFrame(seed_metrics)
        row = {
            "ablation_variant": abl_label,
            "rfr_long_mean": df_m["rfr_long"].mean(),
            "overshoot_30d_mean": df_m["overshoot_30d_max"].mean(),
            "cefl_shock_mean": df_m["cefl_shock"].mean(),
            "net_value_mean": df_m["net_value_per_1k"].mean(),
            "theta_chatter_mean": df_m["theta_chatter"].mean(),
        }
        table2_rows.append(row)
        print(
            f"  {abl_label:<45} | RFR_long: {row['rfr_long_mean']:.2f}x | "
            f"Peak30d: {row['overshoot_30d_mean']:.2f}x | "
            f"Chatter: {row['theta_chatter_mean']:.4f}"
        )

    table2_df = pd.DataFrame(table2_rows)
    table2_path = results_dir / "table2_stage0_ablation.csv"
    table2_df.to_csv(table2_path, index=False)
    print(f"[Saved Table 2] -> {table2_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dual-Channel Conformal Fraud Benchmark.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds (1..10)")
    args = parser.parse_args()
    run_benchmark_suite(n_seeds=args.n_seeds)