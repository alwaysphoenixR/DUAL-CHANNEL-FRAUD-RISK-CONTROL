r"""Generates Publication Figures 1-4 in results/figures/ from experimental_protocol.md."""

import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked via `python notebooks/generate_paper_figures.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.config import SPARKOV_CONFIG
from src.controllers import ConformalFraudController
from src.data_loader import apply_stress_regime
from src.metrics import evaluate_trajectory_metrics
from src.scorers import build_and_cache_scored_stream


def compute_rolling_30d_rfr(sim_out: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    r"""Computes trailing 30-day rolling Released Fraud Ratio across the deployment horizon."""
    days = sim_out["eval_days"]
    norm_loss = sim_out["eval_true_norm_loss"]
    alpha_target = float(sim_out["alpha_target"][0])
    grid = sim_out["deploy_days_grid"]

    n_days = len(grid)
    daily_loss = np.zeros(n_days, dtype=np.float64)
    daily_cnt = np.zeros(n_days, dtype=np.float64)
    offset = int(grid[0])
    np.add.at(daily_loss, days - offset, norm_loss)
    np.add.at(daily_cnt, days - offset, 1.0)

    rfr_30d = np.zeros(n_days, dtype=np.float64)
    for i in range(n_days):
        s = max(0, i - 29)
        rfr_30d[i] = (
            np.sum(daily_loss[s : i + 1]) / max(1.0, np.sum(daily_cnt[s : i + 1]))
        ) / alpha_target
    return grid, rfr_30d


def main() -> None:
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_df, splits = build_and_cache_scored_stream(SPARKOV_CONFIG)
    r4_df = apply_stress_regime(base_df, regime="R4_compound", cfg=SPARKOV_CONFIG, seed=42)

    # --- FIGURE 1: Transient Shock Trajectory (Days 240..480) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=200)
    plot_specs = [
        ("B1_static", "B1: Static LARC", "#7f7f7f", "--"),
        ("B5_arora_prog", "B5: Arora et al. (Nov 2026) Prog-IPCW", "#d95f02", "-."),
        ("Ours2_kkt_dr", "Ours-2: Dual-Channel KKT-DR (Proposed)", "#1f78b4", "-"),
        ("Oracle", "Oracle (Instant Ground Truth)", "#2ca02c", ":"),
    ]
    for code, label, color, ls in plot_specs:
        ctrl = ConformalFraudController(policy_mode=code, cfg=SPARKOV_CONFIG)
        out = ctrl.run_stream(r4_df, splits=splits, seed=42)
        grid, rfr_30d = compute_rolling_30d_rfr(out)
        mask = (grid >= 240) & (grid <= 480)
        ax1.plot(grid[mask], rfr_30d[mask], label=label, color=color, linestyle=ls, linewidth=2.0)
        alpha_t = float(out["alpha_target"][0])
        ax2.plot(
            grid[mask],
            np.exp(out["theta_traj"][mask]) / alpha_t,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.8,
        )

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Target Risk Budget (1.0x)")
    ax1.axvline(
        SPARKOV_CONFIG.shock_day, color="red", alpha=0.5, linestyle=":", label="Shock Onset (Day 300)"
    )
    ax2.axvline(SPARKOV_CONFIG.shock_day, color="red", alpha=0.5, linestyle=":")
    ax1.set_ylabel("Rolling 30-Day Released Fraud Ratio")
    ax1.set_title("Figure 1: Transient Shock Recovery Trajectory on Sparkov (Compound Shock R4)")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Nominal Multiplier exp(theta_d) / alpha")
    ax2.set_xlabel("Deployment Day (d)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_shock_trajectory.png")
    plt.close()

    # --- FIGURE 2: Variance & Controller State Chatter Comparison (Ours-1 vs Ours-2) ---
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    for code, label, color, alpha_v in [
        ("Ours1_unclipped_ht", "Ours-1: Uniform Unclipped HT (beta=0)", "#e41a1c", 0.65),
        ("Ours2_kkt_dr", "Ours-2: KKT Value-Stratified DR (beta=1)", "#1f78b4", 0.95),
    ]:
        ctrl = ConformalFraudController(policy_mode=code, cfg=SPARKOV_CONFIG)
        out = ctrl.run_stream(r4_df, splits=splits, seed=42)
        grid = out["deploy_days_grid"]
        alpha_t = float(out["alpha_target"][0])
        ax.plot(
            grid,
            out["theta_traj"] - np.log(alpha_t),
            label=label,
            color=color,
            alpha=alpha_v,
            linewidth=1.6,
        )

    ax.axvline(SPARKOV_CONFIG.shock_day, color="black", linestyle=":", label="Shock Onset (Day 300)")
    ax.set_xlabel("Deployment Day (d)")
    ax.set_ylabel("Log-Budget State Offset (theta_d - log alpha)")
    ax.set_title("Figure 2: Controller State Trajectory & Chatter Suppression (Theorem 1 & Corollary 1)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_chatter_suppression.png")
    plt.close()

    # --- FIGURE 3: Audit Budget q_audit Pareto Frontier ---
    q_grid = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
    overshoots, net_vals = [], []
    for q_val in q_grid:
        ctrl = ConformalFraudController(
            policy_mode="Ours2_kkt_dr",
            cfg=SPARKOV_CONFIG,
            q_audit=q_val,
            e_min=min(0.002, q_val * 0.5),
        )
        out = ctrl.run_stream(r4_df, splits=splits, seed=42)
        m = evaluate_trajectory_metrics(out)
        overshoots.append(m["overshoot_30d_max"])
        net_vals.append(m["net_value_per_1k"])

    fig, ax_l = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax_r = ax_l.twinx()
    q_pct = [q * 100.0 for q in q_grid]
    ax_l.plot(q_pct, overshoots, marker="o", color="#d95f02", linewidth=2, label="Peak 30d Overshoot")
    ax_r.plot(
        q_pct,
        net_vals,
        marker="s",
        color="#1b9e77",
        linewidth=2,
        linestyle="--",
        label="Net Value ($/1k txns)",
    )
    ax_l.set_xlabel("Stage-0 Diagnostic Audit Budget q_audit (%)")
    ax_l.set_ylabel("Peak 30-Day Released Fraud Ratio", color="#d95f02")
    ax_r.set_ylabel("Net Business Value ($ / 1,000 txns)", color="#1b9e77")
    ax_l.set_title("Figure 3: Pareto Trade-Off of Diagnostic Audit Rate q_audit")
    ax_l.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_audit_pareto.png")
    plt.close()

    # --- FIGURE 4: Two-Speed Integral Gain eta Sensitivity ---
    eta_grid = [0.03, 0.06, 0.10, 0.15, 0.20, 0.25]
    cefl_vals, chatter_vals = [], []
    for eta_val in eta_grid:
        ctrl = ConformalFraudController(
            policy_mode="Ours2_kkt_dr", cfg=SPARKOV_CONFIG, eta_dual=eta_val
        )
        out = ctrl.run_stream(r4_df, splits=splits, seed=42)
        m = evaluate_trajectory_metrics(out)
        cefl_vals.append(m["cefl_shock"])
        chatter_vals.append(m["theta_chatter"])

    fig, ax_l = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax_r = ax_l.twinx()
    ax_l.plot(
        eta_grid, cefl_vals, marker="o", color="#2b5c8f", linewidth=2, label="Post-Shock Excess Loss ($/1k)"
    )
    ax_r.plot(
        eta_grid,
        chatter_vals,
        marker="^",
        color="#7b1fa2",
        linewidth=2,
        linestyle="--",
        label="State Chatter",
    )
    ax_l.set_xlabel("Two-Speed Integral Gain (eta)")
    ax_l.set_ylabel("Post-Shock Excess Fraud Loss CEFL ($/1k)", color="#2b5c8f")
    ax_r.set_ylabel("Controller State Chatter (sigma_dtheta)", color="#7b1fa2")
    ax_l.set_title("Figure 4: Two-Speed Gain Sensitivity (Theorem 4 Critical Damping Regime)")
    ax_l.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_gain_sensitivity.png")
    plt.close()
    print(f"[Saved Figures 1-4] -> {fig_dir}")


if __name__ == "__main__":
    main()