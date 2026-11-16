r"""Evaluation metrics from Section 5 of experimental_protocol.md."""

import numpy as np

from src.config import ECONOMIC_CFG


def evaluate_trajectory_metrics(sim_out: dict[str, np.ndarray]) -> dict[str, float]:
    r"""
    Computes the 6 core metrics on a completed deployment trajectory:
      1. rfr_long             : Long-Run Released Fraud Ratio over matured transactions F_T
      2. overshoot_30d_max    : Peak 30-Day Rolling Released Fraud Ratio in [d_shock, d_shock + 90]
      3. cefl_shock           : Cumulative Post-Shock Excess Fraud Loss ($ / 1k txns)
      4. total_workload_pct   : Mandatory Review Rate (%) + Diagnostic Audit Rate (%)
      5. net_value_per_1k     : Net Economic Business Value V(\pi) ($ / 1k txns)
      6. theta_chatter        : Controller State Chatter \sigma_{\Delta\theta}
    """
    days = sim_out["eval_days"]
    amounts = sim_out["eval_amounts"]
    y_true = sim_out["eval_y_true"].astype(np.float64)
    approve = sim_out["eval_approve"]
    audit = sim_out["eval_audit"]
    norm_loss = sim_out["eval_true_norm_loss"]

    alpha_target = float(sim_out["alpha_target"][0])
    a_bar = float(sim_out["a_bar"][0])
    delta_m = int(sim_out["delta_m"][0])
    shock_day = int(sim_out["shock_day"][0])
    theta_traj = sim_out["theta_traj"]
    deploy_days_grid = sim_out["deploy_days_grid"]

    max_day = int(deploy_days_grid[-1])
    # 1. Long-Run Released Fraud Ratio over matured cohort F_T = {t : d_t + \Delta_M <= T}
    matured_mask = days <= (max_day - delta_m)
    if not np.any(matured_mask):
        matured_mask = np.ones_like(days, dtype=bool)
    rfr_long = float(np.mean(norm_loss[matured_mask]) / alpha_target)

    # 2. Daily & Rolling 30-Day Released Fraud Ratio
    n_days = len(deploy_days_grid)
    daily_loss_sum = np.zeros(n_days, dtype=np.float64)
    daily_tx_cnt = np.zeros(n_days, dtype=np.float64)

    day_offset = int(deploy_days_grid[0])
    day_indices = days - day_offset
    np.add.at(daily_loss_sum, day_indices, norm_loss)
    np.add.at(daily_tx_cnt, day_indices, 1.0)

    rfr_30d = np.zeros(n_days, dtype=np.float64)
    for i in range(n_days):
        start_i = max(0, i - 29)
        win_cnt = float(np.sum(daily_tx_cnt[start_i : i + 1]))
        win_loss = float(np.sum(daily_loss_sum[start_i : i + 1]))
        rfr_30d[i] = (win_loss / max(1.0, win_cnt)) / alpha_target

    shock_window_days = (deploy_days_grid >= shock_day) & (deploy_days_grid <= shock_day + 90)
    if np.any(shock_window_days):
        overshoot_30d_max = float(np.max(rfr_30d[shock_window_days]))
    else:
        overshoot_30d_max = float(np.max(rfr_30d))

    # 3. Cumulative Post-Shock Excess Fraud Loss (CEFL_shock in $ per 1,000 txns over 90d window)
    if np.any(shock_window_days):
        daily_target_loss = alpha_target * daily_tx_cnt[shock_window_days]
        daily_excess_dollars = np.maximum(
            0.0, (daily_loss_sum[shock_window_days] - daily_target_loss) * a_bar
        )
        total_shock_tx = float(np.sum(daily_tx_cnt[shock_window_days]))
        cefl_shock = float(1000.0 * np.sum(daily_excess_dollars) / max(1.0, total_shock_tx))
    else:
        cefl_shock = 0.0

    # 4. Shared Analyst Workload Accounting (Mandatory Reviews + Stage-0 Diagnostic Audits)
    mandatory_rev = ~approve
    diag_audit = approve & audit
    human_intervention = mandatory_rev | diag_audit

    mandatory_review_pct = float(100.0 * np.mean(mandatory_rev))
    diagnostic_audit_pct = float(100.0 * np.mean(diag_audit))
    total_workload_pct = float(100.0 * np.mean(human_intervention))

    # 5. Net Economic Business Value V(\pi) ($ per 1,000 transactions)
    c_cb = ECONOMIC_CFG.dispute_fee
    c_r = ECONOMIC_CFG.review_cost
    c_f = ECONOMIC_CFG.friction_cost
    q_rec = ECONOMIC_CFG.recovery_rate

    human_f = human_intervention.astype(np.float64)
    val_per_tx = human_f * (
        q_rec * y_true * (amounts + c_cb) - c_r - (1.0 - y_true) * c_f
    ) - (1.0 - human_f) * y_true * (amounts + c_cb)
    net_value_per_1k = float(1000.0 * np.mean(val_per_tx))

    # 6. Controller State Chatter \sigma_{\Delta\theta}
    if len(theta_traj) > 1:
        theta_chatter = float(np.sqrt(np.mean(np.diff(theta_traj) ** 2)))
    else:
        theta_chatter = 0.0

    return {
        "rfr_long": rfr_long,
        "overshoot_30d_max": overshoot_30d_max,
        "cefl_shock": cefl_shock,
        "mandatory_review_pct": mandatory_review_pct,
        "diagnostic_audit_pct": diagnostic_audit_pct,
        "total_workload_pct": total_workload_pct,
        "net_value_per_1k": net_value_per_1k,
        "theta_chatter": theta_chatter,
    }