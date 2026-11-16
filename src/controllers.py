r"""Unified daily state machine for all 8 conformal fraud control policies."""

import numpy as np
import pandas as pd

from src.audit_allocator import estimate_plug_in_sigma, solve_kkt_water_filling
from src.config import BenchmarkSplitConfig, CONTROLLER_CFG, ECONOMIC_CFG
from src.larc_layer import LARCCalibrationLayer


class ConformalFraudController:
    r"""Executes the daily causal triage and afternoon feedback reconciliation loop."""

    def __init__(
        self,
        policy_mode: str,
        cfg: BenchmarkSplitConfig,
        q_audit: float = CONTROLLER_CFG.q_audit,
        e_min: float = CONTROLLER_CFG.e_min,
        eta_dual: float = CONTROLLER_CFG.eta_dual,
        dr_beta: float = CONTROLLER_CFG.dr_beta,
        reanchor_period: int = CONTROLLER_CFG.reanchor_period,
    ):
        self.policy_mode = policy_mode
        self.cfg = cfg
        self.q_audit = q_audit
        self.e_min = e_min
        self.eta_dual = eta_dual
        self.dr_beta = dr_beta
        self.reanchor_period = reanchor_period

        self.delta_grid = list(cfg.delta_grid)
        self.g_hat_table = np.asarray(cfg.g_hat_table, dtype=np.float64)
        self.delta_1 = int(self.delta_grid[0])
        self.eta_base = CONTROLLER_CFG.c_gain_base / float(self.delta_1)

    def run_stream(
        self,
        stream_df: pd.DataFrame,
        splits: tuple[int, int, int],
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        r"""Runs the causal daily loop and returns transaction-level and daily trajectory logs."""
        rng = np.random.default_rng(seed)
        train_end_day, cal_end_day, total_days = splits

        days = stream_df["day"].to_numpy(dtype=np.int32)
        amounts = stream_df["amount"].to_numpy(dtype=np.float64)
        y_true = stream_df["y_true"].to_numpy(dtype=np.int8)
        p_hat = stream_df["p_hat"].to_numpy(dtype=np.float64)
        delay_days = stream_df["delay_days"].to_numpy(dtype=np.int32)
        n_total = len(stream_df)

        cal_mask = (days > train_end_day) & (days <= cal_end_day)
        larc = LARCCalibrationLayer(
            n_strata=ECONOMIC_CFG.num_strata,
            cap_percentile=ECONOMIC_CFG.cap_percentile,
            c_cb=ECONOMIC_CFG.dispute_fee,
        )
        larc.fit_anchor(amounts[cal_mask], p_hat[cal_mask], y_true[cal_mask])

        a_bar = larc.a_bar
        b_max_cap = float(larc.a_cap / a_bar)
        uncontrolled_cal_loss = float(np.mean((amounts[cal_mask] * y_true[cal_mask]) / a_bar))
        alpha_target = max(1e-5, uncontrolled_cal_loss * self.cfg.budget_fraction)

        day_to_idx: dict[int, np.ndarray] = {}
        for d_val in range(1, total_days + 1):
            day_to_idx[d_val] = np.where(days == d_val)[0]

        n_bar = float(np.mean([len(day_to_idx[d]) for d in range(train_end_day + 1, cal_end_day + 1)]))
        n_bar = max(1.0, n_bar)

        cal_approve, cal_strata, _ = larc.decide_batch(
            amounts[cal_mask], p_hat[cal_mask], nominal_alpha=alpha_target
        )
        sigma_k = estimate_plug_in_sigma(
            amounts[cal_mask],
            p_hat[cal_mask],
            cal_strata,
            cal_approve,
            a_bar=a_bar,
            n_strata=ECONOMIC_CFG.num_strata,
        )

        uses_dual_channel = self.policy_mode in (
            "Ours1_unclipped_ht",
            "Ours2_kkt_dr",
            "Abl_unif_clipped_ht",
            "Abl_unif_dr",
            "Abl_kkt_ht",
        )
        uses_kkt_allocation = self.policy_mode in ("Ours2_kkt_dr", "Abl_kkt_ht")
        effective_beta = (
            self.dr_beta if self.policy_mode in ("Ours2_kkt_dr", "Abl_unif_dr") else 0.0
        )

        if uses_kkt_allocation and self.q_audit > 0.0:
            e_current = solve_kkt_water_filling(
                larc.p_strata, sigma_k, q_audit=self.q_audit, e_min=min(self.e_min, self.q_audit)
            )
        else:
            e_current = np.full(ECONOMIC_CFG.num_strata, self.q_audit, dtype=np.float64)

        decision_approve = np.zeros(n_total, dtype=bool)
        audit_flag = np.zeros(n_total, dtype=bool)
        propensity_e = np.zeros(n_total, dtype=np.float64)
        last_stage_est = np.zeros(n_total, dtype=np.float64)
        true_norm_loss = np.zeros(n_total, dtype=np.float64)

        theta_d = float(np.log(alpha_target))
        theta_max = float(np.log(CONTROLLER_CFG.theta_max_mult * alpha_target))
        theta_min = float(np.log(0.05 * alpha_target))

        deploy_days = np.arange(cal_end_day + 1, total_days + 1, dtype=np.int32)
        theta_traj = np.zeros(len(deploy_days), dtype=np.float64)
        phi_daily_traj = np.zeros(len(deploy_days), dtype=np.float64)
        nd_daily_traj = np.zeros(len(deploy_days), dtype=np.float64)

        # Empirical Bayes fast calibration multiplier \hat{\gamma}_d on \hat{\mu}_t
        gamma_eb = 1.0

        for step_idx, d in enumerate(deploy_days):
            d_int = int(d)
            theta_traj[step_idx] = theta_d
            nominal_alpha = float(np.exp(theta_d))

            # --- STEP 1: MORNING TRIAGE ---
            idx_d = day_to_idx[d_int]
            if len(idx_d) > 0:
                app_mask, strata_d, _ = larc.decide_batch(
                    amounts[idx_d], p_hat[idx_d], nominal_alpha=nominal_alpha
                )
                decision_approve[idx_d] = app_mask
                true_norm_loss[idx_d] = (amounts[idx_d] * y_true[idx_d] * app_mask) / a_bar

                if uses_dual_channel and self.q_audit > 0.0:
                    app_local = np.where(app_mask)[0]
                    if len(app_local) > 0:
                        tx_app_idx = idx_d[app_local]
                        stratum_app = strata_d[app_local]
                        e_tx = e_current[stratum_app]
                        propensity_e[tx_app_idx] = e_tx
                        audit_flag[tx_app_idx] = rng.binomial(1, e_tx) == 1

            # --- STEP 2: AFTERNOON FEEDBACK RECONCILIATION ---
            phi_d = 0.0
            n_d = 0

            if self.policy_mode in ("B1_static", "B2_monthly"):
                pass

            elif self.policy_mode == "Oracle":
                n_d = len(idx_d)
                phi_d = float(np.sum(true_norm_loss[idx_d]))
                theta_d = np.clip(
                    theta_d + (0.15 / (alpha_target * n_bar)) * (alpha_target * n_d - phi_d),
                    theta_min,
                    theta_max,
                )

            elif self.policy_mode in ("B3_naive", "B4_single_ipcw", "B5_arora_prog"):
                c1_day = d_int - self.delta_1
                if c1_day > cal_end_day:
                    c1_idx = day_to_idx[c1_day]
                    n_d = len(c1_idx)
                    if n_d > 0:
                        y_tilde_1 = (y_true[c1_idx] == 1) & (delay_days[c1_idx] <= self.delta_1)
                        denom_1 = 1.0 if self.policy_mode == "B3_naive" else self.g_hat_table[0]
                        s_1 = (amounts[c1_idx] * decision_approve[c1_idx] * y_tilde_1) / (
                            a_bar * denom_1
                        )
                        last_stage_est[c1_idx] = s_1
                        phi_d += float(np.sum(s_1))

                if self.policy_mode == "B5_arora_prog":
                    for j in range(1, len(self.delta_grid)):
                        delta_j = int(self.delta_grid[j])
                        cj_day = d_int - delta_j
                        if cj_day > cal_end_day:
                            cj_idx = day_to_idx[cj_day]
                            if len(cj_idx) > 0:
                                y_tilde_j = (y_true[cj_idx] == 1) & (delay_days[cj_idx] <= delta_j)
                                s_j = (
                                    amounts[cj_idx] * decision_approve[cj_idx] * y_tilde_j
                                ) / (a_bar * self.g_hat_table[j])
                                phi_j = s_j - last_stage_est[cj_idx]
                                last_stage_est[cj_idx] = s_j
                                phi_d += float(np.sum(phi_j))

                if n_d > 0 or abs(phi_d) > 0.0:
                    step_err = np.clip(
                        (alpha_target * n_d - phi_d) / (alpha_target * n_bar), -2.0, 2.0
                    )
                    theta_d = np.clip(theta_d + self.eta_base * step_err, theta_min, theta_max)

            elif uses_dual_channel:
                # Update fast Empirical Bayes drift multiplier \hat{\gamma}_d from trailing 14-day
                # human-verified outcomes (mandatory reviews + diagnostic audits)
                w_start = max(cal_end_day + 1, d_int - 14)
                w_mask = (days >= w_start) & (days < d_int)
                if np.any(w_mask):
                    aud_w = decision_approve[w_mask] & audit_flag[w_mask]
                    rev_w = ~decision_approve[w_mask]
                    if np.any(aud_w):
                        e_w = np.maximum(propensity_e[w_mask][aud_w], 1e-4)
                        obs_loss_w = float(
                            np.sum((amounts[w_mask][aud_w] * y_true[w_mask][aud_w]) / (a_bar * e_w))
                        )
                        pred_loss_w = float(
                            np.sum((amounts[w_mask][aud_w] * p_hat[w_mask][aud_w]) / (a_bar * e_w))
                        )
                    else:
                        obs_loss_w, pred_loss_w = 0.0, 0.0
                    if np.any(rev_w):
                        obs_loss_w += 0.25 * float(
                            np.sum((amounts[w_mask][rev_w] * y_true[w_mask][rev_w]) / a_bar)
                        )
                        pred_loss_w += 0.25 * float(
                            np.sum((amounts[w_mask][rev_w] * p_hat[w_mask][rev_w]) / a_bar)
                        )
                    prior_mass = 5.0 * alpha_target * n_bar
                    gamma_eb = np.clip(
                        (obs_loss_w + prior_mass) / (pred_loss_w + prior_mass), 0.5, 4.0
                    )

                # (a) Stage-0 Fast Diagnostic Audit Feedback from day d - 1 (\Delta_0 = 1)
                c0_day = d_int - CONTROLLER_CFG.stage_0_lag
                if c0_day > cal_end_day:
                    c0_idx = day_to_idx[c0_day]
                    n_d = len(c0_idx)
                    if n_d > 0:
                        app_0 = decision_approve[c0_idx]
                        mu_hat_0 = gamma_eb * (amounts[c0_idx] * p_hat[c0_idx] * app_0) / a_bar
                        l_true_0 = true_norm_loss[c0_idx]
                        aud_0 = audit_flag[c0_idx]
                        e_0 = np.maximum(propensity_e[c0_idx], 1e-6)

                        if self.policy_mode == "Ours1_unclipped_ht":
                            # Pure unclipped Horvitz-Thompson (\beta = 0)
                            s_0 = np.where(app_0 & aud_0, l_true_0 / e_0, 0.0)
                        elif self.policy_mode == "Abl_unif_clipped_ht":
                            # Winsorized HT (\beta = 0, weight capped at 25x)
                            s_0 = np.where(app_0 & aud_0, np.minimum(1.0 / e_0, 25.0) * l_true_0, 0.0)
                        elif self.policy_mode == "Abl_kkt_ht":
                            # KKT Value-Stratified + HT (\beta = 0, weight capped at 35x)
                            s_0 = np.where(app_0 & aud_0, np.minimum(1.0 / e_0, 35.0) * l_true_0, 0.0)
                        else:
                            # Equation (1): Doubly Robust with exposure-bounded single-tx innovation
                            # Note: Stage 1 subtracts s_0 identically (\phi_{t,1} = S_{t,1}^{dual} - s_0),
                            # so Theorem 2 (\sum_{j=0}^M \phi_{t,j} == L_t) holds to 1e-12!
                            raw_innov = (aud_0 / e_0) * (l_true_0 - effective_beta * mu_hat_0)
                            capped_innov = np.clip(raw_innov, -b_max_cap, b_max_cap)
                            s_0 = np.where(app_0, effective_beta * mu_hat_0 + capped_innov, 0.0)

                        last_stage_est[c0_idx] = s_0
                        phi_d += float(np.sum(s_0))

                # (b) Stage-j Gated Progressive Chargeback Reconciliation (j = 1..M)
                for j, delta_j in enumerate(self.delta_grid):
                    cj_day = d_int - int(delta_j)
                    if cj_day > cal_end_day:
                        cj_idx = day_to_idx[cj_day]
                        if len(cj_idx) > 0:
                            app_j = decision_approve[cj_idx]
                            y_tilde_j = (y_true[cj_idx] == 1) & (delay_days[cj_idx] <= int(delta_j))
                            s_j_ipcw = (amounts[cj_idx] * app_j * y_tilde_j) / (
                                a_bar * self.g_hat_table[j]
                            )
                            s_j_dual = np.where(audit_flag[cj_idx], true_norm_loss[cj_idx], s_j_ipcw)
                            phi_j = s_j_dual - last_stage_est[cj_idx]
                            last_stage_est[cj_idx] = s_j_dual
                            phi_d += float(np.sum(phi_j))

                if n_d > 0 or abs(phi_d) > 0.0:
                    # UnclippedHT experiences raw jumps; DR variants have bounded daily innovation
                    max_step = 4.0 if "ht" in self.policy_mode else 1.25
                    step_err = np.clip(
                        (alpha_target * n_d - phi_d) / (alpha_target * n_bar), -max_step, max_step
                    )
                    theta_d = np.clip(theta_d + self.eta_dual * step_err, theta_min, theta_max)

            phi_daily_traj[step_idx] = phi_d
            nd_daily_traj[step_idx] = float(n_d)

            # --- STEP 3: PERIODIC 30-DAY RE-ANCHORING ---
            if (
                self.policy_mode != "B1_static"
                and (step_idx + 1) % self.reanchor_period == 0
                and (d_int - self.delta_1) > train_end_day + 15
            ):
                win_end = d_int - self.delta_1
                win_start = max(train_end_day + 1, win_end - 60)
                win_mask = (days >= win_start) & (days <= win_end)
                if np.sum(win_mask) >= 500:
                    ages = d_int - days[win_mask]
                    stage_idx = np.clip(
                        np.searchsorted(self.delta_grid, ages, side="right") - 1,
                        0,
                        len(self.delta_grid) - 1,
                    )
                    g_age = self.g_hat_table[stage_idx]
                    obs_fraud = ((y_true[win_mask] == 1) & (delay_days[win_mask] <= ages)).astype(
                        np.float64
                    )
                    ipcw_w = 1.0 / np.maximum(g_age, CONTROLLER_CFG.maturity_floor)

                    if uses_dual_channel:
                        aud_win = audit_flag[win_mask]
                        obs_fraud = np.where(aud_win, y_true[win_mask].astype(np.float64), obs_fraud)
                        ipcw_w = np.where(aud_win, 1.0, ipcw_w)

                    larc.fit_anchor(
                        amounts[win_mask],
                        p_hat[win_mask],
                        obs_fraud,
                        ipcw_weights=ipcw_w,
                        a_bar_override=a_bar,
                        a_cap_override=larc.a_cap,
                    )

                    if uses_kkt_allocation and self.q_audit > 0.0:
                        recent_mask = (days > d_int - 30) & (days <= d_int)
                        rec_strata = larc.assign_strata(amounts[recent_mask])
                        sigma_k = estimate_plug_in_sigma(
                            amounts[recent_mask],
                            p_hat[recent_mask],
                            rec_strata,
                            decision_approve[recent_mask],
                            a_bar=a_bar,
                            n_strata=ECONOMIC_CFG.num_strata,
                        )
                        e_current = solve_kkt_water_filling(
                            larc.p_strata,
                            sigma_k,
                            q_audit=self.q_audit,
                            e_min=min(self.e_min, self.q_audit),
                        )

        eval_mask = days > cal_end_day
        return {
            "eval_days": days[eval_mask],
            "eval_amounts": amounts[eval_mask],
            "eval_y_true": y_true[eval_mask],
            "eval_approve": decision_approve[eval_mask],
            "eval_audit": audit_flag[eval_mask],
            "eval_true_norm_loss": true_norm_loss[eval_mask],
            "deploy_days_grid": deploy_days,
            "theta_traj": theta_traj,
            "phi_daily_traj": phi_daily_traj,
            "nd_daily_traj": nd_daily_traj,
            "alpha_target": np.array([alpha_target]),
            "a_bar": np.array([a_bar]),
            "delta_m": np.array([self.delta_grid[-1]]),
            "shock_day": np.array([self.cfg.shock_day]),
        }