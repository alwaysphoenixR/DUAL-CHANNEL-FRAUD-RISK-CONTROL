r"""Automated Pytest Verification Suite for Gates 1-4 (system_architecture.md Section 7)."""

import numpy as np

from src.audit_allocator import (
    cauchy_schwarz_variance_ratio,
    compute_stage0_variance_objective,
    solve_kkt_water_filling,
)
from src.larc_layer import LARCCalibrationLayer


def test_gate1_exact_telescoping_reconciliation():
    r"""
    Gate 1: Verify Theorem 2 Part 1 — For 10,000 transactions with arbitrary misspecified
    predictions \hat{p}_t, random audit draws A_t, and misspecified intermediate \hat{G}(\Delta_j),
    the sum of increments \sum_{j=0}^M \phi_{t,j} equals L_t to within 1e-12.
    """
    rng = np.random.default_rng(42)
    n_tx = 10_000
    a_bar = 75.0

    amounts = rng.lognormal(mean=3.8, sigma=1.1, size=n_tx)
    y_true = rng.binomial(1, 0.02, size=n_tx)
    approve_mask = rng.binomial(1, 0.95, size=n_tx).astype(bool)
    p_hat_corrupted = rng.uniform(0.0, 0.8, size=n_tx)
    e_propensity = rng.uniform(0.005, 0.15, size=n_tx)
    audit_flag = rng.binomial(1, e_propensity)
    dr_beta = 1.0

    true_loss_l = (amounts * y_true * approve_mask) / a_bar
    mu_hat = (amounts * p_hat_corrupted * approve_mask) / a_bar

    s_0_dr = np.where(
        approve_mask,
        dr_beta * mu_hat + (audit_flag / e_propensity) * (true_loss_l - dr_beta * mu_hat),
        0.0,
    )
    phi_0 = s_0_dr

    delta_grid = [15, 30, 45, 60, 90]
    g_hat_misspecified = [0.65, 0.80, 0.91, 0.97, 1.00]
    delay_days = rng.integers(1, 91, size=n_tx)

    prev_s = s_0_dr.copy()
    total_increment_sum = phi_0.copy()

    for j, delta_j in enumerate(delta_grid):
        g_hat_j = g_hat_misspecified[j]
        y_tilde_j = (y_true == 1) & (delay_days <= delta_j)
        s_j_ipcw = (amounts * approve_mask * y_tilde_j) / (a_bar * g_hat_j)

        s_j_dual = np.where(audit_flag == 1, true_loss_l, s_j_ipcw)
        phi_j = s_j_dual - prev_s
        total_increment_sum += phi_j
        prev_s = s_j_dual

    max_abs_err = np.max(np.abs(total_increment_sum - true_loss_l))
    assert max_abs_err < 1e-12, f"Telescoping reconciliation failed: max error = {max_abs_err}"


def test_gate2_kkt_audit_allocator_and_cauchy_schwarz():
    r"""
    Gate 2: Verify Theorem 3 & Corollary 1 — KKT water-filling satisfies budget and floor
    constraints exactly and matches the Cauchy-Schwarz variance ratio 1 / (1 + CV^2(\sigma_k))
    in the unconstrained regime.
    """
    rng = np.random.default_rng(101)

    for _ in range(100):
        k_strata = 5
        raw_p = rng.uniform(0.1, 0.4, size=k_strata)
        p_strata = raw_p / np.sum(raw_p)
        sigma_strata = np.array([0.02, 0.06, 0.15, 0.45, 1.40]) * rng.uniform(0.8, 1.2, size=k_strata)
        q_audit = float(rng.uniform(0.01, 0.08))
        e_min = 0.005

        e_star = solve_kkt_water_filling(p_strata, sigma_strata, q_audit=q_audit, e_min=e_min)

        assert np.all(e_star >= e_min - 1e-12), "Positivity floor violated."
        assert np.all(e_star <= 1.0 + 1e-12), "Upper propensity bound violated."
        assert np.isclose(np.sum(p_strata * e_star), q_audit, atol=1e-12), "Audit budget constraint violated."

        v_kkt = compute_stage0_variance_objective(p_strata, sigma_strata, e_star)
        v_unif = compute_stage0_variance_objective(
            p_strata, sigma_strata, np.full(k_strata, q_audit)
        )
        assert v_kkt <= v_unif + 1e-12, "KKT allocation failed to beat uniform allocation."

        if np.all(e_star > e_min + 1e-9) and np.all(e_star < 1.0 - 1e-9):
            expected_ratio = cauchy_schwarz_variance_ratio(p_strata, sigma_strata)
            assert np.isclose(v_kkt / v_unif, expected_ratio, atol=1e-10)


def test_gate3_stage0_dr_unbiasedness_and_variance():
    r"""
    Gate 3: Verify Theorem 1 — Exact discrete + Monte Carlo verification of finite-sample
    unbiasedness and variance reduction under severe concept drift.
    """
    true_loss_l = 4.50       # True normalized loss of a fraudulent high-value transaction
    mu_hat = 1.20            # Severely under-predicted model expectation under concept drift
    e_t = 0.02               # 2% audit probability
    dr_beta = 1.0

    # Part A: Exact analytical expectation & variance over binary audit support A_t \in {0, 1}
    s0_dr_a0 = dr_beta * mu_hat
    s0_dr_a1 = dr_beta * mu_hat + (1.0 / e_t) * (true_loss_l - dr_beta * mu_hat)
    exact_expected_dr = (1.0 - e_t) * s0_dr_a0 + e_t * s0_dr_a1
    assert np.isclose(exact_expected_dr, true_loss_l, atol=1e-14)

    exact_var_dr = (1.0 - e_t) * ((s0_dr_a0 - true_loss_l) ** 2) + e_t * ((s0_dr_a1 - true_loss_l) ** 2)
    theo_var_dr = ((1.0 - e_t) / e_t) * ((true_loss_l - dr_beta * mu_hat) ** 2)
    theo_var_ht = ((1.0 - e_t) / e_t) * (true_loss_l ** 2)
    assert np.isclose(exact_var_dr, theo_var_dr, atol=1e-12)

    # Part B: High-precision Monte Carlo verification within 3.5 Standard Errors
    rng = np.random.default_rng(2026)
    n_mc = 1_000_000
    audits = rng.binomial(1, e_t, size=n_mc)
    s0_ht = (audits / e_t) * true_loss_l
    s0_dr = dr_beta * mu_hat + (audits / e_t) * (true_loss_l - dr_beta * mu_hat)

    se_dr = np.sqrt(theo_var_dr / n_mc)
    se_ht = np.sqrt(theo_var_ht / n_mc)

    assert abs(np.mean(s0_dr) - true_loss_l) < 3.5 * se_dr
    assert abs(np.mean(s0_ht) - true_loss_l) < 3.5 * se_ht
    assert np.isclose(np.var(s0_dr), theo_var_dr, rtol=2e-2)
    assert np.var(s0_dr) < np.var(s0_ht) * 0.60


def test_gate4_larc_threshold_monotonicity():
    r"""
    Gate 4: Verify LARC stratum thresholds \tau_k(\alpha') are monotonically non-decreasing
    in nominal risk budget \alpha', ensuring negative feedback loop stability.
    """
    rng = np.random.default_rng(7)
    n_cal = 20_000
    amounts = rng.lognormal(mean=3.5, sigma=1.0, size=n_cal)
    p_hat = rng.beta(0.5, 15.0, size=n_cal)
    y_cal = rng.binomial(1, p_hat)

    larc = LARCCalibrationLayer(n_strata=5, cap_percentile=99.0, c_cb=15.0)
    larc.fit_anchor(amounts, p_hat, y_cal)

    alpha_grid = np.geomspace(1e-4, 5e-2, 25)
    prev_tau = larc.get_thresholds(alpha_grid[0])

    for alpha_val in alpha_grid[1:]:
        curr_tau = larc.get_thresholds(alpha_val)
        assert np.all(curr_tau >= prev_tau - 1e-12), (
            f"Monotonicity violated at alpha={alpha_val}: {curr_tau} < {prev_tau}"
        )
        prev_tau = curr_tau