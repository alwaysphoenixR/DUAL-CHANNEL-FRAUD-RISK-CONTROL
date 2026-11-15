r"""Theorem 3 KKT Water-Filling Audit Propensity Solver & Corollary 1 Variance Gain."""

import numpy as np


def solve_kkt_water_filling(
    p_strata: np.ndarray,
    sigma_strata: np.ndarray,
    q_audit: float = 0.01,
    e_min: float = 0.005,
    e_max: float = 1.0,
) -> np.ndarray:
    r"""
    Exact finite-termination KKT water-filling solver for Value-Stratified Diagnostic Audits.
    Minimizes \sum_{k=1}^K p_k \sigma_k^2 / e_k
    subject to \sum_{k=1}^K p_k e_k = q_audit and e_min <= e_k <= e_max.
    """
    p = np.asarray(p_strata, dtype=np.float64)
    sigma = np.asarray(sigma_strata, dtype=np.float64)
    k_count = len(p)

    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        p = p / np.sum(p)
    if not (e_min <= q_audit <= e_max):
        raise ValueError(f"q_audit ({q_audit}) must lie within [e_min={e_min}, e_max={e_max}].")

    e_star = np.zeros(k_count, dtype=np.float64)
    clamped_low = np.zeros(k_count, dtype=bool)
    clamped_high = np.zeros(k_count, dtype=bool)

    for _ in range(k_count + 1):
        free_mask = ~(clamped_low | clamped_high)
        if not np.any(free_mask):
            break

        budget_used = np.sum(p[clamped_low] * e_min) + np.sum(p[clamped_high] * e_max)
        q_rem = max(0.0, q_audit - budget_used)
        denom = np.sum(p[free_mask] * sigma[free_mask])

        if denom <= 1e-15:
            e_cand = np.full(np.sum(free_mask), q_rem / np.sum(p[free_mask]))
        else:
            e_cand = q_rem * sigma[free_mask] / denom

        free_indices = np.where(free_mask)[0]

        # Check lower floor violations first
        if np.any(e_cand < e_min - 1e-14):
            worst_idx = free_indices[np.argmin(e_cand)]
            clamped_low[worst_idx] = True
            e_star[worst_idx] = e_min
            continue

        # Check upper ceiling violations
        if np.any(e_cand > e_max + 1e-14):
            worst_idx = free_indices[np.argmax(e_cand)]
            clamped_high[worst_idx] = True
            e_star[worst_idx] = e_max
            continue

        e_star[free_mask] = e_cand
        break

    return e_star


def compute_stage0_variance_objective(
    p_strata: np.ndarray, sigma_strata: np.ndarray, e_strata: np.ndarray
) -> float:
    r"""Computes V(e) = \sum_{k=1}^K p_k \sigma_k^2 / e_k."""
    p = np.asarray(p_strata, dtype=np.float64)
    sigma = np.asarray(sigma_strata, dtype=np.float64)
    e = np.asarray(e_strata, dtype=np.float64)
    return float(np.sum(p * (sigma ** 2) / e))


def cauchy_schwarz_variance_ratio(p_strata: np.ndarray, sigma_strata: np.ndarray) -> float:
    r"""
    Computes theoretical unconstrained variance ratio V_Neyman / V_unif = 1 / (1 + CV^2(\sigma_k))
    from Corollary 1 (Equation 9).
    """
    p = np.asarray(p_strata, dtype=np.float64)
    p = p / np.sum(p)
    sigma = np.asarray(sigma_strata, dtype=np.float64)
    mean_sigma = float(np.sum(p * sigma))
    second_moment = float(np.sum(p * (sigma ** 2)))
    if second_moment <= 1e-15:
        return 1.0
    return (mean_sigma ** 2) / second_moment


def estimate_plug_in_sigma(
    amounts: np.ndarray,
    p_hat: np.ndarray,
    strata: np.ndarray,
    approve_mask: np.ndarray,
    a_bar: float,
    n_strata: int = 5,
) -> np.ndarray:
    r"""
    Computes strictly \mathcal{F}_{d-1}-measurable plug-in residual standard deviation \hat{\sigma}_{k,d}
    without peeking at future labels y_t.
    """
    sigmas = np.zeros(n_strata, dtype=np.float64)
    norm_amt = amounts / a_bar
    bernoulli_var = np.clip(p_hat * (1.0 - p_hat), 1e-8, 0.25)

    for k in range(n_strata):
        mask_k = (strata == k) & approve_mask
        if np.any(mask_k):
            sigmas[k] = float(np.sqrt(np.mean((norm_amt[mask_k] ** 2) * bernoulli_var[mask_k])))
        else:
            mask_all_k = strata == k
            if np.any(mask_all_k):
                sigmas[k] = float(np.sqrt(np.mean((norm_amt[mask_all_k] ** 2) * bernoulli_var[mask_all_k])))
            else:
                sigmas[k] = 1e-3 * (k + 1)
    return np.maximum(sigmas, 1e-6)