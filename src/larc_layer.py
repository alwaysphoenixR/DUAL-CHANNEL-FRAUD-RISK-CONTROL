r"""Latency-Aware Risk Control (LARC) calibration and water-filling triage layer."""

import numpy as np


class LARCCalibrationLayer:
    r"""
    Implements Zi, Wang, & Zhang (2026) LARC layer:
      1. 99th-percentile tail exposure cap A_cap routed to mandatory review.
      2. Value-ordered triage score g(x, a) = p_hat * (a + c_cb).
      3. K = 5 quantile amount strata with water-filling conformal risk thresholds \tau_k(\alpha').
    """

    def __init__(self, n_strata: int = 5, cap_percentile: float = 99.0, c_cb: float = 15.0):
        self.n_strata = n_strata
        self.cap_percentile = cap_percentile
        self.c_cb = c_cb
        self.a_bar: float = 1.0
        self.a_cap: float = np.inf
        self.stratum_edges: np.ndarray = np.array([])
        self.p_strata: np.ndarray = np.ones(n_strata) / n_strata
        self._sorted_g: list[np.ndarray] = []
        self._cum_risk: list[np.ndarray] = []
        self._stratum_counts: np.ndarray = np.zeros(n_strata, dtype=np.int64)
        self._b_max_stratum: np.ndarray = np.ones(n_strata, dtype=np.float64)
        self._n_cal_total: int = 1

    def compute_triage_score(self, amounts: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
        r"""Computes value-ordered triage index g_t = \hat{p}_t * (a_t + c_cb)."""
        return np.asarray(p_hat, dtype=np.float64) * (np.asarray(amounts, dtype=np.float64) + self.c_cb)

    def assign_strata(self, amounts: np.ndarray) -> np.ndarray:
        r"""Assigns stratum index k \in {0, ..., K-1}, or -1 if amount > A_cap."""
        amt = np.asarray(amounts, dtype=np.float64)
        strata = np.full(len(amt), -1, dtype=np.int8)
        in_cap = amt <= self.a_cap
        if np.any(in_cap):
            interior_edges = self.stratum_edges[1:-1]
            strata[in_cap] = np.searchsorted(interior_edges, amt[in_cap], side="right").astype(np.int8)
        return strata

    def fit_anchor(
        self,
        amounts: np.ndarray,
        p_hat: np.ndarray,
        y_cal: np.ndarray,
        ipcw_weights: np.ndarray | None = None,
        a_bar_override: float | None = None,
        a_cap_override: float | None = None,
    ) -> None:
        r"""Fits quantile strata and monotone conformal risk lookup tables on calibration anchor."""
        amt = np.asarray(amounts, dtype=np.float64)
        p = np.asarray(p_hat, dtype=np.float64)
        y = np.asarray(y_cal, dtype=np.float64)
        w = np.ones_like(y) if ipcw_weights is None else np.asarray(ipcw_weights, dtype=np.float64)

        self.a_bar = float(np.mean(amt)) if a_bar_override is None else float(a_bar_override)
        self.a_cap = (
            float(np.percentile(amt, self.cap_percentile))
            if a_cap_override is None
            else float(a_cap_override)
        )

        in_cap = amt <= self.a_cap
        self._n_cal_total = max(1, int(np.sum(in_cap)))
        quantiles = np.linspace(0.0, 1.0, self.n_strata + 1)
        self.stratum_edges = np.quantile(amt[in_cap], quantiles)
        self.stratum_edges[0] = 0.0
        self.stratum_edges[-1] = self.a_cap

        strata = self.assign_strata(amt)
        g_scores = self.compute_triage_score(amt, p)
        norm_losses = (amt * y * w) / self.a_bar

        self._sorted_g = []
        self._cum_risk = []
        self._stratum_counts = np.zeros(self.n_strata, dtype=np.int64)
        self._b_max_stratum = np.zeros(self.n_strata, dtype=np.float64)

        for k in range(self.n_strata):
            mask_k = strata == k
            n_k = int(np.sum(mask_k))
            self._stratum_counts[k] = n_k
            self.p_strata[k] = n_k / self._n_cal_total
            b_k = float(self.stratum_edges[k + 1] / self.a_bar)
            self._b_max_stratum[k] = b_k

            if n_k == 0:
                self._sorted_g.append(np.array([0.0]))
                self._cum_risk.append(np.array([0.0]))
                continue

            order = np.argsort(g_scores[mask_k], kind="mergesort")
            g_k = g_scores[mask_k][order]
            loss_k = norm_losses[mask_k][order]

            # Global conformal finite-sample slack scaled by total anchor size N_cal
            slack_k = b_k / (self._n_cal_total + 1.0)
            cum_loss = (np.cumsum(loss_k) / max(1.0, float(n_k))) + slack_k
            self._sorted_g.append(g_k)
            self._cum_risk.append(cum_loss)

    def get_thresholds(self, nominal_alpha: float) -> np.ndarray:
        r"""
        Returns water-filling stratum approval thresholds [\tau_0(\alpha'), ..., \tau_{K-1}(\alpha')].
        Unused risk budget from low-amount strata is carried forward to higher-amount strata,
        while preserving strict monotonicity in nominal_alpha.
        """
        alpha_val = max(0.0, float(nominal_alpha))
        thresholds = np.zeros(self.n_strata, dtype=np.float64)

        rem_budget = alpha_val
        rem_prob = 1.0

        for k in range(self.n_strata):
            g_k = self._sorted_g[k]
            risk_k = self._cum_risk[k]
            p_k = float(self.p_strata[k])
            n_k = self._stratum_counts[k]

            if n_k == 0 or p_k <= 0.0:
                thresholds[k] = 0.0
                continue

            target_k = rem_budget / max(rem_prob, 1e-9)
            max_risk_k = float(risk_k[-1])

            if target_k <= float(risk_k[0]):
                thresholds[k] = 0.0
                realized_k = 0.0
            elif max_risk_k <= target_k:
                excess_ratio = target_k / max(max_risk_k, 1e-12)
                thresholds[k] = float(g_k[-1]) * max(1.01, excess_ratio)
                realized_k = max_risk_k
            else:
                idx = int(np.searchsorted(risk_k, target_k, side="right")) - 1
                thresholds[k] = float(g_k[min(idx + 1, len(g_k) - 1)])
                realized_k = float(risk_k[idx])

            rem_budget = max(0.0, rem_budget - p_k * realized_k)
            rem_prob = max(1e-9, rem_prob - p_k)

        return thresholds

    def decide_batch(
        self, amounts: np.ndarray, p_hat: np.ndarray, nominal_alpha: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Applies exposure cap A_cap and stratum thresholds \tau_k(\alpha').
        Returns (decision_approve, strata, g_scores).
        """
        amt = np.asarray(amounts, dtype=np.float64)
        g_scores = self.compute_triage_score(amt, p_hat)
        strata = self.assign_strata(amt)
        thresholds = self.get_thresholds(nominal_alpha)

        decision_approve = np.zeros(len(amt), dtype=bool)
        in_cap = strata >= 0
        if np.any(in_cap):
            tau_per_tx = thresholds[strata[in_cap]]
            decision_approve[in_cap] = g_scores[in_cap] < tau_per_tx

        return decision_approve, strata, g_scores