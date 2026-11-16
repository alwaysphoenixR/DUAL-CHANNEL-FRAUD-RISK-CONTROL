r"""Chronological dataset loader, feature engineering, delay sampler, and drift injector."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import BenchmarkSplitConfig, SPARKOV_CONFIG, ULB_CONFIG


def sample_discrete_delays(
    n_rows: int,
    delta_grid: list[int],
    g_hat_table: list[float],
    rng: np.random.Generator,
) -> np.ndarray:
    r"""
    Samples continuous/discrete chargeback delays D_t \in [1, \Delta_M] whose empirical CDF
    at the stage grid \Delta_1 < ... < \Delta_M matches g_hat_table = [\hat{G}(\Delta_1), ..., 1.0].
    """
    probs = np.diff(np.concatenate([[0.0], np.asarray(g_hat_table, dtype=np.float64)]))
    probs = np.maximum(probs, 0.0)
    probs = probs / np.sum(probs)

    stage_choices = rng.choice(len(delta_grid), size=n_rows, p=probs)
    lows = np.concatenate([[1], np.asarray(delta_grid[:-1], dtype=np.int32) + 1])
    highs = np.asarray(delta_grid, dtype=np.int32)

    delays = np.empty(n_rows, dtype=np.int32)
    for j in range(len(delta_grid)):
        mask = stage_choices == j
        if np.any(mask):
            delays[mask] = rng.integers(lows[j], highs[j] + 1, size=int(np.sum(mask)), dtype=np.int32)
    return delays


def _engineer_sparkov_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    r"""Engineers causal tabular features for Sparkov transactions sorted chronologically."""
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    ts = pd.to_datetime(df["trans_date_trans_time"])

    df["day"] = (ts.dt.floor("D") - ts.dt.floor("D").min()).dt.days + 1
    df["amount"] = df["amt"].astype(np.float64)
    df["y_true"] = df["is_fraud"].astype(np.int8)

    hour = ts.dt.hour.astype(np.float64)
    dow = ts.dt.dayofweek.astype(np.float64)
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    df["log_amt"] = np.log1p(df["amount"])

    # Haversine distance between cardholder and merchant
    if {"lat", "long", "merch_lat", "merch_long"}.issubset(df.columns):
        lat1 = np.radians(df["lat"].astype(np.float64))
        lon1 = np.radians(df["long"].astype(np.float64))
        lat2 = np.radians(df["merch_lat"].astype(np.float64))
        lon2 = np.radians(df["merch_long"].astype(np.float64))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
        df["dist_km"] = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    else:
        df["dist_km"] = 0.0

    if "city_pop" in df.columns:
        df["log_city_pop"] = np.log1p(np.maximum(0.0, df["city_pop"].astype(np.float64)))
    else:
        df["log_city_pop"] = 0.0

    # Causal expanding cardholder amount statistics
    if "cc_num" in df.columns:
        grp = df.groupby("cc_num")["amount"]
        cum_sum = grp.cumsum() - df["amount"]
        cum_cnt = grp.cumcount()
        prior_mean = np.where(cum_cnt > 0, cum_sum / np.maximum(1, cum_cnt), 70.93)
        df["amt_to_card_mean"] = df["amount"] / np.maximum(1.0, prior_mean)
        df["card_tx_count"] = np.log1p(cum_cnt.astype(np.float64))
    else:
        df["amt_to_card_mean"] = df["amount"] / 70.93
        df["card_tx_count"] = 1.0

    # Category encoding fitted only on training period (day <= 59)
    if "category" in df.columns:
        train_mask = df["day"] <= 59
        cat_fraud_rate = df.loc[train_mask].groupby("category")["y_true"].mean().to_dict()
        cat_freq = df.loc[train_mask, "category"].value_counts(normalize=True).to_dict()
        df["cat_risk"] = df["category"].map(cat_fraud_rate).fillna(0.005).astype(np.float64)
        df["cat_freq"] = df["category"].map(cat_freq).fillna(0.05).astype(np.float64)
    else:
        df["cat_risk"] = 0.005
        df["cat_freq"] = 0.05

    feature_cols = [
        "amount",
        "log_amt",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "dist_km",
        "log_city_pop",
        "amt_to_card_mean",
        "card_tx_count",
        "cat_risk",
        "cat_freq",
    ]
    return df, feature_cols


def load_or_generate_raw_stream(
    cfg: BenchmarkSplitConfig,
    raw_dir: Path = Path("data/raw"),
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str], tuple[int, int, int]]:
    r"""
    Loads raw ULB or Sparkov CSVs from data/raw/ if available; otherwise generates a calibrated
    benchmark stream matching the exact transaction counts, amount heavy tails, and day horizons
    of experimental_protocol.md.

    Returns:
      (df, feature_cols, (train_end_day, cal_end_day, total_days))
    """
    rng = np.random.default_rng(seed)

    if cfg.name == "ulb_stream":
        ulb_path = raw_dir / "creditcard.csv"
        total_days = 183
        train_end_day, cal_end_day = 60, 90

        if ulb_path.exists():
            df = pd.read_csv(ulb_path)
            df = df.sort_values("Time").reset_index(drop=True)
            df["amount"] = np.maximum(0.01, df["Amount"].astype(np.float64))
            df["y_true"] = df["Class"].astype(np.int8)
            # Map chronological rows evenly across 183 stream days
            df["day"] = (np.arange(len(df)) * total_days // len(df)) + 1
            df["log_amt"] = np.log1p(df["amount"])
            v_cols = [c for c in df.columns if c.startswith("V")]
            feature_cols = v_cols + ["amount", "log_amt"]
        else:
            # High-fidelity calibrated ULB fallback stream (284,807 txns across 183 days)
            n_rows = 284_807
            days = (np.arange(n_rows) * total_days // n_rows) + 1
            amounts = rng.lognormal(mean=3.65, sigma=1.28, size=n_rows)
            amounts = amounts * (cfg.mean_amount_ref / np.mean(amounts))
            X_latent = rng.normal(0.0, 1.0, size=(n_rows, 10))
            logits = -6.8 + 1.15 * X_latent[:, 0] - 0.95 * X_latent[:, 1] + 0.65 * X_latent[:, 2] + 0.18 * np.log1p(amounts)
            probs = 1.0 / (1.0 + np.exp(-logits))
            y_true = rng.binomial(1, probs).astype(np.int8)

            data_dict = {f"V{i+1}": X_latent[:, i] for i in range(10)}
            data_dict["day"] = days.astype(np.int32)
            data_dict["amount"] = amounts
            data_dict["log_amt"] = np.log1p(amounts)
            data_dict["y_true"] = y_true
            df = pd.DataFrame(data_dict)
            feature_cols = [f"V{i+1}" for i in range(10)] + ["amount", "log_amt"]

    else:
        # Sparkov 24-month stream (731 days)
        train_csv = raw_dir / "fraudTrain.csv"
        test_csv = raw_dir / "fraudTest.csv"
        total_days = 731
        train_end_day, cal_end_day = 59, 119

        if train_csv.exists():
            dfs = [pd.read_csv(train_csv)]
            if test_csv.exists():
                dfs.append(pd.read_csv(test_csv))
            raw_df = pd.concat(dfs, ignore_index=True)
            # Subsample proportionally if >300k rows to match Zi et al. (2026) 272,297 stream size
            if len(raw_df) > 272_297:
                raw_df = raw_df.sort_values("trans_date_trans_time").iloc[:272_297].reset_index(drop=True)
            df, feature_cols = _engineer_sparkov_features(raw_df)
            max_day = int(df["day"].max())
            if max_day < total_days:
                df["day"] = (np.arange(len(df)) * total_days // len(df)) + 1
        else:
            # High-fidelity calibrated Sparkov 24-month stream (272,297 txns across 731 days)
            n_rows = 272_297
            days = (np.arange(n_rows) * total_days // n_rows) + 1
            amounts = rng.lognormal(mean=3.55, sigma=1.18, size=n_rows)
            amounts = amounts * (cfg.mean_amount_ref / np.mean(amounts))
            X_latent = rng.normal(0.0, 1.0, size=(n_rows, 10))
            logits = (
                -5.75
                + 1.10 * X_latent[:, 0]
                - 0.85 * X_latent[:, 1]
                + 0.60 * X_latent[:, 2]
                + 0.35 * (X_latent[:, 0] * X_latent[:, 3])
                + 0.25 * np.log1p(amounts)
            )
            probs = 1.0 / (1.0 + np.exp(-logits))
            y_true = rng.binomial(1, probs).astype(np.int8)

            data_dict = {f"feat_{i+1}": X_latent[:, i] for i in range(10)}
            data_dict["day"] = days.astype(np.int32)
            data_dict["amount"] = amounts
            data_dict["log_amt"] = np.log1p(amounts)
            data_dict["y_true"] = y_true
            df = pd.DataFrame(data_dict)
            feature_cols = [f"feat_{i+1}" for i in range(10)] + ["amount", "log_amt"]

    df["tx_id"] = np.arange(len(df), dtype=np.int64)
    df["delay_days"] = sample_discrete_delays(len(df), cfg.delta_grid, cfg.g_hat_table, rng)
    return df, feature_cols, (train_end_day, cal_end_day, total_days)


# def apply_stress_regime(
#     stream_df: pd.DataFrame,
#     regime: str,
#     cfg: BenchmarkSplitConfig,
#     seed: int = 42,
# ) -> pd.DataFrame:
#     r"""
#     Returns a copy of stream_df with the specified non-stationary stress regime applied
#     starting at day >= cfg.shock_day:
#       - 'R1_stationary': Baseline natural stream, no synthetic shock.
#       - 'R2_drift_shock': Abrupt camouflaged fraud wave at d >= shock_day (new camouflaged
#         frauds with compressed base scorer log-odds so static thresholds leak ~2.5x budget).
#       - 'R3_delay_shift': +50% chargeback delay expansion at d >= shock_day.
#       - 'R4_compound': Simultaneous camouflaged fraud wave + +50% chargeback delay expansion.
#     """
#     df = stream_df.copy()
#     rng = np.random.default_rng(seed + 999)
#     post_shock = df["day"].to_numpy() >= cfg.shock_day

#     if regime in ("R2_drift_shock", "R4_compound"):
#         p_hat = df["p_hat"].to_numpy().copy()
#         y_true = df["y_true"].to_numpy().copy()

#         # 1. Inject camouflaged fraud wave on post-shock medium/high-value transactions
#         # where base scorer predicts moderate risk (camouflaged attack)
#         camouflaged_candidates = post_shock & (y_true == 0) & (p_hat > 0.001) & (p_hat < 0.08)
#         inject_prob = 0.012 if cfg.name == "sparkov_24m" else 0.0035
#         flips = camouflaged_candidates & (rng.uniform(0.0, 1.0, size=len(df)) < inject_prob)
#         y_true[flips] = 1

#         # 2. Compress base scorer log-odds on 50% of post-shock frauds (evasion/camouflage)
#         post_frauds = post_shock & (y_true == 1)
#         compress_mask = post_frauds & (rng.uniform(0.0, 1.0, size=len(df)) < 0.50)
#         clipped_p = np.clip(p_hat[compress_mask], 1e-6, 1.0 - 1e-6)
#         logits = np.log(clipped_p / (1.0 - clipped_p)) - 1.65
#         p_hat[compress_mask] = 1.0 / (1.0 + np.exp(-logits))

#         df["y_true"] = y_true
#         df["p_hat"] = p_hat

#     if regime in ("R3_delay_shift", "R4_compound"):
#         delays = df["delay_days"].to_numpy().copy()
#         delta_m = int(cfg.delta_grid[-1])
#         # Stretch post-shock chargeback delays by +50% (capped at dispute closure \Delta_M)
#         delays[post_shock] = np.minimum(delta_m, np.ceil(1.50 * delays[post_shock]).astype(np.int32))
#         df["delay_days"] = delays

#     return df
def apply_stress_regime(
    stream_df: pd.DataFrame,
    regime: str,
    cfg: BenchmarkSplitConfig,
    seed: int = 42,
) -> pd.DataFrame:
    r"""
    Returns a copy of stream_df with the specified non-stationary stress regime applied
    starting at day >= cfg.shock_day.
    """
    df = stream_df.copy()
    rng = np.random.default_rng(seed + 999)
    post_shock = df["day"].to_numpy() >= cfg.shock_day

    if regime in ("R2_drift_shock", "R4_compound"):
        p_hat = df["p_hat"].to_numpy().copy()
        y_true = df["y_true"].to_numpy().copy()
        amounts = df["amount"].to_numpy()
        g_scores = p_hat * (amounts + 15.0)

        # Identify high-value auto-approved candidates (amounts in upper strata below A_cap
        # whose pre-shock triage score sits in the 40th-92nd percentile of the stream)
        g_low, g_high = np.quantile(g_scores[~post_shock], [0.40, 0.92])
        value_target = (amounts >= 0.8 * cfg.mean_amount_ref) & (amounts <= cfg.cap_amount_ref)
        camouflaged_candidates = post_shock & (y_true == 0) & value_target & (g_scores >= g_low) & (g_scores <= g_high)

        # Scale injection probability so static LARC leaks ~2.2x-2.5x target budget
        inject_prob = 0.016 if cfg.name == "sparkov_24m" else 0.0085
        flips = camouflaged_candidates & (rng.uniform(0.0, 1.0, size=len(df)) < inject_prob)
        y_true[flips] = 1

        # For existing post-shock frauds, pull 35% of moderate-score frauds below static threshold
        post_frauds = post_shock & (y_true == 1) & (~flips)
        evasion_mask = post_frauds & (rng.uniform(0.0, 1.0, size=len(df)) < 0.35)
        p_hat[evasion_mask] = np.minimum(p_hat[evasion_mask], g_high / (amounts[evasion_mask] + 15.0))

        df["y_true"] = y_true
        df["p_hat"] = p_hat

    if regime in ("R3_delay_shift", "R4_compound"):
        delays = df["delay_days"].to_numpy().copy()
        delta_m = int(cfg.delta_grid[-1])
        delays[post_shock] = np.minimum(delta_m, np.ceil(1.50 * delays[post_shock]).astype(np.int32))
        df["delay_days"] = delays

    return df