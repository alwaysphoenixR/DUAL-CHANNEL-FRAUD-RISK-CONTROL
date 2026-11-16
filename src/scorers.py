r"""Trains base fraud classifiers on the historical training split and caches stream predictions."""

from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import BenchmarkSplitConfig, ECONOMIC_CFG
from src.data_loader import load_or_generate_raw_stream


def build_and_cache_scored_stream(
    cfg: BenchmarkSplitConfig,
    raw_dir: Path = Path("data/raw"),
    cache_dir: Path = Path("data/cached_streams"),
    force_rebuild: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    r"""
    Loads or builds the pre-scored transaction stream for cfg.name.
    Caches the resulting table to data/cached_streams/{cfg.name}.parquet.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cfg.name}.parquet"

    if cfg.name == "ulb_stream":
        splits = (60, 90, 183)
    else:
        splits = (59, 119, 731)

    if cache_path.exists() and not force_rebuild:
        df = pd.read_parquet(cache_path)
        return df, splits

    df, feature_cols, splits = load_or_generate_raw_stream(cfg, raw_dir=raw_dir, seed=seed)
    train_end_day, _, _ = splits

    train_mask = df["day"].to_numpy() <= train_end_day
    X_train = df.loc[train_mask, feature_cols].to_numpy(dtype=np.float64)
    y_train = df.loc[train_mask, "y_true"].to_numpy(dtype=np.int8)
    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    y_all = df["y_true"].to_numpy(dtype=np.int8)

    if cfg.name == "ulb_stream":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, random_state=seed),
        )
    else:
        model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50000,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    model.fit(X_train, y_train)
    p_hat = model.predict_proba(X_all)[:, 1].astype(np.float64)
    p_hat = np.clip(p_hat, 1e-6, 1.0 - 1e-6)

    eval_mask = ~train_mask
    auc_score = roc_auc_score(y_all[eval_mask], p_hat[eval_mask])
    ap_score = average_precision_score(y_all[eval_mask], p_hat[eval_mask])
    print(
        f"[{cfg.name}] Base Scorer Out-of-Sample ROC-AUC: {auc_score:.4f} | AP: {ap_score:.4f} "
        f"(Trained on Days 1..{train_end_day}, N_train={int(np.sum(train_mask))})"
    )

    out_df = pd.DataFrame(
        {
            "tx_id": df["tx_id"].to_numpy(dtype=np.int64),
            "day": df["day"].to_numpy(dtype=np.int32),
            "amount": df["amount"].to_numpy(dtype=np.float64),
            "y_true": y_all,
            "p_hat": p_hat,
            "triage_score_g": p_hat * (df["amount"].to_numpy(dtype=np.float64) + ECONOMIC_CFG.dispute_fee),
            "delay_days": df["delay_days"].to_numpy(dtype=np.int32),
        }
    )
    out_df.to_parquet(cache_path, index=False)
    return out_df, splits