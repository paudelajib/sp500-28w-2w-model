# src/backtest.py
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .data import get_top_n_by_market_cap, download_weekly_prices, CACHE_DIR

def make_windows(prices: pd.DataFrame, in_weeks: int = 28, out_weeks: int = 2):
    """
    Convert wide weekly price matrix (index=date, columns=tickers) into a pooled dataset.
    Features: past 28 weekly % changes (per row).
    Target:   forward 2-week return = P_{t+2}/P_t - 1.
    Returns: X (np.ndarray), y (np.ndarray), meta DataFrame ['date','ticker'].
    """
    X_rows, y_rows, meta = [], [], []
    for ticker in prices.columns:
        p = prices[ticker].dropna()
        if len(p) < in_weeks + out_weeks + 5:
            continue
        r = p.pct_change().dropna()
        p = p.loc[r.index]  # align

        for t in range(in_weeks, len(r) - out_weeks):
            feats = r.iloc[t - in_weeks:t].values  # shape (in_weeks,)
            y = float(p.iloc[t + out_weeks] / p.iloc[t] - 1.0)
            X_rows.append(feats)
            y_rows.append(y)
            meta.append((r.index[t], ticker))

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    meta_df = pd.DataFrame(meta, columns=["date", "ticker"])
    return X, y, meta_df

def get_model(name: str):
    name = name.lower()
    if name in ("gb", "gbr", "gradientboosting", "gradient_boosting"):
        return GradientBoostingRegressor(random_state=42)
    if name in ("ridge", "lr"):
        return Ridge(alpha=1.0, random_state=42)
    raise ValueError(f"Unknown model '{name}'. Use 'gb' or 'ridge'.")

def backtest(years: int, top_n: int, model_name: str, out_path: str,
             in_weeks: int = 28, out_weeks: int = 2, n_splits: int = 5):
    # 1) Universe & prices
    top = get_top_n_by_market_cap(n=top_n)
    tickers = top["Symbol"].tolist()
    prices = download_weekly_prices(tickers, years=years)

    # 2) Windows
    X, y, meta = make_windows(prices, in_weeks=in_weeks, out_weeks=out_weeks)
    order = np.argsort(meta["date"].values)
    X, y, meta = X[order], y[order], meta.iloc[order].reset_index(drop=True)

    # 3) Walk-forward CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = np.zeros_like(y, dtype=np.float32)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_metrics = []

    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        model = get_model(model_name)
        model.fit(Xtr, y[tr])
        yp = model.predict(Xte).astype(np.float32)

        preds[te] = yp
        fold_ids[te] = fold

        mae = mean_absolute_error(y[te], yp)
        r2 = r2_score(y[te], yp)
        fold_metrics.append((fold, mae, r2))
        print(f"Fold {fold}: MAE={mae:.5f}, R2={r2:.4f}")

    # 4) Collect per-row outputs
    out_df = meta.copy()
    out_df["y_true"] = y
    out_df["y_pred"] = preds
    out_df["abs_error"] = np.abs(out_df["y_true"] - out_df["y_pred"])
    out_df["fold"] = fold_ids

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote backtest rows: {len(out_df)} -> {out_path}")

    # 5) Summary
    overall_mae = mean_absolute_error(out_df["y_true"], out_df["y_pred"])
    overall_r2 = r2_score(out_df["y_true"], out_df["y_pred"])
    print(f"Overall: MAE={overall_mae:.5f}, R2={overall_r2:.4f}")

    # Save fold metrics
    metrics = pd.DataFrame(fold_metrics, columns=["fold","MAE","R2"])
    metrics.loc["overall"] = ["overall", overall_mae, overall_r2]
    metrics_path = os.path.join(CACHE_DIR, "backtest_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved fold metrics -> {metrics_path}")

def main():
    p = argparse.ArgumentParser(description="Backtest 28->2 week model with walk-forward CV")
    p.add_argument("--years", type=int, default=4)
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--model", type=str, default="gb", choices=["gb","ridge"])
    p.add_argument("--out", type=str, default="artifacts/backtest_50_gb.csv")
    p.add_argument("--in_weeks", type=int, default=28)
    p.add_argument("--out_weeks", type=int, default=2)
    p.add_argument("--splits", type=int, default=5)
    args = p.parse_args()

    backtest(
        years=args.years,
        top_n=args.top_n,
        model_name=args.model,
        out_path=args.out,
        in_weeks=args.in_weeks,
        out_weeks=args.out_weeks,
        n_splits=args.splits,
    )

if __name__ == "__main__":
    main()