# src/predict.py
from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from .data import get_top_n_by_market_cap, download_weekly_prices, CACHE_DIR

# ----------------------------
# Feature/target helpers
# ----------------------------
def make_windows(
    prices: pd.DataFrame, in_weeks: int = 28, out_weeks: int = 2
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build pooled (X, y) windows across all tickers for training.
    Features: past `in_weeks` weekly % returns up to time t.
    Target:   forward `out_weeks` return = P_{t+out}/P_t - 1.
    Returns X (N,in_weeks), y (N,), meta with ['date','ticker'] for each row.
    """
    X_rows, y_rows, meta = [], [], []
    for ticker in prices.columns:
        p = prices[ticker].dropna()
        if len(p) < in_weeks + out_weeks + 5:
            continue
        r = p.pct_change().dropna()
        p = p.loc[r.index]  # align with returns index

        for t in range(in_weeks, len(r) - out_weeks):
            feats = r.iloc[t - in_weeks : t].values
            y = float(p.iloc[t + out_weeks] / p.iloc[t] - 1.0)
            X_rows.append(feats)
            y_rows.append(y)
            meta.append((r.index[t], ticker))

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    meta_df = pd.DataFrame(meta, columns=["date", "ticker"])
    return X, y, meta_df


def latest_feature_vectors(
    prices: pd.DataFrame, in_weeks: int = 28
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Build one feature vector per ticker using the most recent `in_weeks` returns.
    Returns (features_df, asof_date). features_df has index=ticker, columns f0..f{in_weeks-1}.
    """
    rows: Dict[str, np.ndarray] = {}
    asof = None
    for ticker in prices.columns:
        p = prices[ticker].dropna()
        r = p.pct_change().dropna()
        if len(r) >= in_weeks:
            feats = r.iloc[-in_weeks:].values
            rows[ticker] = feats
            asof = r.index[-1]  # last completed weekly return date
    if not rows:
        return pd.DataFrame(), asof
    F = pd.DataFrame.from_dict(rows, orient="index")
    F.columns = [f"f{i}" for i in range(F.shape[1])]
    return F, asof


# ----------------------------
# Model helper
# ----------------------------
def get_model(name: str):
    name = name.lower()
    if name in ("gb", "gbr", "gradientboosting", "gradient_boosting"):
        return GradientBoostingRegressor(random_state=42)
    if name in ("ridge", "lr"):
        return Ridge(alpha=1.0, random_state=42)
    raise ValueError(f"Unknown model '{name}'. Use 'gb' or 'ridge'.")


# ----------------------------
# Main predict routine
# ----------------------------
def predict_next_2w(
    years: int,
    top_n: int,
    model_name: str,
    out_path: str,
    in_weeks: int = 28,
    out_weeks: int = 2,
):
    # 1) Universe & prices
    top = get_top_n_by_market_cap(n=top_n)
    tickers = top["Symbol"].tolist()
    prices = download_weekly_prices(tickers, years=years)

    # 2) Build training windows (historical) and latest features (today)
    X, y, _ = make_windows(prices, in_weeks=in_weeks, out_weeks=out_weeks)
    F_latest, asof = latest_feature_vectors(prices, in_weeks=in_weeks)

    if X.size == 0 or F_latest.empty:
        raise RuntimeError("Not enough data to train or predict.")

    # 3) Scale + fit model on history
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = get_model(model_name)
    model.fit(Xs, y)

    # 4) Predict next 2-week return per ticker
    X_live = scaler.transform(F_latest.values)
    y_pred = model.predict(X_live)

    # 5) Assemble output CSV
    out = pd.DataFrame(
        {
            "Ticker": F_latest.index,
            "Predicted_2w_Return": y_pred.astype(np.float32),
        }
    ).sort_values("Predicted_2w_Return", ascending=False)
    out.insert(1, "AsOfDate", pd.to_datetime(asof).date())
    out.insert(2, "Model", model_name)
    out.insert(3, "InWeeks", in_weeks)
    out.insert(4, "OutWeeks", out_weeks)
    out.insert(5, "Provider", "stooq")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[info] Wrote {len(out)} predictions -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Predict next 2-week returns for top-N tickers.")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--model", type=str, default="gb", choices=["gb", "ridge"])
    ap.add_argument("--out", type=str, default="artifacts/predictions_next2w.csv")
    ap.add_argument("--in_weeks", type=int, default=28)
    ap.add_argument("--out_weeks", type=int, default=2)
    args = ap.parse_args()

    predict_next_2w(
        years=args.years,
        top_n=args.top_n,
        model_name=args.model,
        out_path=args.out,
        in_weeks=args.in_weeks,
        out_weeks=args.out_weeks,
    )


if __name__ == "__main__":
    main()