import os, argparse, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump
from .data import get_top_n_by_market_cap, download_weekly_prices
from .features import build_windows
from .model import make_model

ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

def main(args):
    top = get_top_n_by_market_cap(n=args.top_n)
    tickers = top["Symbol"].tolist()
    adj_close = download_weekly_prices(tickers, years=args.years)
    X, y, meta = build_windows(adj_close)
    order = np.argsort(meta["end_date"].values)
    X, y, meta = X[order], y[order], meta.iloc[order].reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        model = make_model(args.model)
        model.fit(X[tr_idx], y[tr_idx])
        yhat = model.predict(X[te_idx])
        print(f"Fold {fold+1}: MAE={mean_absolute_error(y[te_idx], yhat):.5f}, R2={r2_score(y[te_idx], yhat):.4f}")
    final_model = make_model(args.model)
    final_model.fit(X, y)
    dump(final_model, os.path.join(ART_DIR, "model.joblib"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=4)
    p.add_argument("--top_n", type=int, default=100)
    p.add_argument("--model", type=str, default="gb", choices=["gb", "ridge"])
    main(p.parse_args())
