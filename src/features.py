import numpy as np, pandas as pd

def build_windows(adj_close: pd.DataFrame, window_total=30, window_in=28):
    X_rows, y_rows, meta_rows = [], [], []
    for tkr in adj_close.columns:
        s = adj_close[tkr].dropna()
        if len(s) < window_total + 1:
            continue
        logret = np.log(s).diff().dropna()
        for i in range(29, len(s)):
            feats = logret.iloc[i-28:i].values
            if len(feats) != 28:
                continue
            target = float(s.iloc[i] / s.iloc[i-2] - 1.0)
            X_rows.append(feats)
            y_rows.append(target)
            meta_rows.append({"ticker": tkr, "end_date": s.index[i]})
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float), pd.DataFrame(meta_rows)

def latest_28w_features(adj_close: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for tkr in adj_close.columns:
        s = adj_close[tkr].dropna()
        if len(s) < 29:
            continue
        logret = np.log(s).diff().dropna()
        feats = logret.iloc[-28:].values
        if len(feats) == 28:
            rows[tkr] = feats
    X = pd.DataFrame.from_dict(rows, orient="index")
    X.columns = [f"f{i}" for i in range(28)]
    return X
