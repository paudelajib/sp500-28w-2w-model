from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import List

import pandas as pd

# =============================================================================
# Paths & cache
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# Ticker normalization utilities
#   - Internally we use Yahoo-style classes (e.g., BRK-B).
#   - Stooq often uses dots (BRK.B) and sometimes needs ".US".
# =============================================================================
def to_stooq_symbol(ticker: str) -> str:
    """Map Yahoo class shares to Stooq: BRK-B -> BRK.B; otherwise return as-is."""
    if "-" in ticker:
        parts = ticker.split("-")
        if len(parts) == 2 and len(parts[1]) == 1:  # e.g., BRK-B, BF-B
            return f"{parts[0]}.{parts[1]}"
    return ticker

def normalize_ticker_yahoo(t: str) -> str:
    """Normalize scraped symbols to Yahoo-style (BRK.B -> BRK-B)."""
    if "." in t:
        parts = t.split(".")
        if len(parts) == 2 and len(parts[1]) == 1:
            return f"{parts[0]}-{parts[1]}"
    return t

# Try multiple possible Stooq aliases per ticker (first hit wins).
STOOQ_ALIASES: dict[str, list[str]] = {
    "BRK-B": ["BRK.B", "BRK.B.US", "BRK-B", "BRK-B.US"],
    # Uncomment/add more if needed:
    # "META":  ["META", "FB", "META.US", "FB.US"],
    # "GOOGL": ["GOOGL", "GOOGL.US"],
    # "GOOG":  ["GOOG", "GOOG.US"],
}

# =============================================================================
# S&P 500 constituents (robust with cache + fallbacks)
# =============================================================================
def get_sp500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """
    Get S&P 500 constituents.
      1) cache/sp500_constituents.csv if fresh (<7 days)
      2) Wikipedia scrape with a real User-Agent
      3) local sp500_symbols.csv (project root) if present
      4) curated mega-cap fallback subset (enough to run anywhere)
    Returns: DataFrame with at least 'Symbol' (Yahoo-style, e.g., BRK-B)
    """
    import requests

    cache_path = os.path.join(CACHE_DIR, "sp500_constituents.csv")

    def _is_fresh(path: str, days: int = 7) -> bool:
        return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < days * 86400

    # 1) Cache
    if not force_refresh and _is_fresh(cache_path, 7):
        try:
            df = pd.read_csv(cache_path)
            if not df.empty and "Symbol" in df.columns:
                df["Symbol"] = df["Symbol"].astype(str).apply(normalize_ticker_yahoo)
                return df
        except Exception:
            pass

    # 2) Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        sp500 = tables[0].copy()
        sp500["Symbol"] = sp500["Symbol"].astype(str).apply(normalize_ticker_yahoo)
        sp500.to_csv(cache_path, index=False)
        if not sp500.empty:
            return sp500
    except Exception:
        pass

    # 3) Local CSV fallback (project root)
    local_csv = os.path.join(PROJECT_ROOT, "sp500_symbols.csv")
    if os.path.exists(local_csv):
        try:
            df = pd.read_csv(local_csv)
            if "Symbol" in df.columns and not df.empty:
                df["Symbol"] = df["Symbol"].astype(str).apply(normalize_ticker_yahoo)
                return df
        except Exception:
            pass

    # 4) Curated fallback
    fallback_syms = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","AVGO","LLY","TSLA",
        "JPM","V","UNH","XOM","JNJ","PG","MA","COST","ORCL","HD","MRK","PEP",
        "KO","BAC","ABBV","CVX","WMT","NFLX","ADBE","CRM","AMD","LIN","ACN",
        "DHR","PM","MCD","INTU","TXN","TMUS","UNP","NEE","AMAT","PFE","TMO",
        "IBM","HON","QCOM","VRTX","RTX"
    ]
    return pd.DataFrame({"Symbol": fallback_syms})

# =============================================================================
# Top-N by market cap (no live caps)
# - Uses cached caps if you provide cache/top{n}_caps.csv (Symbol,MarketCap, sorted).
# - Else: curated megacaps, then fill to exactly n using the S&P list order.
# =============================================================================
def get_top_n_by_market_cap(n: int = 100, force_refresh: bool = False) -> pd.DataFrame:
    sp500 = get_sp500_constituents(force_refresh=force_refresh)

    # 1) Try cached caps first
    caps_path = os.path.join(CACHE_DIR, f"top{n}_caps.csv")
    if os.path.exists(caps_path) and not force_refresh:
        try:
            cached = pd.read_csv(caps_path)
            if "Symbol" in cached.columns and not cached.empty:
                return cached.head(n).reset_index(drop=True)
        except Exception:
            pass

    # 2) Curated megacaps + fill to exactly n
    curated = [
        "NVDA","MSFT","AAPL","AMZN","GOOGL","GOOG","META","AVGO","BRK-B","TSLA",
        "JPM","V","UNH","XOM","JNJ","PG","MA","COST","ORCL","HD","MRK","PEP",
        "KO","BAC","ABBV","CVX","WMT","NFLX","ADBE","CRM","AMD","LIN","ACN",
        "DHR","PM","MCD","INTU","TXN","TMUS","UNP","NEE","AMAT","PFE","TMO",
        "IBM","HON","QCOM","VRTX","RTX",
        # extras so we can exceed 50 and still trim to n
        "NOW","COP","LOW","CAT","AMGN","GE","ISRG","BKNG","ELV","BLK"
    ]
    have = set(sp500["Symbol"].tolist())
    base = [s for s in curated if s in have]

    if len(base) < n:
        extras = [s for s in sp500["Symbol"].tolist() if s not in base]
        base = base + extras[: n - len(base)]

    return pd.DataFrame({"Symbol": base[:n]}).reset_index(drop=True)

# =============================================================================
# Price downloader (Stooq only): daily -> resample to weekly (W-FRI)
# =============================================================================
def _stooq_fetch_daily(ticker: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """Fetch daily OHLC from Stooq via pandas_datareader; ensure ascending index."""
    try:
        from pandas_datareader import data as pdr
        df = pdr.DataReader(ticker, "stooq", start, end)  # Open High Low Close Volume
        if df is None or df.empty or "Close" not in df:
            return None
        return df.sort_index()
    except Exception:
        return None

def _daily_to_weekly_close(df: pd.DataFrame, name: str) -> pd.DataFrame | None:
    """Resample daily close to weekly (Friday). Returns 1-col DataFrame named `name`."""
    if df is None or df.empty or "Close" not in df:
        return None
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index)
    w = s.resample("W-FRI").last()
    return w.to_frame(name)

def download_weekly_prices(tickers: List[str], years: int = 4, chunk_size: int = 20) -> pd.DataFrame:
    """
    Download weekly close (approx) from Stooq:
      - For each ticker, try alias list (e.g., BRK-B -> BRK.B, BRK.B.US, ...)
      - Fetch DAILY per ticker, resample to W-FRI
      - Concatenate into wide matrix [date x tickers]
    """
    start = datetime.utcnow() - timedelta(days=365 * max(years, 4) + 14)
    end = datetime.utcnow()

    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"[info] stooq chunk {i//chunk_size + 1}: {chunk}", flush=True)
        per = []
        for t in chunk:
            candidates = STOOQ_ALIASES.get(t, [])
            base = to_stooq_symbol(t)
            if base.endswith(".US"):
                candidates = candidates + [base]
            else:
                candidates = candidates + [base, f"{base}.US"]

            df = None
            for sym in candidates:
                df = _stooq_fetch_daily(sym, start, end)
                if df is not None and not df.empty:
                    break

            wk = _daily_to_weekly_close(df, t)  # keep original Yahoo-style symbol
            if wk is not None and not wk.empty:
                per.append(wk)

            time.sleep(0.12)  # polite pacing
        if per:
            frames.append(pd.concat(per, axis=1))
        time.sleep(0.3)

    if not frames:
        raise RuntimeError("No data from Stooq. Try a different network or a smaller ticker set.")

    adj_close = pd.concat(frames, axis=1)
    adj_close = adj_close.dropna(axis=1, how="all").sort_index().ffill()

    # Cache parquet for reruns
    try:
        adj_close.to_parquet(os.path.join(CACHE_DIR, "adj_close.parquet"))
    except Exception:
        pass

    print(f"[info] acquired weekly data for {adj_close.shape[1]} tickers; shape={adj_close.shape}", flush=True)
    return adj_close