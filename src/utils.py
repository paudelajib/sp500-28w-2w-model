import pandas as pd

def normalize_ticker(ticker: str) -> str:
    return ticker.replace(".", "-") if isinstance(ticker, str) else ticker

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.inferred_type not in ("datetime64", "datetime", "date"):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.resample("W-FRI").last()
