# S&P 500: 28-Week → 2-Week Return Model

A compact ML pipeline that:
- Pulls S&P 500 constituents, selects Top 100 by market cap.
- Downloads weekly prices (last 4+ years) with `yfinance`.
- Builds rolling 30-week blocks per ticker (28 in, 2 out).
- Trains a model with time-series CV.
- Produces a prediction table for today's Top 100.

##How to run the code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.train --years 4 --top_n 50 --model gb
python -m src.backtest --years 4 --top_n 50 --model gb --out artifacts/backtest_50_gb.csv
python -m src.predict  --years 4 --top_n 50 --model gb --out artifacts/predictions_next2w_50_gb.csv
