"""
make_real_payoffs.py
--------------------
Generates or downloads a CSV of real-world payoff data
for the Exponential Weights experiments (Part 2C).

Two modes:
    1. Download actual stock prices with yfinance
    2. Simulate synthetic but realistic returns if offline

Output:
    real_payoffs.csv  (n_days × k_actions)

Each column = one "arm" (stock), each row = one day’s payoff
scaled to [0,1] so that the EW algorithm can use it directly.
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

USE_REAL_DATA = True           # if False → simulate synthetic returns
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "NFLX"]
START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"
OUTFILE = "real_payoffs.csv"

# scaling parameters for return → [0,1]
CLIP_A, CLIP_B = -0.05, 0.05

# ============================================================

def scale_to_unit(x, a=-0.05, b=0.05):
    """clip returns to [a,b] and linearly scale to [0,1]."""
    x = np.clip(x, a, b)
    return (x - a) / (b - a)

def generate_real_data():
    import yfinance as yf
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Adj Close"]
    data = data.dropna()
    returns = data.pct_change().dropna()
    scaled = returns.apply(scale_to_unit, args=(CLIP_A, CLIP_B))
    scaled.to_csv(OUTFILE, index=False)
    print(f"✅ Saved {OUTFILE} with shape {scaled.shape}")

def generate_synthetic_data():
    n_days, k = 356, len(TICKERS)
    rng = np.random.default_rng(42)
    # correlated Gaussian returns, mean 0, std 0.02
    base = rng.normal(0, 0.02, size=(n_days, k))
    scaled = scale_to_unit(base, CLIP_A, CLIP_B)
    pd.DataFrame(scaled, columns=TICKERS).to_csv(OUTFILE, index=False)
    print(f"✅ Generated synthetic data {OUTFILE} with shape {(n_days, k)}")

if __name__ == "__main__":
    if USE_REAL_DATA:
        try:
            generate_real_data()
        except Exception as e:
            print(f"⚠️ Could not fetch data: {e}")
            print("→ Falling back to synthetic data.")
            generate_synthetic_data()
    else:
        generate_synthetic_data()
