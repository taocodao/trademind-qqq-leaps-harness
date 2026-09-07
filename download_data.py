#!/usr/bin/env python3
"""
download_data.py - fetch the input series the harness needs.

Pulls QQQ hourly + daily bars and the daily VIX, VIX3M, and 13-week T-bill
series from Yahoo Finance (yfinance) and writes them to ./data in exactly
the layout the engine reads.

Note: the canonical published run used institutional hourly bars. Free
sources can differ by a few cents per bar, which can nudge results by a
fraction of a percent. DATA_SHA256.txt lists the checksums of the exact
files used for the published record if you want to compare your download.

Usage:
    pip install -r requirements.txt
    python download_data.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path("data")
DATA.mkdir(exist_ok=True)

START = "2018-01-01"
END = "2026-08-15"


def save_daily(symbol: str, filename: str) -> None:
    df = yf.download(symbol, start=START, end=END, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"no data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(DATA / filename)
    print(f"{symbol}: {len(df)} daily rows -> {filename}")


def save_hourly(symbol: str, filename: str) -> None:
    # yfinance limits 1h history to ~730 days; chunk by year and stitch.
    frames = []
    for year in range(2018, 2027):
        start = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(START))
        end = min(pd.Timestamp(f"{year + 1}-01-01"), pd.Timestamp(END))
        if start >= end:
            continue
        print(f"  {symbol} hourly {year}...", flush=True)
        df = yf.download(symbol, start=start.date().isoformat(), end=end.date().isoformat(),
                         interval="1h", auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            frames.append(df)
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.to_csv(DATA / filename)
    print(f"{symbol}: {len(out)} hourly rows -> {filename}")


def main() -> None:
    save_daily("QQQ", "qqq_1d.csv")
    save_daily("^VIX", "vix_1d.csv")
    save_daily("^VIX3M", "vix3m_1d.csv")
    save_daily("^IRX", "irx_1d.csv")
    save_hourly("QQQ", "qqq_1h.csv")
    print("\nDone. Now run: python run.py")


if __name__ == "__main__":
    main()
