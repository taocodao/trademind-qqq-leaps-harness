#!/usr/bin/env python3
"""
run.py - reproduce the published V4 record exactly.

Reads CSVs from ./data, runs the full 2021-01-04 -> 2026-08-14 window with
the canonical configuration (entry_ml_min = 0.43), and writes:

    output/nav_v4_reproduced.csv      daily equity curve
    output/fills_v4_reproduced.csv    every fill, repriced live
    output/metrics_v4_reproduced.json headline metrics

Compare against expected/ to verify bit-for-bit reproduction.
"""
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("QQQ_OUT_DIR", "output")
os.environ.setdefault("QQQ_DATA_DIR", "data")

import numpy as np
import pandas as pd
import logging

logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
import qqq_leaps_enhanced_2y_hourly as M

VAL_START = pd.Timestamp("2021-01-04")
VAL_END = pd.Timestamp("2026-08-14")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# Canonical public-record configuration. The default in engine config is
# 0.45; the published record uses 0.43. See README for the sensitivity note.
M.CFG.entry_ml_min = 0.43


def main() -> None:
    print("loading data + building features (HMM fit takes a minute)...", flush=True)
    data = M.load_market_data()
    features, _, _ = M.build_enhanced_features(data)

    print(f"running full window {VAL_START.date()} -> {VAL_END.date()}...", flush=True)
    res = M.run_enhanced(VAL_START, VAL_END, data, features)

    nav = res["nav_series"]
    nav.to_csv(OUT / "nav_v4_reproduced.csv", index=False)

    for key in ("fills", "ledger", "trades", "fills_df"):
        if key in res and res[key] is not None:
            v = res[key]
            df = v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)
            df.to_csv(OUT / "fills_v4_reproduced.csv", index=False)
            print(f"fills saved from engine key '{key}': {len(df)} rows", flush=True)
            break

    s = nav["nav"].astype(float)
    rets = s.pct_change().dropna()
    curve = s / s.iloc[0]
    dd = curve / curve.cummax() - 1
    n = len(rets)
    mtr = {
        "initial_capital": 30000.0,
        "final_nav": float(s.iloc[-1]),
        "total_return_pct": (s.iloc[-1] / s.iloc[0] - 1) * 100,
        "cagr_pct": ((s.iloc[-1] / s.iloc[0]) ** (252 / n) - 1) * 100,
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(252)),
        "max_drawdown_pct": float(dd.min() * 100),
        "years": n / 252.0,
        "entry_ml_min": M.CFG.entry_ml_min,
    }
    mtr["calmar"] = mtr["cagr_pct"] / abs(mtr["max_drawdown_pct"])
    try:
        mtr["engine_metrics"] = M.compute_metrics(res, M.CFG.initial_capital)
    except Exception as exc:  # noqa: BLE001
        mtr["engine_metrics_error"] = str(exc)

    with open(OUT / "metrics_v4_reproduced.json", "w") as f:
        json.dump(mtr, f, indent=2, default=str)

    print("\n=== reproduced headline ===")
    for k in ("total_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct", "calmar", "final_nav"):
        print(f"  {k}: {mtr[k]:.4f}" if isinstance(mtr[k], float) else f"  {k}: {mtr[k]}")
    print("\ncompare against expected/metrics_v4_canonical.json")
    print("DONE")


if __name__ == "__main__":
    main()
