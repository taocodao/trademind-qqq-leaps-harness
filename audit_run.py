#!/usr/bin/env python3
"""
audit_run.py - log-only audit instrumentation for the canonical QQQ LEAPS run.

Wraps the public engine: records every entry-gate decision (values, limits,
pass/fail) and attaches decision-time market context to every fill. Does not
change any decision logic. Post-run validation diffs the instrumented fills
against the canonical fills CSV on all shared columns - if anything differs,
the audit data is invalid and must not ship.
"""
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("QQQ_DATA_DIR", "data")
os.environ.setdefault("QQQ_OUT_DIR", "output")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import qqq_leaps_enhanced_2y_hourly as M

M.CFG.entry_ml_min = 0.43  # canonical public-record configuration

GATE_ROWS = []


def _num(v):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return str(v)


def _gate_snapshot(engine, row):
    cfg = engine.cfg
    g = {
        "regime": {
            "value": str(row["regime"]),
            "rule": "not BEAR",
            "pass": row["regime"] not in ["BEAR", "BEAR_SMA_FORCED"],
        },
        "vix": {
            "value": _num(row["vix"]),
            "limit": cfg.entry_vix_max,
            "rule": f"VIX < {cfg.entry_vix_max}",
            "pass": bool(row["vix"] < cfg.entry_vix_max),
        },
        "above_sma100": {
            "value": bool(row["above_sma100"]),
            "rule": "price above 100-day average",
            "pass": bool(row["above_sma100"]),
        },
        "rsi_14": {
            "value": _num(row["rsi_14"]),
            "limit": cfg.entry_rsi14_max,
            "rule": f"RSI-14 < {cfg.entry_rsi14_max}",
            "pass": bool(row["rsi_14"] < cfg.entry_rsi14_max),
        },
        "gap_down_pct": {
            "value": _num(row["gap_down_pct"]),
            "limit": cfg.entry_gap_down_min,
            "rule": f"gap down >= {cfg.entry_gap_down_min}",
            "pass": bool(row["gap_down_pct"] >= cfg.entry_gap_down_min),
        },
        "ml_confidence": {
            "value": _num(row["ml_confidence"]),
            "limit": cfg.entry_ml_min,
            "rule": f"confidence >= {cfg.entry_ml_min}",
            "pass": bool(row["ml_confidence"] >= cfg.entry_ml_min),
        },
        "put_demand": {
            "value": _num(row.get("put_demand_proxy")),
            "limit": cfg.entry_put_demand_max,
            "rule": f"put demand <= {cfg.entry_put_demand_max}",
            "pass": not (pd.notna(row.get("put_demand_proxy"))
                         and row["put_demand_proxy"] > cfg.entry_put_demand_max),
        },
    }
    return g


_orig_check_entry = M.EnhancedEngine.check_entry
_orig_pmcc_should_open = M.EnhancedEngine.pmcc_should_open


def check_entry_audit(self, row):
    result = _orig_check_entry(self, row)
    GATE_ROWS.append({
        "ts": str(row.name),
        "decision": "ENTER" if result else "NO_ENTRY",
        "gates": _gate_snapshot(self, row),
    })
    return result


def pmcc_should_open_audit(self, row, pos):
    ok, reason = _orig_pmcc_should_open(self, row, pos)
    GATE_ROWS.append({
        "ts": str(row.name),
        "decision": "PMCC_OPEN" if ok else "PMCC_SKIP",
        "reason": str(reason),
        "context": {
            "regime": str(row.get("regime")),
            "vix": _num(row.get("vix")),
            "adx_14": _num(row.get("adx_14")),
            "iv_rv_ratio": _num(row.get("iv_rv_ratio")),
            "put_demand_proxy": _num(row.get("put_demand_proxy")),
        },
    })
    return ok, reason


# market context attached to every fill at log time
ROW_CONTEXT = {}


def _stash_row(meth):
    orig = getattr(M.EnhancedEngine, meth)

    def wrapped(self, ts, *args, **kw):
        row = None
        for a in args:
            if isinstance(a, pd.Series):
                row = a
                break
        if row is None:
            row = kw.get("row")
        if isinstance(row, pd.Series) and row.name is not None:
            ROW_CONTEXT[str(ts)] = {
                "date": str(row.name),
                "regime": str(row.get("regime")),
                "vix": _num(row.get("vix")),
                "rsi_14": _num(row.get("rsi_14")),
                "adx_14": _num(row.get("adx_14")),
                "gap_down_pct": _num(row.get("gap_down_pct")),
                "ml_confidence": _num(row.get("ml_confidence")),
                "iv_rv_ratio": _num(row.get("iv_rv_ratio")),
                "put_demand_proxy": _num(row.get("put_demand_proxy")),
            }
        return orig(self, ts, *args, **kw)

    setattr(M.EnhancedEngine, meth, wrapped)


for m in ("open_leaps", "close_leaps", "try_open_short", "close_short"):
    _stash_row(m)

M.EnhancedEngine.check_entry = check_entry_audit
M.EnhancedEngine.pmcc_should_open = pmcc_should_open_audit

OUT = Path("output")
OUT.mkdir(exist_ok=True)


def main():
    print("loading data + features...", flush=True)
    data = M.load_market_data()
    features, _, _ = M.build_enhanced_features(data)

    print("running canonical window with audit instrumentation...", flush=True)
    start = pd.Timestamp("2021-01-04")
    end = pd.Timestamp("2026-08-14")
    res = M.run_enhanced(start, end, data, features)

    fills = pd.DataFrame(res["fills"])
    print(f"fills: {len(fills)}", flush=True)

    # ---- validation gate: instrumented fills must match the canonical CSV ----
    canon_path = Path("expected/fills_v4_canonical.csv")
    if canon_path.exists():
        canon = pd.read_csv(canon_path)
        shared = [c for c in canon.columns if c in fills.columns]
        a = canon[shared].astype(str).sort_values(by=list(shared[:1])).reset_index(drop=True)
        b = fills[shared].astype(str).sort_values(by=list(shared[:1])).reset_index(drop=True)
        same = a.equals(b) and len(a) == len(b)
        print(f"canonical fills match on {len(shared)} shared columns: {same}", flush=True)
        if not same:
            print("VALIDATION FAILED - audit output withheld", flush=True)
            sys.exit(1)

    # nav check
    nav = res["nav_series"]
    s = nav["nav"].astype(float)
    final_nav = float(s.iloc[-1])
    total = (s.iloc[-1] / s.iloc[0] - 1) * 100
    curve = s / s.iloc[0]
    dd = curve / curve.cummax() - 1
    print(f"total {total:.1f}% nav {final_nav:,.0f} maxdd {dd.min()*100:.1f}%", flush=True)

    # attach market context to fills
    def ctx(ts):
        return ROW_CONTEXT.get(str(ts)) or {}

    detail = []
    real = fills[~fills["action"].astype(str).str.startswith("SKIP")]
    for _, f in real.iterrows():
        d = {k: _num(v) if not isinstance(v, str) else v for k, v in f.items()}
        d["context"] = ctx(str(f["ts"]))
        detail.append(d)

    with open(OUT / "fills_audit.json", "w") as fh:
        json.dump(detail, fh, indent=1)
    with open(OUT / "entry_gate_audit.json", "w") as fh:
        json.dump(GATE_ROWS, fh, indent=1)

    print(f"audit fills: {len(detail)}, gate decisions: {len(GATE_ROWS)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
