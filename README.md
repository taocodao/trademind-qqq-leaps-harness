# TradeMind QQQ LEAPS Harness

The open reproducibility harness behind the QQQ LEAPS record published at
[trademind.bot/verify](https://trademind.bot/verify): QQQ LEAPS core
positions with a volatility-gated covered-call overlay, evaluated hourly
over January 4, 2021 through August 14, 2026.

**Published record (this repo reproduces it):** +464.2% total, 36.3% CAGR,
Sharpe 1.48, max drawdown -17.8%, from $30,000 to $169,249, versus QQQ buy
and hold at +136.4% and -35.6% over the same window.

Hypothetical backtested performance has many inherent limitations. No
representation is being made that any account will achieve results similar
to those shown. Past performance is not necessarily indicative of future
results. Read the [Limitations](#limitations) section before believing
anything here.

## Quick start

```bash
git clone https://github.com/taocodao/trademind-qqq-leaps-harness
cd trademind-qqq-leaps-harness
pip install -r requirements.txt
python download_data.py   # fetches QQQ/VIX/VIX3M/IRX into ./data
python run.py             # writes output/, prints the headline numbers
```

Then diff your results against the published run:

```bash
diff output/metrics_qqq_leaps_reproduced.json <(cat expected/metrics_qqq_leaps_canonical.json) # metrics
head -5 output/fills_qqq_leaps_reproduced.csv   # every fill, repriced live
```

`expected/` contains the canonical NAV series, ledger, metrics, and run
configuration exactly as published on the site. Small differences in
vendor data can nudge the last decimal; the trade sequence and headline
metrics should match.

## What is open and what is not

**Open:** the full engine (`qqq_leaps_enhanced_2y_hourly.py`), including
the Gaussian-HMM regime classifier, Black-Scholes pricing, strike
selection, position sizing, the PMCC overlay gates (strong-trend,
low-VRP, put-demand, and the QQQ LEAPS trend-times-IV rule), adaptive exits,
slippage and commission modeling, and the complete run configuration.

**Private:** the walk-forward ML confidence model used as one of seven
entry gates. Its precomputed output for the full history ships as
`data/ml_confidence.csv` and is loaded as a column, so every trade in the
ledger reproduces exactly while the model internals stay private. If you
replace that column with your own signal, everything else still runs.

## Method in one paragraph

The engine buys deep in-the-money QQQ LEAPS calls (delta 0.80-0.85, 12-24
months out) when seven entry conditions agree (regime, VIX, trend, RSI,
gap, confidence, put demand), then sells 32-day calls near delta
0.15-0.28 against each LEAPS whenever four skip gates allow it. All
options are priced with Black-Scholes using VIX scaled by regime as the
volatility input and the 13-week T-bill rate as the risk-free rate. Costs
are charged on every fill: $1 per contract plus VIX-scaled slippage of
0.35%-2.0% of premium. Decisions are hourly, in one 3:00 PM ET window per
trading day, with the equity curve marked at each daily close. Full
methodology, limitations, and the self-audit live at
[trademind.bot/verify](https://trademind.bot/verify).

## Sensitivity note

The canonical record runs `entry_ml_min = 0.43`. The engine's default is
0.45, which excludes exactly one trade (July 28, 2026) and yields 35.3%
CAGR instead of 36.3%. Both numbers, and the ledger entry for that trade,
are published. Set `M.CFG.entry_ml_min = 0.45` in `run.py` to see the
difference yourself.

## Limitations

- **Black-Scholes is not a market.** No historical option quotes are used;
  every price is theoretical. Individual fills can deviate from what a
  broker would have quoted, in either direction.
- **Slippage is modeled, not measured.** The VIX-scaled schedule is a
  conservative estimate, not a measurement of real spreads.
- **Hourly granularity.** Intraday movement inside each bar is invisible
  to the test.
- **One window.** 5.6 years containing one deep QQQ correction, mostly a
  rising market.
- **Design selection.** Variants were tried; QQQ LEAPS is the one published.
  Combinatorial cross-validation (18 of 21 paths better on CAGR and
  Sharpe versus the unfiltered engine) mitigates but does not eliminate
  selection effects.

A separate 15-month simulation using real option quotes instead of
Black-Scholes showed a -30.4% maximum drawdown over its window. That
number is published too, on the verify page.

## License

Code: MIT (see LICENSE). Data files are for research and verification
use. Nothing in this repository is investment advice.
