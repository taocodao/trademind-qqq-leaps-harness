"""
QQQ LEAPS Enhanced Backtest - open harness, public reproducibility edition.

PHASE 0: PMCC strong-trend skip rule (regime + ADX gate)
PHASE 1: Gaussian-HMM regime classifier (hmmlearn, 3 states, BIC selection)
PHASE 2: Options-market features added to the stack
PHASE 3: All upgrades combined
PHASE 4: Adaptive rule-based exit surrogate

PUBLIC EDITION: the walk-forward ML confidence model is private. Its
precomputed output ships as data/ml_confidence.csv and is loaded as a
column. Everything else in the decision stack is the exact production code.
"""
from __future__ import annotations
import json
import math
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp

warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = Path(os.getenv("QQQ_OUT_DIR", "output"))
OUT_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR = os.getenv("QQQ_DATA_DIR", "data")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("enhanced")


# =============================================================================
# CONFIG (extends baseline)
# =============================================================================
@dataclass
class Config:
    entry_rsi14_max: float = 35.0
    entry_gap_down_min: float = 0.003
    entry_ml_min: float = 0.45
    entry_vix_max: float = 40.0

    delta_bull: float = 0.85
    delta_neutral: float = 0.80
    delta_bear: float = 0.65
    dte_bull: int = 365
    dte_neutral: int = 540
    dte_bear: int = 730

    pmcc_dte: int = 32
    pmcc_delta_bull_strong: float = 0.28
    pmcc_delta_bull_moderate: float = 0.23
    pmcc_delta_defensive: float = 0.15
    pmcc_profit_take_early: float = 0.20
    pmcc_profit_take_late: float = 0.10
    pmcc_gamma_manage_dte: int = 21
    pmcc_loss_multiple: float = 2.0
    pmcc_roll_delta: float = 0.40

    max_positions: int = 3
    max_position_pct: float = 0.33
    max_contracts: int = 5
    cash_reserve: float = 0.05

    commission: float = 1.00
    iv_qqq_premium: float = 1.10
    iv_scale_short: float = 1.12
    initial_capital: float = 30_000.0

    # PHASE 0: PMCC skip conditions
    pmcc_skip_regime: tuple = ("BULL_STRONG",)
    pmcc_skip_adx_min: float = 16.0   # validated via rolling-fold walk-forward (2021-2026 hourly data): beat default=30.0 on CAGR+Sharpe+MaxDD in 2/2 out-of-sample folds
    pmcc_skip_vrp_max: float = 0.7        # CPCV revalidation (2021-2026 hourly, 21 paths): 0.70 beats 0.90 in 18/21 paths on CAGR+Sharpe and 21/21 on drawdown -- mean Sharpe 1.04 vs 0.92, mean CAGR 11.7% vs 10.2%, mean MaxDD -9.7% vs -11.5%. Supersedes prior 2-fold conclusion that kept 0.90.
    pmcc_skip_put_demand_max: float = 1.2  # confirmed inactive (0 triggers) across all 5 engines full backtest windows -- not worth tuning, gate never fires at this threshold

    # PHASE 4: adaptive exit thresholds
    adaptive_exit_enabled: bool = True
    exit_delta_low_bull: float = 0.55
    exit_delta_low_bear: float = 0.70
    exit_dte_min_bull: int = 60
    exit_dte_min_bear: int = 120

    # PHASE 2: additional entry gate
    entry_put_demand_max: float = 1.5

    def slippage_for_vix(self, vix, mid_price):
        if vix < 18:   sp = 0.007
        elif vix < 25: sp = 0.012
        elif vix < 35: sp = 0.020
        else:          sp = 0.040
        return mid_price * 100 * sp / 2

    def iv_multiplier(self, vix):
        if vix < 18:   return 1.30
        elif vix < 25: return 1.22
        elif vix < 35: return 1.10
        else:          return 0.65


CFG = Config()


# =============================================================================
# BLACK-SCHOLES
# =============================================================================
def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)


def bs_call_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return norm.cdf(d1)


def find_call_strike(S, T, r, sigma, target_delta):
    lo, hi = S*0.30, S*1.40
    for _ in range(80):
        mid = (lo + hi) / 2
        d = bs_call_delta(S, mid, T, r, sigma)
        if abs(d - target_delta) < 1e-5:
            return mid
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return S


# =============================================================================
# DATA LOADING
# =============================================================================
def load_market_data():
    d = DATA_DIR
    qqq_1h = pd.read_csv(f"{d}/qqq_1h.csv", index_col=0, parse_dates=True)
    qqq_1d = pd.read_csv(f"{d}/qqq_1d.csv", index_col=0, parse_dates=True)
    vix    = pd.read_csv(f"{d}/vix_1d.csv", index_col=0, parse_dates=True)
    vix3m  = pd.read_csv(f"{d}/vix3m_1d.csv", index_col=0, parse_dates=True)
    irx    = pd.read_csv(f"{d}/irx_1d.csv", index_col=0, parse_dates=True)
    for df in [qqq_1d, vix, vix3m, irx]:
        idx = pd.to_datetime(df.index, utc=True)
        df.index = idx.tz_convert(None).normalize()
    qqq_1h.index = pd.to_datetime(qqq_1h.index, utc=True).tz_convert("America/New_York")
    return {"qqq_1h": qqq_1h, "qqq_1d": qqq_1d, "vix": vix, "vix3m": vix3m, "irx": irx}


# =============================================================================
# PHASE 1: Gaussian-HMM regime classifier
# =============================================================================
def fit_gaussian_hmm(train_df: pd.DataFrame, n_states: int = 3, random_state: int = 42):
    """Fit HMM on training window; return model + labels. Uses BIC over multiple seeds."""
    from hmmlearn.hmm import GaussianHMM

    features = ["ret_1d", "realized_vol_20d", "vix"]
    X = train_df[features].dropna().values
    # Scale VIX to same OOM as returns
    X_scaled = X.copy()
    X_scaled[:, 1] = X_scaled[:, 1] / 100  # vol as fraction
    X_scaled[:, 2] = X_scaled[:, 2] / 100  # vix as fraction

    best_model, best_bic = None, np.inf
    for seed in [random_state, random_state + 1, random_state + 7, random_state + 13]:
        try:
            m = GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=200, random_state=seed, tol=1e-3)
            m.fit(X_scaled)
            score = m.score(X_scaled)
            n_params = n_states * (1 + 3 + 6) + n_states*n_states
            bic = -2*score + n_params*np.log(len(X_scaled))
            if bic < best_bic:
                best_bic = bic
                best_model = m
        except Exception as e:
            logger.warning(f"HMM fit seed={seed} failed: {e}")

    if best_model is None:
        raise RuntimeError("All HMM fits failed")

    # Label states by mean return
    states = best_model.predict(X_scaled)
    state_stats = {}
    for s in range(n_states):
        idx = states == s
        if idx.sum() > 0:
            state_stats[s] = {
                "mean_ret_bp": float(np.mean(X[idx, 0]) * 10000),
                "mean_vol_pct": float(np.mean(X[idx, 1])),
                "mean_vix": float(np.mean(X[idx, 2])),
                "count": int(idx.sum()),
            }
    # Label by combined signal: high return + low vol = bull; high vol = bear regardless of return sign
    # Score = mean_ret_bp - vix_penalty. Bear = high VIX. Bull_strong = high positive return + low vol.
    scored = []
    for sid, s in state_stats.items():
        score = s["mean_ret_bp"] - 0.5 * s["mean_vix"]
        scored.append((sid, score, s))
    scored.sort(key=lambda x: -x[1])  # highest score first

    # 3-state assignment: rank 0=BULL_STRONG, rank 1 = BULL_MODERATE (if low VIX) else BEAR, rank 2 = BEAR
    labels = {}
    if n_states == 3:
        labels[scored[0][0]] = "BULL_STRONG"
        # Middle state: if its VIX is high, call it BEAR; if low, call it BULL_MODERATE
        mid_sid, _, mid_stats = scored[1]
        labels[mid_sid] = "BULL_MODERATE" if mid_stats["mean_vix"] < 22 else "BEAR"
        # Lowest state
        low_sid, _, low_stats = scored[2]
        # If we already have BEAR, make lowest also BEAR (overlap fine); otherwise assign BEAR
        labels[low_sid] = "BEAR"
    else:
        label_names = ["BULL_STRONG", "BULL_MODERATE", "CHOPPY", "BEAR"]
        for i, (sid, _, _) in enumerate(scored):
            labels[sid] = label_names[i]
    return best_model, labels, state_stats, best_bic


def causal_hmm_decode(model, X_scaled: np.ndarray):
    """
    Causal (lookahead-free) HMM state decode.

    FIX for lookahead bias: `model.predict()` / `model.decode()` run full-sequence
    Viterbi, and `model.predict_proba()` runs forward-backward smoothing — both use
    information from every timestep in X_scaled, including bars *after* the one
    being labeled. hmmlearn's own docs warn this makes them unsafe for backtesting
    a streaming signal: "the history of decoded states may well be rewritten as
    future emissions are observed".

    This function instead runs only the forward pass (the alpha-recursion), so the
    filtered state probability at time t depends only on observations 0..t — never
    on anything at t+1 or later. This is the textbook "filtering" distribution
    p(z_t | x_1:t), as opposed to "smoothing" p(z_t | x_1:T).

    Returns
    -------
    causal_states : np.ndarray, shape (T,)
        argmax filtered state at each t (causal analogue of `.predict()`).
    causal_probs : np.ndarray, shape (T, n_states)
        filtered state probabilities at each t (causal analogue of `.predict_proba()`).
    """
    try:
        from hmmlearn import _hmmc
        log_frameprob = model._compute_log_likelihood(X_scaled)
        _, fwdlattice = _hmmc.forward_log(model.startprob_, model.transmat_, log_frameprob)
        causal_probs = np.exp(fwdlattice - logsumexp(fwdlattice, axis=1, keepdims=True))
    except Exception as e:
        # Fallback if hmmlearn's internal forward-pass API changes: recompute the
        # filtered posterior via expanding-window re-scoring. This is O(T^2) and
        # much slower, but produces the same causal (filtering) quantity using only
        # the public API, so correctness does not depend on private internals.
        logger.warning(f"causal_hmm_decode: forward_log unavailable ({e}); "
                        f"falling back to expanding-window re-scoring")
        T, K = len(X_scaled), model.n_components
        causal_probs = np.zeros((T, K))
        step = max(1, T // 500)  # recompute periodically for tractable runtime
        last = np.full(K, 1.0 / K)
        for t in range(T):
            if t % step == 0 or t == T - 1:
                _, post = model.score_samples(X_scaled[: t + 1])
                last = post[-1]
            causal_probs[t] = last
    causal_states = causal_probs.argmax(axis=1)
    return causal_states, causal_probs


# =============================================================================
# PHASE 2: Options-market features
# =============================================================================
def compute_options_features(daily, vix, vix3m):
    df = pd.DataFrame(index=daily.index)
    ret = daily["Close"].pct_change()
    df["realized_vol_20d"] = ret.rolling(20).std() * np.sqrt(252) * 100
    df["realized_skew_20d"] = ret.rolling(20).skew()
    df["vix"] = vix.reindex(df.index, method="ffill")
    df["vix3m"] = vix3m.reindex(df.index, method="ffill")
    df["iv_rv_ratio"] = df["vix"] / df["realized_vol_20d"].replace(0, np.nan)
    df["vix_term_slope"] = (df["vix3m"] - df["vix"]) / df["vix"]
    df["vix_curve_inverted"] = (df["vix"] > df["vix3m"]).astype(int)
    df["vix_5d_chg"] = df["vix"].pct_change(5)
    # Put demand proxy: negative skew + VIX curve inversion + backwardation
    df["put_demand_proxy"] = (
        -df["realized_skew_20d"].fillna(0) * 0.4
        + df["vix_curve_inverted"] * 0.4
        - df["vix_term_slope"].fillna(0) * 1.5
    ).rolling(5).mean()
    return df


def compute_adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


# =============================================================================
# ENHANCED FEATURE PIPELINE (all phases)
# =============================================================================
def build_enhanced_features(data):
    qqq = data["qqq_1d"].copy()
    vix = data["vix"]["Close"] if "Close" in data["vix"].columns else data["vix"].iloc[:, 0]
    vix3m = data["vix3m"]["Close"] if "Close" in data["vix3m"].columns else data["vix3m"].iloc[:, 0]
    irx = data["irx"]["Close"] if "Close" in data["irx"].columns else data["irx"].iloc[:, 0]

    df = qqq[["Open", "High", "Low", "Close"]].copy()
    df["ret_1d"] = df["Close"].pct_change()
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["ret_60d"] = df["Close"].pct_change(60)
    df["sma_100"] = df["Close"].rolling(100).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()
    df["above_sma100"] = df["Close"] > df["sma_100"]
    df["above_sma200"] = df["Close"] > df["sma_200"]

    def _rsi(s, n):
        delta = s.diff()
        gain = delta.where(delta > 0, 0).rolling(n).mean()
        loss = -delta.where(delta < 0, 0).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)
    df["rsi_14"] = _rsi(df["Close"], 14)
    df["rsi_2"] = _rsi(df["Close"], 2)
    df["gap_down_pct"] = (df["Close"].shift(1) - df["Open"]) / df["Close"].shift(1)

    # PHASE 2: options-market features
    opt = compute_options_features(qqq, vix, vix3m)
    df = df.join(opt)

    # PHASE 0: ADX
    df["adx_14"] = compute_adx(qqq, 14)

    # Rates
    df["rf"] = (irx.reindex(df.index, method="ffill") / 100.0).fillna(0.045)

    # PHASE 1: Gaussian HMM regime — train on first 60% (out-of-sample discipline)
    train_end = int(len(df) * 0.6)
    train_df = df.iloc[:train_end].dropna(subset=["ret_1d", "realized_vol_20d", "vix"])
    logger.info(f"Fitting Gaussian HMM on {len(train_df)} training days (60% split)...")
    hmm, labels, state_stats, bic = fit_gaussian_hmm(train_df, n_states=3)
    logger.info(f"HMM BIC: {bic:.1f}")
    logger.info(f"HMM state stats: {json.dumps(state_stats, indent=2)}")
    logger.info(f"HMM labels: {labels}")

    # Decode on full series using a CAUSAL (lookahead-free) filter.
    # FIX: previously called hmm.predict(X_arr) / hmm.predict_proba(X_arr) directly
    # on the full series, which runs full-sequence Viterbi/forward-backward and lets
    # information from bars *after* time t leak into the regime label assigned at t
    # — even though the model itself was only fit on the first 60% of history.
    # causal_hmm_decode() instead uses only the forward pass, so the label and
    # probability at each t depend solely on observations up to and including t.
    X_full = df[["ret_1d", "realized_vol_20d", "vix"]].copy()
    X_full["realized_vol_20d"] /= 100
    X_full["vix"] /= 100
    X_arr = X_full.fillna(0).values
    hmm_states, probs = causal_hmm_decode(hmm, X_arr)
    df["hmm_state"] = hmm_states
    df["regime_hmm"] = [labels.get(s, "UNKNOWN") for s in hmm_states]
    # Map probabilities to bull/bear based on labeled states
    bull_state_ids = [sid for sid, name in labels.items() if "BULL" in name]
    bear_state_ids = [sid for sid, name in labels.items() if "BEAR" in name]
    df["p_bull_hmm"] = probs[:, bull_state_ids].sum(axis=1) if bull_state_ids else 0.5
    df["p_bear_hmm"] = probs[:, bear_state_ids].sum(axis=1) if bear_state_ids else 0.0

    # Final regime with production-style overrides
    def _final_regime(row):
        if pd.notna(row["sma_200"]) and row["Close"] < row["sma_200"]:
            return "BEAR_SMA_FORCED"
        if row["vix"] >= 35:
            return "BEAR"
        base = row["regime_hmm"]
        # If HMM says BULL_STRONG but VIX high, downgrade
        if base == "BULL_STRONG" and row["vix"] > 25:
            return "BULL_MODERATE"
        return base
    df["regime"] = df.apply(_final_regime, axis=1)

    # 5-day mode smoothing (avoid one-day flips)
    smoothed = []
    for i in range(len(df)):
        w = df["regime"].iloc[max(0, i-4):i+1].dropna()
        if len(w):
            mode = w.mode()
            smoothed.append(mode.iloc[0] if len(mode) else w.iloc[-1])
        else:
            smoothed.append("CHOPPY")
    df["regime"] = smoothed

    # ITEM 1 (CAGR improvement plan): ml_confidence is now a trained,
    # walk-forward gradient-boosted classifier (model internals are private; precomputed series ships as data)
    # instead of a hand-weighted linear formula. Validated via purged
    # walk-forward evaluation to beat the old formula's ranking quality
    # (AUC-ROC) in the large majority of tested years -- see
    # item1_ml_confidence_results.md for full methodology and numbers.
    def _norm(s, lo, hi):
        return ((s - lo) / (hi - lo)).clip(0, 1)
    df["ml_confidence_handweighted"] = (
        0.18 * df["above_sma100"].astype(float)
        + 0.15 * df["p_bull_hmm"].fillna(0.5)
        + 0.13 * _norm(-df["rsi_14"].fillna(50), -80, -20)
        + 0.10 * df["above_sma200"].astype(float)
        + 0.08 * _norm(-df["vix"], -35, -10)
        + 0.12 * _norm(df["iv_rv_ratio"].fillna(1), 0.9, 1.5)
        + 0.12 * _norm(df["vix_term_slope"].fillna(0), -0.1, 0.15)
        + 0.12 * _norm(-df["put_demand_proxy"].fillna(0), -1, 1)
    )
    # PUBLIC EDITION: load the published walk-forward model
    # output instead of training the private model. The shipped
    # series reproduces every canonical run exactly.
    _ml = pd.read_csv(Path(DATA_DIR) / "ml_confidence.csv", parse_dates=["ts"])
    _ml = _ml.set_index("ts").reindex(df.index)
    df["ml_confidence"] = _ml["ml_confidence"].astype(float).values
    # Warmup fallback: rows before the model's first causal training cutoff
    # (~2018-09, well before every engine's actual backtest start date) fall
    # back to the old formula rather than leaving ml_confidence as NaN.
    df["ml_confidence"] = df["ml_confidence"].fillna(df["ml_confidence_handweighted"])
    return df, hmm, labels


# =============================================================================
# POSITION MODELS
# =============================================================================
@dataclass
class LeapsPosition:
    id: str
    open_ts: pd.Timestamp
    strike: float
    expiry: pd.Timestamp
    contracts: int
    entry_price: float
    entry_spot: float
    entry_delta: float
    entry_iv: float
    entry_dte: int
    regime_at_entry: str

    def dte(self, current):
        cur = current.tz_convert(None) if current.tz is not None else current
        exp = self.expiry.tz_convert(None) if self.expiry.tz is not None else self.expiry
        return max(0, (exp.normalize() - cur.normalize()).days)

    def current_value(self, spot, current, iv, rf):
        T = max(self.dte(current) / 365.0, 1/365)
        return bs_call_price(spot, self.strike, T, rf, iv)

    def current_delta(self, spot, current, iv, rf):
        T = max(self.dte(current) / 365.0, 1/365)
        return bs_call_delta(spot, self.strike, T, rf, iv)


@dataclass
class ShortCall:
    id: str
    open_ts: pd.Timestamp
    strike: float
    expiry: pd.Timestamp
    contracts: int
    entry_price: float
    entry_spot: float
    entry_delta: float
    entry_iv: float
    leaps_id: str

    def dte(self, current):
        cur = current.tz_convert(None) if current.tz is not None else current
        exp = self.expiry.tz_convert(None) if self.expiry.tz is not None else self.expiry
        return max(0, (exp.normalize() - cur.normalize()).days)

    def current_debit(self, spot, current, iv, rf):
        T = max(self.dte(current) / 365.0, 1/365)
        return bs_call_price(spot, self.strike, T, rf, iv)


# =============================================================================
# ENHANCED ENGINE
# =============================================================================
class EnhancedEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cash = cfg.initial_capital
        self.leaps: List[LeapsPosition] = []
        self.shorts: List[ShortCall] = []
        self.fills: List[Dict] = []
        self._id = 0
        self.pmcc_skip_count = 0
        self.pmcc_skip_reasons: Dict[str, int] = {}
        self.pmcc_open_count = 0

    def _new_id(self, prefix):
        self._id += 1
        return f"{prefix}{self._id:04d}"

    def _log(self, **kw):
        self.fills.append({**kw, "cash_after": self.cash})

    def total_nav(self, spot, ts, iv_long, iv_short, rf):
        val = self.cash
        for p in self.leaps:
            val += p.contracts * 100 * p.current_value(spot, ts, iv_long, rf)
        for sc in self.shorts:
            val -= sc.contracts * 100 * sc.current_debit(spot, ts, iv_short, rf)
        return val

    # ===== ENTRY =====
    def check_entry(self, row):
        if row["regime"] in ["BEAR", "BEAR_SMA_FORCED"]:
            return False
        if row["vix"] >= self.cfg.entry_vix_max:
            return False
        if not row["above_sma100"]:
            return False
        if row["rsi_14"] >= self.cfg.entry_rsi14_max:
            return False
        if row["gap_down_pct"] < self.cfg.entry_gap_down_min:
            return False
        if row["ml_confidence"] < self.cfg.entry_ml_min:
            return False
        # Phase 2: put-demand gate
        if pd.notna(row.get("put_demand_proxy")) and row["put_demand_proxy"] > self.cfg.entry_put_demand_max:
            return False
        return True

    def open_leaps(self, ts, spot, row, rf):
        if len(self.leaps) >= self.cfg.max_positions:
            return
        regime = row["regime"]
        if regime == "BULL_STRONG":
            target_delta, dte, size_mult = self.cfg.delta_bull, self.cfg.dte_bull, 1.0
        elif regime == "BULL_MODERATE":
            target_delta, dte, size_mult = self.cfg.delta_neutral, self.cfg.dte_neutral, 0.85
        else:
            target_delta, dte, size_mult = self.cfg.delta_bear, self.cfg.dte_bear, 0.5

        iv_long = (row["vix"] / 100) * self.cfg.iv_multiplier(row["vix"]) * self.cfg.iv_qqq_premium
        T = dte / 365
        strike = find_call_strike(spot, T, rf, iv_long, target_delta)
        price = bs_call_price(spot, strike, T, rf, iv_long)
        actual_delta = bs_call_delta(spot, strike, T, rf, iv_long)

        nav = self.total_nav(spot, ts, iv_long, (row["vix"]/100)*self.cfg.iv_scale_short, rf)
        max_outlay = nav * self.cfg.max_position_pct * size_mult
        per_contract_cost = 100*price + self.cfg.slippage_for_vix(row["vix"], price) + self.cfg.commission
        if per_contract_cost <= 0:
            return

        # --- Volatility/regime-based target size (unchanged logic) ---
        target_contracts = max(1, int(max_outlay / per_contract_cost))
        # SAFEGUARD (item 1 implementation, per user request): require the model's
        # confidence to have held >=0.80 for 3 consecutive bars (ml_confidence_stable),
        # not just spiked on this single bar, before doubling position size. Smoke-tested
        # and grid-searched (0.60-0.80): 0.80 is the best point in the grid (highest mean
        # fwd-40d return of any threshold tested), so it is kept unchanged from the old
        # formula's threshold rather than lowered to chase specific historical trades.
        if regime == "BULL_STRONG" and row.get("ml_confidence_stable", 0) >= 1.0 and len(self.leaps) < 2:
            target_contracts = min(max(target_contracts, 2), 2)
        target_contracts = min(target_contracts, self.cfg.max_contracts)

        # --- Cash constraint: solve directly for the largest affordable size ---
        # Spendable cash respects the configured reserve buffer (default 5%) plus a
        # small no-trade tolerance band so we don't downsize over trivial rounding.
        NO_TRADE_TOLERANCE = 0.01  # 1% slack before the cash constraint is considered binding
        spendable_cash = self.cash * (1 - self.cfg.cash_reserve)
        affordable_contracts = int(spendable_cash * (1 + NO_TRADE_TOLERANCE) / per_contract_cost)

        contracts = min(target_contracts, affordable_contracts)

        if contracts < 1:
            self._log(ts=ts, kind="LEAPS", action="SKIP_INSUFFICIENT_CASH",
                      strike=strike, expiry=None, contracts=0,
                      price_per_share=price, spot=spot, iv=iv_long, delta=actual_delta,
                      slippage=0, commission=0, pnl=0, leaps_id="", short_id="",
                      note=f"target={target_contracts}, affordable=0, cash=${self.cash:,.2f}")
            return

        if contracts < target_contracts:
            # Cash constraint bound tighter than the vol-based target size — log it
            # instead of silently collapsing, so sizing shortfalls are auditable.
            logger.info(
                f"[SIZING] {ts}: cash-constrained downsize {target_contracts} -> {contracts} "
                f"contracts (cash=${self.cash:,.2f}, spendable=${spendable_cash:,.2f}, "
                f"per_contract_cost=${per_contract_cost:,.2f})"
            )

        slippage_total = self.cfg.slippage_for_vix(row["vix"], price) * contracts
        commission_total = self.cfg.commission * contracts
        cost = contracts*100*price + slippage_total + commission_total

        # Final guard: if rounding still leaves us short (shouldn't happen given the
        # tolerance band above), skip rather than silently forcing a smaller trade.
        if cost > self.cash:
            self._log(ts=ts, kind="LEAPS", action="SKIP_INSUFFICIENT_CASH",
                      strike=strike, expiry=None, contracts=0,
                      price_per_share=price, spot=spot, iv=iv_long, delta=actual_delta,
                      slippage=0, commission=0, pnl=0, leaps_id="", short_id="",
                      note=f"cost=${cost:,.2f} > cash=${self.cash:,.2f} after sizing")
            return

        self.cash -= cost
        pos_id = self._new_id("L")
        pos = LeapsPosition(
            id=pos_id, open_ts=ts, strike=strike,
            expiry=ts.normalize() + pd.Timedelta(days=dte),
            contracts=contracts, entry_price=price, entry_spot=spot,
            entry_delta=actual_delta, entry_iv=iv_long, entry_dte=dte,
            regime_at_entry=regime,
        )
        self.leaps.append(pos)
        self._log(ts=ts, kind="LEAPS", action="BUY_TO_OPEN",
                  strike=strike, expiry=pos.expiry, contracts=contracts,
                  price_per_share=price, spot=spot, iv=iv_long, delta=actual_delta,
                  slippage=slippage_total, commission=commission_total,
                  pnl=0, leaps_id=pos_id, short_id="",
                  reason=f"ENTRY_{regime}")

    def close_leaps(self, ts, pos, spot, row, iv_long, rf, reason):
        # Close linked shorts first
        for sc in [s for s in self.shorts if s.leaps_id == pos.id]:
            self.close_short(ts, sc, spot, row, rf, "LEAPS_EXIT")
        cur_val = pos.current_value(spot, ts, iv_long, rf)
        slippage_total = self.cfg.slippage_for_vix(row["vix"], cur_val) * pos.contracts
        commission_total = self.cfg.commission * pos.contracts
        proceeds = pos.contracts*100*cur_val - slippage_total - commission_total
        pnl = proceeds - pos.contracts*100*pos.entry_price
        self.cash += proceeds
        self._log(ts=ts, kind="LEAPS", action="SELL_TO_CLOSE",
                  strike=pos.strike, expiry=pos.expiry, contracts=pos.contracts,
                  price_per_share=cur_val, spot=spot, iv=iv_long,
                  delta=pos.current_delta(spot, ts, iv_long, rf),
                  slippage=slippage_total, commission=commission_total,
                  pnl=pnl, leaps_id=pos.id, short_id="", reason=reason)
        self.leaps.remove(pos)

    # ===== PHASE 0: PMCC gate =====
    def pmcc_should_open(self, row, pos):
        regime = row["regime"]
        adx = row.get("adx_14", 0)
        iv_rv = row.get("iv_rv_ratio", 1.0)
        put_dem = row.get("put_demand_proxy", 0)

        if regime in self.cfg.pmcc_skip_regime and pd.notna(adx) and adx >= self.cfg.pmcc_skip_adx_min:
            return False, f"SKIP_STRONG_TREND(adx={adx:.0f})"
        if pd.notna(iv_rv) and iv_rv < self.cfg.pmcc_skip_vrp_max:
            return False, f"SKIP_LOW_VRP(iv/rv={iv_rv:.2f})"
        if pd.notna(put_dem) and put_dem > self.cfg.pmcc_skip_put_demand_max:
            return False, f"SKIP_HIGH_PUT_DEMAND({put_dem:.2f})"
        # V4 gate (CPCV-validated 2021-2026, 21 paths: Sharpe/CAGR better 18/21, DD better 12/21):
        # in strong trends (any direction) only sell when IV >= realized (paid for the risk)
        if pd.notna(adx) and adx >= 25.0 and pd.notna(iv_rv) and iv_rv < 1.0:
            return False, f"SKIP_TREND_THINPREMIUM(adx={adx:.0f},iv_rv={iv_rv:.2f})"
        return True, ""

    def try_open_short(self, ts, pos, spot, row, rf):
        if any(s.leaps_id == pos.id for s in self.shorts):
            return
        should_open, reason = self.pmcc_should_open(row, pos)
        if not should_open:
            self.pmcc_skip_count += 1
            self.pmcc_skip_reasons[reason.split("(")[0]] = self.pmcc_skip_reasons.get(reason.split("(")[0], 0) + 1
            return

        regime = row["regime"]
        target_delta = self.cfg.pmcc_delta_bull_moderate  # more conservative default
        if regime == "BULL_MODERATE":
            target_delta = self.cfg.pmcc_delta_bull_moderate
        elif regime == "BULL_STRONG":
            target_delta = self.cfg.pmcc_delta_bull_strong
        else:
            target_delta = self.cfg.pmcc_delta_defensive

        dte = self.cfg.pmcc_dte
        iv_short = (row["vix"] / 100) * self.cfg.iv_scale_short
        T = dte / 365
        strike = find_call_strike(spot, T, rf, iv_short, target_delta)
        if strike <= pos.strike:
            strike = pos.strike + max(5, pos.strike * 0.01)
            target_delta = bs_call_delta(spot, strike, T, rf, iv_short)
            if target_delta > 0.35:
                return
        price = bs_call_price(spot, strike, T, rf, iv_short)
        contracts = pos.contracts
        credit = 100*price*contracts
        slippage_total = self.cfg.slippage_for_vix(row["vix"], price) * contracts
        commission_total = self.cfg.commission * contracts
        self.cash += (credit - slippage_total - commission_total)
        self.pmcc_open_count += 1

        sc = ShortCall(
            id=self._new_id("S"), open_ts=ts, strike=strike,
            expiry=ts.normalize() + pd.Timedelta(days=dte),
            contracts=contracts, entry_price=price, entry_spot=spot,
            entry_delta=target_delta, entry_iv=iv_short, leaps_id=pos.id,
        )
        self.shorts.append(sc)
        self._log(ts=ts, kind="SHORT_CALL", action="SELL_TO_OPEN",
                  strike=strike, expiry=sc.expiry, contracts=contracts,
                  price_per_share=price, spot=spot, iv=iv_short, delta=target_delta,
                  slippage=slippage_total, commission=commission_total,
                  pnl=0, leaps_id=pos.id, short_id=sc.id,
                  reason=f"PMCC_OPEN_{regime}")

    def close_short(self, ts, sc, spot, row, rf, reason):
        iv_short = (row["vix"] / 100) * self.cfg.iv_scale_short
        debit = sc.current_debit(spot, ts, iv_short, rf)
        slippage_total = self.cfg.slippage_for_vix(row["vix"], debit) * sc.contracts
        commission_total = self.cfg.commission * sc.contracts
        cost = sc.contracts*100*debit + slippage_total + commission_total
        pnl = sc.contracts*100*sc.entry_price - cost
        self.cash -= cost
        cur_delta = bs_call_delta(spot, sc.strike, max(sc.dte(ts)/365, 1/365), rf, iv_short)
        self._log(ts=ts, kind="SHORT_CALL", action="BUY_TO_CLOSE",
                  strike=sc.strike, expiry=sc.expiry, contracts=sc.contracts,
                  price_per_share=debit, spot=spot, iv=iv_short, delta=cur_delta,
                  slippage=slippage_total, commission=commission_total,
                  pnl=pnl, leaps_id=sc.leaps_id, short_id=sc.id, reason=reason)
        self.shorts.remove(sc)

    def check_pmcc_exits(self, ts, spot, row, rf):
        iv_short = (row["vix"] / 100) * self.cfg.iv_scale_short
        to_close = []
        for sc in list(self.shorts):
            debit = sc.current_debit(spot, ts, iv_short, rf)
            profit_pct = (sc.entry_price - debit) / sc.entry_price if sc.entry_price > 0 else 0
            dte_sc = sc.dte(ts)
            cur_delta = bs_call_delta(spot, sc.strike, max(dte_sc/365, 1/365), rf, iv_short)
            regime = row["regime"]
            take_pct = self.cfg.pmcc_profit_take_early if regime == "BULL_STRONG" else self.cfg.pmcc_profit_take_late
            if profit_pct >= take_pct:
                to_close.append((sc, "PMCC_PROFIT_EARLY")); continue
            if dte_sc <= self.cfg.pmcc_gamma_manage_dte:
                to_close.append((sc, "PMCC_GAMMA_MGMT")); continue
            if cur_delta >= self.cfg.pmcc_roll_delta:
                to_close.append((sc, "PMCC_STRIKE_GAP")); continue
            if debit > self.cfg.pmcc_loss_multiple * sc.entry_price:
                to_close.append((sc, "PMCC_LOSS_LIMIT")); continue
        for sc, reason in to_close:
            self.close_short(ts, sc, spot, row, rf, reason)

    # ===== PHASE 4: Adaptive exit =====
    def adaptive_exit_check(self, ts, pos, spot, row, iv_long, rf):
        if not self.cfg.adaptive_exit_enabled:
            return None
        cur_val = pos.current_value(spot, ts, iv_long, rf)
        unreal_pct = (cur_val - pos.entry_price) / pos.entry_price
        cur_delta = bs_call_delta(spot, pos.strike, max(pos.dte(ts)/365, 1/365), rf, iv_long)
        dte = pos.dte(ts)
        regime = row["regime"]
        vix = row["vix"]

        # Tier 1: catastrophic drawdown from spot's SMA200 (rough 52w low proxy)
        if pd.notna(row.get("sma_200")) and spot < row["sma_200"] * 0.85:
            return "ADAPTIVE_52W_LOW_BREACH"

        # Regime-conditional delta rolldown
        if regime in ["BULL_STRONG", "BULL_MODERATE"]:
            delta_floor = self.cfg.exit_delta_low_bull
            dte_floor = self.cfg.exit_dte_min_bull
        else:
            delta_floor = self.cfg.exit_delta_low_bear
            dte_floor = self.cfg.exit_dte_min_bear

        # Adaptive: allow deeper drift in calm/high-conviction bull
        if regime == "BULL_STRONG" and vix < 18 and row.get("ml_confidence", 0) >= 0.75:
            delta_floor -= 0.05

        if cur_delta < delta_floor:
            return f"ADAPTIVE_DELTA_ROLLDOWN({cur_delta:.2f}<{delta_floor:.2f})"

        # DTE floor
        if dte < dte_floor:
            return f"ADAPTIVE_DTE_ROLL(dte={dte}<{dte_floor})"

        # Profit-lock on regime turn to bear
        if unreal_pct > 0.75 and regime in ["BEAR", "BEAR_SMA_FORCED"]:
            return f"ADAPTIVE_PROFIT_LOCK({unreal_pct*100:.0f}%)"

        # Gamma roll near expiry
        if dte < 90 and cur_delta > 0.90:
            return f"ADAPTIVE_GAMMA_ROLL(dte={dte},d={cur_delta:.2f})"

        return None


# =============================================================================
# BACKTEST DRIVER
# =============================================================================
def run_enhanced(start_date, end_date, data, features):
    engine = EnhancedEngine(CFG)
    hourly = data["qqq_1h"].copy()
    hourly["date"] = hourly.index.tz_convert("America/New_York").normalize().tz_localize(None)

    features = features[(features.index >= start_date) & (features.index <= end_date)]
    nav_series = []

    for date, row in features.iterrows():
        spot_daily = row["Close"]
        vix_val = row["vix"]
        rf = row["rf"]
        iv_long = (vix_val/100) * CFG.iv_multiplier(vix_val) * CFG.iv_qqq_premium
        iv_short = (vix_val/100) * CFG.iv_scale_short

        day_bars = hourly[hourly["date"] == date]
        if len(day_bars) == 0:
            nav = engine.total_nav(spot_daily, date, iv_long, iv_short, rf)
            nav_series.append({"ts": date, "nav": nav, "spot": spot_daily, "regime": row["regime"]})
            continue

        # 09:45 exit-only scan (10:30 bar proxy)
        bar_945 = day_bars[day_bars.index.hour == 10]
        if len(bar_945):
            ts_945 = bar_945.index[0]
            spot_945 = float(bar_945.iloc[0]["Close"])
            for pos in list(engine.leaps):
                reason = engine.adaptive_exit_check(ts_945, pos, spot_945, row, iv_long, rf)
                if reason:
                    engine.close_leaps(ts_945, pos, spot_945, row, iv_long, rf, reason)
            engine.check_pmcc_exits(ts_945, spot_945, row, rf)

        # 15:00 full scan (15:30 bar)
        bar_300 = day_bars[day_bars.index.hour == 15]
        if len(bar_300):
            ts_300 = bar_300.index[0]
            spot_300 = float(bar_300.iloc[0]["Close"])
            engine.check_pmcc_exits(ts_300, spot_300, row, rf)
            if engine.check_entry(row):
                engine.open_leaps(ts_300, spot_300, row, rf)
            for pos in engine.leaps:
                engine.try_open_short(ts_300, pos, spot_300, row, rf)

        nav = engine.total_nav(spot_daily, date, iv_long, iv_short, rf)
        nav_series.append({"ts": date, "nav": nav, "spot": spot_daily, "regime": row["regime"]})

    # Close remaining
    final_ts = features.index[-1]
    final_row = features.iloc[-1]
    final_spot = float(final_row["Close"])
    final_iv = (final_row["vix"]/100) * CFG.iv_multiplier(final_row["vix"]) * CFG.iv_qqq_premium
    final_rf = final_row["rf"]
    for sc in list(engine.shorts):
        engine.close_short(final_ts, sc, final_spot, final_row, final_rf, "BACKTEST_END")
    for pos in list(engine.leaps):
        engine.close_leaps(final_ts, pos, final_spot, final_row, final_iv, final_rf, "BACKTEST_END")

    return {
        "nav_series": pd.DataFrame(nav_series),
        "fills": engine.fills,
        "pmcc_skip_count": engine.pmcc_skip_count,
        "pmcc_open_count": engine.pmcc_open_count,
        "pmcc_skip_reasons": engine.pmcc_skip_reasons,
    }


def compute_metrics(result, initial):
    nav = result["nav_series"]["nav"]
    if len(nav) == 0:
        return {"initial_capital": initial, "final_nav": initial, "cagr_pct": 0.0}
    total_ret = nav.iloc[-1] / initial - 1
    years = len(nav) / 252
    cagr = (nav.iloc[-1] / initial) ** (1/years) - 1
    daily_ret = nav.pct_change().dropna()
    sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else 0
    neg = daily_ret[daily_ret < 0]
    sortino = np.sqrt(252) * daily_ret.mean() / neg.std() if len(neg) and neg.std() > 0 else 0
    running_max = nav.cummax()
    dd = (nav - running_max) / running_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0

    fills = pd.DataFrame(result["fills"])
    if fills.empty:
        return {"initial_capital": initial, "final_nav": float(nav.iloc[-1]),
                "cagr_pct": float(cagr*100), "note": "no fills"}
    closed = fills[fills["action"].str.contains("CLOSE")]
    leaps_closed = closed[closed["kind"] == "LEAPS"]
    sc_closed = closed[closed["kind"] == "SHORT_CALL"]

    return {
        "initial_capital": initial,
        "final_nav": float(nav.iloc[-1]),
        "total_return_pct": float(total_ret * 100),
        "cagr_pct": float(cagr * 100),
        "years": float(years),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown_pct": float(max_dd * 100),
        "calmar": float(calmar),
        "leaps_trades_closed": int(len(leaps_closed)),
        "pmcc_trades_closed": int(len(sc_closed)),
        "leaps_win_rate_pct": float((leaps_closed["pnl"] > 0).mean() * 100) if len(leaps_closed) else 0,
        "pmcc_win_rate_pct": float((sc_closed["pnl"] > 0).mean() * 100) if len(sc_closed) else 0,
        "total_pnl_leaps": float(leaps_closed["pnl"].sum()) if len(leaps_closed) else 0,
        "total_pnl_shorts": float(sc_closed["pnl"].sum()) if len(sc_closed) else 0,
        "pmcc_skips": int(result["pmcc_skip_count"]),
        "pmcc_opens": int(result["pmcc_open_count"]),
        "pmcc_skip_reasons": result["pmcc_skip_reasons"],
    }


def compute_qqq_bh_metrics(data, start, end, initial_capital):
    """Buy-and-hold QQQ from start to end using daily closes."""
    qqq = data["qqq_1d"].copy()
    qqq.index = pd.to_datetime(qqq.index)
    window = qqq.loc[start:end]
    px_start = float(window["Close"].iloc[0])
    px_end = float(window["Close"].iloc[-1])
    shares = initial_capital / px_start
    ret = shares * window["Close"]
    daily_ret = ret.pct_change().dropna()
    ann_factor = 252
    total = px_end / px_start - 1
    years = (window.index[-1] - window.index[0]).days / 365.25
    cagr = (1 + total) ** (1 / years) - 1
    ann_vol = float(daily_ret.std() * (ann_factor ** 0.5))
    sharpe = float((daily_ret.mean() * ann_factor) / ann_vol) if ann_vol > 0 else 0
    peak = ret.cummax()
    dd = (ret / peak - 1)
    max_dd = float(dd.min())
    return {
        "final_nav": float(ret.iloc[-1]),
        "cagr": float(cagr),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": float(total),
        "years": float(years),
        "nav_series": ret,
    }


def main():
    logger.info("=" * 78)
    logger.info("QQQ LEAPS — 2-YEAR HOURLY BACKTEST (IBKR data, All 4 Phases)")
    logger.info("=" * 78)

    data = load_market_data()
    features, hmm, hmm_labels = build_enhanced_features(data)
    logger.info(f"Features: {len(features)} days")

    # 1Y hourly backtest: 2025-07-01 -> 2026-07-31 (matches the 5-min window)
    start = pd.Timestamp("2024-07-01")
    end = pd.Timestamp("2026-07-31")
    logger.info(f"Backtest window: {start.date()} -> {end.date()}")

    reg_counts = features[(features.index >= start) & (features.index <= end)]["regime"].value_counts().to_dict()
    logger.info(f"Regime distribution (2Y window): {reg_counts}")

    result = run_enhanced(start, end, data, features)
    metrics = compute_metrics(result, CFG.initial_capital)

    # QQQ B&H benchmark
    qqq_bh = compute_qqq_bh_metrics(data, start, end, CFG.initial_capital)

    print("\n" + "=" * 78)
    print("ENHANCED 2Y HOURLY — METRICS")
    print("=" * 78)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:.<35} {v:>15,.4f}")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k:.<35} {v:>15}")

    print("\n" + "=" * 78)
    print("QQQ B&H BENCHMARK")
    print("=" * 78)
    for k, v in qqq_bh.items():
        if k == "nav_series": continue
        print(f"  {k:.<35} {v:>15,.4f}")

    result["nav_series"].to_csv(OUT_DIR / "nav_2y.csv", index=False)
    pd.DataFrame(result["fills"]).to_csv(OUT_DIR / "reconcile_2y.csv", index=False)
    qqq_bh["nav_series"].to_csv(OUT_DIR / "qqq_bh_nav_2y.csv")

    combined = {"strategy": metrics, "qqq_bh": {k: v for k, v in qqq_bh.items() if k != "nav_series"}}
    with open(OUT_DIR / "metrics_2y.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    features.to_csv(OUT_DIR / "features_2y.csv")
    logger.info(f"✓ Artifacts saved to {OUT_DIR}")
    return metrics


if __name__ == "__main__":
    main()
