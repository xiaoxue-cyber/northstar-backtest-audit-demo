"""
A deliberately BROKEN backtest, for teaching purposes only.

Everything here runs on SYNTHETIC prices (a seeded random walk). There is no
real ticker, no real market data, and nothing in this file is a trading idea.
The point is to show how a one-line timing bug can make a strategy look
excellent on paper, and how to make that bug visible.

The strategy is a toy trend filter:
    "hold the asset during a session if price is above its 20-day average"

The bug: to decide whether to hold during session t (entering at the OPEN of
day t), the leaky version looks at the CLOSE of day t -- a price that does not
exist yet at 09:30 when the trade would have to be placed.

Besides the P&L, each backtest also writes a *decision log*: one row per
trading decision, recording
    event_time   -- when the decision is acted on (the open of day t)
    feature_time -- timestamp of the newest data point the signal used
That log is what we later send to the audit API. A decision is only honest
if feature_time < event_time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/New_York"
SMA_WINDOW = 20


def make_prices(n_days: int = 500, seed: int = 10) -> pd.DataFrame:
    """Generate synthetic daily open/close prices with exchange-style timestamps.

    Returns one row per business day with columns:
        open, close        -- prices
        open_ts, close_ts  -- timezone-aware timestamps (09:30 / 16:00 New York)
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=n_days)

    # Random walk with no drift: there is genuinely nothing to predict here.
    overnight = rng.normal(0.0, 0.004, n_days)   # close(t-1) -> open(t)
    intraday = rng.normal(0.0, 0.010, n_days)    # open(t)    -> close(t)

    opens, closes = np.empty(n_days), np.empty(n_days)
    prev_close = 100.0
    for i in range(n_days):
        opens[i] = prev_close * (1 + overnight[i])
        closes[i] = opens[i] * (1 + intraday[i])
        prev_close = closes[i]

    return pd.DataFrame(
        {
            "open": opens,
            "close": closes,
            "open_ts": [pd.Timestamp(f"{d.date()} 09:30", tz=TZ) for d in days],
            "close_ts": [pd.Timestamp(f"{d.date()} 16:00", tz=TZ) for d in days],
        },
        index=days,
    )


def run_backtest(prices: pd.DataFrame, leak: bool = True):
    """Run the toy strategy and return (summary_stats, decision_log).

    leak=True  -> signal for session t uses close[t]    (LOOK-AHEAD BUG)
    leak=False -> signal for session t uses close[t-1]  (what was really known)
    """
    sma = prices["close"].rolling(SMA_WINDOW).mean()
    above = prices["close"] > sma            # computed from close[t] and SMA up to t

    if leak:
        # BUG: the signal row t already contains information from 16:00 on day t,
        # but we act on it at 09:30 on day t.
        signal = above
        feature_ts = prices["close_ts"]
    else:
        # FIX: shift by one bar -- at the open of day t we only know close[t-1].
        signal = above.shift(1)
        feature_ts = prices["close_ts"].shift(1)

    # Position held from open(t) to close(t).
    session_ret = prices["close"] / prices["open"] - 1
    valid = sma.notna() & signal.notna()
    if not leak:
        valid &= sma.shift(1).notna()

    position = signal[valid].astype(float)
    strat_ret = position * session_ret[valid]

    equity = (1 + strat_ret).cumprod()
    daily_sd = strat_ret.std(ddof=1)
    stats = {
        "days": int(valid.sum()),
        "total_return_pct": round(float(equity.iloc[-1] - 1) * 100, 1),
        "sharpe_annualized": round(float(strat_ret.mean() / daily_sd * np.sqrt(252)), 2)
        if daily_sd > 0 else 0.0,
        "hit_rate_pct": round(float((strat_ret[position > 0] > 0).mean()) * 100, 1),
    }

    # One audit row per decision: "at event_time I acted on data stamped feature_time".
    decision_log = [
        {
            "event_time": prices.loc[d, "open_ts"].isoformat(),
            "feature_time": feature_ts.loc[d].isoformat(),
        }
        for d in position.index
    ]
    return stats, decision_log


if __name__ == "__main__":
    px = make_prices()
    for leak in (True, False):
        s, _ = run_backtest(px, leak=leak)
        print(("LEAKY " if leak else "FIXED ") + str(s))
