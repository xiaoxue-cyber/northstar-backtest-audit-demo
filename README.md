# Catch look-ahead bias in your backtest before it catches you

A small, runnable demo showing how a one-line timing bug turns a random walk
into a "Sharpe 3" strategy, and how to catch that bug automatically by
auditing the timestamps your backtest actually used.

> Everything here runs on **synthetic** prices. Nothing in this repo is a
> trading idea, a recommendation, or a performance claim.

## The problem: great backtest, disappointing live results

A very common reason a strategy looks great in a backtest and then falls
apart live is **look-ahead bias**: at some point the backtest uses
information that would not have existed yet when the decision was made.

It is rarely obvious. Typical culprits:

- Acting at the open on a signal computed from that day's close
- Forgetting a `.shift(1)` after a rolling indicator
- Fundamentals joined by *period end* date instead of *publication* date
- ML labels (e.g. 5-day forward returns) for training samples that run into
  the test period, so the model has effectively seen part of the test set

Code review catches some of these. But the bug is about **time**, so the most
direct check is on timestamps: for every decision, *when was the newest input
known*, and *when was it acted on*?

## The approach: log timestamps, audit them

1. Have your backtest write a **decision log**: one row per decision with
   - `event_time`: when the decision is acted on
   - `feature_time`: timestamp of the newest data point that went into it
   - for ML train/test splits also `label_end`, `split`, `train_id`, `test_id`
2. Send the log to a rule-based checker that flags every row where
   `feature_time >= event_time`, and every training sample whose label window
   overlaps a test sample.

This demo uses the free
[NorthStar Backtest Audit API](https://app.xuanduzhixuan.com/docs/audit) for
step 2 (no API key, 100 requests per IP per day). The rules are simple
enough that you could reimplement them yourself. The point is to have an
independent check you can't quietly bypass with the same bug.

**What it is not:** the API does not read or execute your code. It only sees
the timestamps you send it. If your log lies, the audit will too. A `passed`
result means "no violations within the checked scope", not "this backtest is
leak-free".

## Quick start

```python
import requests
rows = [{"event_time": "2024-03-01T09:30:00-05:00", "feature_time": "2024-03-01T16:00:00-05:00"}]
print(requests.post("https://app.xuanduzhixuan.com/audit/lookahead", json={"rows": rows}).json()["feature_status"])  # -> 'failed'
```

Timestamps must include a timezone (or be Unix seconds). This one fails
because the decision at 09:30 used data stamped 16:00 the same day.

Run the full demo:

```bash
git clone https://github.com/<your-username>/northstar-backtest-audit-demo.git
cd northstar-backtest-audit-demo
pip install -r requirements.txt
python audit_demo.py
```

Or open `notebook.ipynb` in Google Colab and click *Run all*.

## Example output

This is real output from `python audit_demo.py` against the live API:

```
========================================================================
PART 1 - signal timing (feature_time must be earlier than event_time)
========================================================================

LEAKY backtest (uses today's close at today's open)
  backtest says : total return 82.2%, Sharpe 3.28, hit rate 63.0%
  audit says    : feature_status = 'failed', 481 of 481 decisions used data from the future
  e.g. row 0: acted at 2022-01-28T09:30:00-05:00 using data stamped 2022-01-28T16:00:00-05:00

FIXED backtest (signal shifted by one bar)
  backtest says : total return 0.5%, Sharpe 0.07, hit rate 50.8%
  audit says    : feature_status = 'passed', 0 of 480 decisions used data from the future

========================================================================
PART 2 - train/test split with 5-day forward labels (purging)
========================================================================

NAIVE  split: 346 train rows, 149 test rows
  audit says    : status = 'failed', purge_status = 'failed'
  4 training rows have labels that overlap the test period (10 train/test pairs, reason: 'label_overlap')
  e.g. train row 345: label known only at 2023-05-08T16:00:00-04:00, test starts at 2023-05-03T09:30:00-04:00

PURGED split: 342 train rows, 149 test rows
  audit says    : status = 'passed', purge_status = 'passed'
```

The prices are a driftless random walk, so there is nothing to predict. The
"82% / Sharpe 3.28" comes entirely from the bug. Once the bug is fixed, the
result is what noise should look like.

## What's in the repo

| File | What it does |
|---|---|
| `sample_data/bad_backtest.py` | Synthetic prices plus a toy trend filter with a deliberate look-ahead bug (`leak=True`) and its fix (`leak=False`). Writes a decision log. |
| `audit_demo.py` | Runs both versions, sends the logs to the audit API, then repeats the exercise for an ML-style train/test split with and without purging. Heavily commented. |
| `notebook.ipynb` | Same demo, Colab-ready. |

## Using it on your own backtest

- **Log at the point of decision.** Record `feature_time` as the *availability*
  time of the newest input. For a daily bar that is the bar's close, not its
  date at midnight. For fundamentals it is the filing/publication time.
- **Require a delay if you need one.** Pass `min_lag` (seconds) to require
  `event_time - feature_time >= min_lag`, e.g. to model execution latency.
- **Walk-forward / CV.** Give every row `split` (`train`/`test`), a fold id
  (`train_id`, `test_id`), and `label_end`. Use `embargo` (seconds) when
  training samples come *after* a test fold.
- **Read the right field.** Without fold metadata the overall `status` is
  `not_checked`, because the fold part could not be checked. For pure signal
  timing, read `feature_status`. `violations` and `purge_embargo_violations`
  list zero-based row numbers.
- **Limits:** 100 requests per IP per day (UTC), at most 2000 rows and 64 KB
  per request. `audit_demo.py` shows how to shorten keys and use Unix-second
  timestamps so a few hundred fully annotated rows fit in one request.

Full endpoint reference (also covers evidence-hash, replay-consistency and
sandbox checks): <https://app.xuanduzhixuan.com/docs/audit>

## Disclaimer

Educational code on synthetic data. Audit results describe the consistency
of timestamps you supply. They are not investment advice and say nothing
about the future performance of any strategy.

## License

MIT
