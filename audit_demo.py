"""
audit_demo.py -- catch look-ahead bias with the free NorthStar Backtest Audit API.

Run:
    pip install -r requirements.txt
    python audit_demo.py

What it does
------------
Part 1 - Signal timing.
    Runs the toy backtest in sample_data/bad_backtest.py twice (leaky and
    fixed) on synthetic prices, then sends each run's *decision log* to
    POST /audit/lookahead. The API flags every decision whose input data is
    timestamped at or after the moment the decision is acted on.

Part 2 - Train/test label overlap (for ML-style backtests).
    Builds a simple train/test split where each sample's label is a 5-day
    forward return. Without purging, the last training labels peek into the
    test period. The same endpoint flags this as `label_overlap`; after
    purging those rows, the check passes.

Important: what the API does and does not do
--------------------------------------------
The API does NOT read or parse your source code. It checks *timestamps you
send it*: for each row, when was the newest input known (feature_time) and
when was it used (event_time), plus label windows for train/test rows. So
the real work on your side is logging honest timestamps from your pipeline;
the API then gives you an independent, rule-based check of those timestamps.
A "passed" result means no violations within the checked scope -- it is not
a guarantee that a backtest is free of every possible leak.

The API is free, needs no key, and is limited to 100 requests per IP per day
and 64 KB per request body. This script uses 4 requests. Docs: https://app.xuanduzhixuan.com/docs/audit

All data is synthetic. Nothing here is investment advice.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data"))
from bad_backtest import make_prices, run_backtest  # noqa: E402

API_URL = "https://app.xuanduzhixuan.com/audit/lookahead"
LABEL_HORIZON = 5  # Part 2: label = return over the next 5 sessions


MAX_BODY_BYTES = 64 * 1024  # the public endpoint rejects request bodies larger than 64 KB

# The API lets you name your own columns (event_time="e" tells it that the
# event timestamp lives under key "e"). Short keys keep the payload small.
SHORT_KEYS = {"event_time": "e", "feature_time": "f", "label_end": "l",
              "split": "s", "train_id": "tr", "test_id": "te"}
TIME_FIELDS = {"event_time", "feature_time", "label_end"}


def compact(rows):
    """Rename keys to short names and turn ISO timestamps into Unix seconds.

    Same information, roughly 3x fewer bytes. The API accepts both ISO-8601
    strings (which MUST include a timezone) and Unix seconds (always UTC).
    """
    out = []
    for row in rows:
        out.append({
            SHORT_KEYS[k]: int(pd.Timestamp(v).timestamp()) if k in TIME_FIELDS else v
            for k, v in row.items()
        })
    return out


def audit_lookahead(rows, **options):
    """POST rows to the audit endpoint and return the parsed JSON response.

    rows    -- list of dicts with keys event_time / feature_time and, for the
               train/test check, label_end / split / train_id / test_id.
               Timestamps: ISO-8601 strings with a timezone.
    options -- optional extras accepted by the API, e.g.
               min_lag=<seconds>  require feature_time <= event_time - min_lag
               embargo=<seconds>  gap required between a test fold and
                                  training samples that come after it

    Row numbers in the response refer to positions in `rows` (zero-based).
    """
    payload = {"rows": compact(rows), **SHORT_KEYS, **options}
    body_bytes = json.dumps(payload).encode()
    if len(body_bytes) > MAX_BODY_BYTES:
        sys.exit(f"Payload is {len(body_bytes)} bytes; the endpoint accepts up to "
                 f"{MAX_BODY_BYTES}. Send fewer rows per request (keep each fold together).")

    resp = requests.post(API_URL, data=body_bytes,
                         headers={"Content-Type": "application/json"}, timeout=30)
    if resp.status_code == 429:
        sys.exit("Daily free quota reached (100 requests per IP per day, UTC). Try again tomorrow.")
    if "application/json" not in resp.headers.get("Content-Type", ""):
        sys.exit(f"Unexpected HTTP {resp.status_code} from the audit API.")
    body = resp.json()
    if not body.get("ok"):
        # 422 responses explain what was wrong with the request (bad timestamp, etc.)
        sys.exit(f"Audit request rejected: {body.get('error')}: {body.get('detail')}")
    return body


# ---------------------------------------------------------------------------
# Part 1: is every signal built only from data that existed at decision time?
# ---------------------------------------------------------------------------
def part1_signal_timing(prices):
    print("=" * 72)
    print("PART 1 - signal timing (feature_time must be earlier than event_time)")
    print("=" * 72)

    for leak in (True, False):
        label = "LEAKY backtest (uses today's close at today's open)" if leak \
            else "FIXED backtest (signal shifted by one bar)"
        stats, decision_log = run_backtest(prices, leak=leak)

        print(f"\n{label}")
        print(f"  backtest says : total return {stats['total_return_pct']}%, "
              f"Sharpe {stats['sharpe_annualized']}, hit rate {stats['hit_rate_pct']}%")

        result = audit_lookahead(decision_log)

        # These rows have no train/test fold info (a rule-based strategy has no
        # folds), so the overall `status` is "not_checked" for the fold part.
        # The field that matters for signal timing is `feature_status`.
        bad = result["violations"]
        print(f"  audit says    : feature_status = {result['feature_status']!r}, "
              f"{len(bad)} of {result['rows']} decisions used data from the future")
        if bad:
            first = decision_log[bad[0]["row"]]
            print(f"  e.g. row {bad[0]['row']}: acted at {first['event_time']} "
                  f"using data stamped {first['feature_time']}")


# ---------------------------------------------------------------------------
# Part 2: do training labels overlap the test period?
# ---------------------------------------------------------------------------
def build_ml_rows(prices, purge: bool):
    """One row per sample, with the metadata the API needs for the fold check.

    event_time   -- when the sample's prediction would be made (open of day t)
    feature_time -- newest input used (close of day t-1)
    label_end    -- when the label is fully known (close of day t+4,
                    because the label is the 5-session forward return)
    split        -- "train" or "test"
    train_id / test_id -- which fold the row belongs to (one fold here)
    """
    n = len(prices)
    idx = list(range(1, n - LABEL_HORIZON + 1))      # need t-1 and t+4 to exist
    cut = int(len(idx) * 0.7)                         # first 70% train, rest test
    train_idx, test_idx = idx[:cut], idx[cut:]

    if purge:
        # Drop training samples whose label window reaches into the test period.
        first_test_event = prices["open_ts"].iloc[test_idx[0]]
        train_idx = [t for t in train_idx
                     if prices["close_ts"].iloc[t + LABEL_HORIZON - 1] < first_test_event]

    rows = []
    for split, members in (("train", train_idx), ("test", test_idx)):
        for t in members:
            rows.append({
                "event_time": prices["open_ts"].iloc[t].isoformat(),
                "feature_time": prices["close_ts"].iloc[t - 1].isoformat(),
                "label_end": prices["close_ts"].iloc[t + LABEL_HORIZON - 1].isoformat(),
                "split": split,
                "train_id": "fold1",
                "test_id": "fold1",
            })
    return rows


def part2_label_overlap(prices):
    print("\n" + "=" * 72)
    print("PART 2 - train/test split with 5-day forward labels (purging)")
    print("=" * 72)

    for purge in (False, True):
        rows = build_ml_rows(prices, purge=purge)
        n_train = sum(r["split"] == "train" for r in rows)
        result = audit_lookahead(rows)

        overlaps = result["purge_embargo_violations"]
        leaking_train_rows = sorted({v["row"] for v in overlaps})
        print(f"\n{'PURGED' if purge else 'NAIVE '} split: {n_train} train rows, "
              f"{len(rows) - n_train} test rows")
        print(f"  audit says    : status = {result['status']!r}, "
              f"purge_status = {result['purge_status']!r}")
        if overlaps:
            print(f"  {len(leaking_train_rows)} training rows have labels that overlap the test "
                  f"period ({len(overlaps)} train/test pairs, reason: {overlaps[0]['reason']!r})")
            r = rows[leaking_train_rows[-1]]
            print(f"  e.g. train row {leaking_train_rows[-1]}: label known only at {r['label_end']}, "
                  f"test starts at {rows[n_train]['event_time']}")


def main():
    prices = make_prices()
    part1_signal_timing(prices)
    part2_label_overlap(prices)
    print("\nAll prices are synthetic. Audit results describe timestamp consistency only;")
    print("they are not investment advice and say nothing about future performance.")


if __name__ == "__main__":
    main()
