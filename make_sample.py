#!/usr/bin/env python
"""Generate a SYNTHETIC LLM usage export (sample_export.csv).

Every number here is invented by a seeded random generator. It imitates the
shape of a chat / voice-agent workload (3 models, multi-turn sessions, a few
long shared system prompts, an occasional large retrieval dump) so that
audit.py can be run by anyone. It says nothing about any real system.

Usage: python make_sample.py [out.csv]
"""
import csv
import hashlib
import math
import random
import sys
from datetime import datetime, timedelta, timezone

SEED = 24
ROWS = 2000

# app -> (model, system prompt tokens, median output tokens, caching enabled,
#         share of sessions, per-model timing: ttft base ms, ms per output token, success rate)
APPS = {
    "voice-agent":   dict(model="claude-haiku-4-5",  prefix=1800, out_med=45,  cache=True,  w=0.25, ttft=380, mspt=9,  ok=0.95),
    "support-chat":  dict(model="claude-sonnet-4-5", prefix=3200, out_med=170, cache=False, w=0.15, ttft=650, mspt=22, ok=0.97),
    "intent-router": dict(model="gpt-4o-mini",       prefix=900,  out_med=12,  cache=False, w=0.60, ttft=260, mspt=6,  ok=0.93),
}


def lognorm(rng, median, sigma):
    return max(1, int(rng.lognormvariate(math.log(median), sigma)))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "sample_export.csv"
    rng = random.Random(SEED)
    start = datetime(2026, 11, 9, 0, 0, 0, tzinfo=timezone.utc)
    names = list(APPS)
    weights = [APPS[a]["w"] for a in names]
    rows = []
    sess_n = 0
    while len(rows) < ROWS + 200:
        sess_n += 1
        app = rng.choices(names, weights)[0]
        cfg = APPS[app]
        sid = "s%05d" % sess_n
        # session start: 48h window, busier in the middle of each day
        day = rng.randint(0, 1)
        hour = min(23.99, max(0.0, rng.gauss(13, 5)))
        t = start + timedelta(days=day, hours=hour)
        turns = 1 if app == "intent-router" else rng.randint(2, 12)
        prefix_hash = hashlib.sha1((app + "-system-v3").encode()).hexdigest()[:12]
        history = 0
        for turn in range(1, turns + 1):
            user_msg = lognorm(rng, 40 if app == "voice-agent" else 90, 0.6)
            rag = rng.randint(4000, 9000) if (app == "support-chat" and rng.random() < 0.08) else 0
            inp = cfg["prefix"] + history + user_msg + rag
            if app == "support-chat" and rng.random() < 0.35:
                outp = max(3, int(rng.gauss(14, 4)))        # short acknowledgements
            else:
                outp = lognorm(rng, cfg["out_med"], 0.55)
            cached = 0
            if cfg["cache"] and turn > 1 and rng.random() < 0.7:
                cached = cfg["prefix"]
            ttft = int(cfg["ttft"] * rng.uniform(0.7, 1.4) + inp * 0.03)
            lat = ttft + int(outp * cfg["mspt"] * rng.uniform(0.8, 1.3))
            if rng.random() < 0.03:
                lat *= 4
                ttft = int(ttft * 2.5)
            ok = rng.random() < cfg["ok"]
            if not ok:
                lat = int(lat * 1.5)
            rows.append(dict(
                timestamp=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                model=cfg["model"],
                input_tokens=inp,
                output_tokens=outp,
                cached_input_tokens=cached,
                latency_ms=lat,
                ttft_ms=ttft,
                session_id=sid,
                success="true" if ok else "false",
                prefix_hash=prefix_hash,
                prefix_tokens=cfg["prefix"],
                synthetic="true",
            ))
            history += user_msg + outp
            t += timedelta(seconds=rng.randint(8, 60))
    rows.sort(key=lambda r: r["timestamp"])
    rows = rows[:ROWS]
    cols = ["timestamp", "model", "input_tokens", "output_tokens", "cached_input_tokens",
            "latency_ms", "ttft_ms", "session_id", "success", "prefix_hash", "prefix_tokens",
            "synthetic"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote %d SYNTHETIC rows to %s (seed %d)" % (len(rows), out_path, SEED))


if __name__ == "__main__":
    main()
