#!/usr/bin/env python
"""Generate a SYNTHETIC LLM usage export (sample_export.csv).

Every number is invented by a seeded random generator (fixed seed). It imitates
one month of a company running three workloads on the same price list:

  voice-agent       streamed, multi-turn sessions, TTFT, claude-sonnet-4-5
  support-chat      streamed, multi-turn sessions, TTFT, gpt-4o (provider caches part of the prefix)
  invoice-extract   back-office, nightly bursts, non-streamed, retries, claude-sonnet-4-5
  ticket-summarise  back-office, a slow trickle all day, non-streamed, claude-sonnet-4-5

Each route has one consistent prefix_hash, some failures and retries, and
sessions whose input grows turn by turn. It says nothing about any real system.

Usage: python make_sample.py [out.csv]
"""
import csv
import hashlib
import math
import random
import sys
from datetime import datetime, timedelta, timezone

SEED = 24
DAYS = 30
START = datetime(2026, 10, 12, 0, 0, 0, tzinfo=timezone.utc)   # a Monday

COLS = ["timestamp", "route", "model", "input_tokens", "output_tokens", "cached_input_tokens",
        "latency_ms", "ttft_ms", "task_id", "session_id", "success", "streamed",
        "prefix_hash", "prefix_tokens", "synthetic"]


def lognorm(rng, median, sigma, lo=1, hi=None):
    v = int(rng.lognormvariate(math.log(median), sigma))
    v = max(lo, v)
    return min(hi, v) if hi else v


def phash(route):
    return hashlib.sha1((route + "-system-v7").encode()).hexdigest()[:12]


class Sink(object):
    def __init__(self):
        self.rows = []

    def add(self, t, route, model, inp, out, cached, lat, ttft, task, session, ok, streamed, prefix):
        self.rows.append(dict(
            timestamp=t.strftime("%Y-%m-%dT%H:%M:%SZ"), route=route, model=model, input_tokens=inp,
            output_tokens=out, cached_input_tokens=cached, latency_ms=int(lat),
            ttft_ms="" if ttft is None else int(ttft), task_id=task, session_id=session,
            success="true" if ok else "false", streamed="true" if streamed else "false",
            prefix_hash=phash(route), prefix_tokens=prefix, synthetic="true"))


# route -> settings for the two conversational workloads
CONV = {
    "voice-agent": dict(model="claude-sonnet-4-5", prefix=3500, user=45, out=70, out_sigma=0.5, turns=(3, 15),
                        gap=(6, 25), ttft=620, ms_per_tok=16, fail=0.025, cache_p=0.0, rag_p=0.0, per_day=80),
    "support-chat": dict(model="gpt-4o", prefix=3000, user=90, out=190, out_sigma=0.6, turns=(2, 11),
                         gap=(20, 120), ttft=450, ms_per_tok=14, fail=0.03, cache_p=0.6, rag_p=0.12, per_day=60),
}


def conv_session(rng, sink, route, cfg, sid, t0):
    t, history = t0, 0
    for turn in range(1, rng.randint(*cfg["turns"]) + 1):
        user = lognorm(rng, cfg["user"], 0.6)
        rag = rng.randint(4000, 9000) if rng.random() < cfg["rag_p"] else 0
        inp = cfg["prefix"] + history + user + rag
        out = lognorm(rng, cfg["out"], cfg["out_sigma"], lo=3)
        attempt, ok = 0, False
        while True:
            attempt += 1
            ok = rng.random() > (cfg["fail"] if attempt == 1 else 0.08)
            hour = t.hour
            busy = 1.0 + (0.9 if hour in (17, 18) and rng.random() < 0.12 else 0.0)
            ttft = (rng.lognormvariate(math.log(cfg["ttft"]), 0.3) + inp * 0.035) * busy
            if rng.random() < 0.02:
                ttft *= 3.0
            lat = ttft + out * cfg["ms_per_tok"] * rng.uniform(0.8, 1.3)
            cached = 0
            if cfg["cache_p"] and rng.random() < cfg["cache_p"]:
                cached = min(cfg["prefix"], inp) // 128 * 128
            o = out if ok else max(3, int(out * rng.uniform(0.0, 0.5)))
            if not ok:
                lat *= 2.5
            sink.add(t, route, cfg["model"], inp, o, cached, lat, ttft, sid, sid, ok, True, cfg["prefix"])
            if ok or attempt >= 3 or rng.random() < 0.15:
                break
            t += timedelta(seconds=rng.randint(2, 6))
        if not ok:
            return              # the caller gave up: the session ends on a failed call
        history += user + out
        t += timedelta(seconds=rng.randint(*cfg["gap"]))


def invoice_night(rng, sink, day_start, n_jobs, counter):
    t = day_start + timedelta(hours=1, minutes=rng.randint(0, 20))
    for _ in range(n_jobs):
        counter[0] += 1
        job = "inv-%06d" % counter[0]
        inp = 6000 + lognorm(rng, 20000, 0.45, lo=1500, hi=110000)
        out = lognorm(rng, 650, 0.4, lo=60)
        for attempt, p_ok in enumerate((0.93, 0.6, 0.5), 1):
            ok = rng.random() < p_ok
            o = out if ok else int(out * rng.uniform(1.6, 2.6))     # malformed output runs long
            lat = 900 + o * 18 * rng.uniform(0.8, 1.25)
            sink.add(t, "invoice-extract", "claude-sonnet-4-5", inp, o, 0, lat if ok else lat * 1.8, None,
                     job, "", ok, False, 6000)
            if ok:
                break
            t += timedelta(seconds=rng.randint(3, 8))
        t += timedelta(seconds=rng.uniform(2, 7))


def ticket_day(rng, sink, day_start, mean_per_day, counter):
    t = day_start
    gap = 86400.0 / mean_per_day
    while True:
        t += timedelta(seconds=rng.expovariate(1.0 / gap))
        if t >= day_start + timedelta(days=1):
            return
        counter[0] += 1
        tid = "tkt-%06d" % counter[0]
        inp = 1500 + lognorm(rng, 1500, 0.5, lo=200, hi=30000)
        out = lognorm(rng, 240, 0.4, lo=20)
        tt = t
        for attempt in (1, 2):
            ok = rng.random() > (0.02 if attempt == 1 else 0.05)
            sink.add(tt, "ticket-summarise", "claude-sonnet-4-5", inp, out if ok else 5,
                     0, 700 + out * 17 * rng.uniform(0.8, 1.3), None, tid, "", ok, False, 1500)
            if ok:
                break
            tt += timedelta(seconds=rng.randint(5, 20))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "sample_export.csv"
    rng = random.Random(SEED)
    sink = Sink()
    sess_n, inv_n, tkt_n = 0, [0], [0]
    for d in range(DAYS):
        day = START + timedelta(days=d)
        weekend = day.weekday() >= 5
        f = 0.55 if weekend else 1.0
        for route, cfg in CONV.items():
            for _ in range(max(1, int(rng.gauss(cfg["per_day"] * f, cfg["per_day"] * 0.08)))):
                sess_n += 1
                hour = min(23.99, max(0.0, rng.gauss(15, 4)))
                conv_session(rng, sink, route, cfg, "s%06d" % sess_n, day + timedelta(hours=hour))
        invoice_night(rng, sink, day, max(1, int(rng.gauss(650 * (0.5 if weekend else 1.0), 50))), inv_n)
        ticket_day(rng, sink, day, 72 * f, tkt_n)
    sink.rows.sort(key=lambda r: r["timestamp"])
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(sink.rows)
    print("wrote %d SYNTHETIC rows to %s (seed %d)" % (len(sink.rows), out_path, SEED))


if __name__ == "__main__":
    main()
