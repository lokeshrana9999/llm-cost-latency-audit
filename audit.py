#!/usr/bin/env python
"""LLM usage audit: markdown report from a model / token / timing export.

Stdlib only (Python 3.8+). Reads a CSV or JSONL export (one row per model
call), prices it with prices.json and writes a markdown report to stdout or -o.
It needs no prompts and no user content: counts, model names, timings and an
opaque prefix hash.

    python audit.py export.csv [-o report.md] [--prices prices.json]
                    [--view auto|latency|backoffice|both]
                    [--ttft-budget-ms N] [--window-days N] [--label TEXT]
                    [--input-excludes-cached]

Required columns: model, input_tokens, output_tokens.
Optional: timestamp, latency_ms, ttft_ms, cached_input_tokens, route,
task_id, session_id, success, streamed, prefix_hash, prefix_tokens.
input_tokens is assumed to include cached_input_tokens (the usual API usage
shape); pass --input-excludes-cached if your export counts them separately.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

MIN_P95 = 200          # no p95 for groups with fewer calls than this
MIN_TOP5 = 100         # no top-5% task share for fewer tasks than this
AVOID_LOW = 0.25       # assumption A1: share of retried failed-call spend that is avoidable (low end)
CONTEXT_MIN_CALLS = 3  # a session needs this many calls on a route to measure growth

COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "created_at", "ts"],
    "model": ["model", "model_name"],
    "input_tokens": ["input_tokens", "prompt_tokens"],
    "output_tokens": ["output_tokens", "completion_tokens"],
    "cached_input_tokens": ["cached_input_tokens", "cached_tokens", "cache_read_input_tokens"],
    "latency_ms": ["latency_ms", "latency", "duration_ms"],
    "ttft_ms": ["ttft_ms", "ttft", "time_to_first_token_ms"],
    "route": ["route", "endpoint", "feature"],
    "task_id": ["task_id", "job_id", "trace_id", "conversation_id", "session_id"],
    "session_id": ["session_id", "conversation_id", "task_id"],
    "success": ["success", "ok", "succeeded"],
    "streamed": ["streamed", "stream", "is_streamed"],
    "prefix_hash": ["prefix_hash", "prompt_prefix_hash"],
    "prefix_tokens": ["prefix_tokens", "prefix_len_tokens"],
    "synthetic": ["synthetic"],
}
REQUIRED = ["model", "input_tokens", "output_tokens"]


# ---------------------------------------------------------------- helpers
def pct(values, q):
    """Linear-interpolated percentile, q in 0..100. None if empty."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def num(v, nd=0):
    return "n/a" if v is None else format(round(v, nd), ",.%df" % nd)


def ms(v):
    return "n/a" if v is None else "%s ms" % format(int(round(v)), ",")


def p50(values, fmt=ms):
    return fmt(pct(values, 50)) if values else "n/a"


def p95(values, fmt=ms):
    """p95 only for groups with at least MIN_P95 values."""
    if len(values) < MIN_P95:
        return "n/a (n=%d, under %d)" % (len(values), MIN_P95)
    return fmt(pct(values, 95))


def usd(v):
    if v is None:
        return "n/a"
    if abs(v) >= 100:
        return "$%s" % format(int(round(v)), ",")
    if abs(v) >= 1 or v == 0:
        return "$%s" % format(v, ",.2f")
    return "$%.3f" % v if abs(v) >= 0.01 else "$%.4f" % v


def rng(lo, hi):
    a, b = usd(lo), usd(hi)
    return a if a == b else "%s-%s" % (a, b)


def share(n, d, dp=1):
    return "%.*f%%" % (dp, 100.0 * n / d) if d else "n/a"


def mtok(v):
    return "%.2fM" % (v / 1e6)


def to_bool(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "ok", "success"):
        return True
    if s in ("false", "0", "no", "n", "error", "fail", "failed"):
        return False
    return None


def to_num(v):
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_ts(v):
    """Epoch seconds, or None. Accepts ISO text and unix seconds / milliseconds."""
    if v is None or v == "":
        return None
    n = to_num(v)
    if n is not None:
        if n > 1e12:
            n /= 1000.0
        return n if n > 0 else None
    s = str(v).strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        try:
            d = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


# ---------------------------------------------------------------- loading
def read_raw(path):
    """Return (list of row dicts, count of unreadable lines)."""
    raw, bad = [], 0
    if path.lower().endswith((".jsonl", ".ndjson")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(obj, dict):
                    raw.append(obj)
                else:
                    bad += 1
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            raw = list(csv.DictReader(f))
    return raw, bad


class Call(object):
    __slots__ = ("idx", "ts", "model", "route", "inp", "out", "cached", "lat", "ttft", "task",
                 "session", "ok", "phash", "ptoks", "streamed", "provider")


def parse_rows(raw, bad_lines=0, input_excludes_cached=False):
    """Rows -> dict(rows, colmap, n_read, dropped {reason: count}, synthetic)."""
    dropped = defaultdict(int)
    if bad_lines:
        dropped["line is not a JSON object"] += bad_lines
    if not raw:
        sys.exit("error: no rows to read")
    present = set()
    for r in raw[:50]:
        present |= set(r.keys())
    colmap = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in present:
                colmap[canon] = a
                break
    missing = [c for c in REQUIRED if c not in colmap]
    if missing:
        sys.exit("error: missing required column(s): %s (found: %s)"
                 % (", ".join(missing), ", ".join(sorted(present))))
    rows = []
    for r in raw:
        def g(k):
            return r.get(colmap[k]) if k in colmap else None
        model = g("model")
        inp, outp = to_num(g("input_tokens")), to_num(g("output_tokens"))
        if model is None or str(model).strip() == "":
            dropped["no model name"] += 1
            continue
        if inp is None or outp is None:
            dropped["input or output tokens missing or not a number"] += 1
            continue
        if inp < 0 or outp < 0:
            dropped["negative token count"] += 1
            continue
        cached = to_num(g("cached_input_tokens")) or 0.0
        if cached < 0:
            cached = 0.0
        if input_excludes_cached:
            inp += cached
        c = Call()
        c.ts = parse_ts(g("timestamp"))
        c.model = str(model).strip()
        c.inp, c.out, c.cached = inp, outp, min(cached, inp)
        c.lat, c.ttft = to_num(g("latency_ms")), to_num(g("ttft_ms"))
        task, sess = g("task_id"), g("session_id")
        c.task = str(task) if task not in (None, "") else None
        c.session = str(sess) if sess not in (None, "") else None
        c.ok = to_bool(g("success"))
        c.phash = str(g("prefix_hash")) if g("prefix_hash") not in (None, "") else None
        c.ptoks = to_num(g("prefix_tokens"))
        c.streamed = to_bool(g("streamed"))
        rt = g("route")
        if rt not in (None, ""):
            c.route = str(rt)
        elif c.phash:
            c.route = "prefix " + c.phash[:8]
        else:
            c.route = "all calls"
        c.idx = 0
        c.provider = None
        rows.append(c)
    syn = [to_bool(r.get(colmap["synthetic"])) for r in raw] if "synthetic" in colmap else []
    return dict(rows=rows, colmap=colmap, n_read=len(raw), dropped=dict(dropped),
                synthetic=bool(syn) and all(syn))


def load_prices(path):
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    for m in p["models"].values():
        m.setdefault("cached", m["input"])
    p.setdefault("providers", {})
    return p


def provider_terms(prices, provider):
    t = prices["providers"].get(provider or "", {})
    return dict(ttl=float(t.get("cache_ttl_s", 300)), min_tokens=float(t.get("cache_min_tokens", 1024)),
                write_mult=float(t.get("cache_write_multiplier", 1.0)),
                batch=float(t.get("batch_discount", 0.0)))


def next_tier_down(prices, model):
    """Same provider, the highest tier below this model's tier; None if none."""
    m = prices["models"].get(model)
    if not m or m.get("provider") is None or m.get("tier") is None:
        return None
    best = None
    for name, v in prices["models"].items():
        if v.get("provider") == m["provider"] and v.get("tier") is not None and v["tier"] < m["tier"]:
            if best is None or v["tier"] > prices["models"][best]["tier"]:
                best = name
    return best


# ---------------------------------------------------------------- cost model
class St(object):
    """Mutable pricing state of one call: tokens, rates, kept share, price multiplier."""
    __slots__ = ("u", "c", "o", "rin", "rc", "ro", "w", "m", "extra")

    def __init__(self, u, c, o, rin, rc, ro):
        self.u, self.c, self.o, self.rin, self.rc, self.ro = u, c, o, rin, rc, ro
        self.w, self.m, self.extra = 1.0, 1.0, 0.0

    def copy(self):
        n = St(self.u, self.c, self.o, self.rin, self.rc, self.ro)
        n.w, n.m, n.extra = self.w, self.m, self.extra
        return n


def cost(s):
    return s.w * s.m * ((s.u * s.rin + s.c * s.rc + s.o * s.ro) / 1e6 + s.extra)


def split(s):
    """(uncached input $, cached input $, output $) of a call state."""
    f = s.w * s.m
    return (f * s.u * s.rin / 1e6, f * s.c * s.rc / 1e6, f * s.o * s.ro / 1e6)


def total(states):
    return sum(cost(s) for s in states)


class Acc(object):
    def __init__(self):
        self.saving = 0.0
        self.n = 0
        self.spend = 0.0
        self.by_route = defaultdict(float)
        self.lines = []


def touch(acc, calls, i, before, after):
    d = before - after
    acc.saving += d                      # signed: a cache write premium is a negative saving
    acc.by_route[calls[i].route] += d
    if d > 1e-12:
        acc.n += 1
        acc.spend += before


# ---------------------------------------------------------------- context
class Ctx(object):
    pass


def build_ctx(parsed, prices, window_days=None):
    ctx = Ctx()
    ctx.prices = prices
    ctx.colmap = parsed["colmap"]
    ctx.n_read, ctx.dropped, ctx.synthetic = parsed["n_read"], parsed["dropped"], parsed["synthetic"]
    pm = prices["models"]
    calls, unpriced, unpriced_rows = [], defaultdict(int), 0
    for c in parsed["rows"]:
        if c.model in pm:
            c.provider = pm[c.model].get("provider")
            calls.append(c)
        else:
            unpriced[c.model] += 1
            unpriced_rows += 1
    calls.sort(key=lambda c: (c.ts if c.ts is not None else 0.0))
    for i, c in enumerate(calls):
        c.idx = i
    ctx.calls, ctx.unpriced, ctx.unpriced_rows = calls, dict(unpriced), unpriced_rows
    ctx.base = [St(c.inp - c.cached, c.cached, c.out, pm[c.model]["input"], pm[c.model]["cached"],
                   pm[c.model]["output"]) for c in calls]
    ctx.spend = total(ctx.base)
    # window
    stamps = [c.ts for c in calls if c.ts is not None]
    ctx.window_assumed = False
    ctx.no_ts_rows = len(calls) - len(stamps)
    if window_days:
        ctx.days = float(window_days)
    elif len(stamps) >= 2 and max(stamps) > min(stamps):
        ctx.days = max((max(stamps) - min(stamps)) / 86400.0, 1.0 / 24)
    else:
        ctx.days, ctx.window_assumed = 30.0, True
    ctx.k = 30.0 / ctx.days
    ctx.start = min(stamps) if stamps else None
    ctx.end = max(stamps) if stamps else None
    ctx.monthly = ctx.spend * ctx.k
    ctx.has = dict(success=any(c.ok is not None for c in calls), ttft=any(c.ttft is not None for c in calls),
                   prefix=any(c.phash for c in calls), ts=bool(stamps),
                   route="route" in ctx.colmap, task="task_id" in ctx.colmap, session="session_id" in ctx.colmap)
    _tasks(ctx)
    _groups(ctx)
    _candidates(ctx)
    return ctx


def _order(idxs, calls):
    return sorted(idxs, key=lambda i: (calls[i].ts if calls[i].ts is not None else 0.0, i))


def _tasks(ctx):
    calls = ctx.calls
    tasks = defaultdict(list)
    for c in calls:
        if c.task is not None:
            tasks[c.task].append(c.idx)
    ctx.tasks = {t: _order(v, calls) for t, v in tasks.items()}
    ctx.failed = set(c.idx for c in calls if c.ok is False)
    ctx.retried = set()
    ctx.failed_tasks = []
    for t, order in ctx.tasks.items():
        later = set()
        for i in reversed(order):
            if calls[i].ok is False and calls[i].model in later:
                ctx.retried.add(i)
            later.add(calls[i].model)
        if calls[order[-1]].ok is False:
            ctx.failed_tasks.append(t)
    ctx.failed_task_spend = sum(cost(ctx.base[i]) for t in ctx.failed_tasks for i in ctx.tasks[t])
    ctx.failed_spend = sum(cost(ctx.base[i]) for i in ctx.failed)
    ctx.retried_spend = sum(cost(ctx.base[i]) for i in ctx.retried)


def _groups(ctx):
    calls = ctx.calls
    ctx.by_mr = defaultdict(list)
    ctx.by_route = defaultdict(list)
    for c in calls:
        ctx.by_mr[(c.model, c.route)].append(c.idx)
        ctx.by_route[c.route].append(c.idx)


def _candidates(ctx):
    calls, prices = ctx.calls, ctx.prices
    # routing: same-provider next tier down; short calls on groups of >= MIN_P95
    ctx.route_cand, ctx.route_def = {}, {}
    ctx.caps = {}
    for key, idxs in ctx.by_mr.items():
        if len(idxs) < MIN_P95:
            continue
        outs = [calls[i].out for i in idxs]
        ctx.caps[key] = pct(outs, 95)
        target = next_tier_down(prices, key[0])
        if not target:
            continue
        pin, pout = pct([calls[i].inp for i in idxs], 50), pct(outs, 50)
        ctx.route_def[key] = (target, pin, pout, len(idxs))
        for i in idxs:
            if calls[i].inp <= pin and calls[i].out <= pout:
                ctx.route_cand[i] = target
    # trimming: input growth inside a session on one route
    ctx.excess, ctx.excess_sessions = {}, defaultdict(set)
    sess = defaultdict(list)
    for c in calls:
        if c.session is not None:
            sess[(c.session, c.route)].append(c.idx)
    for key, idxs in sess.items():
        if len(idxs) < CONTEXT_MIN_CALLS:
            continue
        order = _order(idxs, calls)
        first = calls[order[0]].inp
        for i in order[1:]:
            ex = calls[i].inp - first
            if ex > 0:
                ctx.excess[i] = ex
                ctx.excess_sessions[calls[i].route].add(key[0])
    # caching: groups by (prefix_hash, provider)
    cg = defaultdict(list)
    for c in calls:
        if c.phash:
            cg[(c.phash, c.provider)].append(c.idx)
    ctx.cache_groups, ctx.ptoks, ctx.ptoks_estimated = {}, {}, set()
    for key, idxs in cg.items():
        order = _order(idxs, calls)
        ctx.cache_groups[key] = order
        floor = min(calls[i].inp for i in order)
        for i in order:
            if calls[i].ptoks is not None:
                ctx.ptoks[i] = calls[i].ptoks
            else:
                ctx.ptoks[i] = floor
                ctx.ptoks_estimated.add(key)
    # batching: calls nobody streams, on routes with no TTFT
    route_ttft = {}
    for r, idxs in ctx.by_route.items():
        route_ttft[r] = sum(1 for i in idxs if calls[i].ttft is not None) / float(len(idxs)) >= 0.5
    ctx.batch_cand = {}
    for c in calls:
        t = provider_terms(prices, c.provider)
        if t["batch"] <= 0:
            continue
        streamed = c.streamed if c.streamed is not None else (c.ttft is not None or route_ttft[c.route])
        if not streamed:
            ctx.batch_cand[c.idx] = t["batch"]
    ctx.route_ttft = route_ttft


# ---------------------------------------------------------------- opportunity steps
def mo(ctx, v):
    return usd(v * ctx.k)


def top_routes(by_route, n=3):
    items = sorted(by_route.items(), key=lambda kv: -kv[1])
    s = ", ".join(r for r, _ in items[:n])
    return s + (" (+%d more)" % (len(items) - n) if len(items) > n else "") if items else "none"


def step_failed(ctx, st, level):
    acc, calls = Acc(), ctx.calls
    targets = ctx.failed if level == "high" else ctx.retried
    for i in targets:
        b = cost(st[i])
        st[i].w *= 0.0 if level == "high" else (1.0 - AVOID_LOW)
        touch(acc, calls, i, b, cost(st[i]))
    d = "%.1f" % ctx.days
    if level == "low":
        acc.lines.append(
            "Low end: %s failed calls were retried (a later call in the same task on the same model) and cost %s in the "
            "window. Assumption A1: a quarter of that is avoidable. %s x %.2f = %s, x 30/%s = %s a month."
            % (num(len(ctx.retried)), usd(ctx.retried_spend), usd(ctx.retried_spend), AVOID_LOW,
               usd(ctx.retried_spend * AVOID_LOW), d, mo(ctx, ctx.retried_spend * AVOID_LOW)))
    else:
        acc.lines.append(
            "High end: all %s failed calls cost %s in the window, x 30/%s = %s a month. Spend on the other calls inside "
            "failed tasks (%s failed tasks cost %s in total, %s of spend) is not counted; the back-office view shows it."
            % (num(len(ctx.failed)), usd(ctx.failed_spend), d, mo(ctx, ctx.failed_spend), num(len(ctx.failed_tasks)),
               usd(ctx.failed_task_spend), share(ctx.failed_task_spend, ctx.spend)))
    return acc


def step_routing(ctx, st, level):
    acc, calls, pm = Acc(), ctx.calls, ctx.prices["models"]
    if level == "low":
        acc.lines.append("Low end is $0: no call is known to be safe to move until the replay below passes.")
        return acc
    groups = {}
    for i, target in ctx.route_cand.items():
        s, tr = st[i], pm[target]
        b = cost(s)
        g = groups.setdefault((calls[i].route, calls[i].model, target), [0, 0.0, 0.0, 0.0, 0.0, 0.0])
        g[0] += 1
        g[1] += s.u
        g[2] += s.c
        g[3] += s.o
        s.rin, s.rc, s.ro = tr["input"], tr["cached"], tr["output"]
        a = cost(s)
        touch(acc, calls, i, b, a)
        g[4] += b
        g[5] += a
    for (route, model, target), g in sorted(groups.items(), key=lambda kv: kv[1][5] - kv[1][4])[:3]:
        _, pin, pout, n = ctx.route_def[(model, route)]
        mo_, tr = pm[model], pm[target]
        acc.lines.append(
            "High end, %s (%s -> %s): %s of %s calls have input <= %s and output <= %s tokens (the medians of this model and "
            "route). They use %s uncached input, %s cached input and %s output tokens: now %s x $%.2f + %s x $%.2f + %s x "
            "$%.2f (tokens in millions) = %s; at %s ($%.2f / $%.2f / $%.2f) = %s. Saving %s in the window, %s a month."
            % (route, model, target, num(g[0]), num(n), num(pin), num(pout), mtok(g[1]), mtok(g[2]), mtok(g[3]),
               mtok(g[1]), mo_["input"], mtok(g[2]), mo_["cached"], mtok(g[3]), mo_["output"], usd(g[4]), target,
               tr["input"], tr["cached"], tr["output"], usd(g[5]), usd(g[4] - g[5]), mo(ctx, g[4] - g[5])))
    return acc


def step_trimming(ctx, st, level):
    acc, calls = Acc(), ctx.calls
    if level == "low":
        acc.lines.append("Low end is $0: whether the older turns can go is only known after the replay below.")
        return acc
    per = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for i, ex in ctx.excess.items():
        s = st[i]
        b = cost(s)
        take = min(s.u, ex)
        s.u -= take
        rest = min(s.c, ex - take)
        s.c -= rest
        a = cost(s)
        touch(acc, calls, i, b, a)
        p = per[calls[i].route]
        p[0] += 1
        p[1] += take + rest
        p[2] += b
        p[3] += a
    for route, p in sorted(per.items(), key=lambda kv: kv[1][3] - kv[1][2])[:3]:
        acc.lines.append(
            "High end, %s: %s calls in %s sessions carry %s tokens above the first call of their session (input of call k "
            "minus input of call 1). Removing them at the input rates of each call: %s -> %s, saving %s in the window, %s a month."
            % (route, num(p[0]), num(len(ctx.excess_sessions[route])), num(p[1]), usd(p[2]), usd(p[3]),
               usd(p[2] - p[3]), mo(ctx, p[2] - p[3])))
    return acc


def step_caching(ctx, st, level):
    acc, calls = Acc(), ctx.calls
    skipped = []
    per = defaultdict(lambda: dict(h=0, w=0, mv=0.0, hs=0.0, pr=0.0, ttl=0, mult=1.0, rin=0.0, rc=0.0, mn=0))
    for key, order in ctx.cache_groups.items():
        t = provider_terms(ctx.prices, key[1])
        prev_ts, changes, stats = None, [], []
        for pos, i in enumerate(order):
            c, s = calls[i], st[i]
            if level == "low":
                hit = prev_ts is not None and c.ts is not None and (c.ts - prev_ts) <= t["ttl"]
            else:
                hit = pos > 0
            prev_ts = c.ts
            pfx = min(ctx.ptoks[i], c.inp)
            if pfx < t["min_tokens"]:
                continue
            delta = min(max(0.0, pfx - s.c), s.u)
            if delta <= 0:
                continue
            changes.append((i, s.u, s.c, s.extra, cost(s)))
            f = s.w * s.m
            if hit:
                s.u -= delta
                s.c += delta
                stats.append((c.route, True, delta, f * delta * (s.rin - s.rc) / 1e6, 0.0, s.rin, s.rc))
            else:
                prem = delta * s.rin * (t["write_mult"] - 1.0) / 1e6
                s.extra += prem
                stats.append((c.route, False, delta, 0.0, f * prem, s.rin, s.rc))
        net = sum(b - cost(st[i]) for i, _, _, _, b in changes)
        if net <= 1e-12:
            for i, u, cc, ex, _ in changes:
                st[i].u, st[i].c, st[i].extra = u, cc, ex
            if changes and level == "low":
                skipped.append("Low end, %s: repeats are too sparse to pay for the cache write premium (%s of its repeats fall "
                               "inside the %s s time-to-live, n=%s calls); not counted."
                               % (calls[order[0]].route, share(sum(1 for q in range(1, len(order)) if calls[order[q]].ts is not None
                                                                    and calls[order[q - 1]].ts is not None
                                                                    and calls[order[q]].ts - calls[order[q - 1]].ts <= t["ttl"]),
                                                               max(1, len(order) - 1)), int(t["ttl"]), num(len(order))))
            continue
        for i, u, cc, ex, b in changes:
            touch(acc, calls, i, b, cost(st[i]))
        for route, hit, delta, hs, pr, rin, rc in stats:
            p = per[route]
            p["ttl"], p["mult"], p["mn"] = t["ttl"], t["write_mult"], t["min_tokens"]
            p["rin"], p["rc"] = rin, rc
            if hit:
                p["h"] += 1
                p["mv"] += delta
                p["hs"] += hs
            else:
                p["w"] += 1
                p["pr"] += pr
    word = ("repeat calls that follow another call on the same prefix within the %s s cache time-to-live" if level == "low"
            else "repeat calls (every call after the first on its prefix, whatever the gap)")
    for route, p in sorted(per.items(), key=lambda kv: -(kv[1]["hs"] - kv[1]["pr"]))[:3]:
        avg = p["mv"] / p["h"] if p["h"] else 0
        acc.lines.append(
            "%s %s: %s %s x %s prefix tokens moved to the cached rate x ($%.2f - $%.2f) / 1M = %s; minus %s cache writes "
            "paying a write premium of x%.2f on the input rate = %s. Net %s in the window, %s a month."
            % ("Low end," if level == "low" else "High end,", route, num(p["h"]),
               word % int(p["ttl"]) if level == "low" else word, num(avg), p["rin"], p["rc"], usd(p["hs"]),
               num(p["w"]), p["mult"], usd(p["pr"]), usd(p["hs"] - p["pr"]), mo(ctx, p["hs"] - p["pr"])))
    acc.lines.extend(skipped)
    if not per:
        acc.lines.append("%s no prefix repeats often enough to pay for the cache write premium, so the figure is $0."
                         % ("Low end:" if level == "low" else "High end:"))
    return acc


def step_caps(ctx, st, level):
    acc, calls = Acc(), ctx.calls
    if level == "low":
        acc.lines.append("Low end is $0: the cap only helps if truncated answers are acceptable.")
        return acc
    per = {}
    for key, cap in ctx.caps.items():
        for i in ctx.by_mr[key]:
            if calls[i].out > cap:
                s = st[i]
                b = cost(s)
                over = max(0.0, s.o - cap)
                s.o -= over
                touch(acc, calls, i, b, cost(s))
                p = per.setdefault(key, [0, 0.0, 0.0])
                p[0] += 1
                p[1] += over
                p[2] += b - cost(s)
    for (model, route), p in sorted(per.items(), key=lambda kv: -kv[1][2])[:3]:
        rate = ctx.prices["models"][model]["output"]
        acc.lines.append(
            "High end, %s on %s: p95 output is %s tokens (n=%s calls). %s calls are above it and emit %s tokens over the cap, "
            "x $%.2f per 1M = %s in the window, %s a month."
            % (model, route, num(ctx.caps[(model, route)]), num(len(ctx.by_mr[(model, route)])), num(p[0]), num(p[1]),
               rate, usd(p[2]), mo(ctx, p[2])))
    return acc


def step_batching(ctx, st, level):
    acc, calls = Acc(), ctx.calls
    if level == "low":
        acc.lines.append("Low end is $0: the route owner has not said the work can wait.")
        return acc
    per = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for i, disc in ctx.batch_cand.items():
        s = st[i]
        b = cost(s)
        s.m *= (1.0 - disc)
        a = cost(s)
        touch(acc, calls, i, b, a)
        p = per[(calls[i].route, disc)]
        p[0] += 1
        p[1] += b
        p[2] += b - a
        p[3] = disc
    for (route, disc), p in sorted(per.items(), key=lambda kv: -kv[1][2])[:3]:
        acc.lines.append(
            "High end, %s: %s calls are not streamed and the route has no TTFT. They cost %s at the current rates; at a "
            "%.0f%% batch discount: %s x %.2f = %s saved in the window, %s a month."
            % (route, num(p[0]), usd(p[1]), 100 * disc, usd(p[1]), disc, usd(p[2]), mo(ctx, p[2])))
    return acc


STEPS = [
    dict(key="failed", fn=step_failed, short="no: read the retry logs", verb="fix the largest cause of failed calls", name="Stop paying for failed calls and retries", label="MEASURED", bucket="A",
         effort="Low to medium", needs=lambda c: None if c.has["success"] else "no success column",
         eval="Read the retry logs: group 20 failed tasks by cause (timeout, rate limit, malformed output, validation) and fix the largest cause.",
         wrong="Failures caused by provider outages or bad user input cannot be removed on your side; the avoidable share is an assumption, not a measurement."),
    dict(key="routing", fn=step_routing, short="yes: replay 100-200 calls", verb="route easy calls to the next tier down", name="Send easy calls to the next tier down", label="HYPOTHESIS", bucket="B",
         effort="Medium", needs=lambda c: None if c.route_cand else "no route has 200+ calls on a model with a cheaper tier at the same provider",
         eval="Replay a fixed, versioned set of 100-200 candidate calls, stratified by route, on the next tier down and grade the outputs with your own pass criteria.",
         wrong="Short calls can still be hard. If the cheaper tier fails your criteria on more than a few percent of the replayed calls, the figure falls to $0 for that route."),
    dict(key="trimming", fn=step_trimming, short="yes: replay 100-200 calls", verb="trim conversation history", name="Trim conversation history", label="HYPOTHESIS", bucket="B",
         effort="Medium", needs=lambda c: None if c.excess else "no session has 3+ calls on one route with growing input",
         eval="Replay a fixed, versioned set of 100-200 long-session calls, stratified by route, with history summarised or windowed and grade the answers with your own pass criteria.",
         wrong="The growth may be retrieved documents or tool output that the answer needs, not old turns. This method cannot separate the two."),
    dict(key="caching", fn=step_caching, short="no: compare billed cached tokens", verb="enable prompt caching", name="Turn on prompt caching", label="MECHANICAL", bucket="A",
         effort="Low", needs=lambda c: None if c.has["prefix"] else "no prefix_hash column",
         eval="Enable caching on one route for one day and compare billed cached tokens and TTFT with the day before; replay a fixed, versioned set of 100-200 calls stratified by route to confirm outputs are unchanged.",
         wrong="Hits only count when the prefix is identical byte for byte. If a timestamp or user name sits inside the prefix, billed cached tokens will come in under this figure."),
    dict(key="caps", fn=step_caps, short="yes: replay 100-200 calls", verb="cap output length", name="Cap output length per model and route", label="HYPOTHESIS", bucket="B",
         effort="Low", needs=lambda c: None if c.caps else "no model and route group has 200+ calls",
         eval="Replay the fixed, versioned set of 100-200 calls above the cap, stratified by route, with the limit set and grade completeness with your own pass criteria.",
         wrong="A cap truncates long answers; for structured output a cut-off reply is a failed call. The value is mostly latency, not dollars."),
    dict(key="batching", fn=step_batching, short="ask the route owner, then replay", verb="batch the work that can wait", name="Move work that can wait to batch pricing", label="HYPOTHESIS", bucket="B",
         effort="Medium", needs=lambda c: None if c.batch_cand else "no non-streamed calls on a provider with a batch discount",
         eval="Ask each route owner whether the work can wait up to 24 hours, then send a fixed, versioned set of 100-200 calls, stratified by route, through the batch API and compare with the live outputs on your own pass criteria.",
         wrong="Only work that can wait for the batch window and has no user waiting qualifies; batch limits and availability differ by model."),
]


def clone(states):
    return [s.copy() for s in states]


def waterfall(ctx, steps):
    """Fixed-order waterfall. Each step runs standalone on the original spend and
    'after' on the spend left by earlier steps. Only the 'after' figures are summed."""
    out = dict((s["key"], {}) for s in steps)
    final = {}
    for level in ("low", "high"):
        seq = clone(ctx.base)
        for s in steps:
            alone = clone(ctx.base)
            out[s["key"]][level] = dict(alone=s["fn"](ctx, alone, level), after=s["fn"](ctx, seq, level))
        final[level] = total(seq)
    return out, final


# ---------------------------------------------------------------- analysis
def analyze(parsed, prices, ttft_budget=None, view="auto", window_days=None):
    ctx = build_ctx(parsed, prices, window_days)
    ctx.budget, ctx.view = ttft_budget, view
    ctx.not_estimated = []
    active = []
    for s in STEPS:
        why = s["needs"](ctx)
        if why:
            ctx.not_estimated.append((s["name"], why))
        else:
            active.append(s)
    ctx.wf, ctx.final = waterfall(ctx, active)
    ctx.rows = []
    for pos, s in enumerate(active):
        r = ctx.wf[s["key"]]
        row = dict(step=s, pos=pos + 1, alone_low=r["low"]["alone"].saving, alone_high=r["high"]["alone"].saving,
                   after_low=r["low"]["after"].saving, after_high=r["high"]["after"].saving,
                   acc_low=r["low"]["alone"], acc_high=r["high"]["alone"],
                   earlier=[a["name"] for a in active[:pos]])
        # The two ends come from two coherent chains (earlier steps at their low-end or high-end
        # assumptions). A later step can be worth less after an earlier high-end step, so the
        # row's range runs from the smaller to the larger of its two after-earlier figures.
        row["lo"] = min(row["after_low"], row["after_high"])
        row["hi"] = max(row["after_low"], row["after_high"])
        row["acc_show"] = row["acc_low"] if (row["acc_low"].n and s["key"] == "caching") else row["acc_high"]
        ctx.rows.append(row)
    ctx.ranked = sorted([r for r in ctx.rows if r["alone_high"] > 1e-9 or r["hi"] > 1e-9],
                        key=lambda r: (-r["lo"], -r["hi"]))
    ctx.no_gain = [r for r in ctx.rows if r not in ctx.ranked]
    ctx.low_total = sum(r["lo"] for r in ctx.rows) * ctx.k
    ctx.high_total = sum(r["hi"] for r in ctx.rows) * ctx.k
    ctx.a_low = sum(r["lo"] for r in ctx.rows if r["step"]["bucket"] == "A") * ctx.k
    ctx.a_high = sum(r["hi"] for r in ctx.rows if r["step"]["bucket"] == "A") * ctx.k
    ctx.b_high = sum(r["hi"] for r in ctx.rows if r["step"]["bucket"] == "B") * ctx.k
    return ctx


# ---------------------------------------------------------------- workload views
def route_kind(ctx, route):
    idxs = ctx.by_route[route]
    calls = ctx.calls
    t = sum(1 for i in idxs if calls[i].ttft is not None) / float(len(idxs))
    s = sum(1 for i in idxs if calls[i].streamed) / float(len(idxs))
    return "latency" if (t >= 0.5 or s >= 0.5) else "backoffice"


def routes_by_spend(ctx):
    sp = dict((r, sum(cost(ctx.base[i]) for i in idxs)) for r, idxs in ctx.by_route.items())
    return sorted(sp, key=lambda r: -sp[r]), sp


def latency_rows(ctx, routes):
    calls = ctx.calls
    out, blocks = [], []
    for r in routes:
        idxs = ctx.by_route[r]
        n = len(idxs)
        tt = [calls[i].ttft for i in idxs if calls[i].ttft is not None]
        over = share(sum(1 for v in tt if v > ctx.budget), len(tt)) + " of calls (n=%s)" % num(len(tt)) if (ctx.budget and tt) else None
        tps = []
        for i in idxs:
            c = calls[i]
            if c.lat is not None and c.out > 0:
                gen = c.lat - (c.ttft if c.ttft is not None else 0.0)
                if gen > 0:
                    tps.append(c.out / (gen / 1000.0))
        sp = sum(cost(ctx.base[i]) for i in idxs)
        sess = set(calls[i].session for i in idxs if calls[i].session)
        out.append([r, num(n), p50(tt), p95(tt), over if over else "n/a", p50(tps, lambda v: "%s tok/s" % num(v)),
                    p95([calls[i].out for i in idxs], lambda v: "%s tok" % num(v)),
                    usd(sp / len(sess)) + " (n=%s sessions)" % num(len(sess)) if sess else "n/a",
                    usd(sp / n) + " (n=%s calls)" % num(n)])
        if n >= MIN_P95:
            blocks.append(_latency_block(ctx, r, idxs))
    return out, blocks


def _latency_block(ctx, route, idxs):
    calls = ctx.calls
    use_ttft = any(calls[i].ttft is not None for i in idxs)
    metric = (lambda i: calls[i].ttft) if use_ttft else (lambda i: calls[i].lat)
    mname = "TTFT" if use_ttft else "latency"
    L = ["", "**%s: %s by call position in session** (position = order of the call within its session)" % (route, mname), ""]
    sess = defaultdict(list)
    for i in idxs:
        if calls[i].session:
            sess[calls[i].session].append(i)
    buckets = [("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 10 ** 9)]
    rows = []
    for name, lo, hi in buckets:
        vals = []
        for order in sess.values():
            for pos, i in enumerate(_order(order, calls), 1):
                if lo <= pos <= hi and metric(i) is not None:
                    vals.append(metric(i))
        if vals:
            rows.append([name, num(len(vals)), p50(vals), p95(vals)])
    L += table(["call position", "calls", "%s p50" % mname, "%s p95" % mname], rows) if rows else ["No session ids on this route."]
    vals = [(calls[i].inp, metric(i)) for i in idxs if metric(i) is not None]
    if vals:
        ins = sorted(v[0] for v in vals)
        t1, t2 = ins[len(ins) // 3], ins[2 * len(ins) // 3]
        rows = []
        for name, lo, hi in [("smallest third", -1, t1), ("middle third", t1, t2), ("largest third", t2, 1e18)]:
            sel = [v[1] for v in vals if lo < v[0] <= hi]
            if sel:
                rows.append([name, "%s to %s input tokens" % (num(min(v[0] for v in vals if lo < v[0] <= hi)), num(max(v[0] for v in vals if lo < v[0] <= hi))),
                             num(len(sel)), p50(sel), p95(sel)])
        L += ["", "**%s: %s by input size**" % (route, mname), ""]
        L += table(["input size", "range", "calls", "%s p50" % mname, "%s p95" % mname], rows)
        ranked = sorted(vals, key=lambda v: -v[1])
        k = max(1, int(len(ranked) * 0.05))
        tail = [i for i in idxs if metric(i) is not None]
        tail.sort(key=lambda i: -metric(i))
        tail = tail[:k]
        med_all_in = pct([calls[i].inp for i in idxs], 50)
        med_all_out = pct([calls[i].out for i in idxs], 50)
        hours_all, hours_tail = defaultdict(int), defaultdict(int)
        for i in idxs:
            if calls[i].ts is not None:
                hours_all[int(calls[i].ts // 3600 % 24)] += 1
        for i in tail:
            if calls[i].ts is not None:
                hours_tail[int(calls[i].ts // 3600 % 24)] += 1
        hl = ""
        if hours_tail and sum(hours_all.values()):
            ta, tt = float(sum(hours_all.values())), float(sum(hours_tail.values()))
            best = sorted([h for h in hours_tail if hours_tail[h] >= 20],
                          key=lambda h: -(hours_tail[h] / tt) / (hours_all[h] / ta))[:3]
            if best:
                hl = " Busiest UTC hours in the tail, as a multiple of each hour's share of all calls: %s." % ", ".join(
                    "%02d:00 at %.1fx (n=%d tail calls)" % (h, (hours_tail[h] / tt) / (hours_all[h] / ta), hours_tail[h]) for h in best)
        L += ["", "**%s: slowest 5%% by %s** (n=%s calls). Median input %s tokens vs %s for all calls on the route; median output %s vs %s.%s"
              % (route, mname, num(len(tail)), num(pct([calls[i].inp for i in tail], 50)), num(med_all_in),
                 num(pct([calls[i].out for i in tail], 50)), num(med_all_out), hl)]
    return L


def back_rows(ctx, routes):
    calls = ctx.calls
    rows, per_succ_vals = [], []
    for r in routes:
        idxs = ctx.by_route[r]
        groups = defaultdict(list)
        for i in idxs:
            groups[calls[i].task if calls[i].task is not None else "call-%d" % i].append(i)
        costs, succ, dur, fail_sp = [], 0, [], 0.0
        sp = sum(cost(ctx.base[i]) for i in idxs)
        judged = 0
        for t, g in groups.items():
            g = _order(g, calls)
            c_t = sum(cost(ctx.base[i]) for i in g)
            costs.append(c_t)
            last = calls[g[-1]]
            if last.ok is not None:
                judged += 1
                if last.ok:
                    succ += 1
                else:
                    fail_sp += c_t
            if calls[g[0]].ts is not None and calls[g[-1]].ts is not None:
                dur.append(calls[g[-1]].ts - calls[g[0]].ts + (calls[g[-1]].lat or 0) / 1000.0)
        nt = len(groups)
        per_att = usd(sp / nt) + " (n=%s tasks)" % num(nt)
        if judged and succ:
            per_succ = usd(sp / succ) + " (n=%s successful tasks)" % num(succ)
            per_succ_vals.append(sp / succ)
            succ_s = "%s of tasks (n=%s)" % (share(succ, judged), num(judged))
        else:
            per_succ, succ_s = "n/a (no success data)", "n/a"
        top5 = "n/a (n=%s tasks, under %d)" % (num(nt), MIN_TOP5)
        if nt >= MIN_TOP5 and sp > 0:
            k = int(math.ceil(0.05 * nt))
            top5 = "%s of route spend (top %s of %s tasks)" % (share(sum(sorted(costs, reverse=True)[:k]), sp), num(k), num(nt))
        elig = sum(cost(ctx.base[i]) for i in idxs if i in ctx.batch_cand)
        rows.append([r, num(nt), succ_s, "%.1f (n=%s calls)" % (len(idxs) / float(nt), num(len(idxs))), per_att, per_succ,
                     (usd(fail_sp) + " (%s of route spend)" % share(fail_sp, sp)) if judged else "n/a",
                     top5, "%s / %s" % (p50(dur, lambda v: "%s s" % num(v)), p95(dur, lambda v: "%s s" % num(v))),
                     share(elig, sp) + " of route spend"])
    return rows, per_succ_vals


# ---------------------------------------------------------------- report
def first_route(by_route):
    items = sorted(by_route.items(), key=lambda kv: -kv[1])
    return items[0][0] if items else "your top route"


def build_views(ctx):
    order, _ = routes_by_spend(ctx)
    lat, back = [], []
    for r in order:
        kind = route_kind(ctx, r)
        if ctx.view == "latency":
            lat.append(r)
        elif ctx.view == "backoffice":
            back.append(r)
        elif ctx.view == "both":
            lat.append(r)
            back.append(r)
        else:
            (lat if kind == "latency" else back).append(r)
    return lat, back


def money_table(ctx):
    calls = ctx.calls
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for c in calls:
        u, ca, o = split(ctx.base[c.idx])
        a = agg[(c.model, c.route)]
        a[0] += 1
        a[1] += u
        a[2] += ca
        a[3] += o
    return agg


def render(ctx):
    L = []
    calls = ctx.calls
    n = len(calls)
    lat_routes, back_routes = build_views(ctx)
    # ---- headline (line 1) and verdict
    mech_pct = "%s-%s" % (share(ctx.a_low, ctx.monthly).rstrip("%"), share(ctx.a_high, ctx.monthly))
    L.append("# %s a month (%s of spend) can go without changing model output, out of %s a month; up to %s more depends on evals"
             % (rng(ctx.a_low, ctx.a_high), mech_pct, usd(ctx.monthly), usd(ctx.b_high)))
    note = []
    if ctx.synthetic:
        note.append("Synthetic data: this export was generated by make_sample.py with a fixed seed; no real system produced these numbers.")
    if getattr(ctx, "label", None):
        note.append(ctx.label)
    if note:
        L.append(" ".join(note))
    L.append("")
    win = ("%s to %s UTC" % (datetime.fromtimestamp(ctx.start, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                             datetime.fromtimestamp(ctx.end, timezone.utc).strftime("%Y-%m-%d %H:%M"))) if ctx.start else "no timestamps"
    L.append("- **Window:** %.1f days (%s), %s calls, %s priced spend; about %s a month if the window is typical (%s x 30 / %.1f days)."
             % (ctx.days, win, num(n), usd(ctx.spend), usd(ctx.monthly), usd(ctx.spend), ctx.days))
    if ctx.days < 7 or ctx.window_assumed:
        L.append("- **Warning:** %s Monthly figures are less reliable."
                 % ("no usable timestamps, so the window is assumed to be 30 days." if ctx.window_assumed
                    else "the window is under 7 days, so weekday and weekend patterns are missing."))
    L.append("- **Savings that do not change model output:** %s a month (%s of monthly spend). **Further, depends on evals:** $0-%s a month "
             "(up to %s of monthly spend), unproven."
             % (rng(ctx.a_low, ctx.a_high), mech_pct, usd(ctx.b_high), share(ctx.b_high, ctx.monthly)))
    tt = [c.ttft for c in calls if c.ttft is not None]
    if tt and len(tt) >= MIN_P95:
        s = "- **Speed:** TTFT p95 is %s across %s calls with a TTFT" % (ms(pct(tt, 95)), num(len(tt)))
        if ctx.budget:
            s += "; %s of those calls are over your %s ms budget" % (share(sum(1 for v in tt if v > ctx.budget), len(tt)), num(ctx.budget))
        L.append(s + ".")
    if ctx.ranked:
        r = ctx.ranked[0]
        L.append("- **Do first:** %s on %s: n=%s calls, %s a month after de-duplication, %s."
                 % (r["step"]["name"].lower(), top_routes(r["acc_show"].by_route), num(r["acc_show"].n),
                    rng(r["lo"] * ctx.k, r["hi"] * ctx.k), r["step"]["label"]))
    else:
        L.append("- **Do first:** no action clears $0 on this data.")
    drop = sum(ctx.dropped.values())
    L.append("- **Data:** %s rows read, %s dropped, %s on unpriced models%s; prices dated %s. The monthly ranges in the action table are "
             "de-duplicated and add up to the figures above."
             % (num(ctx.n_read), num(drop), num(ctx.unpriced_rows),
                " (" + ", ".join(sorted(ctx.unpriced)) + ")" if ctx.unpriced else "", ctx.prices.get("_date", "unknown")))
    L.append("")

    # ---- action table
    L.append("## %s actions are worth %s a month after de-duplication%s"
             % (len(ctx.ranked), rng(ctx.low_total, ctx.high_total),
                "; first: %s" % ctx.ranked[0]["step"]["name"].lower() if ctx.ranked else ""))
    L.append("")
    if ctx.ranked:
        rows = []
        for rank, r in enumerate(ctx.ranked, 1):
            s = r["step"]
            rows.append([rank, s["name"], top_routes(r["acc_show"].by_route),
                         "n=%s calls, %s of spend" % (num(r["acc_show"].n), share(r["acc_show"].spend, ctx.spend)),
                         rng(r["lo"] * ctx.k, r["hi"] * ctx.k), s["label"], s["short"], s["effort"]])
        rows.append(["", "**Total** (sum of the monthly column)", "", "", "**%s**" % rng(ctx.low_total, ctx.high_total), "", "", ""])
        L += table(["rank", "action", "where", "evidence", "monthly range after de-dup", "confidence", "eval needed", "effort"], rows)
    L.append("")
    L.append("Ranked by the low end of the range, then the high end. Does not change model output: caching, failed-call waste. "
             "Depends on evals (floor $0): routing, trimming, output caps, batching.")
    for name, why in ctx.not_estimated:
        L.append("")
        L.append("Not estimated: %s (%s)." % (name.lower(), why))
    for r in ctx.no_gain:
        L.append("")
        L.append("Checked, nothing to save: %s." % r["step"]["name"].lower())
    L.append("")

    # ---- workload view
    heads = []
    ttp = []
    for rname in lat_routes:
        v = [calls[i].ttft for i in ctx.by_route[rname] if calls[i].ttft is not None]
        if len(v) >= MIN_P95:
            ttp.append(pct(v, 95))
    if lat_routes and ttp:
        heads.append("%s interactive route%s with TTFT p95 of %s-%s ms" % (len(lat_routes), "s" if len(lat_routes) != 1 else "", num(min(ttp)), num(max(ttp))))
    elif lat_routes:
        heads.append("%s interactive route%s" % (len(lat_routes), "s" if len(lat_routes) != 1 else ""))
    brows, bvals = back_rows(ctx, back_routes) if back_routes else ([], [])
    if brows:
        h = "%s back-office route%s" % (len(back_routes), "s" if len(back_routes) != 1 else "")
        if bvals:
            h += " at %s per successful task" % rng(min(bvals), max(bvals))
        heads.append(h)
    L.append("## Workload view: %s" % ("; ".join(heads) if heads else "no routes"))
    L.append("")
    L.append("Each route gets the view that fits it (%s). Interactive routes have TTFT or streamed calls; the rest are back-office."
             % ("chosen by --view " + ctx.view if ctx.view != "auto" else "auto-detected"))
    if lat_routes:
        rows, blocks = latency_rows(ctx, lat_routes)
        L += ["", "### Latency view", "", "One call = one turn."
              + ("" if ctx.budget and tt else (" No TTFT budget line: " + ("no --ttft-budget-ms given." if not ctx.budget else "no TTFT column."))), ""]
        L += table(["route", "calls", "TTFT p50", "TTFT p95", "over TTFT budget", "output speed p50", "output length p95", "cost per session", "cost per turn"], rows)
        for b in blocks:
            L += b
    if back_routes:
        L += ["", "### Back-office view", "",
              "Unit = task (calls sharing a task_id; a call with no task id is its own task). Cost per successful task divides all route "
              "spend, failed tasks included, by successful tasks; cost per attempted task divides it by all tasks.", ""]
        L += table(["route", "tasks", "tasks succeeded (proxy)", "calls per task", "cost per attempted task", "cost per successful task",
                    "spend in failed tasks", "costliest 5% of tasks", "task duration p50 / p95", "batch-eligible"], brows)
    L.append("")

    # ---- cards
    L.append("## The %s opportunities in detail: what we saw, what it is worth, how to check, what would prove it wrong" % len(ctx.ranked))
    for rank, r in enumerate(ctx.ranked, 1):
        s = r["step"]
        a = r["acc_high"]
        L += ["", "### %d. %s (%s)" % (rank, s["name"], s["label"]), ""]
        L.append("**What we saw.** %s" % _saw(ctx, r))
        L += ["", "**What it is worth.** %s a month after the earlier waterfall steps (%s), %s. Standalone it would be %s a month; "
              "only the after-earlier range is counted in the totals."
              % (rng(r["lo"] * ctx.k, r["hi"] * ctx.k), ", ".join(x.lower() for x in r["earlier"]) or "none, this step runs first",
                 s["label"], rng(r["alone_low"] * ctx.k, r["alone_high"] * ctx.k)), ""]
        for line in r["acc_low"].lines + a.lines:
            L.append("- " + line)
        L += ["", "**How to check.** %s" % s["eval"], "", "**What would prove it wrong.** %s" % s["wrong"]]
    L.append("")

    # ---- money
    agg = money_table(ctx)
    models = defaultdict(lambda: [0, 0.0])
    routes = defaultdict(float)
    for (m, rt), v in agg.items():
        models[m][0] += v[0]
        models[m][1] += v[1] + v[2] + v[3]
        routes[rt] += v[1] + v[2] + v[3]
    topm = max(models, key=lambda m: models[m][1])
    topr = max(routes, key=lambda x: routes[x])
    L.append("## %s makes %s of calls and %s of spend; %s is the costliest route at %s of spend"
             % (topm, share(models[topm][0], n), share(models[topm][1], ctx.spend), topr, share(routes[topr], ctx.spend)))
    L.append("")
    items = sorted(agg.items(), key=lambda kv: -(kv[1][1] + kv[1][2] + kv[1][3]))
    rows = []
    for (m, rt), v in items[:15]:
        sp = v[1] + v[2] + v[3]
        rows.append([m, rt, num(v[0]), usd(v[1]), usd(v[2]), usd(v[3]), usd(sp), share(sp, ctx.spend) + " of spend"])
    if len(items) > 15:
        rest = items[15:]
        rows.append(["other", "%d more" % len(rest), num(sum(v[0] for _, v in rest)), usd(sum(v[1] for _, v in rest)),
                     usd(sum(v[2] for _, v in rest)), usd(sum(v[3] for _, v in rest)),
                     usd(sum(v[1] + v[2] + v[3] for _, v in rest)), share(sum(v[1] + v[2] + v[3] for _, v in rest), ctx.spend) + " of spend"])
    tu, tc, to = sum(v[1] for v in agg.values()), sum(v[2] for v in agg.values()), sum(v[3] for v in agg.values())
    rows.append(["**Total**", "", num(n), usd(tu), usd(tc), usd(to), usd(tu + tc + to), "100%"])
    L += table(["model", "route", "calls", "uncached input", "cached input", "output", "spend", "share"], rows)
    L.append("")
    if not ctx.has["route"]:
        L.append("The export has no route column; routes are the first 8 characters of prefix_hash (a proxy)."
                 if ctx.has["prefix"] else "The export has no route or prefix_hash column, so all calls are one route.")
        L.append("")

    # ---- method
    L.append("## Method and limits: %.1f days of data, projected to 30 days" % ctx.days)
    L.append("")
    L.append("- **Projection.** Monthly = window spend x 30 / window days (%s x 30 / %.1f). Under 7 days of data the projection is flagged." % (usd(ctx.spend), ctx.days))
    L.append("- **Prices.** `prices.json` is dated %s with status \"%s\". Verify every price field, cache term and batch discount against the "
             "provider pricing pages before acting on a dollar figure." % (ctx.prices.get("_date", "unknown"), ctx.prices.get("_status", "unknown")))
    L.append("- **Tokens.** input_tokens includes cached_input_tokens. Spend = uncached input x input rate + cached input x cached rate + output x output rate.")
    L.append("- **Labels.** MEASURED: the size is read directly from the export. MECHANICAL: follows from provider price rules and observed "
             "timings and does not change model output. HYPOTHESIS: needs proving on your calls with an eval.")
    L.append("- **Ranges.** Each range comes from named assumptions, not a haircut. Low end: failed-call waste = %d%% of retried failed-call spend "
             "(A1); caching = repeats inside the provider cache time-to-live, net of the write premium, per prefix and never below $0; all other "
             "actions $0. High end: failed-call waste = all failed-call spend; caching = every repeat after the first on each prefix; routing = "
             "short calls (input and output at or under the route medians) at the same provider's next tier down; trimming = input above the "
             "first call of the session at the input rate; caps = output above the p95 of its model and route at the output rate; batching = "
             "non-streamed calls on routes with no TTFT at the provider's batch discount." % int(AVOID_LOW * 100))
    L.append("- **Waterfall.** Steps run in a fixed order: failed-task waste, routing, trimming, caching, output caps, batching. Each step is "
             "computed on the spend the earlier steps left. The totals sum only the after-earlier figures. The waterfall runs twice, once with "
             "every step at its low end and once at its high end; a row shows the smaller and larger of its two after-earlier figures, so a "
             "later step can show a lower low end than the chains alone suggest. The two buckets use the same waterfall: the caching figure "
             "assumes routing and trimming were adopted, and is worth more if they are not.")
    if ctx.has["success"]:
        L.append("- **Success.** A task succeeds when its last call succeeded. This is a proxy: a call can succeed technically and still give a wrong answer.")
    else:
        L.append("- **Success.** The export has no success column, so no cost per successful task and no failed-call waste figure.")
    L.append("- **Limits.** No p95 is printed for groups under %d calls; no caching dollars without a prefix_hash column; routing only goes to the same "
             "provider's next tier down; calls on unpriced models are excluded from every figure." % MIN_P95)
    drops = ", ".join("%s: %s" % (k, num(v)) for k, v in sorted(ctx.dropped.items())) or "none"
    L.append("- **Data quality.** %s rows read; %s dropped (%s); %s rows on unpriced models; %s priced rows have no usable timestamp."
             % (num(ctx.n_read), num(drop), drops, num(ctx.unpriced_rows), num(ctx.no_ts_rows)))
    if ctx.ptoks_estimated:
        L.append("- **Prefix size.** No prefix_tokens for some prefixes; their size is the smallest input seen on that prefix, which can overstate it.")
    L.append("")

    # ---- next step
    L.append("## Next step: a pilot on %s in three steps" % (first_route(ctx.ranked[0]["acc_show"].by_route) if ctx.ranked else "your top route"))
    L.append("")
    if ctx.ranked:
        r = ctx.ranked[0]
        have = dict(route=ctx.has["route"], prefix_hash=ctx.has["prefix"], success=ctx.has["success"], task_id=ctx.has["task"])
        miss = [c for c in ("route", "prefix_hash", "success", "task_id") if not have[c]]
        L.append("1. Share an export of 7-30 days with route, prefix_hash, success and task_id columns%s."
                 % ("" if not miss else " (this export lacks: %s)" % ", ".join(miss)))
        L.append("2. %s on %s for one day and compare the billed result with the day before." % (r["step"]["verb"].capitalize(), first_route(r["acc_show"].by_route)))
        ev = [x for x in ctx.ranked if x["step"]["bucket"] == "B"]
        L.append("3. Replay a fixed, versioned set of 100-200 calls stratified by route%s, graded with your own pass criteria."
                 % (" to test: %s" % ev[0]["step"]["verb"] if ev else ""))
    L.append("")

    # ---- appendix
    L += appendix(ctx)
    return "\n".join(L).rstrip() + "\n"


def _saw(ctx, r):
    key, a = r["step"]["key"], r["acc_show"]
    calls = ctx.calls
    n = len(calls)
    base = "It touches n=%s calls, %s of window spend (%s), on %s." % (num(a.n), share(a.spend, ctx.spend), usd(a.spend), top_routes(a.by_route))
    if key == "failed":
        return ("%s of %s calls failed (%s of calls); %s of the failures were retried. Failed calls cost %s (%s of spend). The %s tasks "
                "that ended in a failed last call cost %s in total (%s of spend)."
                % (num(len(ctx.failed)), num(n), share(len(ctx.failed), n), num(len(ctx.retried)), usd(ctx.failed_spend),
                   share(ctx.failed_spend, ctx.spend), num(len(ctx.failed_tasks)), usd(ctx.failed_task_spend),
                   share(ctx.failed_task_spend, ctx.spend)))
    if key == "caching":
        reps = sum(max(0, len(g) - 1) for g in ctx.cache_groups.values())
        c0 = sum(1 for c in calls if c.cached > 0)
        return ("%s prefixes (by prefix_hash and provider) account for %s repeat calls; %s of calls already carry cached tokens (n=%s calls). %s"
                % (num(len(ctx.cache_groups)), num(reps), share(c0, n), num(n), base))
    if key == "trimming":
        return ("%s calls (%s of calls) sit in sessions whose input grows beyond the first call. %s"
                % (num(len(ctx.excess)), share(len(ctx.excess), n), base))
    if key == "routing":
        return ("%s calls (%s of calls) meet the easy-call rule on %s model-and-route groups with 200+ calls. %s"
                % (num(len(ctx.route_cand)), share(len(ctx.route_cand), n), len(ctx.route_def), base))
    if key == "caps":
        return "%s model-and-route groups have 200+ calls; only calls above their own p95 output are touched. %s" % (len(ctx.caps), base)
    if key == "batching":
        return ("%s calls (%s of calls) are not streamed and sit on routes with no TTFT. %s"
                % (num(len(ctx.batch_cand)), share(len(ctx.batch_cand), n), base))
    return base


def appendix(ctx):
    calls = ctx.calls
    L = ["## Appendix", "", "### Token percentiles by route", ""]
    order, sp = routes_by_spend(ctx)
    rows = []
    for r in order:
        idxs = ctx.by_route[r]
        ins, outs = [calls[i].inp for i in idxs], [calls[i].out for i in idxs]
        rows.append([r, num(len(idxs)), p50(ins, num), p95(ins, num), p50(outs, num), p95(outs, num)])
    L += table(["route", "calls", "input p50", "input p95", "output p50", "output p95"], rows)
    if ctx.has["task"]:
        L += ["", "### Five costliest tasks", ""]
        tc = sorted(((sum(cost(ctx.base[i]) for i in g), t, g) for t, g in ctx.tasks.items()), reverse=True)[:5]
        L += table(["task", "route", "calls", "cost", "share of spend"],
                   [[t, calls[g[0]].route, len(g), usd(c), share(c, ctx.spend) + " of spend"] for c, t, g in tc])
    L += ["", "### Latency per model", ""]
    rows = []
    bym = defaultdict(list)
    for c in calls:
        bym[c.model].append(c)
    for m, cs in sorted(bym.items(), key=lambda kv: -len(kv[1])):
        la = [c.lat for c in cs if c.lat is not None]
        tt = [c.ttft for c in cs if c.ttft is not None]
        rows.append([m, num(len(cs)), p50(la), p95(la), p50(tt), p95(tt)])
    L += table(["model", "calls", "latency p50", "latency p95", "TTFT p50", "TTFT p95"], rows)
    if ctx.cache_groups:
        L += ["", "### Per-prefix table", ""]
        rows = []
        for key, order_ in sorted(ctx.cache_groups.items(), key=lambda kv: -len(kv[1]))[:10]:
            t = provider_terms(ctx.prices, key[1])
            gaps = [calls[b].ts - calls[a].ts for a, b in zip(order_, order_[1:]) if calls[a].ts is not None and calls[b].ts is not None]
            within = share(sum(1 for g in gaps if g <= t["ttl"]), len(gaps)) + " of repeats (n=%s)" % num(len(gaps)) if gaps else "n/a"
            pt = [calls[i].ptoks for i in order_ if calls[i].ptoks is not None]
            rows.append([key[0][:12], calls[order_[0]].route, num(len(order_)), num(pct(pt, 50)) if pt else "n/a (estimated)",
                         share(sum(1 for i in order_ if calls[i].cached > 0), len(order_)) + " of its calls", within])
        L += table(["prefix", "route", "calls", "prefix tokens", "calls with cached tokens", "repeats inside cache TTL"], rows)
    L += ["", "### Candidate-call definitions", "",
          "- Routing candidate: a call on a model and route group with 200+ calls whose provider has a lower tier, with input and output tokens at or under that group's medians.",
          "- Trimming candidate: a call that is not the first of its session on its route, in a session with 3+ calls there; excess = its input tokens minus the first call's.",
          "- Output-cap candidate: a call above the p95 output of its model and route group (200+ calls).",
          "- Batching candidate: a call that is not streamed (streamed column, else no TTFT) on a provider with a batch discount.",
          "- Retry: a failed call followed by a later call in the same task on the same model.",
          "- Cache hit (low end): a call within the provider's cache time-to-live of the previous call on the same prefix_hash and provider, with the prefix at or above the provider's minimum cacheable tokens."]
    return L


# ---------------------------------------------------------------- main
def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="LLM usage audit: markdown report from a call export.")
    ap.add_argument("export")
    ap.add_argument("--prices", default=os.path.join(here, "prices.json"))
    ap.add_argument("-o", "--output")
    ap.add_argument("--label", help="free text printed under the headline")
    ap.add_argument("--input-excludes-cached", action="store_true",
                    help="input_tokens does not include cached_input_tokens")
    ap.add_argument("--view", choices=["auto", "latency", "backoffice", "both"], default="auto")
    ap.add_argument("--ttft-budget-ms", type=float, help="your TTFT budget in ms; omitted lines if not given")
    ap.add_argument("--window-days", type=float, help="override the window length used for the monthly projection")
    args = ap.parse_args(argv)
    raw, bad = read_raw(args.export)
    parsed = parse_rows(raw, bad, args.input_excludes_cached)
    prices = load_prices(args.prices)
    ctx = analyze(parsed, prices, args.ttft_budget_ms, args.view, args.window_days)
    ctx.label = args.label
    text = render(ctx)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote %s (%s calls, %s priced spend)" % (args.output, num(len(ctx.calls)), usd(ctx.spend)))
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
