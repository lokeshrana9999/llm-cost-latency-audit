#!/usr/bin/env python
"""LLM usage audit: markdown report from a model / token / timing export.

Stdlib only. Reads a CSV or JSONL export (one row per model call), prices it
with prices.json and writes a markdown report to stdout (or -o FILE).

Needs no prompts and no user content: only counts, model names and timings.

Usage:
    python audit.py export.csv [--prices prices.json] [-o report.md] [--label TEXT]
                    [--input-excludes-cached]

Column names (aliases in COLUMN_ALIASES). Required: model, input_tokens,
output_tokens, latency_ms. Optional: timestamp, cached_input_tokens, ttft_ms,
task_id or session_id, success, prefix_hash, prefix_tokens.

Assumption: input_tokens INCLUDES cached_input_tokens (the usual API usage
shape). If your export counts them separately, pass --input-excludes-cached.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "created_at", "ts"],
    "model": ["model", "model_name"],
    "input_tokens": ["input_tokens", "prompt_tokens"],
    "output_tokens": ["output_tokens", "completion_tokens"],
    "cached_input_tokens": ["cached_input_tokens", "cached_tokens", "cache_read_input_tokens"],
    "latency_ms": ["latency_ms", "latency", "duration_ms"],
    "ttft_ms": ["ttft_ms", "ttft", "time_to_first_token_ms"],
    "task_id": ["task_id", "session_id", "conversation_id", "trace_id"],
    "success": ["success", "ok", "succeeded"],
    "prefix_hash": ["prefix_hash", "prompt_prefix_hash"],
    "prefix_tokens": ["prefix_tokens", "prefix_len_tokens"],
    "synthetic": ["synthetic"],
}
REQUIRED = ["model", "input_tokens", "output_tokens", "latency_ms"]


# ---------------------------------------------------------------- helpers
def pct(values, q):
    """Linear-interpolated percentile, q in 0..100. None if empty."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fmt_ms(v):
    return "n/a" if v is None else "%d" % round(v)


def usd(v):
    if abs(v) >= 1 or v == 0:
        return "$%s" % format(v, ",.2f")
    return "$%.3f" % v if abs(v) >= 0.01 else "$%.4f" % v


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
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_ts(v):
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


# ---------------------------------------------------------------- loading
def load(path, input_excludes_cached):
    raw = []
    if path.lower().endswith((".jsonl", ".ndjson")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            raw = list(csv.DictReader(f))
    if not raw:
        sys.exit("error: no rows in %s" % path)
    present = set(raw[0].keys())
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
        g = lambda k: r.get(colmap[k]) if k in colmap else None
        inp = to_num(g("input_tokens"))
        outp = to_num(g("output_tokens"))
        if inp is None or outp is None or not g("model"):
            continue
        cached = to_num(g("cached_input_tokens")) or 0.0
        if input_excludes_cached:
            inp += cached
        rows.append(dict(
            ts=parse_ts(g("timestamp")), model=str(g("model")), inp=inp, out=outp,
            cached=min(cached, inp), lat=to_num(g("latency_ms")), ttft=to_num(g("ttft_ms")),
            task=(str(g("task_id")) if g("task_id") not in (None, "") else None),
            ok=to_bool(g("success")),
            phash=(str(g("prefix_hash")) if g("prefix_hash") not in (None, "") else None),
            ptoks=to_num(g("prefix_tokens")),
            synthetic=to_bool(g("synthetic")),
        ))
    return rows, colmap


def load_prices(path):
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    models = p["models"]
    for m in models.values():
        m.setdefault("cached", m["input"])
    return p


class Pricer:
    def __init__(self, prices):
        self.p = prices["models"]
        self.unknown = set()

    def known(self, model):
        return model in self.p

    def rates(self, model):
        if model not in self.p:
            self.unknown.add(model)
            return {"input": 0.0, "output": 0.0, "cached": 0.0}
        return self.p[model]

    def split(self, r):
        """(uncached input $, cached input $, output $) for one call."""
        pr = self.rates(r["model"])
        return ((r["inp"] - r["cached"]) * pr["input"] / 1e6,
                r["cached"] * pr["cached"] / 1e6,
                r["out"] * pr["output"] / 1e6)

    def cost(self, r):
        return sum(self.split(r))


# ---------------------------------------------------------------- analysis
def task_stats(rows, pricer):
    """Group calls by task id. Task success = its last call (by time, then
    file order) has success true. Returns None if there is no task column."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["task"] is not None:
            groups[r["task"]].append(((r["ts"].timestamp() if r["ts"] else 0), i, r))
    if not groups:
        return None
    tasks = []
    for tid, items in groups.items():
        items.sort(key=lambda x: (x[0], x[1]))
        calls = [r for _, _, r in items]
        last = calls[-1]
        tasks.append(dict(id=tid, calls=len(calls), cost=sum(pricer.cost(r) for r in calls), ok=last["ok"]))
    return tasks


def opportunities(rows, pricer, total):
    """Each opportunity: dict(title, hypothesis, whatif, arithmetic, saving, test)."""
    ops = []
    P = pricer.rates

    # 1. Prompt caching on repeated long prefixes (needs prefix_hash)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["phash"]:
            groups[(r["model"], r["phash"])].append((i, r))
    if groups:
        rep_tokens = 0.0
        saving = 0.0
        spend_rep = 0.0
        npref = 0
        for (model, h), items in groups.items():
            items.sort(key=lambda x: (x[1]["ts"].timestamp() if x[1]["ts"] else 0, x[0]))
            ptok = [r["ptoks"] for _, r in items if r["ptoks"]]
            plen = max(ptok) if ptok else min(r["inp"] for _, r in items)  # proxy if no prefix_tokens
            pr = P(model)
            npref += 1
            for _, r in items[1:]:  # first sight of a prefix is a cache write, not a hit
                rep = max(0.0, min(plen, r["inp"] - r["cached"]))
                rep_tokens += rep
                spend_rep += rep * pr["input"] / 1e6
                saving += rep * (pr["input"] - pr["cached"]) / 1e6
        if saving > 0:
            ops.append(dict(
                title="Prompt caching for repeated long prefixes",
                hypothesis="Calls that resend the same long prefix (system prompt, tool schema) and are not yet billed at the cached rate might be eligible for it.",
                whatif="WHAT-IF: if cached-input pricing applied to the %s repeated uncached prefix tokens (%.1f%% of all input tokens) "
                       "across %d model+prefix groups, spend would drop by %s (%.1f%% of total) under these prices."
                       % (format(int(rep_tokens), ","), 100 * rep_tokens / max(1, sum(r["inp"] for r in rows)), npref,
                          usd(saving), 100 * saving / total if total else 0),
                saving=saving,
                test="Enable caching on one route; compare billed cached tokens and TTFT; check TTL, minimum prefix size and cache-write fees against the provider docs (not modelled here).",
            ))
    else:
        unc = sum(r["inp"] - r["cached"] for r in rows)
        sv = sum((r["inp"] - r["cached"]) * 0.5 * (P(r["model"])["input"] - P(r["model"])["cached"]) / 1e6 for r in rows)
        if sv > 0:
            ops.append(dict(
                title="Prompt caching (generic, no prefix_hash in export)",
                hypothesis="Some share of input tokens is likely a repeated prefix. Without a prefix_hash column the share is unknown.",
                whatif="WHAT-IF (arbitrary 50%% assumption): if cached-input pricing applied to 50%% of the %s uncached input tokens, "
                       "spend would drop by %s (%.1f%% of total)." % (format(int(unc), ","), usd(sv), 100 * sv / total if total else 0),
                saving=sv,
                test="Add a prefix_hash column to the export to replace the 50% guess with a measured share.",
            ))

    # 2. Context trimming on the heaviest-input calls
    inputs = [r["inp"] for r in rows]
    p90 = pct(inputs, 90)
    heavy = [r for r in rows if r["inp"] > p90]
    if heavy:
        unc = sum(r["inp"] - r["cached"] for r in heavy)
        sv = sum((r["inp"] - r["cached"]) * 0.20 * P(r["model"])["input"] / 1e6 for r in heavy)
        hs = sum(pricer.cost(r) for r in heavy)
        ops.append(dict(
            title="Context trimming on the largest prompts",
            hypothesis="The top 10%% of calls by input size (> %s tokens) carry history or retrieval text that may not all affect the answer." % format(int(p90), ","),
            whatif="WHAT-IF: those %d calls cost %s (%.1f%% of spend). If 20%% of their uncached input tokens (of %s) could be dropped, spend would drop by %s (%.1f%%)."
                   % (len(heavy), usd(hs), 100 * hs / total if total else 0, format(int(unc), ","), usd(sv), 100 * sv / total if total else 0),
            saving=sv,
            test="Summarise or cut history/retrieval on a sample of these calls; compare task success on an eval set before and after.",
        ))

    # 3. Model routing for short-output calls on the most expensive model
    by_model = defaultdict(float)
    for r in rows:
        by_model[r["model"]] += pricer.cost(r)
    known = [m for m in by_model if pricer.known(m)]
    if len(known) >= 2:
        top = max(known, key=lambda m: by_model[m])
        blended = lambda m: P(m)["input"] + P(m)["output"]
        cheaper = [m for m in known if m != top and blended(m) < blended(top)]
        if cheaper:
            alt = min(cheaper, key=blended)
            top_rows = [r for r in rows if r["model"] == top]
            med = pct([r["out"] for r in top_rows], 50)
            cand = [r for r in top_rows if r["out"] <= med * 0.25]
            if cand:
                cur = sum(pricer.cost(r) for r in cand)
                ap = P(alt)
                alt_cost = sum((r["inp"] * ap["input"] + r["out"] * ap["output"]) / 1e6 for r in cand)
                sv = cur - alt_cost
                if sv > 0:
                    ops.append(dict(
                        title="Route short-output calls to a cheaper model",
                        hypothesis="On %s, calls with very short outputs (<= %d tokens, a quarter of that model's median) may be acknowledgements or simple decisions a cheaper model handles equally well."
                                   % (top, med * 0.25),
                        whatif="WHAT-IF: %d such calls cost %s on %s. Priced as %s (no caching), they would cost %s: a ceiling of %s (%.1f%% of spend). "
                               "If only half pass an eval, about %s."
                               % (len(cand), usd(cur), top, alt, usd(alt_cost), usd(sv), 100 * sv / total if total else 0, usd(sv / 2)),
                        saving=sv,
                        test="Replay a sample of these calls on %s offline; grade against %s outputs; route only the classes that pass." % (alt, top),
                    ))

    # 4. Output length cap
    outs = [r["out"] for r in rows]
    p95o = pct(outs, 95)
    sv = sum(max(0.0, r["out"] - p95o) * P(r["model"])["output"] / 1e6 for r in rows)
    excess = sum(max(0.0, r["out"] - p95o) for r in rows)
    if sv > 0:
        ops.append(dict(
            title="Cap or tighten very long outputs",
            hypothesis="Output tokens cost several times input tokens; the longest replies (> p95 = %d tokens) may be padding or runaway generations." % round(p95o),
            whatif="WHAT-IF: if every output above p95 were cut to p95, %s output tokens would not be generated: spend would drop by %s (%.1f%%)."
                   % (format(int(excess), ","), usd(sv), 100 * sv / total if total else 0),
            saving=sv,
            test="Add a length instruction or max_tokens on one route; check truncation rate and answer quality on an eval set.",
        ))

    # 5. Spend on failed calls
    if any(r["ok"] is not None for r in rows):
        failed = [r for r in rows if r["ok"] is False]
        fs = sum(pricer.cost(r) for r in failed)
        if failed and fs > 0:
            ops.append(dict(
                title="Reduce spend on failed calls",
                hypothesis="Failed calls (%d, %.1f%% of calls) still consume tokens. Retries, malformed output or timeouts may share a few fixable causes."
                           % (len(failed), 100 * len(failed) / len(rows)),
                whatif="WHAT-IF: failed calls cost %s (%.1f%% of spend). If a quarter of that were avoidable, spend would drop by %s."
                       % (usd(fs), 100 * fs / total if total else 0, usd(fs / 4)),
                saving=fs / 4,
                test="Break failures down by error type (needs an error column or logs); fix the largest class first; track the failure rate.",
            ))

    # 6. Latency tails (no dollar figure)
    tails = []
    for m in sorted({r["model"] for r in rows}):
        l = [r["lat"] for r in rows if r["model"] == m and r["lat"] is not None]
        if len(l) >= 20 and pct(l, 50):
            ratio = pct(l, 95) / pct(l, 50)
            if ratio >= 2.5:
                tails.append((m, pct(l, 50), pct(l, 95), ratio))
    if tails:
        ops.append(dict(
            title="Latency tails",
            hypothesis="p95 latency far above p50 suggests a slow minority of calls (long outputs, large prompts, provider spikes, retries).",
            whatif="NOT A COST WHAT-IF. " + "; ".join("%s p50 %d ms, p95 %d ms (%.1fx)" % (m, round(a), round(b), c) for m, a, b, c in tails) + ".",
            saving=-1.0,
            test="Split tail calls by input size, output size and time of day to see which factor explains them; then test streaming, shorter outputs or a fallback.",
        ))

    ops.sort(key=lambda o: o["saving"], reverse=True)
    return ops


# ---------------------------------------------------------------- report
def build_report(rows, colmap, pricer, prices, path, label):
    L = []
    total_cost = sum(pricer.cost(r) for r in rows)
    all_synth = all(r["synthetic"] for r in rows) and "synthetic" in colmap

    L.append("# LLM usage audit")
    if label:
        L.append("**%s**" % label)
    elif all_synth:
        L.append("**SYNTHETIC DATA: this report was generated from a made-up sample (make_sample.py), not from any real system. Numbers illustrate the format only.**")
    else:
        L.append("**Source: %s**" % os.path.basename(path))
    L.append("")
    stamps = [r["ts"] for r in rows if r["ts"]]
    span = ""
    if stamps:
        span = ", %s to %s" % (min(stamps).strftime("%Y-%m-%d %H:%M"), max(stamps).strftime("%Y-%m-%d %H:%M"))
    L.append("%s calls%s. Prices: `prices.json` dated %s, status \"%s\"; **verify against provider pricing pages before acting on any dollar figure.**"
             % (format(len(rows), ","), span, prices.get("_date", "?"), prices.get("_status", "unverified")))
    L.append("Total priced spend: **%s**. Input tokens are assumed to include cached input tokens." % usd(total_cost))
    if pricer.unknown:
        L.append("")
        L.append("WARNING: no price for model(s) %s; their calls are priced at $0 and the totals are understated."
                 % ", ".join(sorted(pricer.unknown)))
    L.append("")

    # Spend by model
    L.append("## Spend by model")
    per = defaultdict(list)
    for r in rows:
        per[r["model"]].append(r)
    trows = []
    for m, rs in sorted(per.items(), key=lambda kv: -sum(pricer.cost(r) for r in kv[1])):
        c = sum(pricer.cost(r) for r in rs)
        ti, to, tc = sum(r["inp"] for r in rs), sum(r["out"] for r in rs), sum(r["cached"] for r in rs)
        oks = [r["ok"] for r in rs if r["ok"] is not None]
        trows.append((m, format(len(rs), ","), format(int(ti), ","), format(int(tc), ","), format(int(to), ","),
                      usd(c), "%.1f%%" % (100 * c / total_cost if total_cost else 0),
                      "$%.2f" % (1000 * c / len(rs)),
                      ("%.1f%%" % (100 * sum(oks) / len(oks))) if oks else "n/a"))
    L += table(["model", "calls", "input tok", "cached tok", "output tok", "spend", "share", "$/1k calls", "success"], trows)
    L.append("")

    # Cost per task
    L.append("## Cost per task")
    tasks = task_stats(rows, pricer)
    if tasks is None:
        L.append("No task_id / session_id column: per-task cost skipped.")
    else:
        costs = [t["cost"] for t in tasks]
        L.append("%d tasks (grouped by `%s`), %.1f calls per task on average. Task success = last call of the task succeeded."
                 % (len(tasks), colmap["task_id"], len(rows) / len(tasks)))
        L.append("")
        hasok = any(t["ok"] is not None for t in tasks)
        ok_tasks = [t for t in tasks if t["ok"]]
        trs = [("mean", usd(sum(costs) / len(costs))), ("p50", usd(pct(costs, 50))), ("p95", usd(pct(costs, 95))),
               ("max", usd(max(costs)))]
        L += table(["cost per task", "USD"], trs)
        if hasok:
            L.append("")
            if ok_tasks:
                L.append("Successful tasks: %d of %d (%.1f%%). **Cost per successful task: %s** (all task spend, including failed tasks, divided by successful tasks); cost per task attempted: %s."
                         % (len(ok_tasks), len(tasks), 100 * len(ok_tasks) / len(tasks), usd(sum(costs) / len(ok_tasks)), usd(sum(costs) / len(tasks))))
            else:
                L.append("No task succeeded: cost per successful task is undefined.")
        L.append("")
        top = sorted(tasks, key=lambda t: -t["cost"])[:5]
        L.append("Five costliest tasks: " + ", ".join("`%s` %s (%d calls)" % (t["id"], usd(t["cost"]), t["calls"]) for t in top) + ".")
    L.append("")

    # Latency
    L.append("## Latency by model (ms)")
    lrows = []
    for m, rs in sorted(per.items()):
        lat = [r["lat"] for r in rs if r["lat"] is not None]
        tt = [r["ttft"] for r in rs if r["ttft"] is not None]
        lrows.append((m, fmt_ms(pct(lat, 50)), fmt_ms(pct(lat, 95)), fmt_ms(pct(tt, 50)), fmt_ms(pct(tt, 95))))
    L += table(["model", "latency p50", "latency p95", "TTFT p50", "TTFT p95"], lrows)
    L.append("")

    # Token distribution
    L.append("## Token distribution")
    unc_in = cached_in = outp = 0.0
    for r in rows:
        a, b, c = pricer.split(r)
        unc_in, cached_in, outp = unc_in + a, cached_in + b, outp + c
    sh = lambda v: "%.1f%%" % (100 * v / total_cost if total_cost else 0)
    L += table(["spend component", "USD", "share of spend"],
               [("uncached input", usd(unc_in), sh(unc_in)), ("cached input", usd(cached_in), sh(cached_in)), ("output", usd(outp), sh(outp))])
    L.append("")
    ins = [r["inp"] for r in rows]
    outs = [r["out"] for r in rows]
    L += table(["tokens per call", "p50", "p90", "p99", "max"],
               [("input", int(pct(ins, 50)), int(pct(ins, 90)), int(pct(ins, 99)), int(max(ins))),
                ("output", int(pct(outs, 50)), int(pct(outs, 90)), int(pct(outs, 99)), int(max(outs)))])
    L.append("")
    heavy = [r for r in rows if r["inp"] >= 2000 and r["inp"] >= 10 * max(1, r["out"])]
    hs = sum(pricer.cost(r) for r in heavy)
    L.append("Input-heavy calls (input >= 2,000 tokens and >= 10x the output): %d calls (%.1f%% of calls), %s (%.1f%% of spend). These are where caching or trimming would pay off first."
             % (len(heavy), 100 * len(heavy) / len(rows), usd(hs), 100 * hs / total_cost if total_cost else 0))
    L.append("")
    groups = defaultdict(list)
    for r in rows:
        if r["phash"]:
            groups[(r["model"], r["phash"])].append(r)
    if groups:
        L.append("### Repeated prefixes (from `prefix_hash`)")
        prow = []
        spend_rep_total = 0.0
        for (m, h), rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ptok = [r["ptoks"] for r in rs if r["ptoks"]]
            plen = max(ptok) if ptok else min(r["inp"] for r in rs)
            pr = pricer.rates(m)
            rep = sum(max(0.0, min(plen, r["inp"] - r["cached"])) for r in rs[1:])
            spend_rep_total += rep * pr["input"] / 1e6
            prow.append(("`%s`" % h, m, len(rs), int(plen), "yes" if ptok else "no (min input)",
                         "%.0f%%" % (100 * sum(r["cached"] > 0 for r in rs) / len(rs)), usd(rep * pr["input"] / 1e6)))
        L += table(["prefix", "model", "calls", "prefix tok", "from column", "calls with cache hits", "spend on repeated uncached prefix"], prow)
        L.append("")
        L.append("Repeated uncached prefix tokens account for %s, **%.1f%% of total spend** (the first call per model+prefix is excluded as a cache write)."
                 % (usd(spend_rep_total), 100 * spend_rep_total / total_cost if total_cost else 0))
        L.append("")

    # Opportunities
    L.append("## Opportunities to test (ranked by what-if dollar size)")
    L.append("Every item is a **hypothesis**, not a promised saving. The dollar figures are what-if arithmetic on this export and these (unverified) prices. Each needs an eval on real traffic before any change ships.")
    L.append("")
    ops = opportunities(rows, pricer, total_cost)
    for i, o in enumerate(ops, 1):
        L.append("%d. **%s**" % (i, o["title"]))
        L.append("   - Hypothesis: %s" % o["hypothesis"])
        L.append("   - %s" % o["whatif"])
        L.append("   - How to test: %s" % o["test"])
    if not ops:
        L.append("No opportunity cleared the minimum thresholds on this export.")
    L.append("")
    L.append("What-if figures overlap (the same tokens can appear in more than one item) and must not be added together.")
    L.append("")
    return "\n".join(L)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="LLM usage audit from a CSV/JSONL export.")
    ap.add_argument("export")
    ap.add_argument("--prices", default=os.path.join(here, "prices.json"))
    ap.add_argument("-o", "--output")
    ap.add_argument("--label", help="override the top-line data-source statement")
    ap.add_argument("--input-excludes-cached", action="store_true",
                    help="input_tokens does not include cached_input_tokens")
    a = ap.parse_args()
    prices = load_prices(a.prices)
    rows, colmap = load(a.export, a.input_excludes_cached)
    report = build_report(rows, colmap, Pricer(prices), prices, a.export, a.label)
    if a.output:
        with open(a.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(report)
        print("wrote %s" % a.output)
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
