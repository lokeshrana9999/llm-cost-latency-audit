# LLM usage audit

Send a usage export (model, tokens and latency per call). Within 48 hours you get a report: where your LLM spend and latency go, which changes cut cost without changing model output, and which need an eval first.

No prompts, completions, user content or API keys. Free. The script is stdlib Python and runs offline, so you can also run it yourself.

The first screen of a report, generated from made-up data by `make_sample.py`:

```text
$427-$684 a month (19.6-31.4% of spend) can go without changing model output,
out of $2,178 a month; up to $915 more depends on evals

Window:  30 days, 49,736 calls, $2,176 priced spend
Savings that do not change model output: $427-$684 a month
Further, depends on evals: $0-$915 a month, unproven
Speed:   TTFT p95 is 1,238 ms; 13.8% of calls are over a 1,000 ms budget
Do first: turn on prompt caching (n=46,283 calls, $381-$470 a month, MECHANICAL)
Data:    49,736 rows read, 0 dropped, 0 on unpriced models
```

The full sample is in [sample_report.md](sample_report.md): a ranked action table, a latency view for voice and chat routes, cost per successful task for back-office routes, and one card per action with its arithmetic, how to check it, and what would prove it wrong.

## Run it in two minutes

```bash
git clone https://github.com/lokeshrana9999/llm-cost-latency-audit
cd llm-cost-latency-audit
python make_sample.py
python audit.py sample_export.csv --ttft-budget-ms 1000 -o sample_report.md
python test_audit.py
```

On your own data: `python audit.py your_export.csv -o report.md` (CSV or JSONL, one row per model call). Add `--ttft-budget-ms 800` to see how many calls miss your first-token budget.

## What each column unlocks

| Column | Required | Unlocks |
|---|---|---|
| `model`, `input_tokens`, `output_tokens`, `latency_ms` | yes | spend by model, latency p50/p95 |
| `timestamp` | recommended | monthly projection, cache-TTL math, retries |
| `task_id` or `session_id` | recommended | cost per task, context growth, retries |
| `success` | recommended | cost per successful task, spend in failed tasks |
| `cached_input_tokens` | optional | what caching already saves |
| `ttft_ms` | optional | voice and chat latency view |
| `prefix_hash`, `prefix_tokens` | optional | prompt-caching dollars |
| `route` | optional | per-route tables (otherwise `prefix_hash` stands in) |

`prefix_hash` is any stable hash of the fixed part of the prompt (system prompt plus tool schema), computed on your side. The text never leaves your system.

## How to read the numbers

Every figure carries one of three labels:

- **MEASURED**: a fact in your export, such as spend inside failed tasks.
- **MECHANICAL**: a saving from billing rules that does not change model output, such as prompt caching.
- **HYPOTHESIS**: a saving that holds only if quality holds, such as moving a route to a smaller model. Its low end is $0 until an eval passes.

Savings are applied in a fixed order (failed-task waste, routing, trimming, caching, output caps, batching), each on the spend left by the one before, so the totals add up.

## Limits

- `prices.json` holds list prices dated in the file. Check them against your provider's pricing page; the report prints the date.
- Without prompts, the report can measure cost and latency but cannot judge answer quality. That is what the eval step is for.
- Not modelled: negotiated discounts, provisioned throughput, and latency outside the model call.

## Get the audit

Book 30 minutes at https://calendly.com/lokeshrana9999/30min, or open an issue with the column names your logs record and I'll reply with what the report can compute from them.

Then send the export. The report arrives within 48 hours, and it and the script are yours whatever you decide. If a change is worth testing, a fixed-price two-week pilot runs evals on your real traffic, ships one or two changes behind them, and adds a CI gate that blocks regressions in quality, cost per successful task or latency.
