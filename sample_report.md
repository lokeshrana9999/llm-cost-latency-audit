# LLM usage audit
**SYNTHETIC DATA: this report was generated from a made-up sample (make_sample.py), not from any real system. Numbers illustrate the format only.**

2,000 calls, 2026-11-09 00:00 to 2026-11-10 17:16. Prices: `prices.json` dated 2026-11-11, status "unverified, edit before use"; **verify against provider pricing pages before acting on any dollar figure.**
Total priced spend: **$11.32**. Input tokens are assumed to include cached input tokens.

## Spend by model
| model | calls | input tok | cached tok | output tok | spend | share | $/1k calls | success |
|---|---|---|---|---|---|---|---|---|
| claude-sonnet-4-5 | 600 | 2,830,922 | 0 | 82,040 | $9.72 | 85.9% | $16.21 | 96.7% |
| claude-haiku-4-5 | 1,014 | 2,244,843 | 1,076,400 | 51,805 | $1.54 | 13.6% | $1.51 | 94.7% |
| gpt-4o-mini | 386 | 387,823 | 0 | 5,216 | $0.061 | 0.5% | $0.16 | 93.8% |

## Cost per task
623 tasks (grouped by `session_id`), 3.2 calls per task on average. Task success = last call of the task succeeded.

| cost per task | USD |
|---|---|
| mean | $0.018 |
| p50 | $0.0002 |
| p95 | $0.135 |
| max | $0.245 |

Successful tasks: 588 of 623 (94.4%). **Cost per successful task: $0.019** (all task spend, including failed tasks, divided by successful tasks); cost per task attempted: $0.018.

Five costliest tasks: `s00294` $0.245 (12 calls), `s00640` $0.245 (12 calls), `s00425` $0.237 (12 calls), `s00434` $0.224 (11 calls), `s00646` $0.213 (12 calls).

## Latency by model (ms)
| model | latency p50 | latency p95 | TTFT p50 | TTFT p95 |
|---|---|---|---|---|
| claude-haiku-4-5 | 898 | 2168 | 463 | 597 |
| claude-sonnet-4-5 | 3548 | 11598 | 825 | 1106 |
| gpt-4o-mini | 389 | 697 | 302 | 392 |

## Token distribution
| spend component | USD | share of spend |
|---|---|---|
| uncached input | $9.72 | 85.9% |
| cached input | $0.108 | 1.0% |
| output | $1.49 | 13.2% |

| tokens per call | p50 | p90 | p99 | max |
|---|---|---|---|---|
| input | 2257 | 4634 | 10710 | 13653 |
| output | 37 | 169 | 446 | 920 |

Input-heavy calls (input >= 2,000 tokens and >= 10x the output): 1278 calls (63.9% of calls), $10.27 (90.8% of spend). These are where caching or trimming would pay off first.

### Repeated prefixes (from `prefix_hash`)
| prefix | model | calls | prefix tok | from column | calls with cache hits | spend on repeated uncached prefix |
|---|---|---|---|---|---|---|
| `de51c90cb990` | claude-haiku-4-5 | 1014 | 1800 | yes | 59% | $1.03 |
| `f5ab92bbac48` | claude-sonnet-4-5 | 600 | 3200 | yes | 0% | $5.75 |
| `c8b51c4c555a` | gpt-4o-mini | 386 | 900 | yes | 0% | $0.052 |

Repeated uncached prefix tokens account for $6.83, **60.3% of total spend** (the first call per model+prefix is excluded as a cache write).

## Opportunities to test (ranked by what-if dollar size)
Every item is a **hypothesis**, not a promised saving. The dollar figures are what-if arithmetic on this export and these (unverified) prices. Each needs an eval on real traffic before any change ships.

1. **Prompt caching for repeated long prefixes**
   - Hypothesis: Calls that resend the same long prefix (system prompt, tool schema) and are not yet billed at the cached rate might be eligible for it.
   - WHAT-IF: if cached-input pricing applied to the 3,288,632 repeated uncached prefix tokens (60.2% of all input tokens) across 3 model+prefix groups, spend would drop by $6.12 (54.1% of total) under these prices.
   - How to test: Enable caching on one route; compare billed cached tokens and TTFT; check TTL, minimum prefix size and cache-write fees against the provider docs (not modelled here).
2. **Route short-output calls to a cheaper model**
   - Hypothesis: On claude-sonnet-4-5, calls with very short outputs (<= 29 tokens, a quarter of that model's median) may be acknowledgements or simple decisions a cheaper model handles equally well.
   - WHAT-IF: 217 such calls cost $3.16 on claude-sonnet-4-5. Priced as gpt-4o-mini (no caching), they would cost $0.157: a ceiling of $3.00 (26.5% of spend). If only half pass an eval, about $1.50.
   - How to test: Replay a sample of these calls on gpt-4o-mini offline; grade against claude-sonnet-4-5 outputs; route only the classes that pass.
3. **Context trimming on the largest prompts**
   - Hypothesis: The top 10% of calls by input size (> 4,634 tokens) carry history or retrieval text that may not all affect the answer.
   - WHAT-IF: those 200 calls cost $4.31 (38.1% of spend). If 20% of their uncached input tokens (of 1,303,923) could be dropped, spend would drop by $0.782 (6.9%).
   - How to test: Summarise or cut history/retrieval on a sample of these calls; compare task success on an eval set before and after.
4. **Cap or tighten very long outputs**
   - Hypothesis: Output tokens cost several times input tokens; the longest replies (> p95 = 251 tokens) may be padding or runaway generations.
   - WHAT-IF: if every output above p95 were cut to p95, 12,035 output tokens would not be generated: spend would drop by $0.180 (1.6%).
   - How to test: Add a length instruction or max_tokens on one route; check truncation rate and answer quality on an eval set.
5. **Reduce spend on failed calls**
   - Hypothesis: Failed calls (98, 4.9% of calls) still consume tokens. Retries, malformed output or timeouts may share a few fixable causes.
   - WHAT-IF: failed calls cost $0.371 (3.3% of spend). If a quarter of that were avoidable, spend would drop by $0.093.
   - How to test: Break failures down by error type (needs an error column or logs); fix the largest class first; track the failure rate.
6. **Latency tails**
   - Hypothesis: p95 latency far above p50 suggests a slow minority of calls (long outputs, large prompts, provider spikes, retries).
   - NOT A COST WHAT-IF. claude-sonnet-4-5 p50 3548 ms, p95 11598 ms (3.3x).
   - How to test: Split tail calls by input size, output size and time of day to see which factor explains them; then test streaming, shorter outputs or a fallback.

What-if figures overlap (the same tokens can appear in more than one item) and must not be added together.
