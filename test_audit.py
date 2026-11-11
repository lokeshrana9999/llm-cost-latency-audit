"""Tests for audit.py (stdlib unittest). Run: python test_audit.py"""
import os
import random
import unittest

import audit

HERE = os.path.dirname(os.path.abspath(__file__))
PRICES = audit.load_prices(os.path.join(HERE, "prices.json"))


def make_rows(n_voice=400, n_chat=300, n_small=50, seed=3):
    """A small mixed export: three routes, two providers, one route under 200 calls."""
    rng = random.Random(seed)
    rows = []
    t = 1_700_000_000
    for i in range(n_voice):
        t += rng.randint(5, 40)
        sess = "v%03d" % (i // 8)
        rows.append(dict(timestamp=t, route="voice", model="claude-sonnet-4-5", input_tokens=3500 + (i % 8) * 120,
                         output_tokens=rng.randint(20, 150), cached_input_tokens=0, latency_ms=2000, ttft_ms=rng.randint(400, 1200),
                         task_id=sess, session_id=sess, success="false" if i % 37 == 0 else "true", streamed="true",
                         prefix_hash="aaa111", prefix_tokens=3200))
    for i in range(n_chat):
        t += rng.randint(5, 40)
        rows.append(dict(timestamp=t, route="chat", model="gpt-4o", input_tokens=rng.randint(1500, 6000),
                         output_tokens=rng.randint(20, 400), cached_input_tokens=0, latency_ms=3000, ttft_ms=rng.randint(300, 900),
                         task_id="c%04d" % i, session_id="c%04d" % i, success="true", streamed="true",
                         prefix_hash="bbb222", prefix_tokens=1500))
    for i in range(n_small):
        t += rng.randint(5, 40)
        rows.append(dict(timestamp=t, route="small", model="claude-sonnet-4-5", input_tokens=2000, output_tokens=100,
                         cached_input_tokens=0, latency_ms=900, ttft_ms=300, task_id="m%03d" % i, session_id="m%03d" % i,
                         success="true", streamed="true", prefix_hash="ccc333", prefix_tokens=1200))
    return rows


def run(rows, **kw):
    parsed = audit.parse_rows(rows)
    return audit.analyze(parsed, PRICES, **kw)


class WaterfallTests(unittest.TestCase):
    def setUp(self):
        self.ctx = run(make_rows())

    def test_after_earlier_sums_equal_total_reduction(self):
        base = self.ctx.spend
        for level in ("low", "high"):
            summed = sum(r["after_" + level] for r in self.ctx.rows)
            self.assertAlmostEqual(summed, base - self.ctx.final[level], places=6)

    def test_standalone_is_never_below_after_for_the_same_chain_start(self):
        first = self.ctx.rows[0]
        self.assertAlmostEqual(first["alone_high"], first["after_high"], places=9)
        self.assertAlmostEqual(first["alone_low"], first["after_low"], places=9)

    def test_range_is_ordered_and_totals_add_up(self):
        for r in self.ctx.rows:
            self.assertLessEqual(r["lo"], r["hi"] + 1e-12)
        self.assertAlmostEqual(self.ctx.low_total, sum(r["lo"] for r in self.ctx.rows) * self.ctx.k, places=6)
        self.assertAlmostEqual(self.ctx.high_total, sum(r["hi"] for r in self.ctx.rows) * self.ctx.k, places=6)

    def test_spend_split_adds_to_total(self):
        agg = audit.money_table(self.ctx)
        uncached = sum(v[1] for v in agg.values())
        cached = sum(v[2] for v in agg.values())
        output = sum(v[3] for v in agg.values())
        self.assertAlmostEqual(uncached + cached + output, self.ctx.spend, places=6)
        # a model's spend includes its output, so it differs from uncached-input spend
        by_model = {}
        for (m, _), v in agg.items():
            by_model[m] = by_model.get(m, 0.0) + v[1] + v[2] + v[3]
        self.assertAlmostEqual(sum(by_model.values()), self.ctx.spend, places=6)
        self.assertGreater(output, 0.0)


class RoutingTests(unittest.TestCase):
    def test_routing_never_crosses_providers(self):
        ctx = run(make_rows())
        self.assertTrue(ctx.route_cand)
        for i, target in ctx.route_cand.items():
            src = ctx.prices["models"][ctx.calls[i].model]
            dst = ctx.prices["models"][target]
            self.assertEqual(src["provider"], dst["provider"])
            self.assertLess(dst["tier"], src["tier"])

    def test_next_tier_down_stays_inside_provider(self):
        self.assertEqual(audit.next_tier_down(PRICES, "claude-sonnet-4-5"), "claude-haiku-4-5")
        self.assertEqual(audit.next_tier_down(PRICES, "gpt-4o"), "gpt-4o-mini")
        self.assertIsNone(audit.next_tier_down(PRICES, "gpt-4o-mini"))
        self.assertIsNone(audit.next_tier_down(PRICES, "claude-haiku-4-5"))


class PercentileTests(unittest.TestCase):
    def test_no_p95_under_200(self):
        self.assertIn("n/a", audit.p95(list(range(199))))
        self.assertNotIn("n/a", audit.p95(list(range(200))))

    def test_small_route_gets_no_p95_in_report(self):
        ctx = run(make_rows())
        text = audit.render(ctx)
        small_group = ("claude-sonnet-4-5", "small")
        self.assertNotIn(small_group, ctx.caps)          # no output cap from 50 calls
        self.assertNotIn(small_group, ctx.route_def)     # no routing from 50 calls
        row = [l for l in text.splitlines() if l.startswith("| small |") and "TTFT" not in l and "n=50, under 200" in l]
        self.assertTrue(row, "small route should show 'n=50, under 200' instead of a p95")

    def test_no_caching_dollars_without_prefix_hash(self):
        rows = make_rows()
        for r in rows:
            r.pop("prefix_hash")
            r.pop("prefix_tokens")
        ctx = run(rows)
        self.assertNotIn("caching", [r["step"]["key"] for r in ctx.rows])
        self.assertTrue(any("prefix_hash" in why for _, why in ctx.not_estimated))


class DataQualityTests(unittest.TestCase):
    def test_dropped_rows_are_counted(self):
        rows = make_rows(n_voice=250, n_chat=0, n_small=0)
        rows.append(dict(rows[0], model=""))                             # no model
        rows.append(dict(rows[0], input_tokens="abc"))                   # not a number
        rows.append(dict(rows[0], output_tokens=-5))                     # negative
        rows.append(dict(rows[0], model="mystery-model-9"))              # kept but unpriced
        parsed = audit.parse_rows(rows)
        self.assertEqual(parsed["n_read"], len(rows))
        self.assertEqual(sum(parsed["dropped"].values()), 3)
        ctx = audit.analyze(parsed, PRICES)
        self.assertEqual(ctx.unpriced_rows, 1)
        self.assertEqual(len(ctx.calls), 250)
        text = audit.render(ctx)
        self.assertIn("%s rows read, 3 dropped, 1 on unpriced models" % format(len(rows), ","), text)
        self.assertIn("mystery-model-9", text)


class ReportShapeTests(unittest.TestCase):
    def test_first_line_states_a_dollar_result_and_sections_in_order(self):
        text = audit.render(run(make_rows()))
        self.assertTrue(text.splitlines()[0].startswith("# $"))
        order = ["## ", "## Workload view", "## The ", "## ", "## Method and limits", "## Next step", "## Appendix"]
        pos = 0
        for marker in ["Workload view", "opportunities in detail", "Method and limits", "Next step", "## Appendix"]:
            nxt = text.find(marker, pos)
            self.assertGreater(nxt, -1, marker)
            pos = nxt

    def test_budget_line_only_when_given(self):
        rows = make_rows()
        self.assertNotIn("budget", audit.render(run(rows)).split("## ")[0])
        self.assertIn("over your 1,000 ms budget", audit.render(run(rows, ttft_budget=1000.0)))

    def test_projection_uses_window(self):
        ctx = run(make_rows())
        self.assertAlmostEqual(ctx.monthly, ctx.spend * 30.0 / ctx.days, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
