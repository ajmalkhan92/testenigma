---
title: "The Tokens Are the Budget: Cost and Latency Regression Testing"
description: "A prompt change or model upgrade can triple your cost per request or blow through a latency SLA, and every other eval in this series would call it a pass. Treat token spend and response time as testable, budgetable numbers."
pubDate: 2026-07-23
category: reliability
tags: ["cost", "latency", "CI"]
---

Every eval on this blog so far asks "is the output correct." None of them ask "what did that cost, and how long did it take" — and a change can pass every correctness check while quietly tripling your per-request spend or pushing p95 latency past what your product can tolerate. Nobody notices until the bill arrives or users start complaining, because nothing in CI was watching either number.

**What it is:** cost and latency treated as testable metrics with budgets, the same threshold-assertion pattern as every eval in this series — pointed at `response.usage` and a stopwatch instead of output quality.

**The problem it solves:** correctness evals and cost/latency regressions are orthogonal. A prompt change can hold accuracy steady while doubling token usage, and nothing in a quality-only eval suite will ever flag it.

## How it works

### Measuring a call

```python
# perf/measure.py
import time
from dataclasses import dataclass

import anthropic

client = anthropic.Anthropic()

# $/million tokens — update when pricing changes
PRICING = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float

def timed_call(prompt: str, model: str = "claude-opus-4-8") -> CallResult:
    start = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=300,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    latency_s = time.monotonic() - start

    rates = PRICING[model]
    cost_usd = (
        response.usage.input_tokens / 1_000_000 * rates["input"]
        + response.usage.output_tokens / 1_000_000 * rates["output"]
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return CallResult(text, response.usage.input_tokens, response.usage.output_tokens, latency_s, cost_usd)
```

`response.usage` is the source of truth for tokens — never estimate cost from `len(text)` or a rough token-per-word guess. It's exact, it's free to read, and it's already in every response.

### A golden set to measure against

```python
# perf/golden_prompts.py
GOLDEN_PROMPTS = [
    "Summarize this ticket in one sentence: I was charged twice for my subscription this month.",
    "Summarize this ticket in one sentence: The export button does nothing when I click it.",
    "Summarize this ticket in one sentence: I can't log in even after resetting my password.",
]
```

Same golden-dataset discipline as the [eval posts](/articles/golden-dataset-for-llm-evals/) — representative prompts, committed to the repo, so cost and latency numbers are measured against something consistent over time instead of whatever happened to be typed into a terminal that day.

## Common issues

**Effort level moves both numbers, in the same direction, a lot.** `output_config.effort` trades thinking depth for cost and latency — bumping from `low` to `high` on a route that doesn't need it is one of the most common silent cost regressions, and it won't show up in a correctness eval because the higher-effort answer is often *also* correct, just needlessly expensive to produce.

**Token counts aren't directly comparable across model versions.** Different models can use different tokenizers, so "prompt X used 340 tokens on the old model, 410 on the new one" isn't automatically a regression — it can just be a different, non-buggy count. Compare *cost in dollars* and *wall-clock latency* across a model change, not raw token counts.

**A cache hit is not the same request twice.** If prompt caching is in play, `cache_read_input_tokens` bills at a fraction of the normal input rate — averaging cost across a mix of cached and uncached calls without accounting for that will make a perfectly healthy cache-warm system look like it regressed.

## What to test, and how

**Absolute budget, on every change:**

```python
# tests/test_cost_latency.py
import statistics

from perf.golden_prompts import GOLDEN_PROMPTS
from perf.measure import timed_call

COST_PER_REQUEST_CEILING = 0.01  # USD
P95_LATENCY_CEILING_S = 6.0

def test_cost_and_latency_stay_within_budget():
    results = [timed_call(p) for p in GOLDEN_PROMPTS]

    avg_cost = statistics.mean(r.cost_usd for r in results)
    latencies = sorted(r.latency_s for r in results)
    p95_index = max(0, int(len(latencies) * 0.95) - 1)
    p95_latency = latencies[p95_index]

    assert avg_cost <= COST_PER_REQUEST_CEILING, f"Avg cost ${avg_cost:.4f} exceeds budget ${COST_PER_REQUEST_CEILING}"
    assert p95_latency <= P95_LATENCY_CEILING_S, f"p95 latency {p95_latency:.1f}s exceeds ceiling {P95_LATENCY_CEILING_S}s"
```

p95, not average, for latency — an average hides the slow tail that's actually responsible for user complaints; a budget on the average alone will happily pass while one in twenty requests times out.

**Relative budget, when evaluating a model upgrade.** A quality-driven upgrade (say, moving a route from Haiku to Opus) is *expected* to cost more — the question isn't "did cost increase," it's "did it increase more than the quality gain justifies":

```python
# tests/test_cost_regression_across_models.py
from perf.golden_prompts import GOLDEN_PROMPTS
from perf.measure import timed_call

MAX_COST_INCREASE = 3.0  # candidate allowed to cost up to 3x the baseline, no more

def test_upgrade_candidate_does_not_blow_the_cost_budget():
    baseline_cost = sum(timed_call(p, model="claude-haiku-4-5").cost_usd for p in GOLDEN_PROMPTS)
    candidate_cost = sum(timed_call(p, model="claude-opus-4-8").cost_usd for p in GOLDEN_PROMPTS)

    multiplier = candidate_cost / baseline_cost
    assert multiplier <= MAX_COST_INCREASE, f"Candidate costs {multiplier:.1f}x baseline, budget is {MAX_COST_INCREASE}x"
```

Run this alongside the correctness eval from the [regression harness post](/articles/catching-llm-hallucinations-regression-harness/) when considering any model swap — quality and cost are two different axes, and a model upgrade decision needs both numbers, not just the accuracy delta.

## Takeaways

- Read cost from `response.usage`, never estimate it — it's exact and it's already there.
- Budget p95 latency, not average — the average hides exactly the slow tail users notice.
- A model upgrade is expected to shift cost; test that the shift stays within a multiplier you've actually decided is acceptable, not that cost stayed flat.
