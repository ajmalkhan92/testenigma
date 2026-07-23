---
title: "The Flakiness Score: Finding Your Worst Tests With Statistics"
description: "'This test is flaky' is usually a feeling, not a measurement. A formula that turns rerun history into an objective flakiness score, plus a way to cluster dozens of flaky failures down to the handful of root causes actually behind them."
pubDate: 2026-07-23
category: reliability
tags: ["flaky tests", "statistics", "Python"]
---

"That test is just flaky" is the sentence that lets a real bug hide in plain sight for months. It's also usually a feeling, not a measurement: nobody's actually comparing pass rates, they're remembering the last time it annoyed them. A flaky suite erodes trust in CI the exact same way an all-judge [test pyramid](/articles/llm-test-pyramid/) does: people stop believing red means something, and start re-running until it's green.

**What it is:** an objective flakiness score computed from rerun history, plus a way to cluster many flaky tests down to the small number of root causes actually behind them.

**The problem it solves:** without a number, "flaky" is a vibe and prioritization is guesswork. With one, you can rank every test by how unreliable it actually is, and tell the difference between forty flaky tests and one shared root cause wearing forty different test names.

## How it works

### The flakiness formula

A test that's always green or always red isn't flaky: it's either reliable or simply broken. A flakiness score needs to be zero at both of those extremes and highest for a test that's a coin flip:

```python
# flaky/score.py
def flakiness_score(results: list[bool]) -> float:
    """2p(1-p): zero for a test that's always passing or always failing,
    maximized at 1.0 for a test that's a coin flip."""
    if not results:
        return 0.0
    p = sum(results) / len(results)
    return 2 * p * (1 - p)
```

This is the variance of a coin-flip (Bernoulli) outcome, scaled to peak at 1.0: a standard, defensible way to turn a pass/fail sequence into a single number without inventing a metric from scratch.

### Ranking a suite

```python
# flaky/rerun_data.py
RERUN_HISTORY = {
    "test_checkout_flow": [True, True, True, True, True, True, True, True, True, True],
    "test_search_pagination": [True, False, True, False, True, True, False, True, False, True],
    "test_password_reset": [False, False, False, False, False, False, False, False, False, False],
    "test_upload_large_file": [True, True, False, True, True, True, True, False, True, True],
}
```

```python
# flaky/rank.py
from flaky.rerun_data import RERUN_HISTORY
from flaky.score import flakiness_score

def rank_by_flakiness() -> list[tuple[str, float]]:
    scored = [(name, flakiness_score(results)) for name, results in RERUN_HISTORY.items()]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
```

`test_password_reset` fails every single time: score 0.0, correctly. It's not flaky, it's broken, and it belongs on a different list entirely from the ones this score is built to surface.

### Clustering failures by likely root cause

A ranked list of forty flaky tests is still forty things to investigate one at a time, unless several of them share a root cause, which is common: one race condition in a shared test fixture can make a dozen unrelated-looking tests flaky simultaneously. Grouping by textual similarity of the failure message, the same `difflib` approach from the [regression harness post](/articles/catching-llm-hallucinations-regression-harness/), turns forty investigations into however many distinct clusters actually exist:

```python
# flaky/cluster.py
from difflib import SequenceMatcher

SIMILARITY_THRESHOLD = 0.7

def cluster_failures(messages_by_test: dict[str, list[str]]) -> list[dict]:
    """Group tests whose most recent failure message is textually similar,
    on the theory that similar error text often shares a root cause."""
    representative = {test: msgs[-1] for test, msgs in messages_by_test.items() if msgs}
    clusters: list[dict] = []

    for test, message in representative.items():
        placed = False
        for cluster in clusters:
            similarity = SequenceMatcher(None, cluster["representative_message"], message).ratio()
            if similarity >= SIMILARITY_THRESHOLD:
                cluster["tests"].append(test)
                placed = True
                break
        if not placed:
            clusters.append({"representative_message": message, "tests": [test]})

    return clusters
```

## Common issues

**Rerun history has to hold code and environment fixed to mean anything.** Flakiness is "same code, same environment, different outcome." Pulling pass/fail history across weeks of commits conflates that with "this test actually got fixed" or "this test actually broke": neither of which is flakiness. Measure it by actually rerunning N times against one fixed commit, not by mining CI history across a moving codebase.

**A 100%-failing test is not flaky: it's broken, and the score correctly says so.** Worth stating plainly because it's the single most common misreading of any flakiness metric: zero score at either extreme is a feature of the formula, not a gap in it. Route always-failing tests to a bug tracker, not a flakiness dashboard.

**Lexical clustering only catches similar-looking error text.** Two failures with the same underlying root cause (a shared race condition, say) but different surface assertion messages won't cluster together with `difflib` alone: the same ceiling the [regression harness post](/articles/catching-llm-hallucinations-regression-harness/) hit with lexical diffing. A semantic clustering pass (embeddings, or an LLM reading a batch of failure messages and grouping them by likely cause) is the natural next step once lexical clustering stops being enough, the same escalation path the [LLM-as-judge post](/articles/llm-as-judge-eval-pipeline/) makes for scoring open-ended text.

## What to test, and how

**The formula itself, against known sequences**: the cheapest, most important test here, because a wrong flakiness formula silently misprioritizes everything downstream of it:

```python
# tests/test_flakiness_score.py
from flaky.score import flakiness_score

def test_always_passing_test_scores_zero():
    assert flakiness_score([True] * 10) == 0.0

def test_always_failing_test_also_scores_zero():
    # Broken, not flaky: a real distinction worth encoding directly in the test.
    assert flakiness_score([False] * 10) == 0.0

def test_coin_flip_test_scores_highest():
    assert flakiness_score([True, False] * 5) == 1.0

def test_mostly_passing_with_occasional_failure_scores_low_but_nonzero():
    score = flakiness_score([True] * 9 + [False])
    assert 0 < score < 0.3
```

**Then, a CI gate on the whole suite**: the check that turns "we know some tests are flaky" into an enforced ceiling instead of a fact everyone's aware of and nobody acts on:

```python
# tests/test_flakiness_gate.py
from flaky.rank import rank_by_flakiness

FLAKINESS_CEILING = 0.4

def test_no_test_exceeds_the_flakiness_ceiling():
    offenders = [(name, score) for name, score in rank_by_flakiness() if score > FLAKINESS_CEILING]
    assert not offenders, f"Tests over the flakiness ceiling: quarantine or fix: {offenders}"
```

A test that trips this doesn't get silently re-run until green: it gets quarantined (excluded from the blocking suite, still tracked) or fixed, which is the entire point of having a number instead of a feeling.

## Takeaways

- `2p(1-p)` is zero at both extremes on purpose: a test that always fails isn't flaky, and a good flakiness metric has to say so explicitly, not just implicitly.
- Rerun against one fixed commit to measure flakiness; mining pass/fail history across a moving codebase measures something else entirely.
- Clustering failure messages turns a long flaky-test list into the much shorter list of root causes actually behind it: investigate clusters, not individual test names, when several tests fail for the same underlying reason.
