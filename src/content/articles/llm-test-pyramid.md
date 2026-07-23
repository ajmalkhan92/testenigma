---
title: "The LLM Test Pyramid"
description: "The classic unit/integration/e2e pyramid, redrawn for systems with an LLM in the loop — and a CI config that actually gates each layer at the right frequency."
pubDate: 2026-07-23
category: reliability
tags: ["testing strategy", "CI", "LLM evals"]
---

The classic test pyramid says: lots of fast, cheap unit tests at the bottom; fewer, slower integration tests in the middle; a handful of expensive end-to-end tests at the top. The shape encodes a tradeoff — cost and confidence both increase as you go up, so you want most of your signal coming from the cheap layer and the expensive layer reserved for what only it can catch.

LLM-powered systems don't break that shape, but they do add layers most teams haven't built yet. The last three posts here — a [regression harness](/articles/catching-llm-hallucinations-regression-harness/), an [eval runner](/articles/write-your-first-llm-eval/), and a [golden dataset](/articles/golden-dataset-for-llm-evals/) — are each one rung on this ladder. This post is where they fit together.

## The six layers

<div class="mermaid-wrap">
<svg viewBox="0 0 640 500" role="img" aria-label="Six-layer pyramid: deterministic unit tests, component evals, regression harness, LLM-as-judge evals, end-to-end agent evals, human spot-check, from bottom to top.">
<defs><marker id="dg3arrow" viewBox="0 0 10 10" refX="5" refY="8" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,0 L5,10 z" fill="var(--muted)"/></marker></defs>
<style>.dg3-node{fill:var(--card);stroke:var(--border-2);stroke-width:1.5}.dg3-text{fill:var(--text);font-family:var(--sans);font-size:13px}.dg3-edge{stroke:var(--muted);stroke-width:1.5;fill:none;marker-end:url(#dg3arrow)}</style>
<rect class="dg3-node" x="20" y="10" width="600" height="56" rx="10"/>
<text class="dg3-text" x="320" y="42" text-anchor="middle">1. Deterministic unit tests &#8212; no model calls</text>
<path class="dg3-edge" d="M320,66 V86"/>
<rect class="dg3-node" x="20" y="88" width="600" height="56" rx="10"/>
<text class="dg3-text" x="320" y="120" text-anchor="middle">2. Component evals &#8212; golden dataset, single call</text>
<path class="dg3-edge" d="M320,144 V164"/>
<rect class="dg3-node" x="20" y="166" width="600" height="56" rx="10"/>
<text class="dg3-text" x="320" y="198" text-anchor="middle">3. Regression harness &#8212; drift detection</text>
<path class="dg3-edge" d="M320,222 V242"/>
<rect class="dg3-node" x="20" y="244" width="600" height="56" rx="10"/>
<text class="dg3-text" x="320" y="276" text-anchor="middle">4. LLM-as-judge evals &#8212; semantic scoring</text>
<path class="dg3-edge" d="M320,300 V320"/>
<rect class="dg3-node" x="20" y="322" width="600" height="56" rx="10"/>
<text class="dg3-text" x="320" y="354" text-anchor="middle">5. End-to-end agent evals &#8212; multi-turn, tool use</text>
<path class="dg3-edge" d="M320,378 V398"/>
<rect class="dg3-node" x="20" y="400" width="600" height="56" rx="10"/>
<text class="dg3-text" x="320" y="432" text-anchor="middle">6. Human spot-check &#8212; sampled production review</text>
</svg>
</div>

| Layer | What it checks | Model calls? | Runs |
|---|---|---|---|
| 1. Deterministic unit tests | Prompt templates render correctly, output parsing doesn't crash on malformed JSON, tool-call arguments validate against a schema | No | Every commit |
| 2. Component evals | A single LLM call against a golden dataset — exact match or keyword coverage | Yes, cheap | Every PR that touches a prompt |
| 3. Regression harness | Output stability across runs and model versions | Yes | Nightly, and on model upgrades |
| 4. LLM-as-judge evals | Semantic quality — tone, completeness, correctness a keyword check can't see | Yes, expensive (two models) | Nightly, or on merge to main |
| 5. End-to-end agent evals | Full multi-turn tasks: did the agent call the right tools, reach the right final state | Yes, expensive | Pre-release |
| 6. Human spot-check | Whatever nothing above is built to catch | No (a person) | Weekly sample of real traffic |

Confidence goes up as you climb. So does cost, latency, and flakiness. The point of the pyramid isn't "run everything, always" — it's matching each layer's expense to how often you actually need to pay it.

## Layer 1: deterministic unit tests

No model call, no API key, no flakiness. These test the code *around* the LLM call — the part you fully control and that a probabilistic model can't excuse:

```python
# tests/test_prompt_rendering.py
from eval.classify import CATEGORIES, build_system_prompt

def test_system_prompt_lists_every_category():
    prompt = build_system_prompt()
    for category in CATEGORIES:
        assert category in prompt


def test_classify_ticket_rejects_empty_input():
    from eval.classify import classify_ticket
    import pytest
    with pytest.raises(ValueError):
        classify_ticket("")
```

If your LLM call's output feeds into a parser — JSON, a regex, a tool-call schema — that parser deserves the same unit-test treatment as any other function. It runs in milliseconds and needs zero budget for API calls.

## Layer 2 & 3: what the last three posts already built

Component evals and the regression harness are the two layers this blog has already covered end to end — a [golden-dataset-backed eval](/articles/golden-dataset-for-llm-evals/) checking correctness on every prompt change, and a [snapshot-diff harness](/articles/catching-llm-hallucinations-regression-harness/) checking stability nightly and on model upgrades. If you've been following along, those two layers are done. This post is about wiring them — and the layers above them — into a schedule that doesn't run everything on every commit.

## Layer 4 and 5 in one line each

Layer 4 (LLM-as-judge) exists for output where "correct" is a judgment call, not a string match — covered in the [next post](/articles/llm-as-judge-eval-pipeline/) along with the specific ways judge models lie to you. Layer 5 (end-to-end agent evals) exists for systems that take more than one step — multi-turn conversations, tool calls, branching state — covered in the post on [testing LangGraph agents](/articles/testing-langgraph-agents/).

## Wiring it into CI

The layers only pay off if each one runs at the right cadence. Everything on every push is slow and expensive; only the cheap layer on every push means real regressions ship silently. A minimal split:

```yaml
# .github/workflows/eval-pipeline.yml
name: Eval pipeline

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: "0 6 * * *"   # nightly at 6am UTC

jobs:
  unit:
    # Layer 1 — every push, every PR. No API key needed.
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/test_prompt_rendering.py -v

  component-evals:
    # Layer 2 — every PR. Cheap enough to gate merges on.
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/test_golden_eval.py -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

  nightly-suite:
    # Layers 3 & 4 — scheduled only. Too slow/expensive for every push.
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/test_regression.py tests/test_judge_eval.py -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Layer 5 (end-to-end agent evals) typically runs as a separate, manually-triggered or pre-release job — it's the most expensive and slowest layer, and gating every merge on it trades CI speed for a kind of confidence you usually only need right before a release, not on every commit.

## The anti-pattern this avoids

Two failure modes show up in nearly every team that skips this and just adds LLM tests ad hoc:

- **Only end-to-end LLM-judge tests.** Every check is slow, expensive, and a little flaky, so people stop trusting red CI and start re-running until it's green. The suite technically exists but has lost its authority.
- **Only deterministic tests.** CI is fast and green, and a prompt change that makes every answer subtly worse ships straight through, because nothing in the suite was ever positioned to catch a quality regression — only a crash.

The pyramid shape is the fix for both: cheap layers catch the cheap failures on every commit, expensive layers catch the expensive failures on a schedule that matches what they cost.

## Takeaways

- Match each layer's cost to how often you actually need to pay it — not everything belongs on every push.
- Deterministic tests around the LLM call (parsing, schema validation, prompt rendering) are still real tests and still belong at the bottom of the pyramid.
- A pyramid that's all top (judge/e2e) erodes trust in CI; a pyramid that's all bottom (deterministic only) misses real quality regressions. You need both ends.
