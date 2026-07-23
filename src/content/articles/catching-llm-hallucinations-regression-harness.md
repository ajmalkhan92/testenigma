---
title: "Catching LLM Hallucinations With a Regression Harness"
description: "A pytest harness that runs the same prompts across model calls, snapshots the outputs, and flags drift automatically, so a silent hallucination regression doesn't ship unnoticed."
pubDate: 2026-07-23
category: evaluation
tags: ["Python", "pytest", "LLM evals"]
---

Unit tests assume a function returns the same output for the same input. LLM calls break that assumption on purpose: ask the same question twice and you can get two different, both-defensible answers. That's fine for a chatbot. It's a problem the moment an LLM call sits inside a pipeline someone depends on: a support-ticket classifier, a summarizer, a regex explainer bolted onto a linter. When you swap model versions, tweak a prompt, or just get unlucky on a temperature roll, you want to know whether the output *drifted*, and you want to know in CI, not from a user's bug report.

A regression harness gives you that signal cheaply: run each prompt several times, snapshot the outputs, and diff future runs against the snapshot. It won't catch every hallucination (nothing lexical will) but it catches the failure mode that actually ships most often: a change that quietly makes answers less consistent.

## The pipeline

<div class="mermaid-wrap">
<svg viewBox="0 0 800 200" role="img" aria-label="Flowchart: Prompt leads to calling the model N times, then diffing against the baseline snapshot, which either passes or flags drift.">
<defs>
<marker id="dg1arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/>
</marker>
</defs>
<style>
.dg1-node{fill:var(--card);stroke:var(--border-2);stroke-width:1.5}
.dg1-good{stroke:var(--good)}
.dg1-bad{stroke:var(--bad)}
.dg1-text{fill:var(--text);font-family:var(--sans);font-size:13px}
.dg1-edge{stroke:var(--muted);stroke-width:1.5;fill:none;marker-end:url(#dg1arrow)}
.dg1-label{fill:var(--faint);font-family:var(--mono);font-size:10px}
</style>
<rect class="dg1-node" x="10" y="78" width="100" height="44" rx="8"/>
<text class="dg1-text" x="60" y="105" text-anchor="middle">Prompt</text>
<path class="dg1-edge" d="M110,100 H158"/>
<rect class="dg1-node" x="160" y="78" width="170" height="44" rx="8"/>
<text class="dg1-text" x="245" y="105" text-anchor="middle">Call model N times</text>
<path class="dg1-edge" d="M330,100 H378"/>
<rect class="dg1-node" x="380" y="78" width="190" height="44" rx="8"/>
<text class="dg1-text" x="475" y="96" text-anchor="middle">Diff vs. baseline</text>
<text class="dg1-text" x="475" y="112" text-anchor="middle">snapshot</text>
<path class="dg1-edge" d="M570,90 C610,90 610,40 648,40"/>
<text class="dg1-label" x="575" y="62">similarity &#8805; floor</text>
<rect class="dg1-node dg1-good" x="650" y="18" width="100" height="40" rx="8"/>
<text class="dg1-text" x="700" y="43" text-anchor="middle">Pass</text>
<path class="dg1-edge" d="M570,110 C610,110 610,160 648,160"/>
<text class="dg1-label" x="575" y="152">similarity &lt; floor</text>
<rect class="dg1-node dg1-bad" x="650" y="138" width="120" height="40" rx="8"/>
<text class="dg1-text" x="710" y="163" text-anchor="middle">Flag drift</text>
</svg>
</div>

Three pieces: a thin client that calls the model, a snapshot store for "what a good answer used to look like," and a pytest test that fails loudly when a new run drifts too far from the baseline.

## The model client

```python
# harness/model_client.py
import anthropic

client = anthropic.Anthropic()

def call_model(prompt: str, model: str = "claude-opus-4-8") -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
```

`model` is a parameter, not a constant: that's what lets the same harness answer "did this drift over time" (rerun against the same model) and "did this drift *because I upgraded*" (rerun against a new model ID and diff both snapshot sets).

## Snapshotting a baseline

```python
# harness/snapshot.py
import json
from pathlib import Path

SNAPSHOT_DIR = Path("snapshots")

def snapshot_path(prompt_id: str) -> Path:
    return SNAPSHOT_DIR / f"{prompt_id}.json"

def load_baseline(prompt_id: str) -> list[str]:
    path = snapshot_path(prompt_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())["outputs"]

def save_baseline(prompt_id: str, outputs: list[str]) -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    snapshot_path(prompt_id).write_text(json.dumps({"outputs": outputs}, indent=2))
```

Snapshots are committed to the repo, same as any other fixture. The first run for a prompt has nothing to compare against, so it records one: every run after that has something to hold the line against.

## The test

```python
# tests/test_regression.py
from difflib import SequenceMatcher

import pytest

from harness.model_client import call_model
from harness.snapshot import load_baseline, save_baseline

PROMPTS = {
    "capital-of-france": "What is the capital of France? Answer in one sentence.",
    "regex-explain": r"Explain what this regex does: ^\d{3}-\d{2}-\d{4}$",
}

SIMILARITY_FLOOR = 0.75
RUNS_PER_PROMPT = 5


@pytest.mark.parametrize("prompt_id,prompt", PROMPTS.items())
def test_output_stability(prompt_id, prompt):
    baseline = load_baseline(prompt_id)
    outputs = [call_model(prompt) for _ in range(RUNS_PER_PROMPT)]

    if not baseline:
        save_baseline(prompt_id, outputs)
        pytest.skip(f"No baseline for {prompt_id}: recorded {RUNS_PER_PROMPT} runs as the new baseline.")

    worst = min(
        SequenceMatcher(None, base, out).ratio()
        for base in baseline
        for out in outputs
    )
    assert worst >= SIMILARITY_FLOOR, (
        f"Output drifted for {prompt_id}: worst-case similarity {worst:.2f} < {SIMILARITY_FLOOR}"
    )
```

`SequenceMatcher.ratio()` is a lexical similarity score: cheap, deterministic, no extra API calls. Comparing every new output against every baseline output (not just the first one) means the assertion respects the model's normal run-to-run variance instead of getting tripped up by it: the floor only fires when a new run falls outside the range the baseline already demonstrated.

## What a flagged run looks like

Two prompts, five runs each, one clean and one flagged:

| Prompt | Runs | Worst-case similarity | Result |
|---|---|---|---|
| capital-of-france | 5 | 0.98 | pass |
| regex-explain | 5 | 0.61 | flagged |

The capital-of-france prompt has almost no room to drift: there's one correct answer and a handful of ways to phrase it. The regex-explanation prompt has much more surface area: one run described the pattern as "a US Social Security number," another called it "a phone-number-like ID," and a third just walked through the character classes without naming what it matches. All three are defensible. `SequenceMatcher` doesn't know that: it sees three different strings and scores them low. That's not a false alarm exactly; it's the harness telling you this prompt's output isn't pinned down enough to regression-test lexically, which is useful information on its own.

## Running it across model versions

The same harness answers the "should I upgrade?" question. Point it at the new model, write outputs to a separate snapshot directory, and diff the two sets before touching production:

```bash
$ SNAPSHOT_DIR=snapshots/opus-4-8 MODEL=claude-opus-4-8 pytest tests/test_regression.py
$ SNAPSHOT_DIR=snapshots/candidate MODEL=claude-sonnet-5 pytest tests/test_regression.py
```

Diff the two snapshot directories and you get a concrete list of exactly which prompts changed behavior under the candidate model: not a vague sense that "the new model feels different," but the specific outputs that differ and by how much.

## Where this breaks down

Lexical similarity catches phrasing drift and gross regressions: an answer that used to be three sentences and is now an apology, a prompt that used to return JSON and now returns prose. It does not catch a fluent, confident, wrong answer that happens to be *worded* like the baseline. Two responses can score 0.95 on `SequenceMatcher` and differ in the one number that matters.

That's a real limitation, not a rounding error, and it's the reason this harness is a floor, not a ceiling. The fix is a judge that reads for *meaning* instead of *characters*: a second model scoring each output against a rubric instead of against a string. That's a harder problem (judges have their own biases and blind spots), and it's next on the list here.

## Takeaways

- Nondeterminism isn't a reason to skip testing LLM output: it's a reason to test the *range* of output instead of a single expected value.
- Snapshot against every recorded baseline run, not just the first one, or you'll flag normal variance as drift.
- Lexical diffing is cheap and catches real regressions, but it's blind to confident, well-phrased wrongness: pair it with a semantic check before you trust it fully.
