---
title: "Write Your First LLM Eval"
description: "The difference between an eval and a regression test, and a minimal pytest-based eval runner you can point at any LLM call: from exact-match scoring to a first pass at semantic grading."
pubDate: 2026-07-23
category: evaluation
tags: ["Python", "pytest", "LLM evals"]
---

The [regression harness from the last post](/articles/catching-llm-hallucinations-regression-harness/) answers one question: *did this output change?* An **eval** answers a different, harder question: *is this output correct?* You need both. A regression harness catches drift with no opinion about quality: it would happily pass a model that's consistently, confidently wrong. An eval needs a ground truth to grade against, and a way to score how close the model got to it.

This post builds the smallest useful version of that: a labeled set of examples, a scoring function, and a pytest test that turns "roughly correct most of the time" into a number you can gate CI on.

## Anatomy of an eval

Every eval has four parts, whether it's five examples or five thousand:

1. **Input**: what you send the model.
2. **Expected output**: what a correct response looks like, or the criteria a response must satisfy.
3. **Scorer**: a function that compares actual output to expected output and returns a score.
4. **Threshold**: the bar the aggregate score has to clear for the eval to pass.

The scorer is where evals get interesting. For closed-form answers (a category, a label, a yes/no) exact match works. For open-ended text it doesn't, and reaching for a full LLM-judge on day one is overkill. Start with the cheapest scorer that's honest about what it's measuring.

## A closed-form eval: ticket classification

Say you have an LLM call that routes support tickets into categories. That's closed-form: exact match is the right scorer.

```python
# eval/classify.py
import anthropic

client = anthropic.Anthropic()

CATEGORIES = ["billing", "bug", "feature-request", "account-access"]

def classify_ticket(text: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=20,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=f"Classify the support ticket into exactly one category: {', '.join(CATEGORIES)}. Respond with only the category name.",
        messages=[{"role": "user", "content": text}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip().lower()
```

```python
# eval/golden.py
GOLDEN_TICKETS = [
    {"text": "I was charged twice for my subscription this month.", "expected": "billing"},
    {"text": "The export button does nothing when I click it.", "expected": "bug"},
    {"text": "Could you add dark mode to the dashboard?", "expected": "feature-request"},
    {"text": "I can't log in: it says my password is wrong but I just reset it.", "expected": "account-access"},
    {"text": "Why does my invoice show a different total than the pricing page?", "expected": "billing"},
]
```

## The eval runner

```python
# tests/test_classify_eval.py
from eval.classify import classify_ticket
from eval.golden import GOLDEN_TICKETS

ACCURACY_FLOOR = 0.8

def test_classification_accuracy():
    results = [
        {"expected": ex["expected"], "actual": classify_ticket(ex["text"]), "text": ex["text"]}
        for ex in GOLDEN_TICKETS
    ]
    correct = sum(1 for r in results if r["actual"] == r["expected"])
    accuracy = correct / len(results)

    failures = [r for r in results if r["actual"] != r["expected"]]
    assert accuracy >= ACCURACY_FLOOR, (
        f"Accuracy {accuracy:.0%} below floor {ACCURACY_FLOOR:.0%}. "
        f"Misclassified: {[(r['text'][:40], r['expected'], r['actual']) for r in failures]}"
    )
```

Notice the assertion is on the **aggregate**, not per-example. A single misclassified ticket in a five-example set is noise; asserting per-example would make the suite flaky for the wrong reason. The threshold is the contract: individual examples are diagnostic detail you only need when the threshold trips.

## When exact match isn't enough

Most interesting LLM calls don't produce a category: they produce prose. "Summarize this ticket in one sentence" has no single correct string to match against. Exact match will fail every reasonable paraphrase, which makes it worse than useless: it fails loudly on correct output, so people start ignoring it.

The honest floor for open-ended text is **keyword/criteria coverage**: check that the response contains the facts it must contain, without requiring exact phrasing:

```python
# eval/summarize.py
def summarize_ticket(text: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system="Summarize the support ticket in one sentence.",
        messages=[{"role": "user", "content": text}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()
```

```python
# eval/golden_summaries.py
GOLDEN_SUMMARIES = [
    {
        "text": "I was charged $49 twice on the 3rd, my card statement shows two identical charges.",
        "must_mention": ["charged", "twice"],
    },
    {
        "text": "Clicking export on the reports page just spins forever and nothing downloads.",
        "must_mention": ["export"],
    },
]
```

```python
# tests/test_summarize_eval.py
from eval.summarize import summarize_ticket
from eval.golden_summaries import GOLDEN_SUMMARIES

COVERAGE_FLOOR = 0.8

def test_summary_covers_required_facts():
    scores = []
    for ex in GOLDEN_SUMMARIES:
        summary = summarize_ticket(ex["text"]).lower()
        hits = sum(1 for kw in ex["must_mention"] if kw in summary)
        scores.append(hits / len(ex["must_mention"]))

    coverage = sum(scores) / len(scores)
    assert coverage >= COVERAGE_FLOOR, f"Keyword coverage {coverage:.0%} below floor {COVERAGE_FLOOR:.0%}"
```

Keyword coverage is crude: it can't tell "charged twice" from "not charged twice," and it says nothing about tone, length, or whether the summary is actually readable. It's a floor, not a quality bar. It catches the failure mode that matters most in practice: a summary that silently drops the one fact the ticket was actually about.

## What comes next

For output where correctness is genuinely a judgment call ("is this explanation clear," "does this response sound appropriately empathetic") keyword coverage runs out fast, and the honest next step is scoring with a second model against an explicit rubric. That's a bigger topic on its own (rubric design, and the specific ways LLM judges lie to you) covered in a later post on [building an LLM-as-judge pipeline](/articles/llm-as-judge-eval-pipeline/).

## Takeaways

- A regression harness and an eval answer different questions: drift detection doesn't need ground truth, correctness scoring does.
- Match the scorer to the output shape: exact match for closed-form answers, keyword/criteria coverage for open-ended text, semantic judging only when you actually need it.
- Assert on the aggregate score against a threshold, not on individual examples: that's what keeps the suite meaningful instead of flaky.
