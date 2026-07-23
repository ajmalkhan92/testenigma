---
title: "Building a Golden Dataset for LLM Evals"
description: "Five examples hardcoded in a test file works until it doesn't. How to structure, version, and grow a golden dataset that your evals actually scale on."
pubDate: 2026-07-23
category: evaluation
tags: ["LLM evals", "data", "Python"]
---

The [last post](/articles/write-your-first-llm-eval/) hardcoded five examples directly in a Python list. That's fine for a demo and wrong for anything you're actually going to maintain. Five examples can't represent a real input distribution, there's no record of *why* each example is in there, and adding a sixth means editing source code instead of just... adding data. A **golden dataset** fixes all three: it's a versioned, structured set of labeled examples that your evals load instead of hardcode — the actual asset the evals are built on top of.

## What goes in a golden example

More than input/expected-output. Each example should carry enough metadata to answer "why is this here" a year from now:

```python
# golden/schema.py
from dataclasses import dataclass, field

@dataclass
class GoldenExample:
    id: str                     # stable identifier — never reused, never renumbered
    input: str                  # what gets sent to the model
    expected: str                # expected output, or the key fact(s) it must contain
    category: str                # billing, bug, feature-request, account-access, ...
    source: str                  # "synthetic" | "prod-incident-4821" | "support-review-2026-06"
    difficulty: str = "normal"   # "easy" | "normal" | "hard"
    tags: list[str] = field(default_factory=list)
```

`source` is the field people skip and regret skipping. An example pulled from a real production failure is worth more than a synthetic one you wrote to look plausible — it's proof the model actually gets this wrong in the wild, not a guess that it might. Tag every example that came from a real bug report with the ticket or incident ID. When someone asks "do we actually have evidence this matters," that field is the answer.

## Storage format

JSONL — one example per line, diffable, appendable without touching existing rows:

```jsonl
{"id": "billing-001", "input": "I was charged twice for my subscription this month.", "expected": "billing", "category": "billing", "source": "synthetic", "difficulty": "easy", "tags": ["double-charge"]}
{"id": "bug-001", "input": "The export button does nothing when I click it.", "expected": "bug", "category": "bug", "source": "prod-incident-4821", "difficulty": "easy", "tags": ["export", "ui"]}
{"id": "access-001", "input": "I can't log in — it says my password is wrong but I just reset it.", "expected": "account-access", "category": "account-access", "source": "prod-incident-5103", "difficulty": "hard", "tags": ["password-reset", "auth"]}
```

Committed to the repo like any other test fixture. A pull request that adds five golden examples is reviewable the same way a pull request that adds five unit test cases is — someone can look at the diff and ask "is this actually representative" before it merges.

## Loading and validating it

```python
# golden/loader.py
import json
from pathlib import Path

from golden.schema import GoldenExample

def load_golden_set(path: Path) -> list[GoldenExample]:
    examples = []
    seen_ids = set()
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example = GoldenExample(**row)
            if example.id in seen_ids:
                raise ValueError(f"Duplicate id {example.id!r} at line {line_no}")
            seen_ids.add(example.id)
            examples.append(example)
    return examples
```

Fail loudly on a duplicate ID at load time, not silently at eval time. A duplicate is almost always a copy-paste mistake that would otherwise double-weight one example in every aggregate score without anyone noticing.

## Scoring against it, with a breakdown by category

A single aggregate accuracy number hides where the model actually struggles. Breaking scores out by `category` (or `difficulty`, or `tags`) turns "82% accuracy" into "82% accuracy, but 40% on account-access tickets specifically" — which is the version of that number someone can actually act on.

```python
# tests/test_golden_eval.py
from collections import defaultdict
from pathlib import Path

from eval.classify import classify_ticket
from golden.loader import load_golden_set

GOLDEN_PATH = Path("golden/tickets.jsonl")
ACCURACY_FLOOR = 0.8
CATEGORY_FLOOR = 0.6  # no single category allowed to fall further than this

def test_classification_accuracy_by_category():
    examples = load_golden_set(GOLDEN_PATH)
    by_category: dict[str, list[bool]] = defaultdict(list)

    for ex in examples:
        actual = classify_ticket(ex.input)
        by_category[ex.category].append(actual == ex.expected)

    overall = sum(sum(v) for v in by_category.values()) / len(examples)
    breakdown = {cat: sum(hits) / len(hits) for cat, hits in by_category.items()}

    failing_categories = {cat: acc for cat, acc in breakdown.items() if acc < CATEGORY_FLOOR}

    assert overall >= ACCURACY_FLOOR, f"Overall accuracy {overall:.0%} < {ACCURACY_FLOOR:.0%}. Breakdown: {breakdown}"
    assert not failing_categories, f"Categories below floor: {failing_categories}"
```

Two assertions, two different failure signatures: the first catches broad regressions, the second catches a model that's fine on average but has quietly collapsed on one category — the kind of regression an aggregate-only check would let straight through.

| Category | Examples | Accuracy | Status |
|---|---|---|---|
| billing | 14 | 0.93 | pass |
| bug | 11 | 0.91 | pass |
| feature-request | 9 | 0.89 | pass |
| account-access | 8 | 0.50 | **below floor** |

That's the shape of report worth pasting into a PR description — not "evals passed," but exactly which slice of the input space regressed and by how much.

## Growing the dataset without overfitting to it

A golden dataset that never grows stops being useful the moment the model starts overfitting to its specific phrasing — high score on the dataset, unchanged (or worse) behavior on real traffic. Two habits keep that from happening:

- **Mine failures, don't invent them.** Every time a real user hits a case the model got wrong, that's a new golden example, tagged with its source incident. This is the single highest-signal way to grow the set — it's not guessing at edge cases, it's recording ones that already happened.
- **Hold a slice out.** Split the dataset into a set you optimize prompts against and a set you never look at while iterating — check the held-out slice only before shipping. If a prompt change improves the visible set but not the held-out one, that's a prompt tuned to the dataset's specific wording, not to the underlying task.

## Where this breaks down

A golden dataset is only as good as whoever wrote it — it encodes their assumptions about what a "correct" answer looks like, including their blind spots. A small dataset (tens of examples) has enough variance that a couple of flipped results can swing the aggregate several points, so treat small-sample accuracy numbers as directional, not precise. And a model can genuinely ace a hundred golden examples while failing on the long tail of real traffic those hundred examples don't cover — a golden dataset raises your confidence, it doesn't replace watching production.

## Takeaways

- Structure beats a hardcoded list the moment you need to know *why* an example exists, not just what it expects.
- JSONL, committed to the repo, reviewed like code — that's what makes a dataset something a team can actually grow together.
- Break scores out by category or tag; an aggregate number hides exactly the regression you need to see.
