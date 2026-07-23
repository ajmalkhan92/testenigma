---
title: "Can Your LLM Catch Its Own Bugs? Mutation Testing Meets AI"
description: "Classic mutation testing injects small bugs and checks whether your test suite catches them. Borrow the same method to benchmark an LLM code reviewer instead, and check its false-positive rate while you're at it."
pubDate: 2026-07-23
category: evaluation
tags: ["mutation testing", "code review", "Python"]
---

Mutation testing is an old idea from outside the LLM world: deliberately inject small bugs into working code (flip a comparison operator, off-by-one a boundary, drop a constant) and run your test suite against each mutant. A test suite that kills 100% of mutants is actually testing behavior; one that lets most of them survive is just executing lines without checking anything meaningful.

If you're using an LLM as a code reviewer (flagging bugs in a diff before a human looks at it) the exact same method answers a different, useful question: not "is my test suite any good," but "is my AI reviewer any good." Same mutants, same kill-rate scoring, different thing being graded.

**What it is:** mutation testing's inject-a-bug-and-measure-detection method, retargeted from grading a test suite to grading an AI code reviewer.

**The problem it solves:** "our AI reviewer looks helpful in demos" isn't a measurement. A kill rate against known mutants is.

## How it works

### The function under review

```python
# mutants/target.py
def calculate_discount(price: float, quantity: int) -> float:
    """Apply a 10% discount for orders of 5 or more items."""
    if quantity >= 5:
        return price * quantity * 0.9
    return price * quantity
```

### Generating mutants

Three classic mutation operators, applied as plain string substitutions against the source:

```python
# mutants/generator.py
from dataclasses import dataclass

@dataclass
class Mutant:
    id: str
    operator: str
    original: str
    mutated: str

ORIGINAL_SOURCE = '''def calculate_discount(price: float, quantity: int) -> float:
    """Apply a 10% discount for orders of 5 or more items."""
    if quantity >= 5:
        return price * quantity * 0.9
    return price * quantity
'''

MUTATIONS = [
    Mutant(
        id="boundary-flip",
        operator="relational operator replacement",
        original="if quantity >= 5:",
        mutated="if quantity > 5:",
    ),
    Mutant(
        id="discount-drop",
        operator="constant replacement",
        original="return price * quantity * 0.9",
        mutated="return price * quantity * 1.0",
    ),
    Mutant(
        id="negate-condition",
        operator="condition negation",
        original="if quantity >= 5:",
        mutated="if quantity < 5:",
    ),
]

def apply_mutant(mutant: Mutant) -> str:
    if mutant.original not in ORIGINAL_SOURCE:
        raise ValueError(f"Mutation target not found for {mutant.id}")
    return ORIGINAL_SOURCE.replace(mutant.original, mutant.mutated, 1)
```

Each of these silently changes behavior at the boundary (`quantity == 5`) without touching anything a casual glance would flag: exactly the shape of bug a reviewer is supposed to exist for.

### The reviewer

```python
# mutants/reviewer.py
import json
import anthropic

client = anthropic.Anthropic()

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "has_bug": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["has_bug", "explanation"],
    "additionalProperties": False,
}

def review_diff(original: str, modified: str) -> dict:
    result = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": REVIEW_SCHEMA},
        },
        system="You are a code reviewer. Compare the original and modified function. Determine whether the modification introduces a behavioral bug.",
        messages=[{
            "role": "user",
            "content": f"Original:\n```python\n{original}\n```\n\nModified:\n```python\n{modified}\n```",
        }],
    )
    text = "".join(b.text for b in result.content if b.type == "text")
    return json.loads(text)
```

## What to test, and how

### Detection rate

```python
# tests/test_mutation_detection.py
from mutants.generator import MUTATIONS, ORIGINAL_SOURCE, apply_mutant
from mutants.reviewer import review_diff

DETECTION_FLOOR = 0.8

def test_reviewer_catches_injected_bugs():
    missed = []
    for mutant in MUTATIONS:
        mutated_source = apply_mutant(mutant)
        verdict = review_diff(ORIGINAL_SOURCE, mutated_source)
        if not verdict["has_bug"]:
            missed.append((mutant.id, mutant.operator))

    detection_rate = 1 - len(missed) / len(MUTATIONS)
    assert detection_rate >= DETECTION_FLOOR, f"Detection rate {detection_rate:.0%} below floor. Missed: {missed}"
```

### The check nobody adds: false positives

A reviewer that flags `has_bug: true` on literally everything would ace the test above with a 100% detection rate, and be useless, because a reviewer whose every comment is a false alarm gets ignored within a week. Mutation testing alone can't catch that; you need a matching control case with *no* behavioral change:

```python
# tests/test_no_false_positive.py
from mutants.generator import ORIGINAL_SOURCE
from mutants.reviewer import review_diff

# Docstring wording only: zero behavior change.
HARMLESS_EDIT = ORIGINAL_SOURCE.replace("Apply a 10% discount", "Apply a ten percent discount")

def test_reviewer_does_not_flag_a_harmless_change():
    verdict = review_diff(ORIGINAL_SOURCE, HARMLESS_EDIT)
    assert not verdict["has_bug"], f"False positive on a no-op change: {verdict['explanation']}"
```

Detection rate and false-positive rate are the same precision/recall tradeoff from the [RAG evaluation post](/articles/evaluating-rag-pipelines/), applied to a reviewer instead of a retriever: report both, because a reviewer optimized for one alone is easy to build and not the thing you actually want.

### Reading the results by operator

| Operator | Mutants | Caught | Detection rate |
|---|---|---|---|
| relational operator replacement | 1 | 1 | 100% |
| constant replacement | 1 | 1 | 100% |
| condition negation | 1 | 0 | 0% |
| **False positives** | 1 harmless edit | 0 flagged |: |

A breakdown by operator, not just an aggregate, tells you *what kind* of bug the reviewer misses: in this shape, "negate the condition entirely" is a categorically different mistake from "shift the boundary by one," and a reviewer can be reliably good at one while blind to the other.

## Common issues

This is mutation testing borrowed for a different purpose than it was built for: classic mutation testing scores a *test suite's* ability to kill mutants by actually running them; this scores a *reviewer's* ability to spot mutants by reading a diff, with no execution involved. Both are useful, and they're not interchangeable: a reviewer that's excellent at reading code for likely bugs says nothing about whether your tests would actually catch a real one in production, and vice versa.

The operator set here is also small and hand-picked. Real mutation-testing tools generate dozens of operator types across a whole codebase automatically; three operators against one function is a fast, illustrative starting point, not a serious coverage claim. Scaling this to a real reviewer eval means running many operators across a representative sample of your actual codebase, not one toy function.

## Takeaways

- The mutation-testing method (inject a known bug, measure whether it's caught) transfers cleanly from grading test suites to grading an AI code reviewer.
- Detection rate alone is gameable by a reviewer that flags everything; pair it with a false-positive check on a harmless, no-op change.
- Break results down by mutation operator, not just the aggregate: it tells you which *category* of bug the reviewer actually misses, which is the part you can act on.
