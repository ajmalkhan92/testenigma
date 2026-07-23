---
title: "AI Wrote the Test Cases: Now Who Tests the Test Cases?"
description: "Turning a user story into Gherkin scenarios with an LLM sounds like free QA coverage. Ungraded, it's also a great way to ship a test suite that looks thorough and only covers the happy path. A coverage rubric for the generator itself."
pubDate: 2026-07-23
category: automation
tags: ["test generation", "BDD", "LLM evals"]
---

"Turn this user story into test scenarios" is one of the more tempting LLM-for-QA use cases: feed in a user story, get Gherkin scenarios back, save the hour of manually writing them. It's also a great way to quietly ship a test suite that *looks* thorough (nicely formatted Given/When/Then, plausible-sounding scenario titles) while only covering the happy path, because nothing checked whether the generator actually thought about anything else.

**What it is:** a coverage rubric graded against generated test scenarios: checking the *generator*, not just trusting its output because it's formatted correctly.

**The problem it solves:** a test-case generator that's 100% happy-path and 0% negative/boundary cases produces output that reads as complete. Nobody catches that by skimming it; you catch it by grading for the categories that are supposed to be there.

## How it works

### Generating scenarios from a user story

```python
# gen/generate_scenarios.py
import anthropic

client = anthropic.Anthropic()

def generate_scenarios(user_story: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=600,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=(
            "Convert the user story into Gherkin test scenarios (Given/When/Then). "
            "Cover the happy path, at least one negative/error case, and at least one boundary case. "
            "Output only the Gherkin scenarios."
        ),
        messages=[{"role": "user", "content": user_story}],
    )
    return "".join(b.text for b in response.content if b.type == "text")
```

### A golden set of user stories

```python
# gen/golden_stories.py
GOLDEN_STORIES = [
    "As a user, I want to reset my password via email so that I can regain access to my account if I forget it.",
    "As a shopper, I want to apply a discount code at checkout so that I pay the reduced price.",
]
```

### Grading the generated scenarios for coverage

The generation prompt above already *asks* for happy-path, negative, and boundary coverage, but a system prompt is an instruction, not a guarantee. Verifying it happened is a separate step, using the same structured-judge pattern from the [LLM-as-judge post](/articles/llm-as-judge-eval-pipeline/):

```python
# gen/coverage_judge.py
import json
import anthropic

client = anthropic.Anthropic()

COVERAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_happy_path": {"type": "boolean"},
        "has_negative_or_error_path": {"type": "boolean"},
        "has_boundary_case": {"type": "boolean"},
        "scenario_count": {"type": "integer"},
        "notes": {"type": "string"},
    },
    "required": ["has_happy_path", "has_negative_or_error_path", "has_boundary_case", "scenario_count", "notes"],
    "additionalProperties": False,
}

def grade_coverage(user_story: str, scenarios: str) -> dict:
    result = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=500,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": COVERAGE_SCHEMA},
        },
        system=(
            "You are grading generated Gherkin test scenarios against a user story for coverage. "
            "Check whether the scenarios include: a happy path, a negative/error path, and a boundary "
            "case. Count distinct scenarios, not steps: two scenarios that test the same path with "
            "only cosmetic differences count as one."
        ),
        messages=[{"role": "user", "content": f"User story:\n{user_story}\n\nGenerated scenarios:\n{scenarios}"}],
    )
    text = "".join(b.text for b in result.content if b.type == "text")
    return json.loads(text)
```

That last instruction (count distinct scenarios, not near-duplicates as separate ones) exists because of exactly the failure mode below.

## Common issues

**Scenario count inflation.** A generator under instruction to "be thorough" will sometimes produce three scenarios that are cosmetically different phrasings of the same happy path, padding a naive scenario-count metric without adding real coverage. Grading for named *categories* (happy/negative/boundary) rather than a raw count sidesteps this: count is a vanity metric here, category coverage is the real one.

**The negative-path blind spot.** Generators trained on far more happy-path examples than error-path ones tend to under-produce negative and boundary cases by default, even when explicitly asked: which is exactly why this needs a *separate* grading pass instead of trusting the generation prompt's instructions to have worked. Belt and suspenders: ask for it in the prompt, then verify it actually happened.

**Gherkin text isn't a test.** This grades the shape of the generated scenarios (do the right categories exist) not whether someone (or something) has actually wired up step definitions that make them executable. Generated Gherkin is a draft a human still reviews and implements, not a finished automated suite.

## What to test, and how

```python
# tests/test_generated_coverage.py
from gen.coverage_judge import grade_coverage
from gen.generate_scenarios import generate_scenarios
from gen.golden_stories import GOLDEN_STORIES

COVERAGE_FLOOR = 0.8  # fraction of stories that must hit each category

def test_generated_scenarios_cover_happy_negative_and_boundary_cases():
    grades = [grade_coverage(story, generate_scenarios(story)) for story in GOLDEN_STORIES]

    happy_rate = sum(g["has_happy_path"] for g in grades) / len(grades)
    negative_rate = sum(g["has_negative_or_error_path"] for g in grades) / len(grades)
    boundary_rate = sum(g["has_boundary_case"] for g in grades) / len(grades)

    assert happy_rate >= COVERAGE_FLOOR, f"Happy-path coverage {happy_rate:.0%} below floor"
    assert negative_rate >= COVERAGE_FLOOR, f"Negative-path coverage {negative_rate:.0%} below floor"
    assert boundary_rate >= COVERAGE_FLOOR, f"Boundary-case coverage {boundary_rate:.0%} below floor"
```

| Category | Coverage rate | Status |
|---|---|---|
| Happy path | 100% | pass |
| Negative / error path | 100% | pass |
| Boundary case | 50% | below floor |

Three separate thresholds, not one blended score: a generator that's perfect on happy-path and negative-path but consistently skips boundary cases fails exactly the assertion that names the gap, instead of passing on a decent-looking average.

## Takeaways

- Grade generated test cases for named coverage categories, not a raw scenario count: count is gameable by near-duplicate scenarios that add no real coverage.
- The negative-path blind spot is real and systematic, not occasional: verify it with a separate grading pass rather than trusting the generation prompt's instructions alone.
- Generated Gherkin is a draft, not a finished suite: this eval checks whether the draft covers the right ground, not whether it's wired up to run.
