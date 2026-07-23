---
title: "Building an LLM-as-Judge Eval Pipeline (and Why You Shouldn't Trust It Blindly)"
description: "Keyword coverage runs out the moment 'correct' becomes a judgment call. A rubric-based judge pipeline with structured output — plus the specific ways judge models lie to you, and how to measure it before you trust one in CI."
pubDate: 2026-07-23
category: evaluation
tags: ["LLM evals", "LLM-as-judge", "Python"]
---

Keyword coverage, from the [first eval post](/articles/write-your-first-llm-eval/), gets you surprisingly far — it catches the failure mode that matters most, a response silently dropping the one fact it was supposed to contain. But it has a ceiling. "Is this explanation actually clear," "does this response sound appropriately empathetic," "did the agent solve the problem or just acknowledge it" — none of those are keyword-checkable. They're judgment calls, and the honest way to automate a judgment call is to have a second model make it, against an explicit rubric, every time.

That second model is a **judge**. It's also the eval layer people trust the most and should trust the least, without calibration. This post builds the pipeline, then spends real time on why.

## The rubric

A rubric is a list of independently gradeable yes/no criteria — not "is this response good," which a judge will happily rate on vibes, but specific, checkable claims:

```python
# eval/judge.py
RUBRIC = [
    {"id": "accurate", "criterion": "The response's factual claims are consistent with the source ticket — no invented details."},
    {"id": "complete", "criterion": "The response addresses the user's actual question or problem, not a tangential one."},
    {"id": "concise", "criterion": "The response is no longer than necessary — no padding, no repeated points."},
]
```

Each criterion should be answerable by looking at the ticket and the response alone, with no outside context. If two people would disagree on whether a response passes a given criterion, the criterion is too vague — narrow it before you hand it to a judge, not after.

## The judge call, with structured output

The judge's whole job is producing a machine-parseable verdict, so the response format matters as much as the prompt. Rather than asking for JSON and hoping, constrain the response with `output_config.format` so it's guaranteed to match a schema:

```python
import json
import anthropic

client = anthropic.Anthropic()

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "passes": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "passes", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}

def judge_response(ticket: str, response: str) -> dict:
    rubric_text = "\n".join(f"- {r['id']}: {r['criterion']}" for r in RUBRIC)
    result = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
        },
        system=(
            "You are grading a support-ticket response against a rubric. "
            "Score every criterion independently and explain each score in one sentence.\n\n"
            f"Rubric:\n{rubric_text}"
        ),
        messages=[{
            "role": "user",
            "content": f"Ticket:\n{ticket}\n\nResponse to grade:\n{response}",
        }],
    )
    text = "".join(b.text for b in result.content if b.type == "text")
    return json.loads(text)
```

Requiring a one-sentence `reason` per criterion isn't decoration — a judge that has to justify a score in writing produces a materially better score than one that just emits `true`/`false`. It's also what makes a failure actionable: `"complete": false, "reason": "Response says the charge will be reviewed but never confirms whether a refund is coming"` tells you exactly what to fix. A bare `false` tells you nothing.

## Wiring it into the eval suite

```python
# tests/test_judge_eval.py
from eval.golden_tickets import GOLDEN_TICKET_TEXTS
from eval.support_agent import answer_ticket
from eval.judge import judge_response

PASS_RATE_FLOOR = 0.85

def test_responses_pass_rubric():
    total, passed = 0, 0
    failures = []
    for ticket in GOLDEN_TICKET_TEXTS:
        response = answer_ticket(ticket)
        result = judge_response(ticket, response)
        for score in result["scores"]:
            total += 1
            if score["passes"]:
                passed += 1
            else:
                failures.append((ticket[:40], score["id"], score["reason"]))

    pass_rate = passed / total
    assert pass_rate >= PASS_RATE_FLOOR, f"Rubric pass rate {pass_rate:.0%} < {PASS_RATE_FLOOR:.0%}. Failures: {failures}"
```

Same shape as every eval on this blog so far: aggregate a per-criterion pass rate across the golden set, assert against a floor, surface the specific failures with their reasons when it trips.

## Why you shouldn't trust it blindly

Everything above works. It also has three specific, well-documented failure modes that a plausible-looking pass rate can hide entirely:

- **Verbosity bias.** Judges systematically rate longer responses higher, independent of whether the extra length adds anything. A response padded with hedges and caveats can out-score a shorter, more direct one that says the same thing.
- **Position bias.** If a judge ever compares two responses side by side (A/B grading instead of the single-response rubric grading above), the order they're presented in measurably shifts the verdict. Rubric grading like the code above sidesteps this by never showing the judge two candidates at once — worth knowing if you're tempted to add comparative grading later.
- **Self-preference bias.** A model judging output from its own family tends to rate it slightly higher than equivalent output from a different model. If you're using a judge to compare *across* model providers, this alone can decide the comparison.

None of these show up in a single pass-rate number. They show up as a judge that's quietly, consistently wrong in one direction — which is worse than a judge that's randomly wrong, because it doesn't average out.

### Run-to-run inconsistency

Separately from bias, judges aren't perfectly deterministic even at low effort. Run the same judge on the same input twice and check:

```python
# scripts/check_judge_stability.py
from eval.judge import judge_response

ticket = "I was charged twice for my subscription this month."
response = "Sorry about that — I've refunded the duplicate charge, you'll see it in 3-5 business days."

first = judge_response(ticket, response)
second = judge_response(ticket, response)

for a, b in zip(first["scores"], second["scores"]):
    if a["passes"] != b["passes"]:
        print(f"UNSTABLE: {a['id']} — first={a['passes']} second={b['passes']}")
```

If a criterion flips between runs on identical input, that criterion's phrasing is ambiguous enough that the judge is guessing, not grading — a signal to rewrite it more narrowly, not a bug to route around.

### Calibrate before you trust it

The only real fix for all of the above is measurement: before a judge gates CI, check how often it agrees with a human on a small labeled sample.

```python
# scripts/calibrate_judge.py
# eval/human_labels.jsonl: {"ticket": ..., "response": ..., "criterion_id": ..., "human_passes": true|false}
import json
from pathlib import Path

from eval.judge import judge_response

agree, total = 0, 0
for line in Path("eval/human_labels.jsonl").read_text().splitlines():
    row = json.loads(line)
    result = judge_response(row["ticket"], row["response"])
    judge_score = next(s for s in result["scores"] if s["id"] == row["criterion_id"])
    total += 1
    if judge_score["passes"] == row["human_passes"]:
        agree += 1

print(f"Judge/human agreement: {agree / total:.0%} ({agree}/{total})")
```

Thirty to fifty hand-labeled examples is enough to get a real signal. An agreement rate in the 90s means the judge is trustworthy for that criterion; anywhere much lower means you're gating CI on something closer to a coin flip with extra steps, no matter how confident the pass-rate number looks. Re-run this calibration whenever you change the rubric wording or swap the judge model — both can shift agreement without changing anything else about the pipeline.

## Takeaways

- A rubric of specific, independently-gradeable criteria produces a usable judge; "rate this 1–10" produces vibes with a number attached.
- Structured output (`output_config.format` with a JSON schema) turns the judge into something you can actually assert on in a test, instead of hoping it formatted its answer correctly.
- Verbosity bias, position bias, and self-preference bias are real and won't show up in an aggregate pass rate — calibrate the judge against a small human-labeled sample before you let it gate anything, and re-check that calibration whenever the rubric or the judge model changes.
