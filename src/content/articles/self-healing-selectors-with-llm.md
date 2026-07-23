---
title: "Teaching Your Test Suite to Heal Itself"
description: "A renamed CSS class shouldn't fail your test suite the same way a real regression does, but brittle selectors make sure it does anyway. A Playwright wrapper that asks an LLM for a replacement selector when one breaks, tested deterministically without a real browser."
pubDate: 2026-07-23
category: automation
tags: ["Playwright", "self-healing", "Python"]
---

Most UI test failures aren't regressions: they're a class name that got renamed, a `<div>` that became a `<button>`, a DOM restructure that had nothing to do with the feature the test was actually checking. That distinction matters to a human reading the failure and matters not at all to the CI run turning red, which is exactly why brittle selectors are the single biggest source of wasted test-maintenance time in most UI suites.

**What it is:** a locator wrapper that, when a selector fails to find its element, asks an LLM to propose a replacement from the current page's HTML and a plain-English description of what it's looking for: then retries once.

**The problem it solves:** it turns "the test suite is red because markup changed" from a maintenance chore into a self-correcting fallback, while still surfacing the healing event so a human can fix the source selector instead of relying on the fallback forever.

## How it works

### Proposing a replacement selector

```python
# healing/propose_selector.py
import anthropic

client = anthropic.Anthropic()

def propose_selector(page_html: str, intent: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=(
            "You are given the HTML of a web page and a description of an element a test needs "
            "to interact with. Propose a single CSS selector that targets that element in the "
            "current HTML. Prefer stable attributes (data-testid, aria-label, role) over classes "
            "or generated IDs. Respond with only the selector, nothing else."
        ),
        messages=[{
            "role": "user",
            "content": f"Element to find: {intent}\n\nCurrent page HTML:\n{page_html[:8000]}",
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()
```

### The resilient wrapper

```python
# healing/resilient_locator.py
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from healing.propose_selector import propose_selector

class SelectorHealingFailed(Exception):
    pass

def resilient_click(page: Page, selector: str, intent: str, timeout_ms: int = 3000) -> str:
    """Click `selector`. If it can't be found, ask an LLM to propose a replacement based on
    `intent` and the current page HTML, then retry once. Returns whichever selector actually
    worked, so callers can log it and update the source."""
    try:
        page.locator(selector).click(timeout=timeout_ms)
        return selector
    except PlaywrightTimeoutError:
        pass

    healed_selector = propose_selector(page.content(), intent)
    try:
        page.locator(healed_selector).click(timeout=timeout_ms)
    except PlaywrightTimeoutError as e:
        raise SelectorHealingFailed(
            f"Original selector {selector!r} failed and healed selector {healed_selector!r} also failed"
        ) from e

    return healed_selector
```

Callers get the selector that actually worked back, not just a boolean: that return value is what makes the healing event visible instead of invisible.

## Common issues

**Silent healing becomes permanent tech debt.** If nothing surfaces a heal event, the suite keeps passing forever on a selector that's technically wrong in the source: nobody ever goes back and fixes `#old-submit-btn`. Log every heal (selector, page, timestamp) somewhere a human actually looks, and treat a rising heal rate as a signal the source selectors need a real update, not proof the fallback is working well.

**A vague `intent` can heal to the wrong element.** "The submit button" on a page with two submit buttons is exactly the kind of ambiguity that produces a selector matching the wrong one: the test then clicks something, "passes," and verifies nothing. Write `intent` as specifically as you'd want a human to describe the element: "the submit button in the payment form," not "the submit button."

**This adds a real-time LLM call to the failure path of an already-slow suite.** Fine as an occasional fallback; a bad sign if it's firing on every run: see the budget check below.

## What to test, and how

The fallback logic is a pure control-flow question: does it retry once, does it return the right selector, does it fail loudly when both attempts miss, and none of that needs a real browser or a real API call to verify. Fake the two dependencies and test the logic directly:

```python
# tests/test_resilient_locator.py
import healing.resilient_locator as healing_module
from healing.resilient_locator import resilient_click, SelectorHealingFailed


class FakeLocator:
    def __init__(self, should_succeed):
        self.should_succeed = should_succeed

    def click(self, timeout=None):
        if not self.should_succeed:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            raise PlaywrightTimeoutError("element not found")


class FakePage:
    def __init__(self, working_selector):
        self.working_selector = working_selector

    def locator(self, selector):
        return FakeLocator(should_succeed=(selector == self.working_selector))

    def content(self):
        return "<button data-testid='submit-order'>Submit</button>"


def test_heals_to_a_working_selector_when_the_original_breaks(monkeypatch):
    monkeypatch.setattr(healing_module, "propose_selector", lambda html, intent: "[data-testid=submit-order]")

    page = FakePage(working_selector="[data-testid=submit-order]")
    used = resilient_click(page, selector="#old-submit-btn", intent="the order submit button")

    assert used == "[data-testid=submit-order]"


def test_raises_when_both_original_and_healed_selector_fail(monkeypatch):
    monkeypatch.setattr(healing_module, "propose_selector", lambda html, intent: "#still-wrong")

    page = FakePage(working_selector="[data-testid=submit-order]")
    try:
        resilient_click(page, selector="#old-submit-btn", intent="the order submit button")
        assert False, "expected SelectorHealingFailed"
    except SelectorHealingFailed:
        pass
```

**Then, a suite-level budget check** against the healing log: the thing that catches "healing is quietly propping up a suite full of stale selectors" before it becomes normal:

```python
# tests/test_healing_budget.py
import json
from pathlib import Path

MAX_HEAL_RATE = 0.05  # no more than 5% of clicks in a run should need healing

def test_healing_rate_stays_low(tmp_path_or_real_log=Path("logs/healing_events.jsonl")):
    if not tmp_path_or_real_log.exists():
        return  # no runs logged yet
    events = [json.loads(line) for line in tmp_path_or_real_log.read_text().splitlines()]
    total_clicks = sum(1 for e in events if e["type"] in ("click", "heal"))
    heals = sum(1 for e in events if e["type"] == "heal")
    if total_clicks == 0:
        return
    assert heals / total_clicks <= MAX_HEAL_RATE, f"Heal rate {heals/total_clicks:.1%} exceeds {MAX_HEAL_RATE:.0%}: selectors need real fixes"
```

## Takeaways

- Return (and log) the selector that actually worked: a wrapper that swallows heal events into a plain boolean turns a maintenance signal into permanent hidden debt.
- Test the retry logic with fakes, not a real browser and a real model call: it's pure control flow and deserves a fast, deterministic test like everything else in this series.
- Track a heal rate across runs and budget it, the same way the [cost/latency post](/articles/cost-latency-regression-testing/) budgets tokens: a rising heal rate is a maintenance signal, not evidence the fallback is doing its job well.
