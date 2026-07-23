---
title: "Turn It Loose: Building an Exploratory Testing Agent You Can Actually Trust"
description: "An agent that wanders your app clicking things sounds appealing right up until it reports 'looks fine' having tried the same two buttons fifty times. A minimal exploratory agent, plus the one check that actually verifies it's exploring: does it find a bug you know is there?"
pubDate: 2026-07-23
category: automation
tags: ["agents", "exploratory testing", "Python"]
---

Exploratory testing — a person wandering an app, trying weird inputs, poking at things a test plan wouldn't think to check — is valuable precisely because it's unscripted. That's also exactly what makes it hard to automate credibly: an agent that "explores" your app and comes back with "everything looks fine" is either telling you something real, or it clicked the same two buttons fifty times and never went anywhere near the actual bug. From the outside, those look identical.

**What it is:** an agent that picks actions from what's available on a page instead of following a fixed script, plus a verification method borrowed from [mutation testing](/articles/mutation-testing-meets-ai/) — seed a known bug, check whether exploration actually finds it.

**The problem it solves:** "the agent explored for 25 steps and reported no issues" is meaningless on its own. It's only meaningful once you know the agent reliably finds problems that are actually there.

## How it works

### A stand-in app to explore

A tiny in-memory state machine takes the place of a real browser here — the agent and verification code below only depend on `get_available_actions()` / `perform_action()`, so swapping this for Playwright driving a real page is a change to this one file, not to the agent.

```python
# explorer/app.py
from dataclasses import dataclass

@dataclass
class AppState:
    page: str = "cart"
    coupon_applied: bool = False
    total: float = 15.99
    crashed: bool = False
    crash_reason: str | None = None

PAGES = {
    "cart": ["view_item", "apply_coupon", "go_to_checkout"],
    "item": ["add_to_cart", "back_to_cart"],
    "checkout": ["apply_coupon", "confirm_order", "back_to_cart"],
}

def get_available_actions(state: AppState) -> list[str]:
    return PAGES[state.page]

def perform_action(state: AppState, action: str) -> AppState:
    if action == "view_item":
        return AppState(page="item", coupon_applied=state.coupon_applied, total=state.total)
    if action == "back_to_cart":
        return AppState(page="cart", coupon_applied=state.coupon_applied, total=state.total)
    if action == "add_to_cart":
        return AppState(page="cart", coupon_applied=state.coupon_applied, total=state.total + 19.99)
    if action == "go_to_checkout":
        return AppState(page="checkout", coupon_applied=state.coupon_applied, total=state.total)
    if action == "apply_coupon":
        if state.coupon_applied:
            # Seeded bug: the coupon should only be appliable once per order.
            return AppState(page=state.page, coupon_applied=True, total=state.total - 10,
                             crashed=True, crash_reason="coupon applied twice to the same order")
        return AppState(page=state.page, coupon_applied=True, total=state.total - 10)
    if action == "confirm_order":
        return AppState(page="checkout", coupon_applied=state.coupon_applied, total=state.total)
    raise ValueError(f"Unknown action: {action}")
```

There's a real bug seeded in there on purpose — applying the coupon twice should be blocked and isn't. That's the target the agent needs to find.

### The exploring agent

```python
# explorer/agent.py
import anthropic

from explorer.app import AppState, get_available_actions, perform_action

client = anthropic.Anthropic()

def choose_next_action(current_page: str, available_actions: list[str], tried_before: set[str]) -> str:
    untried = [a for a in available_actions if a not in tried_before]
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=50,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=(
            "You are an exploratory testing agent. You're on a page with a list of available "
            "actions. Prefer actions you haven't tried on this page before, to maximize coverage. "
            "Respond with only the action name, nothing else."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Current page: {current_page}\n"
                f"Available actions: {available_actions}\n"
                f"Already tried on this page: {sorted(tried_before) or 'none'}\n"
                f"Untried: {untried or 'all tried — pick any to test a repeat interaction'}"
            ),
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def explore(steps: int) -> dict:
    state = AppState()
    tried_by_page: dict[str, set[str]] = {}
    log = []

    for _ in range(steps):
        if state.crashed:
            log.append({"action": "CRASH", "reason": state.crash_reason})
            break

        available = get_available_actions(state)
        tried_here = tried_by_page.setdefault(state.page, set())
        action = choose_next_action(state.page, available, tried_here)

        if action not in available:
            continue  # model picked something invalid — skip rather than crash the harness

        tried_here.add(action)
        state = perform_action(state, action)
        log.append({"action": action, "page": state.page})

    return {"final_state": state, "log": log, "tried_by_page": tried_by_page}
```

The prompt nudges toward untried actions instead of hard-forcing them — that's deliberate. An agent that mechanically enumerates every action once and stops never tries the *sequences* that real bugs hide in, like applying the same coupon twice.

## Common issues

**Aimless wandering that looks like exploration.** With no pressure toward novelty, an LLM will often gravitate to the same one or two "safe" actions repeatedly — technically exploring, practically not. The `tried_before` bookkeeping and the system prompt's explicit preference for untried actions exist specifically to counter this; without them, "explored for 25 steps" can mean 25 repeats of the same click.

**Coverage without depth.** An agent that visits every action exactly once has broad coverage and zero depth — and the coupon bug in this example only appears on the *second* application of an action already tried. Breadth-only exploration would completely miss it. Real exploratory testing bugs disproportionately live in sequences and repeats, not first visits.

**"No crash" isn't the same as "no bug."** This example's verification is deliberately narrow — it only catches bugs that flip a `crashed` flag. A real target app can fail more quietly: wrong data displayed, a price silently miscalculated with no error raised, a state that's wrong but doesn't crash anything. An exploration agent needs assertions beyond "did it crash" to catch those, the same way a human exploratory tester is checking for wrongness, not just breakage.

## What to test, and how

The verification question isn't "did the agent do things" — it's "did the agent's exploration actually work," checked the same way [mutation testing](/articles/mutation-testing-meets-ai/) checks a code reviewer: seed a known defect, measure whether the process finds it.

```python
# tests/test_exploration.py
from explorer.agent import explore
from explorer.app import PAGES

STEP_BUDGET = 25
MIN_COVERAGE = 0.7  # fraction of all (page, action) pairs tried at least once

def test_agent_finds_the_seeded_bug_within_budget():
    result = explore(steps=STEP_BUDGET)
    assert result["final_state"].crashed, (
        f"Exploration ran {STEP_BUDGET} steps without finding the seeded double-coupon bug. "
        f"Log: {result['log']}"
    )


def test_agent_achieves_meaningful_coverage():
    result = explore(steps=STEP_BUDGET)
    total_actions = sum(len(actions) for actions in PAGES.values())
    tried_actions = sum(len(tried) for tried in result["tried_by_page"].values())
    coverage = tried_actions / total_actions
    assert coverage >= MIN_COVERAGE, f"Coverage {coverage:.0%} below floor {MIN_COVERAGE:.0%}"
```

Both assertions matter and check different things: the coverage test catches an agent that's stuck repeating a narrow loop; the bug-detection test catches an agent that has broad coverage but never revisits anything long enough to trigger a sequence-dependent bug. An agent that passes coverage but fails bug-detection is exploring wide and shallow — exactly the failure mode the depth point above describes.

Run this against every change to the agent's prompt or model, the same way any other eval in this series gets re-run on a prompt change — an exploration agent that used to reliably find the seeded bug and stops is a regression worth catching before it ships.

## Takeaways

- "The agent explored and found nothing" is not evidence of a clean app — it's ambiguous between a clean app and an agent that never actually explored anything meaningful.
- Verify the exploration process itself with a seeded, known bug — the same method as mutation testing, aimed at a test *process* instead of a code reviewer.
- Track coverage and bug-detection as two separate assertions; an agent can have broad coverage and still miss the sequence-dependent bugs that only show up on a repeat visit.
