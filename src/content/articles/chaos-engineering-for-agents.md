---
title: "Chaos Monkey for Agents: Breaking Tool Calls on Purpose"
description: "The Testing LangGraph Agents post proved the graph is wired correctly under ideal conditions. Real tools fail: timeouts, malformed responses, outages. Inject those failures on purpose and verify the agent degrades gracefully instead of crashing or looping forever."
pubDate: 2026-07-23
category: reliability
tags: ["LangGraph", "chaos engineering", "Python"]
---

The [LangGraph testing post](/articles/testing-langgraph-agents/) proved the agent's graph is wired correctly: the right node runs, the right tool gets called, the loop terminates on a clean answer. All of that assumed the tool actually works. In production, tools fail: a weather API times out, a search backend returns malformed JSON, a rate limit kicks in mid-conversation. An agent that's never been tested against a failing tool has an unknown, and probably bad, answer to "what happens then."

**What it is:** chaos engineering's core method (deliberately inject a failure and observe the system's response) applied to an agent's tool calls instead of infrastructure.

**The problem it solves:** "the happy path works" tells you nothing about whether the agent crashes, hangs, or silently corrupts state the one time a tool call fails. That's exactly the case that shows up in production and never shows up in a demo.

## How it works

### Injecting the failure

```python
# agent/chaos_tools.py
import random

class ToolFailureInjected(Exception):
    """Raised by a chaos-wrapped tool to simulate a real backend failure."""

def flaky(fn, failure_rate: float):
    def wrapped(*args, **kwargs):
        if random.random() < failure_rate:
            raise ToolFailureInjected(f"{fn.__name__} failed (simulated {failure_rate:.0%} failure rate)")
        return fn(*args, **kwargs)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped
```

### The agent, with one flaky tool

`ToolNode` has built-in error handling: by default it catches an exception raised inside a tool and returns it to the model as a `ToolMessage` instead of letting it crash the graph. That default is what makes the rest of this testable at all: a failing tool becomes information the agent can react to, not a stack trace.

```python
# agent/chaos_graph.py
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.chaos_tools import flaky


def _get_weather(location: str) -> str:
    """Look up the current weather for a city."""
    return f"It's 68°F and clear in {location}."


get_weather = tool(flaky(_get_weather, failure_rate=0.5))
TOOLS = [get_weather]
model = init_chat_model("claude-opus-4-8").bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def call_model(state: State) -> dict:
    system = SystemMessage(
        "You are a support assistant. If a tool call fails, apologize once and tell "
        "the user you're unable to complete the request right now: do not retry silently forever."
    )
    response = model.invoke([system, *state["messages"]])
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", ToolNode(TOOLS))  # handle_tool_errors=True by default
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()
```

The system prompt explicitly tells the model to stop after a failure instead of retrying forever: that instruction is itself something worth testing, not just trusting, which is exactly what the next section does.

## Common issues

A model can (and sometimes will) treat a tool failure the same way it treats an ambiguous result: try again. Left unchecked, "call a tool, see it fail, call it again" can loop for as long as the graph lets it: burning tokens and latency on a request that was never going to succeed. Graceful degradation isn't automatic just because `ToolNode` doesn't crash; the agent still needs a path to actually stop.

The other common gap is treating "doesn't crash" as the whole bar. A graph that survives a tool failure but returns an empty or nonsensical final message hasn't actually degraded gracefully: it's just failed in a way that didn't raise an exception.

## What to test, and how

**The worst case: a model that never gives up.** LangGraph's `recursion_limit` is the safety net under an agent that keeps retrying: cap the steps and assert the graph trips the limit instead of running forever:

```python
# tests/test_chaos_structural.py
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

import agent.chaos_graph as chaos_graph


class AlwaysRetryModel:
    """Worst case: a model that keeps calling the tool no matter how many times it fails."""
    def invoke(self, messages):
        return AIMessage(content="", tool_calls=[
            {"name": "get_weather", "args": {"location": "Boston"}, "id": f"call_{len(messages)}"}
        ])


def test_recursion_limit_stops_an_infinite_retry_loop(monkeypatch):
    monkeypatch.setattr(chaos_graph, "model", AlwaysRetryModel())

    try:
        chaos_graph.graph.invoke(
            {"messages": [HumanMessage("What's the weather in Boston?")]},
            config={"recursion_limit": 6},
        )
        assert False, "Expected the graph to hit its recursion limit"
    except GraphRecursionError:
        pass  # this is the safety net catching a model that never gives up
```

**The well-behaved case: a model that stops after one failure.** Force the tool to fail exactly once, script a model that reads the error and gives up cleanly, and assert the final message actually reflects that:

```python
# tests/test_chaos_recovery.py
from langchain.messages import AIMessage, HumanMessage, ToolMessage

import agent.chaos_graph as chaos_graph


class GivesUpAfterFailureModel:
    """A well-behaved model: sees the tool error and stops instead of retrying forever."""
    def __init__(self):
        self._called = False

    def invoke(self, messages):
        if not self._called:
            self._called = True
            return AIMessage(content="", tool_calls=[
                {"name": "get_weather", "args": {"location": "Boston"}, "id": "call_1"}
            ])
        return AIMessage(content="I wasn't able to check the weather right now: please try again shortly.")


def test_graph_recovers_gracefully_from_a_single_tool_failure(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.0)  # force the tool to fail this one time
    monkeypatch.setattr(chaos_graph, "model", GivesUpAfterFailureModel())

    result = chaos_graph.graph.invoke(
        {"messages": [HumanMessage("What's the weather in Boston?")]},
        config={"recursion_limit": 10},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "Expected the failed tool call to still produce a ToolMessage"
    assert "try again" in result["messages"][-1].content.lower()
```

Both tests are fully deterministic and API-free: same scripted-model pattern as the [structural tests in the LangGraph post](/articles/testing-langgraph-agents/), pointed at failure paths instead of the happy path.

Common issues these two tests are built to catch is worth stating for a broader takeaway: without the recursion-limit test, a pathological model can burn an unbounded budget retrying a dead tool; without the recovery test, "doesn't crash" can hide a final answer that's blank, garbled, or stuck mid-thought.

This only covers one failure shape: a tool that raises an exception synchronously. Real backends also hang without ever raising (a timeout, not an exception) and return malformed-but-valid data (a 200 response with a corrupted payload): both need different injection techniques (an async timeout wrapper, a mangled fixture response) and aren't exercised by anything here.

## Takeaways

- `ToolNode`'s default error handling turns a tool exception into information the agent can act on, but "doesn't crash" and "recovers gracefully" are two different claims, and only the second one is worth trusting without a test.
- Test the worst case explicitly: a model that never stops retrying needs a recursion limit to catch it, and that limit needs its own test, not just an assumption that it's configured.
- A scripted fake model that fails once and a scripted fake model that never gives up are two different, cheap, deterministic tests: write both.
