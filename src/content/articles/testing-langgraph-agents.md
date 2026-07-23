---
title: "Testing LangGraph Agents"
description: "Agents add state, branching, and tool calls on top of everything that already made LLM output hard to test. A LangGraph support agent, and three layers of tests that don't require a live API key for the ones that matter most."
pubDate: 2026-07-23
category: automation
tags: ["LangGraph", "Python", "agents"]
---

Everything on this blog so far has tested a single LLM call: one input, one output, one thing to score. An **agent** is a loop — the model decides whether to call a tool, the tool result feeds back in, and the model decides again, until it has enough to answer. That loop is exactly what makes agents useful and exactly what makes them harder to test: there's state that persists across steps, branching that depends on what the model decided, and a failure can happen at any node in the graph, not just at the final answer.

This post builds a small [LangGraph](https://github.com/langchain-ai/langgraph) agent and three layers of tests for it — two of which need no API key and run in milliseconds, because the fast, deterministic layers are where most of an agent test suite's value actually lives.

## The agent

A support assistant with two tools: weather lookup and a docs search stub.

```python
# agent/graph.py
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def get_weather(location: str) -> str:
    """Look up the current weather for a city."""
    return f"It's 68°F and clear in {location}."


@tool
def search_docs(query: str) -> str:
    """Search internal documentation for a query."""
    return f"3 docs found for '{query}': onboarding.md, faq.md, changelog.md"


TOOLS = [get_weather, search_docs]
model = init_chat_model("claude-opus-4-8").bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def call_model(state: State) -> dict:
    system = SystemMessage("You are a support assistant. Use tools when you need current info.")
    response = model.invoke([system, *state["messages"]])
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", ToolNode(TOOLS))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()
```

<div class="mermaid-wrap">
<svg viewBox="0 0 560 240" role="img" aria-label="Flowchart: START leads to the agent node, which either loops to the tools node when tool calls are present and back, or ends when there are none.">
<defs>
<marker id="dg2arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/>
</marker>
</defs>
<style>
.dg2-node{fill:var(--card);stroke:var(--border-2);stroke-width:1.5}
.dg2-endpoint{fill:var(--bg-2);stroke:var(--border-2);stroke-width:1.5}
.dg2-text{fill:var(--text);font-family:var(--sans);font-size:13px}
.dg2-edge{stroke:var(--muted);stroke-width:1.5;fill:none;marker-end:url(#dg2arrow)}
.dg2-label{fill:var(--faint);font-family:var(--mono);font-size:10px}
</style>
<ellipse class="dg2-endpoint" cx="55" cy="52" rx="38" ry="24"/>
<text class="dg2-text" x="55" y="57" text-anchor="middle">START</text>
<path class="dg2-edge" d="M93,52 H148"/>
<rect class="dg2-node" x="150" y="28" width="200" height="48" rx="8"/>
<text class="dg2-text" x="250" y="48" text-anchor="middle">agent node</text>
<text class="dg2-text" x="250" y="64" text-anchor="middle">calls the model</text>
<path class="dg2-edge" d="M350,52 H418"/>
<text class="dg2-label" x="358" y="42">no tool_calls</text>
<ellipse class="dg2-endpoint" cx="460" cy="52" rx="34" ry="24"/>
<text class="dg2-text" x="460" y="57" text-anchor="middle">END</text>
<path class="dg2-edge" d="M230,76 V148"/>
<rect class="dg2-node" x="150" y="150" width="200" height="48" rx="8"/>
<text class="dg2-text" x="250" y="170" text-anchor="middle">tools node</text>
<text class="dg2-text" x="250" y="186" text-anchor="middle">runs the tool</text>
<text class="dg2-label" x="238" y="115">tool_calls present</text>
<path class="dg2-edge" d="M270,150 V78"/>
</svg>
</div>

`tools_condition` is a prebuilt LangGraph helper: it looks at the last message and routes to `"tools"` if the model asked for a tool call, or ends the graph if it didn't. That single function is also the reason this agent is testable at three different levels — it's a pure routing decision you can call directly with a fabricated message, with no model involved.

## Layer 1: test the routing logic without touching the model

`tools_condition` takes a state dict and returns a string. No LLM call, no network, no API key — just construct the message shapes it branches on and assert:

```python
# tests/test_routing.py
from langchain.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition


def test_routes_to_tools_when_model_requests_a_tool_call():
    state = {
        "messages": [
            HumanMessage("What's the weather in Boston?"),
            AIMessage(content="", tool_calls=[
                {"name": "get_weather", "args": {"location": "Boston"}, "id": "call_1"}
            ]),
        ]
    }
    assert tools_condition(state) == "tools"


def test_ends_when_model_answers_directly():
    state = {
        "messages": [
            HumanMessage("Hi there"),
            AIMessage(content="Hello! How can I help?"),
        ]
    }
    assert tools_condition(state) == "__end__"
```

## Layer 1, continued: test each tool as a plain function

A `@tool`-decorated function is still callable directly — test it the same way you'd test any function, independent of whether a model ever decides to call it:

```python
# tests/test_tools.py
from agent.graph import get_weather, search_docs


def test_get_weather_mentions_the_requested_city():
    result = get_weather.invoke({"location": "Boston"})
    assert "Boston" in result


def test_search_docs_returns_a_result_list():
    result = search_docs.invoke({"query": "password reset"})
    assert "docs found" in result
```

Both of these test files run in milliseconds, need zero budget, and never flake — they're the base of the [test pyramid](/articles/llm-test-pyramid/) applied to an agent specifically. Most bugs in an agent's control flow (wrong tool picked for a given routing state, a tool that crashes on an edge-case argument) get caught right here, before a single token gets generated.

## Layer 2: test the whole graph with a scripted fake model

The routing tests above check `tools_condition` in isolation. What they don't check is the *loop* — does the graph actually route back to the agent after a tool runs, and does the final state contain everything it should. For that, swap the real model for a fake that returns a fixed script of responses, so the whole graph runs deterministically with no API key:

```python
# tests/test_graph_structure.py
from langchain.messages import AIMessage, HumanMessage, ToolMessage

import agent.graph as agent_graph


class ScriptedModel:
    """Returns one prepared response per call, in order."""
    def __init__(self, responses):
        self._responses = iter(responses)

    def invoke(self, messages):
        return next(self._responses)


def test_graph_calls_weather_tool_then_answers(monkeypatch):
    scripted = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "get_weather", "args": {"location": "Boston"}, "id": "call_1"}
        ]),
        AIMessage(content="It's 68°F and clear in Boston."),
    ])
    monkeypatch.setattr(agent_graph, "model", scripted)

    result = agent_graph.graph.invoke({"messages": [HumanMessage("What's the weather in Boston?")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Boston" in tool_messages[0].content
    assert result["messages"][-1].content == "It's 68°F and clear in Boston."
```

This is the layer that catches wiring bugs — a missing edge, a node that doesn't return the right state shape, a tool result that never makes it back into the message list — without paying for a single real model call. The scripted model's second response only gets used if the graph actually loops back to the agent node after the tool runs, so the test is a genuine structural check, not just a shallow smoke test.

## Layer 3: the real model, sparingly

Everything above tests that the agent is *wired correctly*. None of it tests whether the real model actually decides to call `get_weather` when asked about the weather — that's a model-behavior question, not a graph-structure question, and it's the one thing only a real call can answer. Keep this layer small and explicitly gated:

```python
# tests/test_agent_e2e.py
import os
import pytest
from langchain.messages import HumanMessage, ToolMessage

from agent.graph import graph

pytestmark = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ, reason="requires a live API key"
)


def test_agent_calls_weather_tool_for_a_weather_question():
    result = graph.invoke({"messages": [HumanMessage("What's the weather in Boston right now?")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("Boston" in m.content for m in tool_messages)
```

This is layer 5 from the [test pyramid](/articles/llm-test-pyramid/) — expensive and slow relative to the two layers above it, so it runs nightly or pre-release, not on every commit.

## Where this breaks down

The scripted-model test proves the graph is wired correctly; it says nothing about whether the real model reaches for the right tool given real, messier phrasing — "what's it like outside in Boston" instead of "what's the weather in Boston." That's exactly the gap the golden-dataset and eval patterns from [earlier posts](/articles/golden-dataset-for-llm-evals/) exist to close, applied here to routing decisions instead of single-call outputs: a golden set of realistic queries, each labeled with which tool (if any) should get called, scored against the real model on a schedule — not on every commit.

The other real gap is multi-turn state. This example agent answers in one loop; a longer conversation accumulates message history across many turns, and bugs there — a tool result that should have been dropped from context, a system message that gets duplicated — don't show up in a single-turn structural test at all. Testing that requires driving the graph through a multi-turn script and asserting on state at each step, not just the final result.

## Takeaways

- Most of an agent's testable surface — routing logic, individual tools, graph wiring — doesn't require calling the real model at all. Test that surface first; it's fast, free, and where most control-flow bugs actually live.
- A scripted fake model turns "does the whole graph work" into a deterministic, zero-cost test, as long as you assert on structure (which tools got called, what ended up in state) rather than exact wording.
- Reserve real model calls for the one question only they can answer: does the model actually make the right decision on realistic input. Keep that layer small and scheduled, not on every push.
