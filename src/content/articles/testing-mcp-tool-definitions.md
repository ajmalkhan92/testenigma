---
title: "Schema or It Didn't Happen: Testing Your MCP Tool Definitions"
description: "An agent only knows what a tool does through its schema and description. If the schema promises something the handler doesn't actually do, the agent finds out mid-conversation: as a crash. Contract tests that catch the gap first."
pubDate: 2026-07-23
category: automation
tags: ["MCP", "schema validation", "Python"]
---

An agent never sees your tool's source code. It sees a name, a description, and a JSON Schema describing the inputs: that's the entire contract it has to work with. If the schema says a field is optional and the handler actually requires it, or the schema allows a type the handler crashes on, the agent has no way to know until it's already mid-conversation and the call fails.

**What it is:** contract tests for MCP tool definitions: checking that the `inputSchema` is well-formed, and separately, that it actually matches what the handler behind it does.

**The problem it solves:** a tool schema and its handler are two separate pieces of code that can drift apart silently. Nothing forces them to stay in sync except a test that checks both.

## How it works

### The tool definition

MCP tools use `inputSchema` (camelCase, one word) which is easy to typo as the Anthropic Messages API's own tool format (`input_schema`, snake_case) if you're hand-writing definitions for both surfaces in the same codebase.

```python
# mcp_tools/definitions.py
GET_ORDER_STATUS_TOOL = {
    "name": "get_order_status",
    "description": "Look up the current status of a customer order by order ID.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order ID, e.g. 'ORD-1042'"},
            "include_history": {"type": "boolean", "description": "Include status change history"},
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
}
```

### The handler behind it

```python
# mcp_tools/handlers.py
ORDERS = {
    "ORD-1042": {"status": "shipped", "history": ["placed", "packed", "shipped"]},
}

def get_order_status(order_id: str, include_history: bool = False) -> dict:
    if order_id not in ORDERS:
        raise ValueError(f"No such order: {order_id}")
    order = ORDERS[order_id]
    result = {"status": order["status"]}
    if include_history:
        result["history"] = order["history"]
    return result
```

Nothing links these two files together automatically. The schema is a promise; the handler's actual signature is what gets kept, or not.

## Common issues

**Over-constrained schema.** `required` lists a field the handler treats as optional (or defaults). Harmless to correctness, but it forces every agent call to supply something it didn't actually need to: friction with no payoff.

**Under-constrained schema.** The opposite, and the dangerous direction: the handler genuinely needs a field the schema doesn't mark `required`. An agent omits it, validation doesn't catch anything, and the handler throws a raw, unhandled exception mid-conversation instead of a clean rejection.

**`additionalProperties` left unset.** JSON Schema defaults this to `true`: meaning a typo'd field name from the agent (or from the model hallucinating a plausible-sounding parameter) is silently dropped instead of rejected. Set it to `false` explicitly, every time, the same way the [LLM-as-judge post's](/articles/llm-as-judge-eval-pipeline/) output schema does.

## What to test, and how

**First, the schema has to actually be valid JSON Schema**: a malformed schema fails in whatever tool-loading code parses it, before an agent ever gets near it:

```python
# tests/test_schema_validity.py
from jsonschema import Draft7Validator
from mcp_tools.definitions import GET_ORDER_STATUS_TOOL

def test_input_schema_is_valid_json_schema():
    Draft7Validator.check_schema(GET_ORDER_STATUS_TOOL["inputSchema"])


def test_tool_has_required_fields():
    tool = GET_ORDER_STATUS_TOOL
    assert tool["name"]
    assert tool["description"], "A tool with no description gives the model nothing to decide when to call it"
    assert "inputSchema" in tool
```

**Then, the contract between schema and handler**: every input the schema accepts should also work against the real handler, and every input the schema rejects should actually be rejected, not silently accepted:

```python
# tests/test_schema_contract.py
import pytest
from jsonschema import validate, ValidationError

from mcp_tools.definitions import GET_ORDER_STATUS_TOOL
from mcp_tools.handlers import get_order_status

SCHEMA = GET_ORDER_STATUS_TOOL["inputSchema"]

VALID_INPUTS = [
    {"order_id": "ORD-1042"},
    {"order_id": "ORD-1042", "include_history": True},
]

INVALID_INPUTS = [
    {},                                              # missing required order_id
    {"order_id": 1042},                               # wrong type: schema says string
    {"order_id": "ORD-1042", "extra_field": "nope"},  # additionalProperties: False
]

@pytest.mark.parametrize("payload", VALID_INPUTS)
def test_schema_accepts_and_handler_succeeds_on_valid_input(payload):
    validate(instance=payload, schema=SCHEMA)   # should not raise
    result = get_order_status(**payload)         # handler should accept what the schema accepts
    assert "status" in result


@pytest.mark.parametrize("payload", INVALID_INPUTS)
def test_schema_rejects_invalid_input(payload):
    with pytest.raises(ValidationError):
        validate(instance=payload, schema=SCHEMA)
```

The first test class is the one that catches the dangerous drift direction: if someone adds a new required parameter to `get_order_status()` and forgets to update the schema, `test_schema_accepts_and_handler_succeeds_on_valid_input` breaks immediately, with a `TypeError` from the handler, not a mystery agent failure discovered three weeks later in production.

This checks the boundary contract, not runtime correctness under load: it doesn't cover a handler that validates fine but returns wrong data, timeouts, or the tool-level chaos-testing questions from the [fault-injection post](/articles/chaos-engineering-for-agents/). Schema conformance is necessary and cheap to test; it isn't a substitute for testing what the tool actually does once it's called.

## Takeaways

- A tool's schema and its handler are two independent pieces of code: nothing keeps them in sync except a test that exercises both.
- Test both directions: valid-per-schema inputs should work against the real handler, and invalid-per-schema inputs should actually be rejected by the schema, not just by hoping the handler happens to guard against them too.
- Always set `additionalProperties: false`: the default (`true`) means a typo'd or hallucinated field name gets silently dropped instead of caught.
