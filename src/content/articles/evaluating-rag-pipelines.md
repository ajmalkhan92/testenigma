---
title: "Precision, Recall, and Vibes: Evaluating RAG Pipelines Properly"
description: "A RAG system fails in two different places — retrieval and generation — and testing it as one opaque blob tells you nothing about which one broke. Separate metrics, separate tests, and a faithfulness check that catches ungrounded answers."
pubDate: 2026-07-23
category: evaluation
tags: ["RAG", "LLM evals", "Python"]
---

"The RAG answers are worse this week" is a symptom, not a diagnosis. A retrieval-augmented pipeline has two failure surfaces stacked on top of each other: the retriever can fetch the wrong documents, or the generator can be handed the right documents and still write an answer they don't support. Score the whole pipeline as one number and you can't tell which one broke — which means you can't tell whether to fix the index or fix the prompt.

This post splits the two apart, with a metric and a test for each.

**What it is:** two separate evals — retrieval quality (precision/recall) and generation faithfulness — instead of one blended "RAG quality" score.

**The problem it solves:** a single combined score tells you something regressed; it can't tell you whether to fix the index or fix the prompt. Splitting the metrics localizes the failure.

## How it works

### The retriever

A toy retriever stands in for a real vector store here — the eval code below doesn't care what's behind `retrieve()`, only that it returns document IDs:

```python
# rag/retriever.py
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    text: str

CORPUS = [
    Document("doc-1", "To reset your password, go to Settings > Security > Reset Password."),
    Document("doc-2", "Refunds are processed within 3-5 business days to the original payment method."),
    Document("doc-3", "Two-factor authentication can be enabled under Settings > Security > 2FA."),
    Document("doc-4", "Our support hours are Monday-Friday, 9am-6pm ET."),
]

def retrieve(query: str, k: int = 2) -> list[Document]:
    """Ranks by word overlap. Swap in a real vector index for production —
    everything below only depends on retrieve() returning document IDs."""
    query_words = set(query.lower().split())
    scored = [(len(query_words & set(doc.text.lower().split())), doc) for doc in CORPUS]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:k]]
```

### Retrieval metrics: precision@k and recall@k

Precision@k asks "of what we fetched, how much was actually relevant." Recall@k asks "of what's relevant, how much did we actually fetch." A retriever can be great at one and bad at the other — fetching one perfect document scores 100% precision and terrible recall if there were three relevant documents; fetching everything in the corpus scores 100% recall and terrible precision. You need both numbers, not one.

```python
# rag/golden_retrieval.py
GOLDEN_RETRIEVAL = [
    {"query": "How do I reset my password?", "relevant_ids": {"doc-1"}},
    {"query": "When will I get my refund?", "relevant_ids": {"doc-2"}},
    {"query": "How do I turn on two-factor authentication?", "relevant_ids": {"doc-3"}},
]
```

```python
# tests/test_retrieval_eval.py
from rag.golden_retrieval import GOLDEN_RETRIEVAL
from rag.retriever import retrieve

K = 2
PRECISION_FLOOR = 0.4
RECALL_FLOOR = 0.8

def test_retrieval_precision_and_recall():
    precisions, recalls = [], []
    for ex in GOLDEN_RETRIEVAL:
        retrieved = {doc.id for doc in retrieve(ex["query"], k=K)}
        hits = retrieved & ex["relevant_ids"]
        precisions.append(len(hits) / len(retrieved) if retrieved else 0)
        recalls.append(len(hits) / len(ex["relevant_ids"]))

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    assert avg_precision >= PRECISION_FLOOR, f"Precision@{K} {avg_precision:.2f} < {PRECISION_FLOOR}"
    assert avg_recall >= RECALL_FLOOR, f"Recall@{K} {avg_recall:.2f} < {RECALL_FLOOR}"
```

`GOLDEN_RETRIEVAL` is the same discipline as the [golden dataset post](/articles/golden-dataset-for-llm-evals/) — labeled examples, committed to the repo, grown from real query logs over time. A precision/recall eval is only as good as the relevance labels it's scored against.

### The generator: faithfulness, not correctness

Retrieval passing doesn't mean the generated answer is any good — the model can be handed exactly the right document and still write something that document doesn't say. That's a **faithfulness** (or groundedness) check: does every claim in the answer trace back to the retrieved context, independent of whether the context itself was the right context to retrieve.

```python
# rag/generate.py
import anthropic

client = anthropic.Anthropic()

def generate_answer(query: str, context_docs: list) -> str:
    context = "\n\n".join(f"[{doc.id}] {doc.text}" for doc in context_docs)
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=200,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system="Answer the question using only the provided context. If the context doesn't contain the answer, say so.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text")
```

Checking faithfulness is a judging task, not a scoring task — same pattern as the [LLM-as-judge post](/articles/llm-as-judge-eval-pipeline/), pointed at "is this claim in the context" instead of "does this pass the rubric":

```python
# rag/faithfulness.py
import json
import anthropic

client = anthropic.Anthropic()

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "fully_supported": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["fully_supported", "unsupported_claims"],
    "additionalProperties": False,
}

def check_faithfulness(context: str, answer: str) -> dict:
    result = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": FAITHFULNESS_SCHEMA},
        },
        system=(
            "You are checking whether an answer's claims are supported by the given context. "
            "List any claim in the answer that is not directly supported by the context — "
            "including claims the context doesn't mention at all."
        ),
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"}],
    )
    text = "".join(b.text for b in result.content if b.type == "text")
    return json.loads(text)
```

```python
# tests/test_faithfulness_eval.py
from rag.golden_retrieval import GOLDEN_RETRIEVAL
from rag.retriever import retrieve
from rag.generate import generate_answer
from rag.faithfulness import check_faithfulness

FAITHFULNESS_FLOOR = 0.9

def test_answers_are_grounded_in_context():
    results = []
    for ex in GOLDEN_RETRIEVAL:
        docs = retrieve(ex["query"], k=2)
        context = "\n\n".join(f"[{d.id}] {d.text}" for d in docs)
        answer = generate_answer(ex["query"], docs)
        results.append(check_faithfulness(context, answer)["fully_supported"])

    faithfulness_rate = sum(results) / len(results)
    assert faithfulness_rate >= FAITHFULNESS_FLOOR
```

### Reading the two numbers together

| Layer | Metric | This run | Diagnosis if it fails |
|---|---|---|---|
| Retrieval | Precision@2 | 0.83 | Index or embedding change — the model was never given the right material |
| Retrieval | Recall@2 | 0.91 | Same as above |
| Generation | Faithfulness | 0.95 | Prompt or model change — the model had the right material and didn't use it correctly |

That's the entire point of splitting these two tests: a regression in the top rows means look at the retriever; a regression in the bottom row means look at the prompt. A single blended "RAG quality" score can't tell you that — it can only tell you something got worse.

## What to test, and how

Two independent pytest suites, one per failure surface, both shown above:

```bash
pytest tests/test_retrieval_eval.py     # precision@k / recall@k against golden_retrieval.py
pytest tests/test_faithfulness_eval.py  # faithfulness rate via the LLM judge
```

Run both on every change to the retriever *or* the generation prompt. A regression that shows up only in the first file means look at the index or embeddings; a regression only in the second means look at the prompt — that split is the entire reason these are two suites instead of one.

## Common issues

Precision and recall are only as trustworthy as the relevance labels behind them, and labeling "which documents are relevant to this query" is itself a judgment call someone made once — the same golden-dataset maintenance burden from earlier posts, not a one-time setup cost. And faithfulness checking inherits every caveat from the [LLM-as-judge post](/articles/llm-as-judge-eval-pipeline/): verbosity and self-preference bias apply here too, and a faithfulness judge should get the same human-calibration pass before it gates anything. A faithful answer also isn't automatically a *good* answer — it can accurately reflect context that was itself the wrong context, which is exactly why this post keeps the two failure surfaces separate instead of collapsing them into one trust score.

## Takeaways

- Retrieval and generation fail independently — test them with separate metrics, or a regression in either one just looks like "the RAG got worse" with no way to localize it.
- Precision@k and recall@k measure different things and a retriever can ace one while failing the other; report both.
- Faithfulness checks whether an answer is grounded in what was retrieved, not whether what was retrieved was correct — you need both the retrieval eval and the faithfulness eval, not one instead of the other.
