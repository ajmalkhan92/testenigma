#!/usr/bin/env python3
"""Generate a cover image for every article in src/content/articles/, plus a
site-wide default OG image, using Gemini's image generation model. Requires
GEMINI_API_KEY in .env (see .env.example). Prints running and total cost
based on actual token usage returned by the API (Gemini 3.1 Flash Image
standard-tier pricing: $0.50 / 1M input tokens, $60.00 / 1M output tokens,
as of https://ai.google.dev/gemini-api/docs/pricing).

Usage:
    python3 scripts/generate_article_images.py
"""
import os
from pathlib import Path

from google import genai

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
ARTICLES_DIR = ROOT / "public" / "images" / "articles"
MODEL = "gemini-3.1-flash-image"

INPUT_PRICE_PER_M = 0.50   # $ per 1M input tokens (text/image), standard tier
OUTPUT_PRICE_PER_M = 60.00  # $ per 1M output tokens (image), standard tier

STYLE = (
    "A bold, catchy, fun flat character illustration for a software "
    "engineering blog about testing LLMs and AI systems, modern flat "
    "illustration style with a friendly human character, like the "
    "illustrations on Slack, Mailchimp, or Dropbox's websites. Feature a "
    "simply-drawn person (expressive pose, simple friendly face, casual "
    "clothes) actively interacting with a computer, laptop, or giant screen. "
    "Playful and energetic, NOT corporate or stiff. The background is a "
    "solid, rich, saturated color, not white, not light gray, filling the "
    "frame edge to edge. Bold saturated colors throughout, strong contrast. "
    "No text, no letters, no numbers, no UI chrome in the image."
)

MIME_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}

# slug -> subject description (combined with STYLE to form the full prompt).
# The "_default" entry generates the site-wide OG fallback image instead of
# an article-specific one, saved directly to public/ as og-default.<ext>.
ARTICLES = {
    "write-your-first-llm-eval": "A person happily checking off boxes with a giant pencil on a checklist floating beside a laptop screen showing a chat bubble",
    "golden-dataset-for-llm-evals": "A person carefully filing glowing golden folders into an open cabinet drawer next to a laptop",
    "llm-test-pyramid": "A person climbing a small pyramid built out of stacked laptop and phone screens, reaching for the top",
    "testing-langgraph-agents": "A person at a control desk with levers, directing a few small friendly robot characters through a branching maze on a big screen behind them",
    "llm-as-judge-eval-pipeline": "A person in a judge's robe holding a gavel, standing between two big speech-bubble screens balanced on a scale",
    "red-teaming-prompt-injection-tests": "A person in a detective coat holding a magnifying glass up to a laptop screen, spotting a sneaky masked bug character hiding in the code",
    "evaluating-rag-pipelines": "A person with a magnifying glass digging through a tall stack of giant books next to a laptop showing a chat bubble",
    "mutation-testing-meets-ai": "A person with tweezers sneaking a small glitchy bug icon into lines of code on a big screen, while a robot character watches with a checklist",
    "chaos-engineering-for-agents": "A person in a hard hat swinging a wrench at spinning gears next to a startled little robot character, sparks flying",
    "cost-latency-regression-testing": "A person watching a giant stopwatch and a tall stack of coins next to a laptop, looking at a dial gauge",
    "testing-mcp-tool-definitions": "A person stamping a huge checkmark onto a blueprint document taped to the wall next to their laptop",
    "ai-generated-test-cases": "A friendly robot character handing a checklist to a person, who is reviewing it with a raised eyebrow and a pen",
    "exploratory-testing-agent": "A person dressed as an explorer with a compass and backpack, wandering through a maze made of giant floating app-screen walls",
    "self-healing-selectors-with-llm": "A person with a wrench and a giant bandage, repairing a broken robot arm next to a laptop showing code",
    "flaky-test-detection-with-statistics": "A person scratching their head looking confused at a wall of screens showing wobbly, inconsistent bar charts, with a couple of dice on the desk",
    "_default": "A friendly person giving a thumbs up while sitting on a giant checkmark, laptop open beside them, in the exact same bold flat character illustration style as the other images",
}


def load_env() -> None:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def call_cost(usage) -> float:
    if usage is None:
        return 0.0
    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    return (
        input_tokens * INPUT_PRICE_PER_M + output_tokens * OUTPUT_PRICE_PER_M
    ) / 1_000_000


def output_path(slug: str, ext: str) -> Path:
    if slug == "_default":
        return ROOT / "public" / f"og-default{ext}"
    return ARTICLES_DIR / f"{slug}{ext}"


def main() -> None:
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set: add it to .env first.")

    client = genai.Client(api_key=api_key)
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0
    generated = 0
    failed: list[str] = []

    for slug, subject in ARTICLES.items():
        # An image with either extension already present counts as done.
        if any(output_path(slug, ext).exists() for ext in MIME_EXT.values()):
            print(f"skip (already exists): {slug}")
            continue

        prompt = f"{STYLE} Subject: {subject}. 16:9 landscape composition."
        print(f"generating: {slug} ...")
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
        except Exception as e:
            print(f"  ! generation failed for {slug}: {e}")
            failed.append(slug)
            continue

        image_bytes, mime_type = None, None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                mime_type = part.inline_data.mime_type
                break

        if not image_bytes:
            print(f"  ! no image returned for {slug}, skipping")
            failed.append(slug)
            continue

        cost = call_cost(response.usage_metadata)
        total_cost += cost
        generated += 1

        ext = MIME_EXT.get(mime_type, ".jpg")
        out_path = output_path(slug, ext)
        out_path.write_bytes(image_bytes)
        print(f"  saved -> {out_path.relative_to(ROOT)}  (${cost:.4f}, running total ${total_cost:.4f})")

    print()
    print(f"Done. Generated {generated} image(s), {len(failed)} failed. Total cost: ${total_cost:.4f}")
    if failed:
        print(f"Failed: {', '.join(failed)}. Re-run the script to retry just these (existing images are skipped).")


if __name__ == "__main__":
    main()
