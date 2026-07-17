#!/usr/bin/env python3
"""Email the daily content-idea digest via Resend.

Usage:
    python3 send_idea_email.py ideas/2026-07-18.md

Reads RESEND_API_KEY, RESEND_FROM_EMAIL, RECIPIENT_EMAIL from the environment.
"""
import os
import sys
import markdown
import requests

RESEND_ENDPOINT = "https://api.resend.com/emails"


def send(md_path: str):
    api_key = os.environ["RESEND_API_KEY"]
    from_email = os.environ["RESEND_FROM_EMAIL"]
    to_email = os.environ["RECIPIENT_EMAIL"]

    with open(md_path) as f:
        body_md = f.read()

    date_label = os.path.basename(md_path).removesuffix(".md")
    html_body = markdown.markdown(body_md, extensions=["fenced_code"])

    resp = requests.post(
        RESEND_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "from": from_email,
            "to": [to_email],
            "subject": f"10 blog ideas for {date_label}",
            "html": html_body,
            "text": body_md,
        },
    )

    if resp.status_code >= 300:
        sys.exit(f"Resend API error {resp.status_code}: {resp.text}")

    print(f"Idea digest emailed to {to_email} ({resp.json().get('id')})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: send_idea_email.py <path-to-ideas.md>")
    send(sys.argv[1])
