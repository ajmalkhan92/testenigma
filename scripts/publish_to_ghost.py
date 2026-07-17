#!/usr/bin/env python3
"""Publish or update a single Markdown post on a self-hosted Ghost blog.

Usage:
    python3 publish_to_ghost.py posts/my-post.md

Reads GHOST_API_URL and GHOST_ADMIN_API_KEY from the environment.
The Markdown file must have YAML front matter with at least: title, slug, status.
"""
import os
import sys
import time
import jwt
import requests
import markdown
import frontmatter

API_VERSION = "v5.0"


def build_admin_token(admin_api_key: str) -> str:
    key_id, secret = admin_api_key.split(":")
    iat = int(time.time())
    payload = {"iat": iat, "exp": iat + 5 * 60, "aud": "/admin/"}
    header = {"alg": "HS256", "typ": "JWT", "kid": key_id}
    return jwt.encode(payload, bytes.fromhex(secret), algorithm="HS256", headers=header)


def find_existing_post(base_url: str, token: str, slug: str):
    resp = requests.get(
        f"{base_url}/ghost/api/admin/posts/slug/{slug}/",
        headers={"Authorization": f"Ghost {token}"},
        params={"formats": "html"},
    )
    if resp.status_code == 200:
        return resp.json()["posts"][0]
    return None


def publish(md_path: str):
    base_url = os.environ["GHOST_API_URL"].rstrip("/")
    admin_api_key = os.environ["GHOST_ADMIN_API_KEY"]

    post = frontmatter.load(md_path)
    title = post.get("title")
    slug = post.get("slug")
    status = post.get("status", "draft")
    tags = post.get("tags", [])
    excerpt = post.get("excerpt", "")
    feature_image = post.get("feature_image") or None

    if not title or not slug:
        sys.exit(f"{md_path}: front matter must include 'title' and 'slug'")

    html_body = markdown.markdown(post.content, extensions=["fenced_code", "tables"])

    token = build_admin_token(admin_api_key)
    existing = find_existing_post(base_url, token, slug)

    payload = {
        "title": title,
        "slug": slug,
        "status": status,
        "html": html_body,
        "tags": [{"name": t} for t in tags],
        "custom_excerpt": excerpt or None,
    }
    if feature_image:
        payload["feature_image"] = feature_image

    # Ghost requires a fresh token per request and, for updates, the current updated_at.
    if existing:
        payload["updated_at"] = existing["updated_at"]
        token = build_admin_token(admin_api_key)
        resp = requests.put(
            f"{base_url}/ghost/api/admin/posts/{existing['id']}/?source=html",
            headers={"Authorization": f"Ghost {token}"},
            json={"posts": [payload]},
        )
        action = "Updated"
    else:
        resp = requests.post(
            f"{base_url}/ghost/api/admin/posts/?source=html",
            headers={"Authorization": f"Ghost {token}"},
            json={"posts": [payload]},
        )
        action = "Created"

    if resp.status_code not in (200, 201):
        sys.exit(f"Ghost API error {resp.status_code}: {resp.text}")

    result = resp.json()["posts"][0]
    print(f"{action} '{title}' -> {result.get('url', '(no url, draft)')} [status={result['status']}]")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: publish_to_ghost.py <path-to-post.md>")
    publish(sys.argv[1])
