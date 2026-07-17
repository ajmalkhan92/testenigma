# Testing & AI Blog

A self-hosted [Ghost](https://ghost.org) blog about AI for software testing
and testing AI systems, with:

- an automated publish pipeline (push a Markdown post -> GitHub Actions ->
  live on the blog),
- a documented manual step to cross-post to Medium with a canonical link
  (see [CROSSPOST.md](CROSSPOST.md) — Medium's API is closed to new
  integrations, so this can't be automated),
- a daily scheduled agent that emails you 10 fresh post ideas to pick from.

## Repo layout

```
posts/      Markdown posts (front matter: title, slug, tags, excerpt, status)
ideas/      Daily idea digests, dated (also emailed to you)
scripts/    publish_to_ghost.py, send_idea_email.py
infra/      docker-compose.yml + Caddyfile for self-hosting Ghost
```

## 1. Set up the VPS and Ghost

1. **Buy a domain** (Namecheap, Porkbun, etc.) if you don't have one.
2. **Provision a small Ubuntu 22.04 VPS** — a Hetzner CX22 or DigitalOcean
   $6/mo droplet is plenty. Point the domain's `A` record at the VPS's IP.
3. **Install Docker** on the VPS:
   ```
   curl -fsSL https://get.docker.com | sh
   ```
4. **Copy `infra/` to the VPS** (`scp -r infra your-vps:~/blog-infra` or
   `git clone` this repo there), then create `infra/.env` from
   `infra/.env.example` with your real domain and a strong MySQL password.
5. **Start it**:
   ```
   cd blog-infra && docker compose --env-file .env up -d
   ```
   Caddy will automatically request a Let's Encrypt certificate for your
   domain the first time it's reached over HTTPS — give DNS a few minutes to
   propagate first.
6. Visit `https://your-domain.com/ghost/` and complete Ghost's setup wizard
   (creates your admin user).
7. In Ghost Admin: **Settings -> Advanced -> Integrations -> Add custom
   integration**. Name it e.g. "Publish pipeline". Copy the **Admin API
   Key** (`id:secret` format) and the site URL — these become
   `GHOST_ADMIN_API_KEY` and `GHOST_API_URL`.

## 2. Wire up the publish pipeline

1. Push this repo to a **GitHub repository**.
2. In the GitHub repo settings -> **Secrets and variables -> Actions**, add:
   - `GHOST_API_URL`
   - `GHOST_ADMIN_API_KEY`
3. That's it — `.github/workflows/publish.yml` runs on every push to `main`
   that touches `posts/*.md` and publishes/updates the changed post(s) via
   the Ghost Admin API.

### Local testing (optional, before relying on CI)

```
cd scripts
pip install -r requirements.txt
export GHOST_API_URL=https://your-domain.com
export GHOST_ADMIN_API_KEY=id:secret
python3 publish_to_ghost.py ../posts/your-post.md
```

## 3. Writing and publishing a post

1. Copy `posts/_template.md` to `posts/your-post-slug.md`.
2. Fill in `title`, `slug`, `tags`, `excerpt`, write the body in Markdown.
3. Set `status: draft` while you're still writing (safe to push — it'll
   sync to Ghost as a draft, not go live).
4. When ready, set `status: publish`, commit, and push to `main`. The post
   goes live automatically.
5. Cross-post to Medium manually — see [CROSSPOST.md](CROSSPOST.md).

## 4. Daily 10-idea email

Set up once as a Claude Code scheduled routine (`/schedule` or the
`CronCreate` tool) that runs once a day and:

1. Reads existing titles in `posts/` and `ideas/` to avoid repeats.
2. Brainstorms 10 post ideas spanning both "AI for testing" and "testing AI
   systems."
3. Writes them to `ideas/YYYY-MM-DD.md`.
4. Runs `python3 scripts/send_idea_email.py ideas/YYYY-MM-DD.md` to email
   the digest.

Requires these environment variables available to the routine (a
[Resend](https://resend.com) account, free tier is enough for 1 email/day):

- `RESEND_API_KEY`
- `RESEND_FROM_EMAIL` (an address on a domain verified in Resend)
- `RECIPIENT_EMAIL` (ajmalkhan92@gmail.com)

Pick one idea from the digest each day and follow the workflow in section 3.
