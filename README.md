# TestEnigma

An [Astro](https://astro.build) static blog at [testenigma.com](https://testenigma.com)
about AI for software testing and testing AI systems, with:

- an automated publish pipeline (push a Markdown post -> GitHub Actions ->
  builds with Astro -> deploys to GitHub Pages, free, no server to maintain),
- a documented manual step to cross-post to Medium with a canonical link
  (see [CROSSPOST.md](CROSSPOST.md) — Medium's API is closed to new
  integrations, so this can't be automated),
- a daily scheduled agent that emails you 10 fresh post ideas to pick from.

## Repo layout

```
src/content/articles/  One Markdown/MDX file per post
src/pages/              Routes (home, articles, learn, services, about, search)
src/components/         Header, Footer, Giscus comments
src/layouts/             BaseLayout.astro — shared <head>, nav, footer, theme toggle
src/styles/global.css   Design tokens and site-wide styles
public/                 Static assets served as-is (favicon, CNAME, robots.txt)
ideas/                  Daily idea digests, dated (also emailed to you)
scripts/                send_idea_email.py — emails the daily digest via Resend
```

## 1. Writing and publishing a post

Add a new Markdown file under `src/content/articles/your-post-slug.md`:

```yaml
---
title: "Your post title"
description: "One or two sentences for cards and SEO."
pubDate: 2026-07-23
draft: true
category: field-notes # reliability | evaluation | automation | field-notes
---

Post body in Markdown.
```

1. Leave `draft: true` while you're still writing — drafts are excluded
   from the build, so it's safe to commit/push mid-draft.
2. When ready, set `draft: false`, commit, and push to `main`.
   `.github/workflows/publish.yml` builds the site with Astro and deploys it
   to GitHub Pages automatically — no server, no manual deploy step.
3. Cross-post to Medium manually — see [CROSSPOST.md](CROSSPOST.md).

### Preview locally before pushing

```
npm install
npm run dev
```

Opens a live-reloading preview (including drafts) at `localhost:4321`.

```
npm run build   # astro build + pagefind search index -> dist/
npm run preview # serve the production build locally
```

## 2. One-time GitHub Pages setup

1. Push this repo to GitHub (already done — `ajmalkhan92/testenigma`).
2. Repo **Settings -> Pages -> Build and deployment -> Source**: set to
   **GitHub Actions**.
3. Repo **Settings -> Pages -> Custom domain**: enter `testenigma.com`,
   then enable **Enforce HTTPS** once it's verified (may take a few minutes
   after DNS is configured).
4. In your DNS provider (GoDaddy) for `testenigma.com`, add four `A` records
   for the apex domain (`@`) pointing to GitHub Pages:
   ```
   185.199.108.153
   185.199.109.153
   185.199.110.153
   185.199.111.153
   ```
   Optionally add a `CNAME` record for `www` -> `ajmalkhan92.github.io`.
5. Push to `main` (or re-run the workflow) — GitHub Actions builds and
   deploys automatically from here on.

## 3. Daily 10-idea email

Set up once as a Claude Code scheduled routine (`/schedule` or the
`CronCreate` tool) that runs once a day and:

1. Reads existing titles in `src/content/articles/` and `ideas/` to avoid
   repeats.
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

Pick one idea from the digest each day and follow the workflow in section 1.
