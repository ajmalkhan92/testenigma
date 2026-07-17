# Cross-posting to Medium

Medium closed its Publishing API to new developer integrations, so there is no
way to automate this step in 2026. Do it manually per post (~2 minutes):

1. Publish the post on the blog first (push to `main`, let the
   `publish.yml` workflow build and deploy it via GitHub Pages, confirm
   it's live at its URL).
2. On Medium, click your profile photo -> **Stories** -> **Import a story**
   (or go directly to `https://medium.com/p/import`).
3. Paste the live post's URL and import.
4. Review formatting (headings, code blocks, images) and hit Publish.

Medium automatically sets a canonical link back to the original blog URL and
backdates the story to match, so SEO credit stays with the blog.
