// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://testenigma.com',
  integrations: [mdx(), sitemap()],

  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },

  redirects: {
    '/posts/welcome-to-testenigma/': '/articles/welcome-to-testenigma/',
    '/posts/catching-llm-hallucinations-regression-harness/': '/articles/catching-llm-hallucinations-regression-harness/',
  },

  vite: {
    plugins: [tailwindcss()]
  }
});