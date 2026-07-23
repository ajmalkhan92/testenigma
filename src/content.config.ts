import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const articles = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/articles' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    draft: z.boolean().default(false),
    category: z.enum(['reliability', 'evaluation', 'automation', 'field-notes']),
    tags: z.array(z.string()).default([]),
    youtubeUrl: z.string().url().optional(),
  }),
});

export const collections = { articles };
