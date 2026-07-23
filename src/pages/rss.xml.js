import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const articles = (await getCollection('articles', ({ data }) => !data.draft))
    .sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());

  return rss({
    title: 'TestEnigma',
    description: 'Testing software with AI, and testing AI itself.',
    site: context.site,
    items: articles.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.pubDate,
      link: `/articles/${post.id}/`,
    })),
    customData: `<language>en-us</language>`,
  });
}
