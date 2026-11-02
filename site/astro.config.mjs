// The written docs at https://dshakes.github.io/onpar/docs/, generated from the
// repo's Markdown by scripts/sync-docs.mjs. The landing page, generated assets and
// the interactive lab are plain static files copied in by scripts/postbuild.mjs.
// The build fails on any broken internal link (starlight-links-validator).
// React renders the interactive diagrams in src/components/diagrams/.
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';

export default defineConfig({
	site: 'https://dshakes.github.io',
	base: '/onpar',
	trailingSlash: 'always',
	integrations: [
		react(),
		starlight({
			title: 'onpar',
			description: 'Prove which open-weight LLM can replace your closed-model API, on your own traffic, hardware and budget.',
			customCss: ['./src/styles/onpar.css'],
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/dshakes/onpar' }],
			editLink: { baseUrl: 'https://github.com/dshakes/onpar/edit/main/' },
			head: [
				// Dark-first, like the landing page: default to dark until the reader picks a theme.
				{
					tag: 'script',
					content: "try{if(!localStorage.getItem('starlight-theme'))localStorage.setItem('starlight-theme','dark')}catch(e){}",
				},
			],
			plugins: [
				starlightLinksValidator({
					// ponytail: these are copied into dist after the build, so the validator
					// cannot see them. sync-docs.mjs checks every /onpar/assets/ target exists.
					exclude: ['/onpar/', '/onpar/assets/**', '/onpar/lab/', '/onpar/lab/**'],
				}),
			],
			sidebar: [
				{ label: 'Overview', slug: 'docs' },
				{
					label: 'Start here',
					items: [
						{ slug: 'docs/05-field-guide' },
						{ slug: 'docs/00-verdict' },
						{ slug: 'docs/60-pitch' },
					],
				},
				{
					label: 'Product',
					items: ['10-landscape', '20-prd', '30-architecture', '40-ux', '50-roadmap'].map((s) => ({ slug: `docs/${s}` })),
				},
				{
					label: 'Operating',
					items: ['80-implementation-plan', '90-ci-gating', '95-ga-readiness', '70-naming'].map((s) => ({ slug: `docs/${s}` })),
				},
				{ label: 'Decisions', collapsed: true, items: [{ autogenerate: { directory: 'docs/adr' } }] },
				// Starlight prefixes `base` onto internal sidebar links itself. Writing
				// '/onpar/' here shipped as /onpar/onpar/ (404) — scripts/check-links.mjs
				// now crawls dist after every build so that cannot recur.
				{ label: 'Landing page', link: '/' },
				{ label: 'Interactive lab', link: '/lab/' },
			],
		}),
	],
});
