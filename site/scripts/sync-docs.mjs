// Sync the repo's Markdown docs into Starlight's content tree.
//
//   docs/[0-9][0-9]-*.md -> src/content/docs/docs/<name>.md      (/onpar/docs/<name>/)
//   docs/adr/*.md        -> src/content/docs/docs/adr/<name>.md  (/onpar/docs/adr/<name>/)
//
// The output is build output (gitignored). docs/*.md stays the source of truth and
// keeps working on GitHub, so links are written for GitHub and rewritten here.
// Every relative link is resolved against the source file's directory and then
// mapped by where it lands in the repo. Anything unmappable fails the sync.
// No dependencies: runs on plain Node.
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, posix, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Sidebar labels: a doc titled "onpar — pitch" reads as "pitch" next to twelve siblings that all belong to onpar.
const cleanTitle = (t) => t.trim().replace(/^onpar\s*[—:-]\s*/i, '').replace(/^\w/, (c) => c.toUpperCase());

const SITE = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REPO = resolve(SITE, '..');
const OUT = join(SITE, 'src/content/docs/docs');
const BASE = '/onpar';
const GITHUB = 'https://github.com/dshakes/onpar';

const errors = [];

// Map one link target, written relative to repo file `from`, to its site URL.
function rewrite(target, from) {
	if (/^([a-z][a-z0-9+.-]*:|#|\/\/)/i.test(target) || target.startsWith('/')) return target;
	const [pathPart, hash] = splitHash(target);
	if (pathPart === '') return target;
	const abs = posix.normalize(posix.join(posix.dirname(from), pathPart));
	if (abs.startsWith('..')) {
		errors.push(`${from}: link escapes the repo: ${target}`);
		return target;
	}
	const suffix = hash === undefined ? '' : `#${anchor(abs, hash, from)}`;
	let m;
	if ((m = abs.match(/^docs\/guide\/([a-z-]+)\/(?:\d\d-)?([^/]+)\.md$/)))
		return `${BASE}/docs/${m[1]}/${m[2]}/${suffix}`;
	if ((m = abs.match(/^docs\/(\d\d-[^/]+)\.md$/))) return `${BASE}/docs/${m[1]}/${suffix}`;
	if ((m = abs.match(/^docs\/adr\/([^/]+)\.md$/))) return `${BASE}/docs/adr/${m[1]}/${suffix}`;
	if ((m = abs.match(/^docs\/assets\/([^/]+)$/))) {
		// ponytail: the validator cannot see these (postbuild copies them), so check here.
		// postbuild serves site/assets/ plus docs/assets/ (which wins) at /onpar/assets/.
		if (!existsSync(join(SITE, 'assets', m[1])) && !existsSync(join(REPO, 'docs/assets', m[1])))
			errors.push(`${from}: asset missing: ${target}`);
		return `${BASE}/assets/${m[1]}${suffix}`;
	}
	if (abs === 'README.md') return `${GITHUB}#${hash ?? 'readme'}`;
	if (!existsSync(join(REPO, abs))) errors.push(`${from}: link to a path that does not exist: ${target}`);
	return `${GITHUB}/blob/main/${abs}${suffix}`;
}

// GitHub-style heading slug (what both GitHub and Starlight's github-slugger produce
// for the plain-text headings these docs use).
const slug = (h) =>
	h
		.trim()
		.toLowerCase()
		.replace(/[^\p{L}\p{N}\s_-]/gu, '')
		.replace(/\s/g, '-');

// An anchor that names no heading but uniquely prefixes one ("#grader-stack" for
// "## Grader stack (highest-risk component)") resolves to that heading, with a warning
// so the source can be fixed. Anything else is left for the links validator to fail.
function anchor(abs, hash, from) {
	if (!abs.endsWith('.md') || !existsSync(join(REPO, abs))) return hash;
	const slugs = [...readFileSync(join(REPO, abs), 'utf8').matchAll(/^#{2,6} (.+)$/gm)].map((m) => slug(m[1]));
	if (slugs.includes(hash)) return hash;
	const hits = slugs.filter((s) => s.startsWith(`${hash}-`));
	if (hits.length !== 1) return hash;
	console.warn(`sync-docs: warning: ${from}: #${hash} names no heading in ${abs}; using #${hits[0]}`);
	return hits[0];
}

function splitHash(t) {
	const i = t.indexOf('#');
	return i < 0 ? [t, undefined] : [t.slice(0, i), t.slice(i + 1)];
}

// Rewrite markdown links/images and HTML src/href outside fenced code and inline code.
function rewriteBody(body, from) {
	let fenced = false;
	return body
		.split('\n')
		.map((line) => {
			if (/^\s*(```|~~~)/.test(line)) {
				fenced = !fenced;
				return line;
			}
			if (fenced) return line;
			return line
				.split(/(`[^`]*`)/)
				.map((part, i) =>
					i % 2
						? part
						: part
								.replace(/\]\(\s*([^)\s]+)((?:\s+"[^"]*")?\s*)\)/g, (_, t, rest) => `](${rewrite(t, from)}${rest})`)
								.replace(/\b(src|href)="([^"]+)"/g, (_, a, t) => `${a}="${rewrite(t, from)}"`),
				)
				.join('');
		})
		.join('\n');
}

// First prose paragraph, flattened to plain text, for <meta name="description">.
function describe(body) {
	for (const block of body.split(/\n\s*\n/)) {
		const text = block.trim();
		if (!text || /^(<|#|!\[|\||```|~~~|---|\* \* \*)/.test(text)) continue;
		// Status and version lines ("Status: accepted · date", "Version: 0.1 · Date: …")
		// say nothing about the page.
		if (/^[*_]*(status|version|date|format)\b/i.test(text) || text.includes(' · ')) continue;
		// Lists, and the one-line lead-ins to them ("Two forcing functions:"), aren't summaries.
		if (/^(\d+\.|[-*+]) /.test(text) || /:$/.test(text)) continue;
		const plain = text
			.replace(/^>\s?/gm, '')
			.replace(/!\[[^\]]*\]\([^)]*\)/g, '')
			.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
			.replace(/<[^>]+>/g, '')
			.replace(/[`*_]/g, '')
			.replace(/\s+/g, ' ')
			.trim();
		if (!plain) continue;
		return plain.length > 240 ? `${plain.slice(0, 237).replace(/\s+\S*$/, '')}...` : plain;
	}
	return '';
}

// `<!-- run: name -->` on its own line embeds docs/guide/_runs/<name>.txt, which
// tools/story_run.sh records from a real run. A missing recording fails the sync:
// a page may not ship a transcript that was typed.
function includeRuns(body, from) {
	return body.replace(/^<!-- run: ([\w-]+) -->$/gm, (_, name) => {
		const file = join(REPO, 'docs/guide/_runs', `${name}.txt`);
		if (!existsSync(file)) {
			errors.push(`${from}: no recording docs/guide/_runs/${name}.txt — run tools/story_run.sh`);
			return '';
		}
		const text = readFileSync(file, 'utf8').replace(/\s+$/, '');
		return '```text\n' + text.replace(/```/g, "'''") + '\n```';
	});
}

function convert(srcRel, { order } = {}) {
	const text = readFileSync(join(REPO, srcRel), 'utf8').replace(/\r\n/g, '\n');
	const m = text.match(/^# (.+)\n/m);
	if (!m) {
		errors.push(`${srcRel}: no "# " title heading`);
		return '';
	}
	const body = includeRuns(text.slice(0, m.index) + text.slice(m.index + m[0].length), srcRel);
	// JSON strings are valid YAML scalars, so this quoting is always safe.
	const side = order === undefined ? '' : `sidebar:\n  order: ${order}\n`;
	const fm = `---\ntitle: ${JSON.stringify(cleanTitle(m[1]))}\ndescription: ${JSON.stringify(describe(body))}\n${side}editUrl: ${JSON.stringify(`${GITHUB}/edit/main/${srcRel}`)}\n---\n`;
	return fm + rewriteBody(body, srcRel);
}

function list(dir, re) {
	return readdirSync(join(REPO, dir))
		.filter((f) => re.test(f))
		.sort()
		.map((f) => `${dir}/${f}`);
}

const docs = list('docs', /^\d\d-.+\.md$/);
const adrs = list('docs/adr', /^.+\.md$/);

// The reader's docs: docs/guide/<section>/<NN-name>.md. NN orders the sidebar and
// is dropped from the URL. Sections become sidebar groups (astro.config.mjs).
const GUIDE = 'docs/guide';
const sections = existsSync(join(REPO, GUIDE))
	? readdirSync(join(REPO, GUIDE)).filter((d) => !d.startsWith('_') && statSync(join(REPO, GUIDE, d)).isDirectory()).sort()
	: [];
const guidePages = sections.flatMap((sec) =>
	list(`${GUIDE}/${sec}`, /\.md$/).map((f) => {
		const base = posix.basename(f, '.md');
		const mm = base.match(/^(\d\d)-(.+)$/);
		return { file: f, section: sec, slug: mm ? mm[2] : base, order: mm ? Number(mm[1]) : 100 };
	}),
);

rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, 'adr'), { recursive: true });
for (const f of docs) writeFileSync(join(OUT, posix.basename(f)), convert(f));
for (const f of adrs) writeFileSync(join(OUT, 'adr', posix.basename(f)), convert(f));
for (const pg of guidePages) {
	mkdirSync(join(OUT, pg.section), { recursive: true });
	writeFileSync(join(OUT, pg.section, `${pg.slug}.md`), convert(pg.file, { order: pg.order }));
}

// /onpar/docs/ itself: a splash page built from the docs' own titles and first
// paragraphs, so it cannot drift from them. Starlight components, hence .mdx.
const titleOf = (f) => cleanTitle(readFileSync(join(REPO, f), 'utf8').match(/^# (.+)$/m)?.[1] ?? f);
// MDX reads `<` and `{` inside attributes as JSX, so card text is plain prose only.
const plain = (t) => t.replace(/[<>{}]/g, '').replace(/\s+/g, ' ').trim();
// JSX attribute strings have no backslash escapes, so quotes become entities.
const attr = (t) => '"' + plain(t).replace(/&/g, '&amp;').replace(/"/g, '&quot;') + '"';
const descOf = (f) => plain(describe(readFileSync(join(REPO, f), 'utf8').replace(/^# .+$/m, ''))).slice(0, 220);
const urlOf = (f) => rewrite(posix.basename(f), f);
const card = (f, title = titleOf(f)) =>
	`  <LinkCard title=${attr(title)} description=${attr(descOf(f))} href=${attr(urlOf(f))} />`;
const grid = (heading, files, extra = []) => ['', `## ${heading}`, '', '<CardGrid>', ...files.map((f) => card(f)), ...extra, '</CardGrid>'];
// The splash follows the reader's path: one grid per guide section, in the sidebar's
// order and labels (astro.config.mjs GUIDE_LABELS), pages in NN order.
const GUIDE_LABELS = [
	['start', 'Start here'],
	['loop', 'The loop'],
	['agents', 'For agents'],
	['learn', 'Learn'],
	['reference', 'Reference'],
];
const guideIn = (sec) =>
	guidePages
		.filter((p) => p.section === sec)
		.sort((a, b) => a.order - b.order || a.file.localeCompare(b.file))
		.map((p) => p.file);
const labCard = `  <LinkCard title="The interactive lab" description="A memory lab that sizes weights, KV cache and overhead as you move the sliders, and the written curriculum with diagrams." href=${JSON.stringify(`${BASE}/lab/`)} />`;
const guideGrids = GUIDE_LABELS.flatMap(([sec, label]) => {
	const files = guideIn(sec);
	return files.length ? grid(label, files, sec === 'learn' ? [labCard] : []) : [];
});
// The specs that built the product: titles only, so history stays small and last.
const projectCard = (f) => `  <LinkCard title=${attr(titleOf(f))} href=${attr(urlOf(f))} />`;
const byNum = (...nums) => nums.map((n) => docs.find((f) => posix.basename(f).startsWith(n)) ?? `docs/${n}`);
const latestAdr = adrs[adrs.length - 1];
writeFileSync(
	join(OUT, 'index.mdx'),
	[
		'---',
		'title: onpar docs',
		'description: "The onpar guide: a first fit and a first proof, the seven-stage loop, what agents may call, the curriculum and reference, then every decision record and the project specs."',
		'template: splash',
		'editUrl: false',
		'hero:',
		'  title: Is the open model on&nbsp;par?',
		'  tagline: Prove which open-weight model can replace your closed-model API, on your own traffic and hardware, then migrate with a quality gate and a rollback. These pages are generated from the repo, so they cannot drift from it.',
		'  actions:',
		'    - text: Start with the field guide',
		`      link: ${urlOf(byNum('05')[0])}`,
		'      icon: right-arrow',
		'    - text: Open the interactive lab',
		`      link: ${BASE}/lab/`,
		'      icon: rocket',
		'      variant: secondary',
		'    - text: Landing page',
		`      link: ${BASE}/`,
		'      variant: minimal',
		'---',
		"import { CardGrid, LinkCard } from '@astrojs/starlight/components';",
		// The seven-stage walkthrough, a React island (src/components/diagrams/).
		"import { LoopWalkthrough } from '../../../components/diagrams';",
		'',
		'## Step through the loop',
		'',
		'Press play, or pick a scenario, to follow your traffic from capture to a receipt and see where the loop says no.',
		'',
		'<LoopWalkthrough client:visible />',
		...guideGrids,
		'',
		'## Decisions',
		'',
		'<CardGrid>',
		card(latestAdr, `Latest: ${titleOf(latestAdr)}`),
		`  <LinkCard title=${JSON.stringify(`All ${adrs.length} architecture decision records`)} description="Every load-bearing choice, including the two later reversed. Each one records the context, the options and why." href=${JSON.stringify(urlOf(adrs[0]))} />`,
		'</CardGrid>',
		'',
		'## Project',
		'',
		'The specs that built the product: history and intent, kept out of the reader’s path.',
		'',
		'<CardGrid>',
		...docs.map(projectCard),
		'</CardGrid>',
		'',
	].join('\n'),
);

if (errors.length) {
	console.error(`sync-docs: ${errors.length} unmappable link(s):\n  ${errors.join('\n  ')}`);
	process.exit(1);
}
console.log(`sync-docs: ${guidePages.length} guide pages + ${docs.length} project docs + ${adrs.length} ADRs -> ${relative(SITE, OUT)}`);
