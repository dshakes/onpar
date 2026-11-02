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
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
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

function convert(srcRel) {
	const text = readFileSync(join(REPO, srcRel), 'utf8').replace(/\r\n/g, '\n');
	const m = text.match(/^# (.+)\n/m);
	if (!m) {
		errors.push(`${srcRel}: no "# " title heading`);
		return '';
	}
	const body = text.slice(0, m.index) + text.slice(m.index + m[0].length);
	// JSON strings are valid YAML scalars, so this quoting is always safe.
	const fm = `---\ntitle: ${JSON.stringify(cleanTitle(m[1]))}\ndescription: ${JSON.stringify(describe(body))}\neditUrl: ${JSON.stringify(`${GITHUB}/edit/main/${srcRel}`)}\n---\n`;
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

rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, 'adr'), { recursive: true });
for (const f of docs) writeFileSync(join(OUT, posix.basename(f)), convert(f));
for (const f of adrs) writeFileSync(join(OUT, 'adr', posix.basename(f)), convert(f));

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
const grid = (heading, files) => ['', `## ${heading}`, '', '<CardGrid>', ...files.map((f) => card(f)), '</CardGrid>'];
const byNum = (...nums) => nums.map((n) => docs.find((f) => posix.basename(f).startsWith(n)) ?? `docs/${n}`);
const latestAdr = adrs[adrs.length - 1];
writeFileSync(
	join(OUT, 'index.mdx'),
	[
		'---',
		'title: onpar docs',
		'description: "The written docs for onpar: the field guide, product specs, operating docs and every architecture decision record."',
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
		...grid('Start here', byNum('05', '00', '60')),
		...grid('Product', byNum('10', '20', '30', '40', '50')),
		...grid('Operating', byNum('80', '90', '95', '70')),
		'',
		'## Decisions',
		'',
		'<CardGrid>',
		card(latestAdr, `Latest: ${titleOf(latestAdr)}`),
		`  <LinkCard title=${JSON.stringify(`All ${adrs.length} architecture decision records`)} description="Every load-bearing choice, including the two later reversed. Each one records the context, the options and why." href=${JSON.stringify(urlOf(adrs[0]))} />`,
		'</CardGrid>',
		'',
		'## Learn on your own numbers',
		'',
		'<CardGrid>',
		`  <LinkCard title="The interactive lab" description="A memory lab that sizes weights, KV cache and overhead as you move the sliders, the written curriculum with diagrams, and the solver-checked CLI reference." href=${JSON.stringify(`${BASE}/lab/`)} />`,
		`  <LinkCard title="onpar learn basics" description="Fifteen lessons in the terminal, basics then intermediate, where every figure is computed by the solver for the machine you are sitting at." href=${JSON.stringify(`${BASE}/lab/#edu-why`)} />`,
		'</CardGrid>',
		'',
	].join('\n'),
);

if (errors.length) {
	console.error(`sync-docs: ${errors.length} unmappable link(s):\n  ${errors.join('\n  ')}`);
	process.exit(1);
}
console.log(`sync-docs: ${docs.length} docs + ${adrs.length} ADRs -> ${relative(SITE, OUT)}`);
