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
		// ADR status lines ("Status: accepted · date") say nothing about the page.
		if (/^[*_]*status\b/i.test(text)) continue;
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

// /onpar/docs/ itself: a generated index of everything above.
const titleOf = (f) => cleanTitle(readFileSync(join(REPO, f), 'utf8').match(/^# (.+)$/m)?.[1] ?? f);
const item = (f) => `- [${titleOf(f).replace(/[[\]]/g, '\\$&')}](${rewrite(posix.basename(f), f)})`;
writeFileSync(
	join(OUT, 'index.md'),
	[
		'---',
		'title: "onpar docs"',
		'description: "The written docs for onpar: the field guide, product specs, operating docs and every architecture decision record."',
		'editUrl: false',
		'---',
		'',
		`Generated from the repo's [docs/](${GITHUB}/tree/main/docs) folder. The interactive memory lab and the CLI reference live at [the lab](${BASE}/lab/).`,
		'',
		'## Docs',
		'',
		...docs.map(item),
		'',
		'## Decisions',
		'',
		...adrs.map(item),
		'',
	].join('\n'),
);

if (errors.length) {
	console.error(`sync-docs: ${errors.length} unmappable link(s):\n  ${errors.join('\n  ')}`);
	process.exit(1);
}
console.log(`sync-docs: ${docs.length} docs + ${adrs.length} ADRs -> ${relative(SITE, OUT)}`);
