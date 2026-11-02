// Crawl the built site the way a visitor does: start at dist/index.html, follow
// every href, src and data-svg across every HTML page, and fail if any internal
// target is not a file in dist. starlight-links-validator only sees Starlight's
// own pages; the landing page, the lab and the manual sidebar links are outside
// it, which is how two sidebar links shipped as /onpar/onpar/ (404) on 2026-11-01.
//
// Usage: node scripts/check-links.mjs [dist] [base]
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, posix, resolve } from 'node:path';

const DIST = resolve(process.argv[2] ?? 'dist');
const BASE = (process.argv[3] ?? '/onpar').replace(/\/$/, '');

const htmlFiles = [];
const walk = (dir) => {
	for (const name of readdirSync(dir)) {
		const p = join(dir, name);
		if (statSync(p).isDirectory()) walk(p);
		else if (name.endsWith('.html')) htmlFiles.push(p);
	}
};
walk(DIST);

// A URL path under BASE → does a file exist for it in dist?
const resolves = (urlPath) => {
	let p = urlPath.split('#')[0].split('?')[0];
	if (!p.startsWith(BASE + '/') && p !== BASE) return { ok: false, why: `outside base ${BASE}` };
	p = p.slice(BASE.length) || '/';
	const candidates = p.endsWith('/') ? [p + 'index.html'] : [p, p + '/index.html', p + '.html'];
	return candidates.some((c) => existsSync(join(DIST, c))) ? { ok: true } : { ok: false, why: 'no such file' };
};

const bad = [];
let checked = 0;
for (const file of htmlFiles) {
	const pageUrl = BASE + '/' + posix.relative(DIST, file).replace(/\\/g, '/');
	const html = readFileSync(file, 'utf8');
	for (const [, , value] of html.matchAll(/\b(href|src|data-svg)=["']([^"']+)["']/g)) {
		if (/^(https?:|mailto:|javascript:|data:|tel:)/.test(value)) continue;
		if (value.startsWith('#') || value.startsWith('//')) continue;
		const target = value.startsWith('/') ? value : posix.resolve(posix.dirname(pageUrl), value);
		checked += 1;
		const r = resolves(target);
		if (!r.ok) bad.push(`${pageUrl} → ${value} (${target}: ${r.why})`);
	}
}

if (bad.length) {
	console.error(`check-links: ${bad.length} dead link(s) in ${htmlFiles.length} pages (${checked} checked):`);
	for (const b of [...new Set(bad)].sort()) console.error('  ' + b);
	process.exit(1);
}
console.log(`check-links: ${checked} internal links across ${htmlFiles.length} pages all resolve under ${BASE}`);
