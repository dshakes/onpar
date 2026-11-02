// Copy the hand-written static site into Astro's output so it keeps its URLs:
//   index.html -> /onpar/   install.sh -> /onpar/install.sh
//   assets/**  -> /onpar/assets/**   lab/** -> /onpar/lab/**
//   ../docs/assets/** -> /onpar/assets/** (copied last: the diagrams' source of truth)
// Starlight owns only /onpar/docs/**; there is no index.md at the content root, so
// it emits no /onpar/ page for the landing page to collide with. Checked below.
import { cpSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SITE = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIST = join(SITE, 'dist');

if (existsSync(join(SITE, 'src/content/docs/index.md')) || existsSync(join(SITE, 'src/content/docs/index.mdx'))) {
	console.error('postbuild: Starlight has a root index page; it would overwrite the landing page');
	process.exit(1);
}
for (const p of ['index.html', 'install.sh', 'assets', 'lab']) {
	cpSync(join(SITE, p), join(DIST, p), { recursive: true });
}
// docs/assets/ is the diagrams' source of truth (tools/diagrams.py writes it), so it is
// copied last and wins, the same precedence the Pages workflow has always had.
cpSync(join(SITE, '../docs/assets'), join(DIST, 'assets'), { recursive: true });
console.log('postbuild: landing page, install.sh, assets/ and lab/ copied into dist/');
