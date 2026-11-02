// Shared shapes for the interactive diagrams. Every claim a diagram makes
// carries a `Src` so readers can jump to the code that backs it.
// Ported from github.com/dshakes/pal (site/src/components/diagrams, Apache-2.0).

export type Src = { p: string };

const REPO = "https://github.com/dshakes/onpar/blob/main";

export function srcHref(s: Src): string {
  // No #L anchors: they point at moving lines on main and drift. Link the file.
  return `${REPO}/${s.p}`;
}

export function srcLabel(s: Src): string {
  return s.p.split("/").slice(-2).join("/");
}

export type Tone = "host" | "guest" | "deny" | "store" | "client" | "plain";
