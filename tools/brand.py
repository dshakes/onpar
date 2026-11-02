#!/usr/bin/env python3
"""Generate onpar's brand: the mark, the wordmark, the lockups, the icons and the tokens.

Every brand file is written by this script. Nobody hand-edits an SVG under
site/assets/brand/ or docs/assets/brand/; change the geometry or the palette
here and run it again. Why it is shaped the way it is: docs/brand/README.md.

    python3 tools/brand.py

Writes the same files to site/assets/brand/ (what the site serves) and
docs/assets/brand/ (what the README links), plus site/public/favicon.svg for
the Starlight docs. Then it re-reads every file it wrote and checks it.

## The idea

The name is a claim: "on par", two things level. In onpar the two things are
the incumbent (the closed model you pay for, amber) and the candidate (the open
model you are proving, teal), and what decides whether they are level is the
gate. So the mark is a two-bar chart under a gate line: both bars reach the
line, and the line is level because they do. The wordmark is drawn from the
same stroke, so its x-height is the height of the bars.

## Three candidates were drawn; one was kept

A. SPLIT DISC. A disc cut down the middle by a gap, amber half and teal half,
   brought level into one circle. Strongest silhouette at 16 px. Rejected: it is
   the ◐ glyph, which is already the site's theme toggle, and it reads as a
   yin-yang or a contrast icon before it reads as anything about models.

B. DOT, GATE, DOT. The embryo on the old site header and og card: an amber dot,
   a short gate line, a teal dot, side by side. Rejected: it does not survive
   16 px (each dot becomes two pixels and the line one), and side by side says
   "connected", not "level". It reads as a dumbbell.

C. THRESHOLD BARS (kept). Two bars, amber and teal, both reaching a gate line
   above them. It is what an onpar result looks like: the candidate scored up to
   the line the incumbent set. It is rectangles on a pixel grid, so the 16 px
   favicon is drawn on whole pixels and stays crisp, and in one colour it still
   reads (two equal bars under a line). Animated, the teal bar rises and the
   tilted gate settles level: the whole product in one second.

`candidates.svg` draws all three side by side at 64 px and 16 px.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTS = (ROOT / "site" / "assets" / "brand", ROOT / "docs" / "assets" / "brand")
FAVICON = ROOT / "site" / "public" / "favicon.svg"

# --- palette ------------------------------------------------------------------
# The site's validated two-pole palette (formerly site/index.html :root). Dark
# is the designed mode; light is re-picked, not inverted, because #2fd4c4 on
# white is a pastel nobody can read.
DARK = {
    "void": "#050a0f",
    "deep": "#08111a",
    "panel": "#0b1620",
    "panel2": "#0f1e2b",
    "ink": "#e9f4f8",
    "ink2": "#93aab8",
    "ink3": "#5b7385",
    "line": "#152633",
    "line2": "#1e3546",
    "incumbent": "#f0a340",
    "incumbent-dim": "#7a5420",
    "candidate": "#2fd4c4",
    "candidate-dim": "#186b64",
    "candidate-wash": "#0c2a2a",
    "proof": "#8fb3ff",
    "alarm": "#ff6b81",
}
LIGHT = {
    "void": "#fbfcfd",
    "deep": "#f3f6f8",
    "panel": "#ffffff",
    "panel2": "#f7f9fb",
    "ink": "#0d1620",
    "ink2": "#42586a",
    "ink3": "#6b8296",
    "line": "#dde5ea",
    "line2": "#c9d6de",
    "incumbent": "#b45309",
    "incumbent-dim": "#e0b184",
    "candidate": "#0d7d72",
    "candidate-dim": "#7cc4bc",
    "candidate-wash": "#c9f2ee",
    "proof": "#3d5fb8",
    "alarm": "#c02a41",
}

# (gate, incumbent bar, candidate bar, wordmark) per colourway.
WAYS = {
    "light": (LIGHT["ink"], LIGHT["incumbent"], LIGHT["candidate"], LIGHT["ink"]),
    "dark": (DARK["ink"], DARK["incumbent"], DARK["candidate"], DARK["ink"]),
    "black": ("#000000",) * 4,
    "white": ("#ffffff",) * 4,
}

# --- the mark (candidate C), on a 64-unit grid --------------------------------
# Drawn at 4x the 16 px favicon grid, so every edge lands on a whole favicon
# pixel: gate 1..15 x 2..4, bars 2..7 and 9..14 x 6..14 (in 16 px units).
GATE = (4, 8, 56, 8)  # x, y, w, h
INC = (8, 24, 20, 32)
CAN = (36, 24, 20, 32)
RADIUS = 2


def rect(r: tuple[int, int, int, int], fill: str, cls: str = "", rx: float = RADIUS) -> str:
    c = f' class="{cls}"' if cls else ""
    x, y, w, h = r
    return f'<rect{c} x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>'


def mark_body(way: str, animated: bool = False) -> str:
    gate, inc, can, _ = WAYS[way]
    return (
        rect(GATE, gate, "g" if animated else "")
        + rect(INC, inc)
        + rect(CAN, can, "c" if animated else "")
    )


# --- the wordmark ---------------------------------------------------------------
# "onpar" built, not typeset: every letter is a ring, a stem or an arch in one
# stroke weight, so it renders identically everywhere with no font to load, and
# its geometry matches the mark's. Designed at x-height 40, stroke 8.
XH, STROKE, DESC, TRACK = 40, 8, 16, 7


def _ring(cx: float) -> str:
    r = (XH - STROKE) / 2
    return f"M{cx - r} {XH / 2}a{r} {r} 0 1 0 {2 * r} 0a{r} {r} 0 1 0 {-2 * r} 0"


def _letters() -> tuple[str, float]:
    """The wordmark path (top of x-height at y=0, baseline at y=XH) and its width."""
    h, s, r = XH, STROKE / 2, (XH - STROKE) / 2
    arch = 14  # n and r shoulders: centre line radius
    d, x = [], 0.0
    # o
    d.append(_ring(x + XH / 2))
    x += XH + TRACK
    # n: stem, then an arch to a second stem
    w = 36
    d.append(f"M{x + s} 0V{h}M{x + s} {s + arch}a{arch} {arch} 0 0 1 {w - 2 * s} 0V{h}")
    x += w + TRACK
    # p: descending stem and a ring
    d.append(f"M{x + s} 0V{h + DESC}")
    d.append(_ring(x + s + r))
    x += XH + TRACK
    # a: ring and a closing stem
    d.append(_ring(x + XH / 2))
    d.append(f"M{x + XH - s} 0V{h}")
    x += XH + TRACK
    # r: stem and a quarter shoulder ending in a short flat terminal
    d.append(f"M{x + s} {h}V0M{x + s} {s + arch}a{arch} {arch} 0 0 1 {arch} {-arch}h6")
    x += s + arch + 6
    return "".join(d), x


WORD_D, WORD_W = _letters()
WORD_H = XH + DESC


def word(color: str, x: float, y: float, scale: float) -> str:
    """The wordmark with its x-height top at (x, y)."""
    return (
        f'<path transform="translate({x:g} {y:g}) scale({scale:g})" d="{WORD_D}" '
        f'fill="none" stroke="{color}" stroke-width="{STROKE}"/>'
    )


# --- documents --------------------------------------------------------------------

ANIM_CSS = (
    "<style>"
    # Frame one is the rested, level mark; the uneven state starts 2% in and
    # is gone by the end. So a renderer that ignores CSS animation, one that
    # freezes an <img> SVG on its first frame, and a reader who asked for less
    # motion all see the level mark. It plays once, then rests.
    ".c,.g{transform-box:fill-box;animation:1.5s linear .2s 1}"
    ".c{transform-origin:50% 100%;animation-name:rise}"
    ".g{transform-origin:0% 50%;animation-name:settle}"
    "@keyframes rise{0%{transform:none}"
    "2%{transform:scaleY(.62);animation-timing-function:cubic-bezier(.2,.7,.3,1)}"
    "100%{transform:none}}"
    "@keyframes settle{0%{transform:none}"
    "2%{transform:rotate(8deg);animation-timing-function:cubic-bezier(.2,.7,.3,1)}"
    "100%{transform:none}}"
    "@media (prefers-reduced-motion:reduce){.c,.g{animation:none}}"
    "</style>"
)


def svg(w: float, h: float, label: str, body: str, vb: str | None = None) -> str:
    vb = vb or f"0 0 {w:g} {h:g}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{w:g}" '
        f'height="{h:g}" role="img" aria-label="{label}">{body}</svg>\n'
    )


def mark_svg(way: str, animated: bool = False, size: int = 64) -> str:
    css = ANIM_CSS if animated else ""
    return svg(size, size, "onpar", css + mark_body(way, animated), vb="0 0 64 64")


# Horizontal lockup: the mark's content box (x 4..60, y 8..56), then the
# wordmark with its x-height spanning the bars (y 24..56), so the letters stand
# on the same baseline as the bars and the gate line rides above both.
H_SCALE = INC[3] / XH  # x-height == bar height
H_GAP = 14


def lockup_h_body(way: str, animated: bool = False) -> tuple[str, float, float]:
    color = WAYS[way][3]
    x0 = GATE[0] + GATE[2] + H_GAP
    body = mark_body(way, animated) + word(color, x0, INC[1], H_SCALE)
    w = x0 + WORD_W * H_SCALE + GATE[0]
    return body, w, 64 + DESC * H_SCALE - 8  # room for the p's descender


def lockup_group(way: str, x: float, y: float, scale: float) -> str:
    """The horizontal lockup as a <g>, its gate line's top-left at (x, y): for
    composing into another SVG (the og card) without a second copy of the art."""
    body, _, _ = lockup_h_body(way)
    return (
        f'<g transform="translate({x - GATE[0] * scale:g} {y - GATE[1] * scale:g}) '
        f'scale({scale:g})">{body}</g>'
    )


def lockup_h_svg(way: str, animated: bool = False) -> str:
    body, w, h = lockup_h_body(way, animated)
    css = ANIM_CSS if animated else ""
    # Trim the mark's top padding: the gate line is the top of the lockup.
    w, h = round(w), round(h - 4)
    return svg(w, h, "onpar", css + body, vb=f"0 4 {w} {h}")


def lockup_v_svg(way: str) -> str:
    color = WAYS[way][3]
    scale = 0.5
    ww = WORD_W * scale
    w = max(ww, 64) + 16
    mx = (w - 64) / 2
    body = f'<g transform="translate({mx:g} 0)">{mark_body(way)}</g>' + word(
        color, (w - ww) / 2, 70, scale
    )
    h = 70 + WORD_H * scale + 6
    return svg(round(w), round(h), "onpar", body, vb=f"0 0 {round(w)} {round(h)}")


def app_icon_svg() -> str:
    # Mark at 62% of the tile on the deep surface, on-dark colours.
    s = 512 * 0.62 / 64
    off = (512 - 64 * s) / 2
    body = (
        f'<rect width="512" height="512" rx="112" fill="{DARK["deep"]}"/>'
        f'<rect x="1.5" y="1.5" width="509" height="509" rx="110.5" fill="none" '
        f'stroke="{DARK["line2"]}" stroke-width="3"/>'
        f'<g transform="translate({off:g} {off:g}) scale({s:g})">{mark_body("dark")}</g>'
    )
    return svg(512, 512, "onpar", body)


def favicon_svg() -> str:
    # Whole-pixel geometry on the 16 grid, square corners (a 0.5 px radius is
    # blur at this size). The gate follows the browser chrome's theme.
    body = (
        "<style>.g{fill:" + LIGHT["ink"] + "}"
        "@media (prefers-color-scheme:dark){.g{fill:" + DARK["ink"] + "}}</style>"
        '<rect class="g" x="1" y="2" width="14" height="2"/>'
        f'<rect x="2" y="6" width="5" height="8" fill="{DARK["incumbent"]}"/>'
        f'<rect x="9" y="6" width="5" height="8" fill="{DARK["candidate"]}"/>'
    )
    return svg(16, 16, "onpar", body)


def candidates_svg() -> str:
    """A, B and C side by side, at 64 px and at 16 px, on both surfaces."""
    _, inc, can, _ = WAYS["dark"]
    a = (
        f'<path d="M30 8a24 24 0 0 0 0 48z" fill="{inc}"/>'
        f'<path d="M34 8a24 24 0 0 1 0 48z" fill="{can}"/>'
    )
    b = (
        f'<circle cx="12" cy="32" r="8" fill="{inc}"/>'
        f'<rect x="24" y="31" width="16" height="2" fill="{DARK["line2"]}"/>'
        f'<circle cx="52" cy="32" r="8" fill="{can}"/>'
    )
    c = mark_body("dark")
    out = [f'<rect width="600" height="260" fill="{DARK["deep"]}"/>']
    for i, (body, name) in enumerate(((a, "A split disc"), (b, "B dot gate dot"), (c, "C kept"))):
        x = 40 + i * 190
        out.append(f'<g transform="translate({x} 40)">{body}</g>')
        out.append(f'<g transform="translate({x + 90} 64) scale(0.25)">{body}</g>')
        # Labels as shapes would be absurd; this sheet is the one brand file
        # with text, and it carries no presentation attributes.
        out.append(
            f'<text x="{x}" y="150" style="fill:{DARK["ink2"]};font:14px ui-sans-serif,'
            f'system-ui,sans-serif">{name}</text>'
        )
    return svg(600, 180, "Three mark candidates; C was kept", "".join(out), vb="0 0 600 180")


# --- tokens.css ---------------------------------------------------------------------

TYPE = {
    "font-sans": "ui-sans-serif,-apple-system,'Segoe UI',Inter,system-ui,sans-serif",
    "font-mono": "ui-monospace,SFMono-Regular,'SF Mono',Menlo,monospace",
    "fs-2xs": "11px",
    "fs-xs": "12.5px",
    "fs-sm": "14.5px",
    "fs-base": "16px",
    "fs-md": "17.5px",
    "fs-lg": "19.5px",
    "fs-h2": "clamp(26px,3.5vw,40px)",
    "fs-h1": "clamp(35px,5.4vw,64px)",
    "track-tight": "-.03em",
    "track-caps": ".16em",
}
SHAPE = {
    "r-xs": "5px",
    "r-sm": "8px",
    "r-md": "10px",
    "r-lg": "12px",
    "r-xl": "14px",
    "sp-1": "4px",
    "sp-2": "8px",
    "sp-3": "12px",
    "sp-4": "16px",
    "sp-5": "24px",
    "sp-6": "32px",
    "sp-7": "48px",
    "sp-8": "86px",
}
MOTION = {
    "dur-fast": "200ms",
    "dur-base": "350ms",
    "dur-slow": "750ms",
    "ease-out": "cubic-bezier(.2,.7,.3,1)",
}


def _decls(d: dict[str, str], indent: str = "  ") -> str:
    return "".join(f"{indent}--{k}:{v};\n" for k, v in d.items())


def tokens_css() -> str:
    return (
        "/* onpar design tokens. GENERATED by tools/brand.py; edit there, not here.\n"
        "   The single source for colour, type, radius, spacing and motion across the\n"
        "   landing page, the lab and the Starlight docs. Dark is the default;\n"
        "   [data-theme='light'] selects light, and with no attribute set the\n"
        "   reader's system preference decides. */\n"
        ":root{\n" + _decls(DARK) + _decls(TYPE) + _decls(SHAPE) + _decls(MOTION) + "}\n"
        ":root[data-theme='light']{\n" + _decls(LIGHT) + "}\n"
        "@media (prefers-color-scheme:light){\n"
        "  :root:not([data-theme='dark']){\n" + _decls(LIGHT, "    ") + "  }\n}\n"
        "@media (prefers-reduced-motion:reduce){\n"
        "  :root{--dur-fast:0ms;--dur-base:0ms;--dur-slow:0ms}\n}\n"
        "/* Brand images come in a dark and a light colourway; show the one that\n"
        "   matches the theme by the same rule the tokens follow. */\n"
        ".for-light{display:none}\n"
        ":root[data-theme='light'] .for-light{display:inline-block}\n"
        ":root[data-theme='light'] .for-dark{display:none}\n"
        "@media (prefers-color-scheme:light){\n"
        "  :root:not([data-theme='dark']) .for-light{display:inline-block}\n"
        "  :root:not([data-theme='dark']) .for-dark{display:none}\n}\n"
    )


# --- write and check ---------------------------------------------------------------


def files() -> dict[str, str]:
    out = {
        "mark.svg": mark_svg("light"),
        "mark-on-dark.svg": mark_svg("dark"),
        "mark-mono-black.svg": mark_svg("black"),
        "mark-mono-white.svg": mark_svg("white"),
        "lockup-horizontal.svg": lockup_h_svg("light"),
        "lockup-horizontal-on-dark.svg": lockup_h_svg("dark"),
        "lockup-horizontal-mono-black.svg": lockup_h_svg("black"),
        "lockup-horizontal-mono-white.svg": lockup_h_svg("white"),
        "lockup-stacked.svg": lockup_v_svg("light"),
        "lockup-stacked-on-dark.svg": lockup_v_svg("dark"),
        "logo-animated.svg": lockup_h_svg("light", animated=True),
        "logo-animated-on-dark.svg": lockup_h_svg("dark", animated=True),
        "app-icon.svg": app_icon_svg(),
        "favicon.svg": favicon_svg(),
        "candidates.svg": candidates_svg(),
        "tokens.css": tokens_css(),
    }
    return out


def check(name: str, text: str) -> None:
    """The repo's SVG rules (tests/test_docs_lab.py), applied to brand files."""
    if not name.endswith(".svg"):
        assert "--candidate:" in text and "[data-theme='light']" in text, name
        return
    head = text[:600]
    vb = re.search(r'viewBox="([\d.\- ]+)"', head)
    wh = re.search(r'width="([\d.]+)"\s+height="([\d.]+)"', head)
    assert vb and wh, f"{name}: needs viewBox, width and height"
    _, _, vw, vh = map(float, vb.group(1).split())
    w, h = map(float, wh.groups())
    assert abs(w / h - vw / vh) < 0.01, f"{name}: {w}x{h} vs viewBox {vw}x{vh}"
    for tag in re.findall(r"<text[^>]*>", text):
        assert not re.search(r'\s(fill|font-size)="', tag), f"{name}: attribute on {tag}"
    for style in re.findall(r'style="([^"]*)"', text):
        for decl in style.split(";"):
            prop, _, v = decl.partition(":")
            bad = prop.strip() in ("font-size", "stroke-width", "letter-spacing")
            assert not (bad and re.fullmatch(r"[0-9.]+", v.strip())), f"{name}: unitless {decl}"
    if "animation" in text:
        assert "prefers-reduced-motion" in text, name
        # Frame one must be the rested mark (see ANIM_CSS).
        assert all(
            k.startswith("0%{transform:none}")
            for k in re.findall(r"@keyframes \w+\{(.*?)\}\}", text)
        ), name


def main() -> None:
    out = files()
    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)
        for name, text in out.items():
            (d / name).write_text(text)
    FAVICON.write_text(out["favicon.svg"])
    for d in OUTS:
        for name in out:
            check(name, (d / name).read_text())
    check("favicon.svg", FAVICON.read_text())
    print(f"brand: {len(out)} files -> " + ", ".join(str(d.relative_to(ROOT)) for d in OUTS))


if __name__ == "__main__":
    main()
