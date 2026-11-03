"""Draws the ProRate figures as SVG with computed geometry.

Run from the project root:  python figures/make_overview.py
Writes figures/prorate_overview.svg (encoding) and figures/prorate_boundary.svg (boundary coding and
decoding), plus .png / .pdf for each when cairosvg is installed.
Log: logs/make_overview.log
"""
import logging
import math
import random
from pathlib import Path
from xml.sax.saxutils import escape as esc
import cairocffi as cairo

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"

TEXT_SCALE = 1.1
W1, H1 = 960, 780
W2, H2 = 840, 800
BOX_Y, BOX_W = 520, 450
PART_X, TOK_X = 20, 490
PANEL_POS = {"geometry": (20, 20), "quant": (440, 20), "loop": (20, 385), "decoder": (440, 385)}

FONT = "Arial, Helvetica, sans-serif"
INK, GREY = "#222222", "#8a8a8a"
BLUE_FILL, BLUE_EDGE = "#CFE0F7", "#4C78C8"
GREEN_FILL, GREEN_EDGE = "#CDE8C8", "#4C9A45"
RED = "#B03030"

N_RESIDUES = 12
LENGTHS = [2, 4, 3, 3]
FRAGMENT_IDS = [97, 437, 221, 163]
BOUNDARY_IDS = [318540127, 41872390, 502216854]
LEVELS = (16, 16, 64, 16, 32, 64)
AXIS_NAMES = ("\u03b8(C\u03b1\u2013C\u2013N)", "d(C\u2013N)", "\u03c8", "\u03b8(C\u2013N\u2013C\u03b1)", "\u03c9", "\u03c6")
BOUNDARY_BITS = 29
VOCAB = 1577
CODEBOOK_SIZE = {L: (64 if L <= 2 else 256 if L <= 6 else 128) for L in range(1, 17)}

ANGLE_CA_C_N = "#D9680F"
ANGLE_C_N_CA = "#7A3DB8"
DIST_C_N = "#00859B"

# Bond lengths N-CA, CA-C, C-N at 17 px per angstrom and the vertical step of the planar zig-zag.
BOND_PX = (24.8, 25.9, 22.6)
STEP_Y = 14.0
KINDS = ("N", "CA", "C")
# Substituent of each backbone atom: kind, x offset, bond length in px, drawn on the outer side of the zig-zag.
SUBST = {"N": ("H", -1.0, 17.0), "CA": ("R", 0.0, 26.0), "C": ("O", 0.0, 21.0)}
# kind: fill, text color, label, radius
STYLE = {"N": ("#4C6EF5", "#FFFFFF", "N", 9), "CA": ("#B8B8B8", "#222222", "C\u03b1", 9),
         "C": ("#B8B8B8", "#222222", "C", 9), "O": ("#E8423F", "#FFFFFF", "O", 8),
         "H": ("#FFFFFF", "#222222", "H", 6.5), "R": ("#CDB4DB", "#222222", "R", 9)}

X0, BY = 200.0, 110.0
AXIS_Y, BAR_Y, BAR_H = 172, 210, 34
GLYPH_Y, FRAME_H = 308, 96
FID_Y, BID_Y = 384, 444


# ---------------------------------------------------------------- checks
def block_offsets():
    off, out = 0, {}
    for L in range(1, 17):
        out[L] = off
        off += CODEBOOK_SIZE[L]
    return out


def decode_cells(index):
    basis, b = [], 1
    for n in LEVELS:
        basis.append(b)
        b *= n
    cells = [(index // bs) % n for n, bs in zip(LEVELS, basis)]
    assert sum(c * bs for c, bs in zip(cells, basis)) == index
    return cells


def check_inputs():
    assert sum(LENGTHS) == N_RESIDUES, "fragment lengths must cover every residue"
    assert len(FRAGMENT_IDS) == len(LENGTHS)
    assert len(BOUNDARY_IDS) == len(LENGTHS) - 1, "one boundary cell per junction"
    assert math.prod(LEVELS) == 2 ** BOUNDARY_BITS
    off = block_offsets()
    for L, g in zip(LENGTHS, FRAGMENT_IDS):
        assert off[L] <= g < off[L] + CODEBOOK_SIZE[L], f"fragment id {g} is outside the length-{L} block"
    for b in BOUNDARY_IDS:
        assert 0 <= b < 2 ** BOUNDARY_BITS
    logging.info("inputs verified; first boundary cells %s", decode_cells(BOUNDARY_IDS[0]))


SCRIPT_SCALE = 0.7
SCRIPT_SHIFT = {"sub": 3.0, "sup": -4.0}


def text_width(s, px, weight="normal", italic=False):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
    ctx = cairo.Context(surface)
    ctx.select_font_face("Arial", cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if weight == "bold" else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(px)
    return ctx.text_extents(s)[4]


def rich_text(x, y, parts, size=12, anchor="start", weight="normal", fill=INK):
    """parts: list of (string, mode) with mode in 'n', 'it', 'sub', 'sup'. Each part is drawn as its own text element."""
    px = size * TEXT_SCALE
    items, total = [], 0.0
    for s, mode in parts:
        k = px * SCRIPT_SCALE if mode in SCRIPT_SHIFT else px
        italic = mode == "it"
        core = s.strip()
        lead = text_width(" ", k, weight, italic) if s != s.lstrip() else 0.0
        trail = text_width(" ", k, weight, italic) if s != s.rstrip() else 0.0
        w = text_width(core, k, weight, italic) if core else 0.0
        items.append((core, mode, lead, w, trail))
        total += lead + w + trail
    cur = x - {"start": 0.0, "middle": total / 2, "end": total}[anchor]
    out = []
    for core, mode, lead, w, trail in items:
        cur += lead
        if core:
            if mode in SCRIPT_SHIFT:
                out.append(text(cur, y + SCRIPT_SHIFT[mode] * TEXT_SCALE, core, size * SCRIPT_SCALE,
                                "start", weight, fill))
            else:
                out.append(text(cur, y, core, size, "start", weight, fill, italic=(mode == "it")))
        cur += w + trail
    return "".join(out)

# ---------------------------------------------------------------- svg primitives
def text(x, y, s, size=12, anchor="start", weight="normal", fill=INK, italic=False, rotate=None, raw=False,
         scaled=True):
    body = s if raw else esc(s)
    px = size * TEXT_SCALE if scaled else size
    extra = ' font-style="italic"' if italic else ""
    if rotate is not None:
        extra += f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{px:.2f}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{extra}>{body}</text>')


def sub(s, after="", size=12):
    px = size * TEXT_SCALE * 0.7
    return (f'<tspan dy="{3 * TEXT_SCALE:.2f}" font-size="{px:.2f}">{esc(s)}</tspan>'
            f'<tspan dy="{-3 * TEXT_SCALE:.2f}" font-size="{size * TEXT_SCALE:.2f}">{esc(after)}</tspan>')


def sup(s, after="", size=12):
    px = size * TEXT_SCALE * 0.7
    return (f'<tspan dy="{-4 * TEXT_SCALE:.2f}" font-size="{px:.2f}">{esc(s)}</tspan>'
            f'<tspan dy="{4 * TEXT_SCALE:.2f}" font-size="{size * TEXT_SCALE:.2f}">{esc(after)}</tspan>')


def rect(x, y, w, h, fill="none", stroke=GREY, sw=1.2, r=4):
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def segment(x1, y1, x2, y2, stroke=GREY, sw=1.2, marker=None, dash=None):
    m = f' marker-end="url(#{marker})"' if marker else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linecap="round"{d}{m}/>')


def path(d, stroke=GREY, sw=1.2, fill="none", marker=None, transform=None):
    m = f' marker-end="url(#{marker})"' if marker else ""
    t = f' transform="{transform}"' if transform else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{m}{t}/>'


def circle(cx, cy, r, fill, stroke="#555555", sw=0.8):
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def id_box(cx, y, w, h, label, kind, size=13):
    fill, edge = (BLUE_FILL, BLUE_EDGE) if kind == "fragment" else (GREEN_FILL, GREEN_EDGE)
    return rect(cx - w / 2, y, w, h, fill, edge, 1.3, 5) + text(cx, y + h / 2 + size * 0.35, str(label), size, "middle")


def text_box(x, y, w, h, rows, fill="#F6F6F6", stroke="#888888", size=12, gap=14):
    y0 = y + h / 2 - (len(rows) - 1) * gap / 2 + size * 0.35
    out = [rect(x, y, w, h, fill, stroke, 1.2, 6)]
    for i, r in enumerate(rows):
        parts = [(r, "n")] if isinstance(r, str) else r
        out.append(rich_text(x + w / 2, y0 + i * gap, parts, size, "middle"))
    return "".join(out)


def defs():
    def m(i, c):
        return (f'<marker id="{i}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
                f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{c}"/></marker>')
    return "<defs>" + m("arrow-dark", "#333333") + m("arrow-red", RED) + "</defs>"


# ---------------------------------------------------------------- backbone drawing
def chain_atoms(n_res, x0, y0):
    """Planar trans zig-zag of N, CA, C atoms. out is -1 for atoms on the upper vertex, +1 for the lower."""
    atoms, x = [], x0
    for g in range(3 * n_res):
        if g > 0:
            b = BOND_PX[(g - 1) % 3]
            x += math.sqrt(b * b - STEP_Y ** 2)
        up = g % 2 == 0
        atoms.append({"kind": KINDS[g % 3], "x": x, "y": y0 + (-STEP_Y / 2 if up else STEP_Y / 2),
                      "out": -1 if up else 1, "res": g // 3})
    assert len(atoms) == 3 * n_res
    return atoms


def atom_svg(x, y, kind, r=None):
    fill, tc, label, rad = STYLE[kind]
    rad = r or rad
    size = 8.5 if kind == "H" else 11.5
    return circle(x, y, rad, fill) + text(x, y + size * 0.36, label, size, "middle", fill=tc, scaled=False)


def chain_svg(atoms, substituents=True, junction_bonds=()):
    out = []
    if substituents:
        faded = []
        for a in atoms:
            kind, dx, length = SUBST[a["kind"]]
            sx, sy = a["x"] + dx, a["y"] + a["out"] * length
            faded.append(segment(a["x"], a["y"], sx, sy, "#666666", 2))
            faded.append(atom_svg(sx, sy, kind))
        out.append('<g opacity="0.5">' + "".join(faded) + "</g>")
    for i in junction_bonds:
        out.append(segment(atoms[i]["x"], atoms[i]["y"], atoms[i + 1]["x"], atoms[i + 1]["y"], GREEN_EDGE, 9))
    for a, b in zip(atoms, atoms[1:]):
        out.append(segment(a["x"], a["y"], b["x"], b["y"], "#555555", 3))
    for a in atoms:
        out.append(atom_svg(a["x"], a["y"], a["kind"]))
    return "".join(out)


def prototype_atoms(seg, y0, bend):
    """Copy of a backbone stretch with every bond length and the local zig-zag preserved, plus a gentle
    curvature set by bend (radians), centered vertically on y0."""
    mid = max((len(seg) - 1) / 2, 1)
    out, x, y = [], seg[0]["x"], 0.0
    for g, a in enumerate(seg):
        if g > 0:
            dx, dy = a["x"] - seg[g - 1]["x"], a["y"] - seg[g - 1]["y"]
            length = math.hypot(dx, dy)
            ang = math.atan2(dy, dx) + bend * (g - mid) / mid
            x, y = x + length * math.cos(ang), y + length * math.sin(ang)
        out.append(dict(a, x=x, y=y))
    shift = y0 - sum(a["y"] for a in out) / len(out)
    return [dict(a, y=a["y"] + shift) for a in out]


# ---------------------------------------------------------------- figure 1: encoding
def main_diagram():
    atoms = chain_atoms(N_RESIDUES, X0, BY)
    res_x = [sum(a["x"] for a in atoms if a["res"] == r) / 3 for r in range(N_RESIDUES)]
    cut_idx, r0 = [], 0
    for L in LENGTHS[:-1]:
        r0 += L
        cut_idx.append(3 * r0 - 1)                       # the C atom just before each cut
    cut_x = [(atoms[i]["x"] + atoms[i + 1]["x"]) / 2 for i in cut_idx]
    edges = [atoms[0]["x"] - 16] + cut_x + [atoms[-1]["x"] + 16]
    p = [text(30, 32, "Encoding", 20, weight="bold")]

    for x in cut_x:
        p.append(segment(x, 156, x, BID_Y, "#777777", 1, dash="4 3"))

    p.append(chain_svg(atoms, junction_bonds=cut_idx))
    p.append(text(30, BY + 5, "Backbone", 13.5))

    p.append(segment(X0 - 10, AXIS_Y, edges[-1], AXIS_Y, "#444444", 1.2))
    for r, x in enumerate(res_x):
        p.append(segment(x, AXIS_Y - 4, x, AXIS_Y + 4, "#444444", 1))
        p.append(text(x, AXIS_Y + 21, str(r + 1), 12.5, "middle"))
    p.append(text(30, AXIS_Y + 5, "Residue", 13.5))

    p.append(text(30, BAR_Y + 16, "Fragment length", 13.5))
    p.append(text(30, BAR_Y + 32, "(optimal partitioning)", 11.5, fill="#555555"))
    p.append(text(30, GLYPH_Y - 2, "Nearest codebook", 13.5))
    p.append(text(30, GLYPH_Y + 14, "entry (Kabsch RMSD)", 13.5))
    p.append(text(30, FID_Y + 20, "Fragment ID", 13.5))
    p.append(text(30, BID_Y + 20, "Boundary cell ID", 13.5))

    rng = random.Random(11)
    start = 0
    for i, (L, fid) in enumerate(zip(LENGTHS, FRAGMENT_IDS)):
        left, right = edges[i] + 2, edges[i + 1] - 2
        cx = (left + right) / 2
        p.append(rect(left, BAR_Y, right - left, BAR_H, BLUE_FILL if i % 2 == 0 else "#B9D2F2", BLUE_EDGE, 1.3, 5))
        p.append(text(cx, BAR_Y + BAR_H / 2 + 7, str(L), 19, "middle", "bold"))
        p.append(segment(cx, BAR_Y + BAR_H, cx, GLYPH_Y - FRAME_H / 2 - 2, "#333333", 1.2, marker="arrow-dark"))
        p.append(rect(left - 1.5, GLYPH_Y - FRAME_H / 2, right - left + 3, FRAME_H, "#F3F7FD", BLUE_EDGE, 1.2, 5))
        seg = atoms[start:start + 3 * L]
        glyph = prototype_atoms(seg, GLYPH_Y, rng.choice((-1, 1)) * rng.uniform(0.08, 0.14))
        assert len(glyph) == 3 * L
        p.append(chain_svg(glyph))
        start += 3 * L
        p.append(segment(cx, GLYPH_Y + FRAME_H / 2, cx, FID_Y - 2, "#333333", 1.2, marker="arrow-dark"))
        p.append(id_box(cx, FID_Y, 58, 28, fid, "fragment", 14.5))
        logging.info("fragment %d: length %d, glyph atoms %d, id %d", i + 1, L, len(glyph), fid)
    for x, bid in zip(cut_x, BOUNDARY_IDS):
        p.append(id_box(x, BID_Y, 100, 28, bid, "boundary", 13.5))

    yb = BID_Y + 44
    p.append(path(f"M {X0 - 12},{yb - 8} L {X0 - 12},{yb} L {edges[-1]:.1f},{yb} L {edges[-1]:.1f},{yb - 8}",
                  stroke=INK, sw=1.6))
    xc = (X0 - 12 + edges[-1]) / 2
    p.append(segment(xc, yb, xc, BOX_Y - 4, "#333333", 1.5, marker="arrow-dark"))
    return "".join(p)


def partition_box(bx, by):
    bw, bh, o = BOX_W, 196, 7
    p = [rect(bx, by, bw, bh, "#F3F7FD", BLUE_EDGE, 1.3, 8),
         text(bx + bw / 2, by + o + 30, "Optimal partitioning", 20, "middle", "bold"),
         text(bx + bw / 2, by + o + 56, "Cost of each fragment =", 15, "middle")]
    chips = [
        ("Fragment error", [("3L d", "n"), ("2", "sup")], BLUE_FILL, BLUE_EDGE),
        ("Boundary penalty", [("\u03b7 L", "n"), ("\u03b3", "sup")], GREEN_FILL, GREEN_EDGE),
        ("Rate", [("\u03bb \u00d7 bits", "n")], "#EDEDED", "#888888"),
    ]
    cw, gap = 136, 14
    x0 = bx + (bw - (3 * cw + 2 * gap)) / 2
    for i, (label, formula, fill, edge) in enumerate(chips):
        x = x0 + i * (cw + gap)
        p.append(rect(x, by + o + 68, cw, 56, fill, edge, 1.3, 6))
        p.append(text(x + cw / 2, by + o + 90, label, 13.5, "middle", "bold"))
        p.append(rich_text(x + cw / 2, by + o + 112, formula, 16, "middle"))
        if i < len(chips) - 1:
            p.append(text(x + cw + gap / 2, by + o + 104, "+", 20, "middle"))
    p.append(text(bx + bw / 2, by + o + 148, "Dynamic programming finds the cuts with the lowest total.", 14.5, "middle"))
    p.append(text(bx + bw / 2, by + o + 167, "Larger \u03bb gives fewer, longer fragments.", 14.5, "middle"))
    return "".join(p)


def token_stream(bx, by):
    bw, bh = BOX_W, 196
    p = [rect(bx, by, bw, bh, "#FAFAFA", "#888888", 1.3, 8),
         text(bx + 16, by + 32, "Integer token stream", 20, weight="bold"),
         text(bx + 16, by + 78, "Fragment IDs", 15),
         text(bx + 16, by + 124, "Boundary cell IDs", 15)]
    for i, fid in enumerate(FRAGMENT_IDS):
        p.append(id_box(bx + 183 + i * 74, by + 56, 66, 32, fid, "fragment", 15))
    for i, bid in enumerate(BOUNDARY_IDS):
        p.append(id_box(bx + 204 + i * 100, by + 102, 92, 32, bid, "boundary", 14))
    p.append(rich_text(bx + 16, by + 160, [("Fragment ID: log", "n"), ("2", "sub"),
                                           (f"|V| = {math.log2(VOCAB):.1f} bits (|V| = {VOCAB} codes)", "n")], 14))
    p.append(text(bx + 16, by + 180, f"Boundary cell ID: {BOUNDARY_BITS} bits (six quantized values)", 14))
    return "".join(p)


# ---------------------------------------------------------------- figure 2: boundary coding and decoding
def panel(key, letter, title, body, subtitle=None):
    x, y = PANEL_POS[key]
    sub_svg = text(34, 43, subtitle, 13.5) if subtitle else ""
    return (f'<g transform="translate({x} {y})">' + text(0, 24, letter, 26, weight="bold")
            + text(34, 24, title, 17, weight="bold") + sub_svg + body + "</g>")


def angle_arc(c, p1, p2, r, stroke=INK):
    a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
    a2 = math.atan2(p2[1] - c[1], p2[0] - c[0])
    d = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    s = (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1))
    e = (c[0] + r * math.cos(a2), c[1] + r * math.sin(a2))
    mid = a1 + d / 2
    svg = path(f"M {s[0]:.2f},{s[1]:.2f} A {r} {r} 0 0 {1 if d > 0 else 0} {e[0]:.2f},{e[1]:.2f}", stroke=stroke, sw=1.5)
    return svg, (c[0] + r * math.cos(mid), c[1] + r * math.sin(mid))


def torsion_arrow(p, q, rx=9, ry=21):
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
    t0, t1 = math.radians(-125), math.radians(125)
    d = (f"M {rx * math.cos(t0):.2f},{ry * math.sin(t0):.2f} "
         f"A {rx} {ry} 0 1 1 {rx * math.cos(t1):.2f},{ry * math.sin(t1):.2f}")
    return path(d, stroke=INK, sw=1.6, marker="arrow-dark", transform=f"translate({mx:.2f} {my:.2f}) rotate({ang:.2f})")


def panel_boundary_geometry():
    order = ["N0", "CA0", "C0", "N1", "CA1", "C1"]
    xy = {"N0": (45, 79), "CA0": (108, 100), "C0": (132, 137),
          "N1": (234, 137), "CA1": (255, 212), "C1": (316, 230)}
    name = {"N0": "N", "CA0": "C\u03b1", "C0": "C", "N1": "N", "CA1": "C\u03b1", "C1": "C"}
    p = []
    for a, b in zip(order, order[1:]):
        if (a, b) == ("C0", "N1"):
            color = "#8f8f8f"
        else:
            color = BLUE_EDGE if a.endswith("0") else GREEN_EDGE
        p.append(segment(*xy[a], *xy[b], color, 5))

    # psi about CA_i-C_i, omega about C_i-N_i+1, phi about N_i+1-CA_i+1 (Section 2.1)
    p.append(torsion_arrow(xy["CA0"], xy["C0"], 7, 15))
    p.append(torsion_arrow(xy["C0"], xy["N1"], 8, 18))
    p.append(torsion_arrow(xy["N1"], xy["CA1"], 8, 18))
    p.append(text(94, 138, "\u03c8", 17, "middle", "bold"))
    p.append(text(183, 112, "\u03c9", 17, "middle", "bold"))
    p.append(text(277, 174, "\u03c6", 17, "middle", "bold"))

    arc1, mid1 = angle_arc(xy["C0"], xy["CA0"], xy["N1"], 31, ANGLE_CA_C_N)
    p.append(arc1)
    p.append(segment(mid1[0], mid1[1], 172, 93, ANGLE_CA_C_N, 1.2, dash="3 3"))
    p.append(text(176, 96, "\u03b8(C\u03b1\u2013C\u2013N)", 13.5, "start", "bold", ANGLE_CA_C_N))

    arc2, mid2 = angle_arc(xy["N1"], xy["C0"], xy["CA1"], 24, ANGLE_C_N_CA)
    p.append(arc2)
    p.append(segment(mid2[0], mid2[1], 195, 194, ANGLE_C_N_CA, 1.2, dash="3 3"))
    p.append(text(195, 207, "\u03b8(C\u2013N\u2013C\u03b1)", 13.5, "middle", "bold", ANGLE_C_N_CA))

    p.append(segment(148, 146, 148, 236, DIST_C_N, 1.4, dash="4 3"))
    p.append(text(153, 244, "d(C\u2013N)", 13.5, "start", "bold", DIST_C_N))

    for key in order:
        fill, edge = (BLUE_FILL, BLUE_EDGE) if key.endswith("0") else (GREEN_FILL, GREEN_EDGE)
        x, y = xy[key]
        p.append(circle(x, y, 14, fill, edge, 1.6))
        p.append(text(x, y + 4.8, name[key], 13.5, "middle"))

    p.append(path("M 18,263 L 18,271 M 18,267 L 146,267 M 146,263 L 146,271", stroke=BLUE_EDGE, sw=1.5))
    p.append(path("M 196,263 L 196,271 M 196,267 L 358,267 M 358,263 L 358,271", stroke=GREEN_EDGE, sw=1.5))
    p.append(rich_text(82, 290, [("Residue ", "n"), ("i", "it")], 14, "middle", "bold", "#2d4f8f"))
    p.append(text(82, 306, "(end of fragment)", 12.5, "middle", "bold", "#2d4f8f"))
    p.append(rich_text(277, 290, [("Residue ", "n"), ("i", "it"), (" + 1", "n")], 14, "middle", "bold", "#2f6b2a"))
    p.append(text(277, 306, "(start of next fragment)", 12.5, "middle", "bold", "#2f6b2a"))
    return panel("geometry", "a", "Boundary geometry", "".join(p), "(6 internal coordinates)")


def panel_quantization():
    cells = decode_cells(BOUNDARY_IDS[0])
    strip_w, mark_w = 130.0, 7.0
    p = []
    for i, (name, lv, z) in enumerate(zip(AXIS_NAMES, LEVELS, cells)):
        y = 52 + 34 * i
        center = min(max((z + 0.5) * strip_w / lv, mark_w / 2), strip_w - mark_w / 2)
        p.append(text(100, y + 17, name, 13.5, "end"))
        p.append(rect(108, y, strip_w, 24, "#FFFFFF", "#999999", 1, 3))
        p.append(rect(108 + center - mark_w / 2, y - 3, mark_w, 30, GREEN_EDGE, "#2f6b2a", 0.8, 1.5))
        p.append(text(248, y + 17, f"z = {z} of {lv}", 13.5))
    p.append(path("M 108,256 L 108,262 L 238,262 L 238,256", stroke=INK, sw=1.4))
    p.append(segment(173, 262, 173, 280, "#333333", 1.3, marker="arrow-dark"))
    p.append(id_box(173, 282, 120, 30, BOUNDARY_IDS[0], "boundary", 14.5))
    p.append(text(173, 334, f"packed into one {BOUNDARY_BITS}-bit integer", 13, "middle"))
    return panel("quant", "b", "6D boundary quantization", "".join(p))


def panel_closed_loop():
    bw, bh = 150, 50
    xl, xr, yt, yb = 10, 190, 56, 176
    boxes = [
        (xl, yt, ["Reconstructed end", "of fragment k\u22121"], "#F6F6F6", "#888888"),
        (xr, yt, [[("Exact g", "n"), ("k", "sub"), (" to target", "n")], "pose of fragment k"],
         "#F6F6F6", "#888888"),
        (xr, yb, [[("Quantize to \u011d", "n"), ("k", "sub")], f"({BOUNDARY_BITS} bits)"],
         GREEN_FILL, GREEN_EDGE),
        (xl, yb, ["Place fragment k", [("from \u011d", "n"), ("k", "sub")]], "#F6F6F6", "#888888"),
    ]
    p = [text_box(x, y, bw, bh, rows, fill, stroke, 13, 16) for x, y, rows, fill, stroke in boxes]
    ym_t, ym_b = yt + bh / 2, yb + bh / 2
    p.append(segment(xl + bw + 2, ym_t, xr - 3, ym_t, "#333333", 1.4, marker="arrow-dark"))
    p.append(segment(xr + bw / 2, yt + bh + 2, xr + bw / 2, yb - 3, "#333333", 1.4, marker="arrow-dark"))
    p.append(segment(xr - 2, ym_b, xl + bw + 3, ym_b, "#333333", 1.4, marker="arrow-dark"))
    p.append(segment(xl + bw / 2, yb - 2, xl + bw / 2, yt + bh + 3, RED, 2.2, marker="arrow-red"))
    p.append(text(xl + bw / 2 + 8, (yt + bh + yb) / 2 + 4, "next boundary", 12.5, "start", fill=RED))
    p.append(text(175, 266, "Each boundary corrects the rounding", 13, "middle", italic=True))
    p.append(text(175, 283, "error of the one before it.", 13, "middle", italic=True))
    return panel("loop", "c", "Closed-loop boundary fit", "".join(p))


def panel_decoder():
    p = [id_box(62, 58, 64, 28, FRAGMENT_IDS[1], "fragment", 14.5),
         id_box(80, 108, 104, 28, BOUNDARY_IDS[0], "boundary", 13),
         text_box(195, 70, 130, 54, [esc("Forward"), esc("kinematics")], BLUE_FILL, BLUE_EDGE, 15, 18),
         segment(96, 72, 193, 88, "#333333", 1.3, marker="arrow-dark"),
         segment(134, 122, 193, 106, "#333333", 1.3, marker="arrow-dark"),
         segment(260, 124, 260, 170, "#333333", 1.4, marker="arrow-dark")]

    orig = chain_atoms(LENGTHS[1], 40, 224)
    recon = [dict(a, x=a["x"] + 1.2 * math.cos(0.7 * g), y=a["y"] + 1.8 * math.sin(0.9 * g))
             for g, a in enumerate(orig)]
    assert len(orig) == len(recon) == 3 * LENGTHS[1]

    for a, b in zip(orig, orig[1:]):
        p.append(segment(a["x"], a["y"], b["x"], b["y"], "#E2E2E2", 11))
    p += [circle(a["x"], a["y"], 12.5, "#E2E2E2", "#D0D0D0", 0.6) for a in orig]
    p.append(chain_svg(recon))

    p.append(circle(70, 298, 9, "#E2E2E2", "#D0D0D0", 0.6))
    p.append(text(86, 303, "original", 13.5))
    p.append(circle(190, 298, 7, STYLE["N"][0], sw=0.8))
    p.append(text(204, 303, "reconstructed (N, C\u03b1, C)", 13.5))
    p.append(text(185, 330, "Deterministic from the integers,", 13, "middle", italic=True))
    p.append(text(185, 347, "no learned decoder.", 13, "middle", italic=True))
    return panel("decoder", "d", "Decoding", "".join(p))


# ---------------------------------------------------------------- legend, assembly and export
def legend(cx, y, items):
    """Legend centered on cx. items: 'N', 'CA', 'C', 'faded', 'fragment', 'boundary', 'feedback'."""
    label = {"faded": "O, H, R (not encoded)", "fragment": "fragment ID",
             "boundary": "boundary cell ID", "feedback": "closed-loop feedback"}
    lead = {"faded": 54, "fragment": 34, "boundary": 34, "feedback": 44}
    width = {k: 52 for k in ("N", "CA", "C")}
    width.update({k: lead[k] + 6.3 * len(label[k]) + 12 for k in label})
    x = cx - sum(width[i] for i in items) / 2 + 7
    p = []
    for i in items:
        if i in ("N", "CA", "C"):
            p.append(circle(x, y, 7, STYLE[i][0], sw=0.8))
            p.append(text(x + 12, y + 4, STYLE[i][2], 12))
        elif i == "faded":
            p.append('<g opacity="0.5">' + "".join(circle(x + 16 * j, y, 7, STYLE[k][0], sw=0.8)
                                                   for j, k in enumerate(("O", "H", "R"))) + "</g>")
            p.append(text(x + 54, y + 4, label[i], 12))
        elif i in ("fragment", "boundary"):
            fill, edge = (BLUE_FILL, BLUE_EDGE) if i == "fragment" else (GREEN_FILL, GREEN_EDGE)
            p.append(rect(x, y - 8, 26, 16, fill, edge, 1.2, 3))
            p.append(text(x + 34, y + 4, label[i], 12))
        else:
            p.append(segment(x, y, x + 34, y, RED, 2, marker="arrow-red"))
            p.append(text(x + 44, y + 4, label[i], 12))
        x += width[i]
    return "".join(p)


def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}">')


def build_overview():
    return "\n".join([svg_open(W1, H1), defs(), f'<rect width="{W1}" height="{H1}" fill="#FFFFFF"/>',
                      main_diagram(), partition_box(PART_X, BOX_Y), token_stream(TOK_X, BOX_Y),
                      legend(W1 / 2, H1 - 28, ["N", "CA", "C", "faded", "fragment", "boundary"]), "</svg>"])


def build_boundary():
    return "\n".join([svg_open(W2, H2), defs(), f'<rect width="{W2}" height="{H2}" fill="#FFFFFF"/>',
                      panel_boundary_geometry(), panel_quantization(), panel_closed_loop(), panel_decoder(),
                      segment(420, 20, 420, 745, "#CCCCCC", 1.2), segment(20, 372, W2 - 20, 372, "#CCCCCC", 1.2),
                      legend(W2 / 2, H2 - 28, ["N", "CA", "C", "faded", "fragment", "boundary", "feedback"]),
                      "</svg>"])

def export(svg_path):
    try:
        import cairosvg
    except ImportError:
        logging.error("cairosvg is required to write the png; install with: pip install cairosvg")
        raise
    cairosvg.svg2png(url=str(svg_path), write_to=str(svg_path.with_suffix(".png")), output_width=3200)
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(svg_path.with_suffix(".pdf")))
    logging.info("wrote png and pdf for %s", svg_path.name)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(LOG_DIR / "make_overview.log"), filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    check_inputs()
    for stem, svg in (("prorate_overview", build_overview()), ("prorate_boundary", build_boundary())):
        out = FIG_DIR / f"{stem}.svg"
        out.write_text(svg, encoding="utf-8")
        logging.info("wrote %s", out)
        export(out)


if __name__ == "__main__":
    main()