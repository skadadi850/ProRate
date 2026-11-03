"""Token efficiency of the tokenizers (Table 3): utilization rate and max-normalized perplexity of the codebook on CAMEO and CASP14.
Sized to sit next to the link-geometry figure: same height, fonts, panel spacing and panel rows as the scatter of fig_boundary_closeup.py.
Run from the repo root:
  python scripts/fig_token_efficiency.py --out outputs/fig_token_efficiency
Baseline values are from Table 10 of the GeoBPE paper; ProRate is computed from release/checkpoints/protrig_s8k_rw0p4_s0.pt (8.6 BPR, 1577 codes). Only segment ids are counted.
Each row prints CAMEO / CASP14. Writes <out>.pdf and <out>.png at exactly width x height inches."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

BLUE, ORANGE, PINK, PURPLE = "#0072B2", "#D55E00", "#CC79A7", "#6A3D9A"
# method, codebook size, UR (%) CAMEO, CASP14, normalized perplexity CAMEO, CASP14, color
ROWS = (("VQ-VAE", 512, 5.55, 5.60, 0.034, 0.0337, "#E69F00"),
        ("AminoAseed", 512, 64.45, 68.87, 0.495, 0.5119, PURPLE),
        ("ESM3", 4096, 27.60, 32.10, 0.249, 0.2841, "#56E96A"),
        ("FoldSeek", 20, 99.00, 100.00, 0.755, 0.7435, "#8C8C8C"),
        ("ProToken", 512, 69.88, 75.56, 0.537, 0.5697, PINK),
        ("GeoBPE", 2500, 58.24, 38.22, 0.274, 0.2641, ORANGE),
        ("ProRate", 1577, 59.35, 38.55, 0.363, 0.2839, BLUE))
HIGHLIGHT = "ProRate"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "xtick.major.size": 3.5, "ytick.major.size": 0, "xtick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "axes.axisbelow": True, "grid.linewidth": 0.4, "grid.alpha": 0.2})


def positions():
    """Row centers from the top; ProRate is set apart by a gap, as the rule above it in the table."""
    y = list(range(len(ROWS) - 1)) + [len(ROWS) - 1 + 0.6]
    return dict(zip([r[0] for r in ROWS], y))


def draw(ax, column, title, xmax, ticks, fmt):
    pos = positions()
    height, offset = 0.36, 0.19
    for r in ROWS:
        name, color = r[0], r[6]
        cameo, casp = r[column], r[column + 1]
        y = pos[name]
        if name == HIGHLIGHT:
            ax.axhspan(y - 0.55, y + 0.55, color="#E3F5E3", zorder=0, lw=0)
        ax.barh(y - offset, cameo, height=height, color=color, edgecolor=color, linewidth=1.0, zorder=3)
        ax.barh(y + offset, casp, height=height, color="white", edgecolor=color, linewidth=1.3, zorder=3)
        ax.text(max(cameo, casp) + 0.015 * xmax, y, f"{fmt(cameo)} / {fmt(casp)}", va="center", ha="left", fontsize=7,
                color="0.2", zorder=4)
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.set_ylim(max(pos.values()) + 0.7, -0.7)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([pos[r[0]] for r in ROWS])
    ax.set_yticklabels([f"{r[0]} ({r[1]})" for r in ROWS])
    for label in ax.get_yticklabels():
        if label.get_text().startswith(HIGHLIGHT):
            label.set_fontweight("bold")


def prorate_row(ckpt, cameo, casp14, device):
    """Codebook size, and per test set (UR %, perplexity / K), of the segment ids ProRate emits; only segment ids are counted."""
    import glob
    import sys
    from collections import Counter

    import numpy as np

    sys.path.insert(0, str(ROOT))
    from prorate import data as D
    from prorate.checkpoint import load_protrig

    model = load_protrig(ckpt, device)
    K = model.codebook.total
    out = []
    for d in (cameo, casp14):
        lays = []
        for f in sorted(glob.glob(f"{d}/*.pdb")):
            try:
                lay = D.chain_layout(D.backbone_from_pdb(f))
            except Exception:
                continue
            if lay["items"]:
                lays.append(lay)
        D.encode_chains(model, lays, device, 64)
        ids = [g for lay in lays for it in lay["items"] if not it["raw"] for g in it["seg"]]
        p = np.array(list(Counter(ids).values()), dtype=float)
        p /= p.sum()
        out.append((100.0 * len(set(ids)) / K, float(np.exp(-(p * np.log(p)).sum())) / K))
    return K, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "fig_token_efficiency"))
    ap.add_argument("--width", type=float, default=4.2)
    ap.add_argument("--height", type=float, default=5.6, help="height of the link-geometry figure in inches")
    ap.add_argument("--ckpt", default=str(ROOT / "release" / "checkpoints" / "protrig_s8k_rw0p4_s0.pt"))
    ap.add_argument("--cameo", default=str(ROOT / "release" / "data" / "cameo"))
    ap.add_argument("--casp14", default=str(ROOT / "release" / "data" / "casp14"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    global ROWS
    K, ((ur_cameo, pp_cameo), (ur_casp, pp_casp)) = prorate_row(args.ckpt, args.cameo, args.casp14, args.device)
    print(f"ProRate K {K}: UR {ur_cameo:.2f} / {ur_casp:.2f}, normalized perplexity {pp_cameo:.4f} / {pp_casp:.4f}", flush=True)
    ROWS = ROWS[:-1] + (("ProRate", K, ur_cameo, ur_casp, pp_cameo, pp_casp, BLUE),)
    style()

    fig, (ax_ur, ax_pp) = plt.subplots(2, 1, figsize=(args.width, args.height))
    fig.subplots_adjust(left=0.36, right=0.98, top=0.94, bottom=0.2, hspace=0.62)
    draw(ax_ur, 2, "Utilization rate", 140, [0, 25, 50, 75, 100], lambda v: f"{v:.2f}")
    draw(ax_pp, 4, "Normalized perplexity", 1.05, [0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:.3f}")
    ax_ur.set_xlabel("Share of codebook used (%)")
    ax_pp.set_xlabel("Perplexity / codebook size")

    handles = [Patch(facecolor="0.45", edgecolor="0.45", label="CAMEO"),
               Patch(facecolor="white", edgecolor="0.45", linewidth=1.3, label="CASP14")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=9,
               columnspacing=1.6, handlelength=2.4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()