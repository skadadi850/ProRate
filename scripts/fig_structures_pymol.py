"""Ground truth against ProtRig (free boundary cells), ProtRig (valid peptide links) and GeoBPE reconstructions, rendered with PyMOL.
One file, two modes. Run from the repo root:
  PYTHONPATH=$PWD python scripts/fig_structures_pymol.py --pymol <pymol executable> --test cameo=<dir> casp14=<dir> --out runs/fig_structures_pymol
The script reconstructs test chains with both ProtRig models and GeoBPE, superposes every reconstruction on the ground truth over all N, CA, C
atoms, writes backbone PDBs with computed carbonyl oxygens to <out>_pdb/, starts PyMOL on this same file (FIG_RENDER=1) to ray-trace cartoons
from one camera per chain, and assembles titles and RMSDs into <out>.pdf and .png. Secondary structure is assigned from the ground-truth CA
trace and applied to all objects.

--select-worst-closeup: in every chain of the test directories that passes the length filter, the link between fragments at which GeoBPE
is farthest from the ground truth over the 2 * --flank residues around it (after superposing the whole chain) is found among the links at
which both ProtRig models are within --protrig-max-rmsd of the ground truth and the link error of GeoBPE is at least
--geobpe-min-link-error; the --num chains with the largest distance are drawn, worst first. fig_boundary_closeup.py ranks the same way
and, once it has run, adds a red square on the ground truth where each of its close-ups comes from.

  bash run.sh   (the full command is the structures step of run.sh)

  python scripts/fig_structures_pymol.py --figure-only --out outputs/fig_structures_pymol_seed1

  PYMOL=$(which pymol) python scripts/fig_structures_pymol.py --render-only --pymol "$PYMOL" --out outputs/fig_structures_pymol_seed1
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

KEYS = ("gt", "protrig", "valid", "geobpe", "protoken", "aminoaseed")
BLUE, ORANGE, GRAY, RED, PINK, PURPLE = "#0072B2", "#D55E00", "#4D4D4D", "#B2182B", "#CC79A7", "#6A3D9A"
WIDTH = 10.5
TITLES = ("Ground truth", "ProRate (free)", "ProRate (valid)", "GeoBPE", "ProToken", "AminoAseed")
COLORS = (GRAY, BLUE, BLUE, ORANGE, PINK, PURPLE)
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()


# ---- render mode: runs inside PyMOL ------------------------------------------------
def render():
    from pymol import cmd

    manifest = json.load(open(os.environ["FIG_MANIFEST"]))
    size = int(os.environ.get("FIG_SIZE", "1200"))
    masks_only = os.environ.get("FIG_MASKS_ONLY") == "1"
    cmd.set_color("c_gt", [0.55, 0.55, 0.55])
    cmd.set_color("c_protrig", [0.54, 0.76, 0.94])
    cmd.set_color("c_valid", [0.54, 0.76, 0.94])
    cmd.set_color("c_geobpe", [0.98, 0.72, 0.47])
    cmd.set_color("c_protoken", [0.86, 0.60, 0.76])
    cmd.set_color("c_aminoaseed", [0.62, 0.48, 0.78])
    cmd.set_color("c_mark", [0.70, 0.09, 0.17])
    cmd.bg_color("white")
    for k, v in (("ray_opaque_background", "on"), ("antialias", 2), ("cartoon_fancy_helices", 1), ("cartoon_smooth_loops", 1),
                 ("cartoon_loop_radius", 0.2), ("cartoon_tube_radius", 0.2), ("depth_cue", 0), ("ray_shadows", "off"),
                 ("specular", 0.2), ("cartoon_transparency", 0.0)):
        cmd.set(k, v)

    def assign(key, ss):
        """Set the secondary structure of every atom of object key; residues are counted in file order."""
        seen = {}

        def ss_of(chain, resi):
            k = (chain, resi)
            if k not in seen:
                seen[k] = len(seen)
            return ss[seen[k]] if seen[k] < len(ss) else "L"

        cmd.alter(key, "ss=f(chain, resi)", space={"f": ss_of})
        return len(seen)

    for rec in manifest["chains"]:
        cmd.delete("all")
        ss = rec.get("ss")
        if not ss:
            cmd.load(rec["orig"], "orig")
            cmd.dss("orig")
            tmp = []
            cmd.iterate("orig and polymer.protein and name CA", "tmp.append(ss or 'L')", space={"tmp": tmp})
            cmd.delete("orig")
            ss = "".join(tmp)
        for key in KEYS:
            cmd.load(rec[key + "_pdb"], key)
        counts = {key: assign(key, ss) for key in KEYS}
        if any(n != len(ss) for n in counts.values()):
            print(f"WARNING {rec['stem']}: {len(ss)} secondary-structure letters, residues per object {counts}")
        print(f"{rec['stem']}: H {ss.count('H')}, S {ss.count('S')}, L {ss.count('L')}", flush=True)
        cmd.hide("everything")
        cmd.show("cartoon")
        cmd.cartoon("automatic", "all")
        cmd.rebuild()
        for key in KEYS:
            cmd.color("c_" + key, key)
        cmd.orient("gt")
        cmd.zoom("all", buffer=2, complete=1)
        view = cmd.get_view()
        if not masks_only:
            for key in KEYS:
                cmd.disable("all")
                cmd.enable("gt")
                cmd.set("cartoon_transparency", 0.0 if key == "gt" else 0.65, "gt")
                if key != "gt":
                    cmd.enable(key)
                cmd.set_view(view)
                cmd.png(os.path.join(manifest["outdir"], f"{rec['stem']}_{key}.png"), width=size, height=size, dpi=300, ray=1)
                print(f"rendered {rec['stem']} {key}", flush=True)
        for m, mark in enumerate(rec.get("marks", [])):
            sel = f"gt and resi {mark['first']}-{mark['last']}"
            cmd.disable("all")
            cmd.enable("gt")
            cmd.hide("cartoon", "gt")
            cmd.show("cartoon", sel)
            cmd.color("c_mark", sel)
            cmd.set("cartoon_transparency", 0.0, "gt")
            cmd.set_view(view)
            cmd.png(os.path.join(manifest["outdir"], f"{rec['stem']}_mark{m}.png"), width=size, height=size, dpi=300, ray=1)
            print(f"rendered {rec['stem']} mark {m}", flush=True)
    cmd.quit()


# ---- main mode: runs -------------------------------------
def superpose(ref, mov):
    """mov rigidly aligned onto ref (both (n,3,3)) and the per-atom RMSD of the fit."""
    import numpy as np

    p, q = ref.reshape(-1, 3), mov.reshape(-1, 3)
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((q - qc).T @ (p - pc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = (q - qc) @ r.T + pc
    return aligned.reshape(mov.shape), float(np.sqrt(((aligned - p) ** 2).sum(1).mean()))


def write_pdb(x, path):
    """Backbone PDB with N, CA, C and a computed carbonyl O per residue. x is (n,3,3). The O lies in the plane of CA, C and the next
    N, 1.231 A from C, on the bisector pointing away from both; for the last residue the next N is placed parallel to N-CA."""
    import numpy as np

    unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    n_, ca, c = x[:, 0], x[:, 1], x[:, 2]
    n_next = np.vstack([n_[1:], (c[-1] + 1.33 * unit(n_[-1] - ca[-1]))[None]])
    o = c + 1.231 * unit(unit(c - ca) + unit(c - n_next))
    lines, serial = [], 1
    for i in range(len(x)):
        for name, xyz in (("N", n_[i]), ("CA", ca[i]), ("C", c[i]), ("O", o[i])):
            lines.append(f"ATOM  {serial:5d}  {name:<3s} ALA A{i + 1:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                         f"  1.00  0.00           {name[0]:>2s}")
            serial += 1
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def ss_string(ca):
    """H, S or L per residue from the CA trace by CA-CA distances (P-SEA windows). A residue starts a helix window when d(i,i+2),
    d(i,i+3) and d(i,i+4) are 5.5 +- 0.5, 5.3 +- 0.5 and 6.4 +- 0.6 A, and a strand window when they are 6.7 +- 0.6, 9.9 +- 0.9 and
    12.4 +- 1.1 A. Helix runs shorter than 4 residues and strand runs shorter than 3 become loop; helix wins over strand."""
    import numpy as np

    ca = np.asarray(ca, dtype=np.float64)
    n = len(ca)
    if n < 6:
        return None
    d = lambda k: np.linalg.norm(ca[k:] - ca[:-k], axis=1)
    d2, d3, d4 = d(2), d(3), d(4)
    ok = lambda x, mu, tol: np.abs(x - mu) <= tol
    m = n - 4
    helix_start = ok(d2[:m], 5.5, 0.5) & ok(d3[:m], 5.3, 0.5) & ok(d4[:m], 6.4, 0.6)
    sheet_start = ok(d2[:m], 6.7, 0.6) & ok(d3[:m], 9.9, 0.9) & ok(d4[:m], 12.4, 1.1)

    def mark(starts, width):
        out = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(starts):
            out[i:i + width] = True
        return out

    def drop_short(mask, minlen):
        out = mask.copy()
        i = 0
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]:
                    j += 1
                if j - i < minlen:
                    out[i:j] = False
                i = j
            else:
                i += 1
        return out

    helix = drop_short(mark(helix_start, 4), 4)
    sheet = drop_short(mark(sheet_start, 5), 3) & ~helix
    return "".join("H" if helix[i] else "S" if sheet[i] else "L" for i in range(n))


def parse_candidates(args, D):
    """Chains of the test directories with a parsed ProtRig layout, in a seeded random order (or in the order of --chains)."""
    import glob

    import numpy as np

    files = []
    for spec in args.test:
        name, d = spec.split("=", 1)
        files += [(name, f) for f in sorted(glob.glob(os.path.join(d, "*.pdb")))]
    wanted = set(args.chains) if args.chains else None
    out = []
    for name, f in files:
        stem = Path(f).stem
        if wanted is not None and stem not in wanted:
            continue
        try:
            lay = D.chain_layout(D.backbone_from_pdb(f))
        except Exception:
            continue
        if not lay["items"] or lay["cuts"]["unresolved"] != 0:
            continue
        if wanted is None and not args.min_len <= lay["n_rows"] <= args.max_len:
            continue
        if args.min_helix > 0:
            ss = ss_string(D.reference_coords(lay)[:, 1])
            if not ss or ss.count("H") / len(ss) < args.min_helix:
                continue
        out.append({"set": name, "file": f, "stem": stem, "n": int(lay["n_rows"])})
    if wanted is None:
        out = [out[i] for i in np.random.RandomState(args.seed).permutation(len(out))]
    return out[: (len(out) if wanted is not None else args.num * args.pool_factor)]


def protrig_coords(ckpt, cands, args, D, segmentation=False):
    """Coordinates decoded from the integer stream of every candidate, on fresh layouts (encoding overwrites a layout)."""
    import numpy as np
    import torch
    from prorate.checkpoint import load_protrig

    model = load_protrig(ckpt, args.device)
    if segmentation:
        if args.segment_rate_weight is not None:
            model.cfg.rate_weight = args.segment_rate_weight
        if args.zero_boundary_cost:
            model.boundary_cost.zero_()
    layouts = [D.chain_layout(D.backbone_from_pdb(c["file"])) for c in cands]
    with torch.no_grad():
        D.encode_chains(model, layouts, args.device, args.batch_size)
        out = []
        for lay in layouts:
            D.encode_breaks(model, lay, args.device)
            out.append(np.asarray(D.decode_chain(model, lay, args.device), dtype=np.float64))
    del model
    return out

def baseline_path(directory, name, stem):
    """File of the decoded backbone that scripts/baselines.py wrote for the chain stem."""
    return Path(directory) / f"{name}_decoded" / f"{stem}.npy"


def baseline_coords(directory, name, stem):
    """Decoded (n,3,3) backbone of the baseline name for the chain stem, or None when it was not decoded."""
    import numpy as np

    path = baseline_path(directory, name, stem)
    return np.load(path).astype(np.float64) if path.exists() else None


def load_geobpe(directory, stem):
    """Decoded (n,3,3) GeoBPE backbone cached for the chain stem, or None."""
    import numpy as np

    path = Path(directory) / f"{stem}.npy"
    return np.load(path).astype(np.float64) if path.exists() else None


def reconstruct_geobpe(directory, cands, count):
    """First count candidates, in order, whose cached GeoBPE rebuild has the residue count of the ground truth."""
    import numpy as np

    shown = []
    for c in cands:
        if len(shown) == count:
            break
        x = load_geobpe(directory, c["stem"])
        if x is None or x.shape != c["gt"].shape or not np.isfinite(x).all():
            print(f"skip {c['stem']}: no GeoBPE rebuild with the shape of the ground truth", flush=True)
            continue
        c["geobpe"] = x
        shown.append(c)
        print(f"GeoBPE rebuilt {c['stem']} ({c['n']} residues)", flush=True)
    return shown


def crop_box(images, pad_frac=0.04, extra=()):
    """(y0, y1, x0, x1) of the union bounding box of the non-white pixels of the images of one row and of the boxes in extra (each
    (x0, y0, x1, y1) in image pixels), with pad_frac of the larger side as margin. Coordinates of the uncropped images; a pixel of
    the cropped images is the original pixel minus (y0, x0)."""
    import numpy as np

    mask = np.zeros(images[0].shape[:2], dtype=bool)
    for im in images:
        mask |= (im[..., :3] < 0.97).any(-1)
    ys, xs = np.where(mask)
    top, bottom, left, right = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    for ex0, ey0, ex1, ey1 in extra:
        top, bottom, left, right = min(top, ey0), max(bottom, ey1), min(left, ex0), max(right, ex1)
    pad = int(pad_frac * max(bottom - top, right - left))
    return top - pad, bottom + 1 + pad, left - pad, right + 1 + pad


def crop_tight(images, col_in, strip_in, pad_frac=0.04, extra=()):
    """Crop every image of one row to the union bounding box of the non-white pixels and of the boxes in extra, so the shared camera
    and scale are kept, add pad_frac of the larger box side as margin, and append a white strip of strip_in inches (at a panel width of
    col_in inches) at the bottom for the label."""
    import numpy as np

    y0, y1, x0, x1 = crop_box(images, pad_frac, extra)
    strip = int(round(strip_in / col_in * (x1 - x0)))
    out = []
    for im in images:
        h, w = im.shape[:2]
        c = im[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
        c = np.pad(c, ((max(-y0, 0), max(y1 - h, 0) + strip), (max(-x0, 0), max(x1 - w, 0)), (0, 0)), constant_values=1.0)
        out.append(c)
    return out


def run_pymol(args, manifest, masks_only=False):
    """Ray-trace every panel of the manifest by starting PyMOL on this file (render mode). With masks_only, only the images of the
    marked residue windows are rendered."""
    if shutil.which(args.pymol) is None:
        raise SystemExit(f"{args.pymol} not found: install PyMOL and pass its path with --pymol")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(FIG_RENDER="1", FIG_MANIFEST=str(manifest), FIG_SIZE=str(args.size))
    if masks_only:
        env["FIG_MASKS_ONLY"] = "1"
    subprocess.run([args.pymol, "-cq", str(Path(__file__).resolve())], env=env, check=True)

def mark_box(pdb_dir, stem, m):
    """(x0, y0, x1, y1) in image pixels of the square around the residue window rendered in <stem>_mark<m>.png, or None when that image
    does not exist or is empty."""
    import matplotlib.image as mpimg
    import numpy as np

    path = pdb_dir / f"{stem}_mark{m}.png"
    if not path.exists():
        return None
    ys, xs = np.where((mpimg.imread(path)[..., :3] < 0.97).any(-1))
    if len(xs) == 0:
        return None
    half = 0.7 * max(xs.max() - xs.min(), ys.max() - ys.min(), 90)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def assemble(args, records, pdb_dir, out, manifest):
    """Crop the rendered panels, add titles and RMSD labels and, on the ground truth, a red square around every residue window that a
    close-up comes from, and write <out>.pdf and <out>.png. The crop includes the squares, so none is cut off."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    cols = len(KEYS)
    col_in, strip_in, title_in, bottom_in, pad = WIDTH / cols, 0.26, 0.48, 0.06, 0.02
    panels, origins, squares = [], [], []
    for rec in records:
        imgs = [mpimg.imread(pdb_dir / f"{rec['stem']}_{key}.png") for key in KEYS]
        boxes = [b for b in (mark_box(pdb_dir, rec["stem"], m) for m in range(len(rec.get("marks", [])))) if b]
        panels.append(crop_tight(imgs, col_in, strip_in, pad, extra=boxes))
        origins.append(crop_box(imgs, pad, extra=boxes))
        squares.append(boxes)
    ratios = [p[0].shape[0] / p[0].shape[1] for p in panels]
    H = col_in * sum(ratios) + title_in + bottom_in
    fig, axes = plt.subplots(len(records), cols, figsize=(WIDTH, H), squeeze=False, gridspec_kw={"height_ratios": ratios})
    fig.subplots_adjust(left=0, right=1, bottom=bottom_in / H, top=1 - title_in / H, wspace=0, hspace=0)
    rates = (None, args.protrig_bpr, args.valid_bpr, args.geobpe_bpr, args.protoken_bpr, args.aminoaseed_bpr)
    for j, (t, rate, color) in enumerate(zip(TITLES, rates, COLORS)):
        fig.text((j + 0.5) / cols, 0.995, t + (f"\n{rate:.1f} bits/res" if rate else ""), ha="center", va="top",
                 fontsize=9, fontweight="bold", color=color)
    for i, rec in enumerate(records):
        h = col_in * ratios[i]
        for j, key in enumerate(KEYS):
            ax = axes[i][j]
            ax.imshow(panels[i][j])
            ax.axis("off")
            label = f"{rec['stem']}, {rec['residues']} residues" if key == "gt" else f"RMSD {rec[key + '_rmsd']:.2f} \u00c5"
            ax.text(0.5, 0.05 / h, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        y0, _, x0, _ = origins[i]
        ax = axes[i][0]
        xl, yl = ax.get_xlim(), ax.get_ylim()
        for bx0, by0, bx1, by1 in squares[i]:
            ax.add_patch(Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0, fill=False, edgecolor=RED, linewidth=1.3))
        ax.set_xlim(xl)
        ax.set_ylim(yl)
    fig.savefig(out.with_suffix(".pdf"), dpi=args.dpi)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi)
    print(f"wrote {out.with_suffix('.pdf')}, {out.with_suffix('.png')} and {manifest}", flush=True)

    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", default=None, metavar="NAME=DIR", help="required unless --figure-only or --render-only")
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "protrig_s8k_rw0p015_s0" / "stage1.pt"), help="ProtRig with free boundary cells")
    ap.add_argument("--valid-ckpt", default=str(ROOT / "runs" / "protrig_s8k_rw0p06_natural21_s0" / "stage1.pt"),
                    help="ProtRig with valid peptide links")
    ap.add_argument("--segment-rate-weight", type=float, default=None, help="applies to --ckpt only")
    ap.add_argument("--zero-boundary-cost", action="store_true", help="applies to --ckpt only")
    ap.add_argument("--geobpe-decoded", default=str(ROOT / "release" / "cache" / "geobpe_decoded"),
                    help="directory of <chain stem>.npy GeoBPE rebuilds")
    ap.add_argument("--protrig-bpr", type=float, default=None)
    ap.add_argument("--valid-bpr", type=float, default=None)
    ap.add_argument("--geobpe-bpr", type=float, default=None)
    ap.add_argument("--protoken-bpr", type=float, default=None)
    ap.add_argument("--aminoaseed-bpr", type=float, default=None)
    ap.add_argument("--baseline-dir", default=str(ROOT / "runs" / "baselines"),
                    help="directory written by scripts/baselines.py: <name>_decoded/<chain stem>.npy")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--chains", nargs="*", default=None, help="chain stems, for example 7x4e_A; overrides the random choice")
    ap.add_argument("--min-len", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--pool-factor", type=int, default=4, help="candidates examined per chain shown")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-helix", type=float, default=0.0, help="keep only chains with at least this fraction of helix residues")
    ap.add_argument("--select-worst-closeup", action="store_true",
                    help="show the --num chains in which GeoBPE is farthest from the ground truth around an eligible link between "
                         "fragments, worst first; the pool is every chain of the test directories that passes the length filter "
                         "(--pool-factor is ignored)")
    ap.add_argument("--protrig-max-rmsd", type=float, default=0.5,
                    help="with --select-worst-closeup, a link is eligible only when both ProtRig models are within this per-atom RMSD (A) "
                         "of the ground truth over the residues around it, after superposing the whole chain")
    ap.add_argument("--geobpe-min-link-error", type=float, default=2.0,
                    help="with --select-worst-closeup, a link is eligible only when the link error of GeoBPE is at least this many sigma")
    ap.add_argument("--flank", type=int, default=3, help="residues on each side of the link; keep equal to --flank of fig_boundary_closeup.py")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pymol", default="pymol", help="PyMOL executable")
    ap.add_argument("--size", type=int, default=1200, help="pixels per panel")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--figure-only", action="store_true",
                    help="only reassemble the figure from the saved panel PNGs and manifest of an earlier run with the same --out")
    ap.add_argument("--render-only", action="store_true",
                    help="re-run PyMOL on the saved PDBs of an earlier run with the same --out, then reassemble the figure")
    ap.add_argument("--out", default=str(ROOT / "runs" / "fig_structures_pymol"))
    args = ap.parse_args()

    out = Path(args.out)
    pdb_dir = Path(str(out) + "_pdb")
    manifest = out.with_suffix(".manifest.json")

    if args.figure_only or args.render_only:
        if not manifest.exists():
            raise SystemExit(f"{manifest} does not exist: run once without --figure-only/--render-only, with the same --out")
        saved = json.load(open(manifest))
        for k in ("protrig_bpr", "valid_bpr", "geobpe_bpr", "protoken_bpr", "aminoaseed_bpr"):
            if getattr(args, k) is None:
                setattr(args, k, saved["args"].get(k))
        if args.render_only:
            run_pymol(args, manifest)
        assemble(args, saved["chains"], pdb_dir, out, manifest)
        return

    if not args.test:
        raise SystemExit("--test is required for a full run")
    if shutil.which(args.pymol) is None:
        raise SystemExit(f"{args.pymol} not found: install PyMOL and pass its path with --pymol")

    sys.path.insert(0, str(ROOT))
    import numpy as np
    from prorate import data as D

    if args.select_worst_closeup:
        args.pool_factor = 10 ** 6
    cands = parse_candidates(args, D)
    listed = len(cands)
    cands = [c for c in cands if all(baseline_path(args.baseline_dir, n, c["stem"]).exists() for n in KEYS[4:])]
    print(f"{len(cands)} of {listed} candidate chains decoded by ProToken and AminoAseed", flush=True)
    if not cands:
        raise SystemExit("no candidate chain: check --test, --min-len, --max-len and --baseline-dir")
    if args.select_worst_closeup:
        from fig_boundary_closeup import worst_window_order

        order = worst_window_order(args, [{"orig": c["file"], "stem": c["stem"], "residues": c["n"]} for c in cands], D, args.flank,
                                   args.geobpe_decoded, args.protrig_max_rmsd, args.geobpe_min_link_error)
        rank = {stem: k for k, (_, stem, _) in enumerate(order)}
        cands = sorted((c for c in cands if c["stem"] in rank), key=lambda c: rank[c["stem"]])[: args.num]
        print(f"selection: {len(order)} chains with an eligible link (ProtRig within {args.protrig_max_rmsd} A, GeoBPE link error >= "
              f"{args.geobpe_min_link_error} sigma); showing the {len(cands)} in which GeoBPE is farthest from the ground truth around the "
              f"link, worst first: " + ", ".join(f"{s} {r:.2f} A" for r, s, _ in order[: args.num]), flush=True)
    free = protrig_coords(args.ckpt, cands, args, D, segmentation=True)
    valid = protrig_coords(args.valid_ckpt, cands, args, D)
    keep = []
    for c, p, v in zip(cands, free, valid):
        gt = np.asarray(D.reference_coords(D.chain_layout(D.backbone_from_pdb(c["file"]))), dtype=np.float64)
        base = {n: baseline_coords(args.baseline_dir, n, c["stem"]) for n in KEYS[4:]}
        if gt.shape == p.shape == v.shape and np.isfinite(gt).all() and all(b.shape == gt.shape for b in base.values()):
            c["gt"], c["protrig"], c["valid"] = gt, p, v
            c.update(base)
            keep.append(c)
    shown = reconstruct_geobpe(args.geobpe_decoded, keep, args.num)
    if not shown:
        raise SystemExit("no chain was rebuilt by all methods")

    pdb_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for c in shown:
        rec = {"stem": c["stem"], "set": c["set"], "residues": c["n"], "orig": c["file"], "ss": ss_string(c["gt"][:, 1])}
        fits = {"gt": c["gt"]}
        for key in KEYS[1:]:
            fits[key], rmsd = superpose(c["gt"], c[key])
            rec[key + "_rmsd"], rec[key + "_rmsd_esm"] = rmsd, rmsd / np.sqrt(3.0)
        for key in KEYS:
            rec[key + "_pdb"] = str(pdb_dir / f"{c['stem']}_{key}.pdb")
            write_pdb(fits[key], rec[key + "_pdb"])
        records.append(rec)
        print(f"{c['stem']:10s} {c['n']:4d} residues  " + "  ".join(f"{k} {rec[k + '_rmsd']:.3f} A" for k in KEYS[1:]), flush=True)
    json.dump({"outdir": str(pdb_dir), "args": vars(args), "chains": records}, open(manifest, "w"), indent=1)
    run_pymol(args, manifest)
    assemble(args, records, pdb_dir, out, manifest)


if os.environ.get("FIG_RENDER") == "1":
    render()
elif __name__ == "__main__":
    main()