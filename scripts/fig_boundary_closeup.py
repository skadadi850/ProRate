"""Close-ups of the links between code fragments in ProRate (free) and ProRate (valid), against the ground truth and GeoBPE, the
distribution of the link error of the three models against the ground truth, and a zoom figure that joins the structures figure to the
close-ups.
Run from the repo root, after fig_structures_pymol.py has written its manifest:
  PYTHONPATH=$PWD python scripts/fig_boundary_closeup.py --manifest runs/fig_structures_pymol_seed1.manifest.json --pymol <pymol executable> --out runs/fig_boundary_closeup_seed1
The first --num chains of that manifest are encoded again with both models. In every chain the link between two fragments is drawn at which
GeoBPE is farthest from the ground truth over the 2 * --flank residues around it, after superposing the whole chain as in the structures
figure, among the links at which both ProtRig models are within --protrig-max-rmsd of the ground truth over the same residues and the link
error of GeoBPE is at least --geobpe-min-link-error; the rows are the chains ordered by that distance, largest first. Each row shows the
ground truth, ProRate (free), ProRate (valid) and GeoBPE in the frame of the whole-chain superposition. Under each panel are the C-N
length, CA-C-N angle, C-N-CA angle and omega deviation from trans of the link, in red when beyond 2 sigma of the Engh and Huber ideals
(omega: more than 10 degrees), the RMSD over the residues shown after the whole-chain superposition, the CA LDDT over the same residues,
and the link error. Backbone PDBs go to <out>_pdb/; PyMOL runs on this same file (CLOSEUP_RENDER=1).

Scatter (<out>_scatter.pdf, .png, .json): for every link between two fragments of a ProtRig model, over the chains drawn plus
--scatter-chains further chains, the cumulative distribution of the error against the ground truth of the C-N length, CA-C-N, C-N-CA,
omega deviation, RMSD and LDDT. ProRate (free) and ProRate (valid) are taken at their own links, GeoBPE at the union of both sets. The RMSD
of the scatter is over the residues around the link after superposing those residues only.

At the end, the residue window of every drawn link is written into the structures manifest, which is put in the row order of the close-up
and reassembled with a red square on the ground truth where each close-up comes from.

Zoom (<out>_zoom.pdf, .png): for every chain, a row of the four full structures with a red box around the residue window, and under it the
close-up of that window in the four columns, joined to the box by lines from its corners.

  bash run.sh   (the full command is the closeup step of run.sh)

  python scripts/fig_boundary_closeup.py --figure-only --out outputs/fig_boundary_closeup_seed1

  PYMOL=$(which pymol) python scripts/fig_boundary_closeup.py --render-only --pymol "$PYMOL" --out outputs/fig_boundary_closeup_seed1
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KEYS = ("gt", "protrig", "valid", "geobpe", "protoken", "aminoaseed")
BLUE, ORANGE, GRAY, RED, PINK, PURPLE = "#0072B2", "#D55E00", "#4D4D4D", "#B2182B", "#CC79A7", "#6A3D9A"
BLUE_STRONG, ORANGE_STRONG = BLUE, ORANGE
WIDTH = 10.5
TITLES = ("Ground truth", "ProRate (free)", "ProRate (valid)", "GeoBPE", "ProToken", "AminoAseed")
COLORS = (GRAY, BLUE, BLUE, ORANGE, PINK, PURPLE)
SIDE = {"protrig": "free", "valid": "valid", "geobpe": "GeoBPE", "protoken": "ProToken", "aminoaseed": "AminoAseed"}
KIND = {0: "inside a fragment", 1: "between fragments", 2: "piece break"}
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
SERIES = (("protrig", "ProRate (free)"), ("valid", "ProRate (valid)"), ("geobpe", "GeoBPE"),
          ("protoken", "ProToken"), ("aminoaseed", "AminoAseed"))
SIGMA = (0.014, 2.0, 1.8, 5.0)

# ---- render mode: runs inside PyMOL ------------------------------------------------
def render():
    from pymol import cmd

    manifest = json.load(open(os.environ["FIG_MANIFEST"]))
    size = int(os.environ.get("FIG_SIZE", "1200"))
    for name, rgb in (("c_gt", [0.55, 0.55, 0.55]), ("c_protrig", [0.50, 0.73, 0.93]), ("c_protrig_b", [0.76, 0.88, 0.98]),
                      ("c_valid", [0.50, 0.73, 0.93]), ("c_valid_b", [0.76, 0.88, 0.98]),
                      ("c_geobpe", [0.98, 0.69, 0.42]), ("c_geobpe_b", [0.99, 0.84, 0.68]),
                      ("c_protoken", [0.86, 0.60, 0.76]), ("c_protoken_b", [0.93, 0.80, 0.88]),
                      ("c_aminoaseed", [0.62, 0.48, 0.78]), ("c_aminoaseed_b", [0.80, 0.73, 0.90]),
                      ("c_link", [0.698, 0.094, 0.169])):
        cmd.set_color(name, rgb)
    cmd.bg_color("white")
    for k, v in (("ray_opaque_background", "on"), ("antialias", 2), ("depth_cue", 0), ("ray_shadows", "off"), ("specular", 0.2)):
        cmd.set(k, v)

    for rec in manifest["records"]:
        cmd.delete("all")
        flank = rec["flank"]
        link = lambda key: f"{key} and ((resi {flank} and name C) or (resi {flank + 1} and name N))"
        for key in KEYS:
            cmd.load(rec["pdb"][key], key)
        cmd.hide("everything")
        cmd.show("sticks")
        for key in KEYS:
            for r in range(1, 2 * flank):
                cmd.bond(f"{key} and resi {r} and name C", f"{key} and resi {r + 1} and name N")
            cmd.set("sphere_scale", 0.2, key)
            if key != "gt":
                cmd.set("stick_radius", 0.16, key)
                cmd.color("c_" + key, key)
                cmd.color("c_" + key + "_b", f"{key} and b > 0.5")
                cmd.color("c_link", link(key))
                cmd.show("spheres", link(key))
        cmd.orient("gt")
        cmd.zoom("all", buffer=1.0, complete=1)
        view = cmd.get_view()
        for key in KEYS:
            ghost = key != "gt"
            cmd.disable("all")
            cmd.enable("gt")
            cmd.set("stick_radius", 0.30 if ghost else 0.16, "gt")
            cmd.set("stick_transparency", 0.6 if ghost else 0.0, "gt")
            cmd.color("c_gt", "gt")
            cmd.hide("spheres", "gt")
            if ghost:
                cmd.enable(key)
            else:
                cmd.show("spheres", link("gt"))
                cmd.color("c_link", link("gt"))
            cmd.set_view(view)
            cmd.png(os.path.join(manifest["outdir"], f"{rec['tag']}_{key}.png"), width=size, height=size, dpi=600, ray=1)
            print(f"rendered {rec['tag']} {key}", flush=True)
    cmd.quit()


# ---- main mode: runs -------------------------------------
def shared():
    """fig_structures_pymol.py, imported for crop_tight, parse_candidates, reconstruct_geobpe, run_pymol and assemble."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import fig_structures_pymol as S
    return S


def link_kinds(model, lay):
    """Per link of the concatenated resolved residues (n-1): 0 inside a fragment or raw piece, 1 between two fragments,
    2 between two pieces."""
    import numpy as np

    kinds = []
    for it in lay["items"]:
        k = np.zeros(max(it["n"] - 1, 0), dtype=np.int8)
        if not it["raw"]:
            pos = 0
            for gid in it["seg"][:-1]:
                pos += model.codebook.length_of(gid)
                k[pos - 1] = 1
        kinds += [k, np.array([2], dtype=np.int8)]
    return np.concatenate(kinds)[:-1]


def decode_chains(ckpt, files, args, D, free):
    """Per file: decoded coordinates x (n,3,3) from the integer stream alone, reference coordinates, and the kind of every link."""
    import numpy as np
    import torch
    from prorate.checkpoint import load_protrig

    model = load_protrig(ckpt, args.device)
    if free:
        if args.segment_rate_weight is not None:
            model.cfg.rate_weight = args.segment_rate_weight
        if args.zero_boundary_cost:
            model.boundary_cost.zero_()
    else:
        if getattr(args, "valid_link_weight", None) is not None:
            model.cfg.link_weight = args.valid_link_weight
        if getattr(args, "valid_link_tol", None) is not None:
            model.cfg.link_tol = args.valid_link_tol
    layouts = [D.chain_layout(D.backbone_from_pdb(f)) for f in files]
    out = []
    with torch.no_grad():
        D.encode_chains(model, layouts, args.device, args.batch_size)
        for lay in layouts:
            D.encode_breaks(model, lay, args.device)
            out.append({"x": np.asarray(D.decode_chain(model, lay, args.device), dtype=np.float64),
                        "ref": np.asarray(D.reference_coords(lay), dtype=np.float64),
                        "kind": link_kinds(model, lay)})
    del model
    return out


def decode_all(args, chains, D, geobpe_decoded):
    """Decode every chain with both ProtRig models, rebuild it with GeoBPE and read the ProToken and AminoAseed decodings cached by
    scripts/baselines.py. Returns decoded, {model: one dict per chain}, and keep, the indices of the chains that all methods
    rebuilt with the residue count of the ground truth."""
    import pickle

    import numpy as np

    S = shared()
    files = [c["orig"] for c in chains]
    decoded = {"protrig": decode_chains(args.ckpt, files, args, D, True), "valid": decode_chains(args.valid_ckpt, files, args, D, False)}
    base = {n: {} for n in KEYS[4:]}
    keep = []
    for ci, c in enumerate(chains):
        a, b = decoded["protrig"][ci], decoded["valid"][ci]
        if not (a["x"].shape == b["x"].shape == a["ref"].shape and np.isfinite(a["ref"]).all()):
            print(f"skip {c['stem']}: decoded shapes {a['x'].shape}, {b['x'].shape}, reference {a['ref'].shape}", flush=True)
            continue
        x = {n: S.baseline_coords(args.baseline_dir, n, c["stem"]) for n in base}
        if any(v is None or v.shape != a["ref"].shape for v in x.values()):
            print(f"skip {c['stem']}: ProToken or AminoAseed has no decoded backbone with the shape of the ground truth", flush=True)
            continue
        for n, v in x.items():
            base[n][ci] = v
        keep.append(ci)

    cands = [{"file": chains[ci]["orig"], "stem": chains[ci]["stem"], "n": chains[ci]["residues"],
              "gt": decoded["protrig"][ci]["ref"]} for ci in keep]
    geo = {c["stem"]: c["geobpe"] for c in S.reconstruct_geobpe(geobpe_decoded, cands, len(cands))}
    keep = [ci for ci in keep if chains[ci]["stem"] in geo]
    for n in KEYS[3:]:
        decoded[n] = [None] * len(chains)
    for ci in keep:
        piece_break = np.where(decoded["protrig"][ci]["kind"] == 2, 2, 0).astype(np.int8)
        ref = decoded["protrig"][ci]["ref"]
        decoded["geobpe"][ci] = {"x": geo[chains[ci]["stem"]], "ref": ref, "kind": piece_break}
        for n in KEYS[4:]:
            decoded[n][ci] = {"x": base[n][ci], "ref": ref, "kind": piece_break}
    return decoded, keep


def usable_links(kind, flank):
    """Every link between two fragments whose 2 * flank residues fit in the chain and hold no piece break."""
    import numpy as np

    n = len(kind) + 1
    return [int(i) for i in np.flatnonzero(kind == 1)
            if i >= flank - 1 and i + flank < n and not (kind[i - flank + 1: i + flank + 1] == 2).any()]


def align_on(ref, mov, rows):
    """mov (n,3,3) rigidly fitted onto ref over the residues in rows; returns all of mov moved and the per-atom RMSD over rows."""
    import numpy as np

    p, q = ref[rows].reshape(-1, 3), mov[rows].reshape(-1, 3)
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((q - qc).T @ (p - pc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = ((mov.reshape(-1, 3) - qc) @ r.T + pc).reshape(mov.shape)
    return aligned, float(np.sqrt(((aligned[rows].reshape(-1, 3) - p) ** 2).sum(1).mean()))


def fit_metrics(ref, mov, rows):
    """mov (n,3,3) fitted onto ref over the residues in rows; returns all of mov moved, the per-atom RMSD and the CA LDDT (ESM's
    ProteinChain.lddt_ca, the form of the paper) over rows."""
    import numpy as np
    from esm.utils.structure.protein_chain import ProteinChain

    aligned, rmsd = align_on(ref, mov, rows)
    pred = ProteinChain.from_backbone_atom_coordinates(mov[rows])
    true = ProteinChain.from_backbone_atom_coordinates(ref[rows])
    return aligned, rmsd, float(np.nanmean(pred.lddt_ca(true)))

def window_rmsd(ref, fitted, rows):
    """Per-atom RMSD over the residues in rows between fitted and ref, both (n,3,3) and already superposed."""
    import numpy as np

    return float(np.sqrt(((fitted[rows] - ref[rows]) ** 2).sum(-1).mean()))

def centered_fit(ref, mov, rows):
    """mov (n,3,3) superposed on ref over all residues as in the structures figure, then translated so that the centroid of its residues
    in rows coincides with that of ref; returns all of mov moved and the per-atom RMSD over rows of the whole-chain superposition, before
    the translation."""
    aligned, rmsd = global_fit(ref, mov, rows)
    shift = ref[rows].reshape(-1, 3).mean(0) - aligned[rows].reshape(-1, 3).mean(0)
    return aligned + shift, rmsd

def global_fit(ref, mov, rows):
    """mov (n,3,3) superposed on ref over all residues, as in the structures figure; returns all of mov moved and the per-atom RMSD over
    the residues in rows."""
    import numpy as np

    aligned = align_on(ref, mov, np.arange(len(ref)))[0]
    return aligned, window_rmsd(ref, aligned, rows)

def rank_links(decoded, ci, flank, count, max_protrig_rmsd, min_link_error):
    """The count links of chain ci at which GeoBPE is farthest from the ground truth, as (RMSD, chain, link), largest first. The RMSD is
    the per-atom RMSD of the 2 * flank residues around the link after superposing the whole chain, as in the structures figure. A link is
    eligible when both ProtRig models are within max_protrig_rmsd of the ground truth over the same residues and the link error of
    GeoBPE (largest deviation of C-N length, CA-C-N, C-N-CA and omega in units of SIGMA) is at least min_link_error. Links between
    fragments in both ProtRig models are preferred; a chain without one falls back to the links between fragments of ProRate (free)."""
    import numpy as np
    from figures import link_values, omega_dev

    free = usable_links(decoded["protrig"][ci]["kind"], flank)
    both = [i for i in free if decoded["valid"][ci]["kind"][i] == 1]
    ref = decoded["protrig"][ci]["ref"]
    fitted = {m: align_on(ref, decoded[m][ci]["x"], np.arange(len(ref)))[0] for m in ("geobpe", "protrig", "valid")}
    values = lambda x: np.column_stack([link_values(x), omega_dev(x)])
    link_error = (np.abs(values(decoded["geobpe"][ci]["x"]) - values(ref)) / np.array(SIGMA)).max(1)
    scored = []
    for i in both or free:
        rows = np.arange(i - flank + 1, i + flank + 1)
        rmsd = {m: window_rmsd(ref, fitted[m], rows) for m in fitted}
        if rmsd["protrig"] <= max_protrig_rmsd and rmsd["valid"] <= max_protrig_rmsd and link_error[i] >= min_link_error:
            scored.append((rmsd["geobpe"], ci, i))
    return sorted(scored, reverse=True)[:count]


def worst_window_order(args, chains, D, flank, geobpe_decoded, max_protrig_rmsd, min_link_error):
    """The chains ordered by the worst close-up: for every chain, the eligible link (rank_links) at which GeoBPE is farthest from the
    ground truth, and the chains sorted by that RMSD, largest first. Returns (RMSD, stem, link) triples. chains hold orig, stem and
    residues."""
    decoded, keep = decode_all(args, chains, D, geobpe_decoded)
    best = []
    for ci in keep:
        top = rank_links(decoded, ci, flank, 1, max_protrig_rmsd, min_link_error)
        if top:
            best.append(top[0])
    return sorted(((rmsd, chains[ci]["stem"], i) for rmsd, ci, i in best), reverse=True)


def carbonyl_oxygen(x):
    """O of every residue of x (n,3,3): in the plane of CA, C and the next N, 1.231 A from C, on the bisector pointing away from both."""
    import numpy as np

    unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    n_, ca, c = x[:, 0], x[:, 1], x[:, 2]
    n_next = np.vstack([n_[1:], (c[-1] + 1.33 * unit(n_[-1] - ca[-1]))[None]])
    return c + 1.231 * unit(unit(c - ca) + unit(c - n_next))


def write_pdb(x, rows, parity, path):
    """PDB of the residues in rows of x, renumbered from 1. The B-factor column holds the parity of the fragment index."""
    o = carbonyl_oxygen(x)
    lines, serial = [], 1
    for r, row in enumerate(rows):
        for name, xyz in (("N", x[row, 0]), ("CA", x[row, 1]), ("C", x[row, 2]), ("O", o[row])):
            lines.append(f"ATOM  {serial:5d}  {name:<3s} ALA A{r + 1:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                         f"  1.00{float(parity[r]):6.2f}           {name[0]:>2s}")
            serial += 1
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def pick(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def junction_stats(items):
    """Share of links between fragments, links inside fragments and reference links beyond 2 sigma of the Engh and Huber ideals, and of
    omega beyond 10 degrees from trans, over the decoded chains in items."""
    import numpy as np
    from figures import link_values, omega_dev
    from prorate.metrics import omega_report, peptide_validity

    pl, rl, po, ro, fl = [], [], [], [], []
    for d in items:
        keep = d["kind"] != 2
        pl.append(link_values(d["x"])[keep])
        rl.append(link_values(d["ref"])[keep])
        po.append(omega_dev(d["x"])[keep])
        ro.append(omega_dev(d["ref"])[keep])
        fl.append(d["kind"][keep] == 1)
    cat = np.concatenate
    pv = peptide_validity(cat(pl), cat(rl), cat(fl))
    ow = omega_report(cat(po), cat(ro), cat(fl))
    return {"n_junction": int(pick(pv, "decoded_boundary", "n_links") or 0),
            "beyond_2sigma_junction": pick(pv, "decoded_boundary", "frac_any_beyond", "2.0"),
            "beyond_2sigma_interior": pick(pv, "decoded_interior", "frac_any_beyond", "2.0"),
            "beyond_2sigma_reference": pick(pv, "reference", "frac_any_beyond", "2.0"),
            "omega_beyond_10_junction": pick(ow, "decoded_boundary", "frac_beyond", "10.0"),
            "omega_beyond_10_interior": pick(ow, "decoded_interior", "frac_beyond", "10.0"),
            "omega_beyond_10_reference": pick(ow, "reference", "frac_beyond", "10.0")}


def scatter_points(decoded, tables, keep, flank):
    """Per series, rows [gt (4), model (4), RMSD, LDDT] of every link between fragments of the ProtRig model (GeoBPE, ProToken and
    AminoAseed: of either ProtRig model) over the decoded chains in keep."""
    import numpy as np

    rows = {key: [] for key, _ in SERIES}
    for ci in keep:
        ref = decoded["protrig"][ci]["ref"]
        links = {m: usable_links(decoded[m][ci]["kind"], flank) for m in ("protrig", "valid")}
        union = sorted(set(links["protrig"]) | set(links["valid"]))
        links.update({m: union for m in KEYS[3:]})
        for m, idx in links.items():
            for i in idx:
                sel = np.arange(i - flank + 1, i + flank + 1)
                _, rmsd, lddt = fit_metrics(ref, decoded[m][ci]["x"], sel)
                rows[m].append(np.concatenate([tables[ci]["gt"][i], tables[ci][m][i], [rmsd, lddt]]))
    return {m: np.asarray(r, dtype=np.float64).reshape(-1, 10) for m, r in rows.items()}


def run_pymol(args, manifest):
    """Ray-trace every panel of the manifest by starting PyMOL on this file (render mode)."""
    if shutil.which(args.pymol) is None:
        raise SystemExit(f"{args.pymol} not found: install PyMOL and pass its path with --pymol")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(CLOSEUP_RENDER="1", FIG_MANIFEST=str(manifest), FIG_SIZE=str(args.size))
    subprocess.run([args.pymol, "-cq", str(Path(__file__).resolve())], env=env, check=True)

def link_error(rec, key):
    """Largest deviation of the link of key from the ground truth among C-N length, CA-C-N, C-N-CA and omega, in units of SIGMA."""
    gt, v = rec["values"]["gt"], rec["values"][key]
    return max(abs(v[k] - gt[k]) / SIGMA[k] for k in range(4))

def assemble(args, records, pdb_dir, out, manifest):
    """Crop the rendered panels, add titles, the link error, the RMSD and LDDT over the residues shown and the signed link values, and
    write <out>.pdf and <out>.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    S = shared()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    cols = len(KEYS)
    col_in, strip_in, title_in, bottom_in, pad = WIDTH / cols, 0.66, 0.32, 0.08, 0.02
    panels = [S.crop_tight([mpimg.imread(pdb_dir / f"{rec['tag']}_{key}.png") for key in KEYS], col_in, strip_in, pad) for rec in records]
    ratios = [p[0].shape[0] / p[0].shape[1] for p in panels]
    H = col_in * sum(ratios) + title_in + bottom_in
    fig, axes = plt.subplots(len(records), cols, figsize=(WIDTH, H), squeeze=False, gridspec_kw={"height_ratios": ratios})
    fig.subplots_adjust(left=0, right=1, bottom=bottom_in / H, top=1 - title_in / H, wspace=0, hspace=0)
    for j, (t, color) in enumerate(zip(TITLES, COLORS)):
        fig.text((j + 0.5) / cols, 0.995, t, ha="center", va="top", fontsize=9, fontweight="bold", color=color)
    names, units, decimals = ("C-N", "CA-C-N", "C-N-CA", "\u03c9 dev"), ("\u00c5", "\u00b0", "\u00b0", "\u00b0"), (2, 1, 1, 1)
    for i, rec in enumerate(records):
        h = col_in * ratios[i]
        gt = rec["values"]["gt"]
        for j, key in enumerate(KEYS):
            ax = axes[i][j]
            ax.imshow(panels[i][j])
            ax.axis("off")
            v = rec["values"][key]
            if key == "gt":
                ax.text(0.5, 0.34 / h, f"{rec['stem']}, residues {rec['link'] + 1}-{rec['link'] + 2}", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7)
            else:
                z = link_error(rec, key)
                ax.text(0.5, 0.48 / h, f"link error {z:.1f} \u03c3", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color=RED if z > 2.0 else "black")
                ax.text(0.5, 0.34 / h, f"RMSD {rec['rmsd'][key]:.2f} \u00c5, LDDT {rec['lddt'][key]:.2f}", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7)
            for k in range(4):
                text = (f"{names[k]} {v[k]:.{decimals[k]}f}{units[k]}" if key == "gt"
                        else f"\u0394{names[k]} {v[k] - gt[k]:+.{decimals[k]}f}{units[k]}")
                ax.text(0.27 if k % 2 == 0 else 0.73, (0.18 if k < 2 else 0.04) / h, text, transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=6.4, color=RED if rec["flags"][key][k] else "black")
    fig.savefig(out.with_suffix(".pdf"), dpi=args.dpi)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi)
    print(f"wrote {out.with_suffix('.pdf')}, {out.with_suffix('.png')} and {manifest}", flush=True)

def assemble_zoom(args, records, pdb_dir, out, structures_manifest):
    """One figure that joins the structures figure to the close-ups: for every chain, a row of the four full structures with a red box
    around the residue window of the close-up, and under it the close-up of that window in the four columns, joined to the box by lines
    from its corners. Writes <out>_zoom.pdf and .png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle

    S = shared()
    sm = json.load(open(structures_manifest))
    s_dir = Path(sm["outdir"])
    chains = {c["stem"]: c for c in sm["chains"]}
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    col_in, strip_full, strip_zoom, title_in, bottom_in, pad = WIDTH / len(KEYS), 0.26, 0.52, 0.48, 0.06, 0.02
    rows, seen = [], {}
    for rec in records:
        m = seen.get(rec["stem"], 0)
        seen[rec["stem"]] = m + 1
        full_imgs = [mpimg.imread(s_dir / f"{rec['stem']}_{key}.png") for key in KEYS]
        zoom_imgs = [mpimg.imread(pdb_dir / f"{rec['tag']}_{key}.png") for key in KEYS]
        box = S.mark_box(s_dir, rec["stem"], m)
        extra = [box] if box else []
        rows.append({"rec": rec, "chain": chains[rec["stem"]], "box": box,
                     "full": S.crop_tight(full_imgs, col_in, strip_full, pad, extra=extra),
                     "origin": S.crop_box(full_imgs, pad, extra=extra),
                     "zoom": S.crop_tight(zoom_imgs, col_in, strip_zoom, pad)})
    ratios = [p[0].shape[0] / p[0].shape[1] for r in rows for p in (r["full"], r["zoom"])]
    H = col_in * sum(ratios) + title_in + bottom_in
    fig, axes = plt.subplots(2 * len(rows), len(KEYS), figsize=(WIDTH, H), squeeze=False, gridspec_kw={"height_ratios": ratios})
    fig.subplots_adjust(left=0, right=1, bottom=bottom_in / H, top=1 - title_in / H, wspace=0, hspace=0)
    sa = sm["args"]
    rates = (None,) + tuple(sa.get(k + "_bpr") for k in KEYS[1:])
    for j, (t, rate, color) in enumerate(zip(TITLES, rates, COLORS)):
        fig.text((j + 0.5) / len(KEYS), 0.995, t + (f"\n{rate:.1f} bits/res" if rate else ""), ha="center", va="top",
                 fontsize=9, fontweight="bold", color=color)
    for i, r in enumerate(rows):
        rec, chain = r["rec"], r["chain"]
        hf, hz = col_in * ratios[2 * i], col_in * ratios[2 * i + 1]
        y0, _, x0, _ = r["origin"]
        for j, key in enumerate(KEYS):
            axf, axz = axes[2 * i][j], axes[2 * i + 1][j]
            axf.imshow(r["full"][j])
            axf.axis("off")
            axz.imshow(r["zoom"][j])
            axz.axis("off")
            label = f"{chain['stem']}, {chain['residues']} residues" if key == "gt" else f"RMSD {chain[key + '_rmsd']:.2f} \u00c5"
            axf.text(0.5, 0.05 / hf, label, transform=axf.transAxes, ha="center", va="bottom", fontsize=8)
            if key == "gt":
                axz.text(0.5, 0.06 / hz, f"residues {rec['link'] + 1}-{rec['link'] + 2}", transform=axz.transAxes, ha="center",
                         va="bottom", fontsize=7)
            else:
                z = link_error(rec, key)
                axz.text(0.5, 0.27 / hz, f"link error {z:.1f} \u03c3", transform=axz.transAxes, ha="center", va="bottom", fontsize=8.5,
                         fontweight="bold", color=RED if z > 2.0 else "black")
                axz.text(0.5, 0.06 / hz, f"RMSD {rec['rmsd'][key]:.2f} \u00c5, LDDT {rec['lddt'][key]:.2f}", transform=axz.transAxes,
                         ha="center", va="bottom", fontsize=7)
            h_zoom, w_zoom = r["zoom"][j].shape[:2]
            h_img = h_zoom - int(round(strip_zoom / col_in * w_zoom))
            xl, yl = axz.get_xlim(), axz.get_ylim()
            axz.add_patch(Rectangle((-0.5, -0.5), w_zoom, h_img, fill=False, edgecolor=RED, linewidth=1.0))
            axz.set_xlim(xl)
            axz.set_ylim(yl)
            if r["box"]:
                bx0, by0, bx1, by1 = r["box"]
                xl, yl = axf.get_xlim(), axf.get_ylim()
                axf.add_patch(Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0, fill=False, edgecolor=RED, linewidth=1.3))
                axf.set_xlim(xl)
                axf.set_ylim(yl)
                for bx, zx in ((bx0, -0.5), (bx1, w_zoom - 0.5)):
                    fig.add_artist(ConnectionPatch(xyA=(bx - x0, by1 - y0), coordsA=axf.transData, xyB=(zx, -0.5), coordsB=axz.transData,
                                                   color=RED, linewidth=0.6, alpha=0.55))
    path = str(out) + "_zoom"
    fig.savefig(path + ".pdf", dpi=args.dpi)
    fig.savefig(path + ".png", dpi=args.dpi)
    print(f"wrote {path}.pdf and {path}.png", flush=True)

def assemble_scatter(args, saved, out):
    """Cumulative distribution of the error against the ground truth of the five models: RMSD and LDDT over the residues around the
    link, and C-N length, C-N-CA, CA-C-N and omega deviation of the link. A curve further to the left is closer to the ground truth; the
    dashed grey line at the top is the ground truth against itself, which has zero error. The median and the share within 2 sigma of
    every curve are printed, not drawn. Writes <out>_scatter.pdf and .png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 10, "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.2})
    series = {k: np.asarray(v, dtype=np.float64).reshape(-1, 10) for k, v in saved["series"].items()}
    if not any(len(p) for p in series.values()):
        raise SystemExit("no link between fragments to plot: lower --flank or raise --scatter-chains")
    label = dict(SERIES)
    look = {"protrig": dict(color=BLUE_STRONG, ls=":", lw=1.6, z=2), "valid": dict(color=BLUE_STRONG, ls="-", lw=2.0, z=6),
            "geobpe": dict(color=ORANGE_STRONG, ls="-", lw=1.7, z=3),
            "protoken": dict(color=PINK, ls="-", lw=1.7, z=4), "aminoaseed": dict(color=PURPLE, ls="-", lw=1.7, z=5)}
    # title, axis label, error of a row [gt C-N, CA-C-N, C-N-CA, omega dev, model C-N, CA-C-N, C-N-CA, omega dev, RMSD, LDDT],
    # lowest value of the log axis, 2 sigma of the Engh and Huber ideals (omega: 10 degrees)
    panels = (("RMSD around the link", "RMSD (\u00c5)", lambda p: p[:, 8], 1e-3, None),
              ("LDDT around the link", "1 - CA LDDT", lambda p: 1.0 - p[:, 9], 1e-3, None),
              ("C-N length", "|\u0394 C-N| (\u00c5)", lambda p: np.abs(p[:, 4] - p[:, 0]), 1e-4, 0.028),
              ("C-N-CA angle", "|\u0394 C-N-CA| (\u00b0)", lambda p: np.abs(p[:, 6] - p[:, 2]), 1e-2, 3.6),
              ("CA-C-N angle", "|\u0394 CA-C-N| (\u00b0)", lambda p: np.abs(p[:, 5] - p[:, 1]), 1e-2, 4.0),
              ("\u03c9 deviation from trans", "|\u0394 \u03c9 dev| (\u00b0)", lambda p: np.abs(p[:, 7] - p[:, 3]), 1e-2, 10.0))
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.6))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.94, bottom=0.2, wspace=0.42, hspace=0.62)
    for n, (ax, (title, xlabel, err, floor, tol)) in enumerate(zip(axes.ravel(), panels)):
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
        errs = {k: err(p) for k, p in series.items() if len(p)}
        errs = {k: e[np.isfinite(e)] for k, e in errs.items()}
        top = max(float(e.max()) for e in errs.values())
        for k, e in errs.items():
            s = look[k]
            x = np.sort(np.maximum(e, floor))
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=s["color"], ls=s["ls"], lw=s["lw"], zorder=s["z"])
            print(f"{title:26s} {label[k]:16s} n {len(e):6d}  median {np.median(e):.4g}  mean {np.mean(e):.4g}  p99 {np.percentile(e, 99):.4g}"
                  + (f"  within 2 sigma {100 * float((e <= tol).mean()):.1f}%" if tol is not None else ""), flush=True)
        bottom = title == "LDDT around the link"
        top_y = 0.04 + 0.085 * (len(errs) + 1) if bottom else 0.95
        x, align = (0.96, "right") if bottom else (0.04, "left")
        ax.text(x, top_y, "median error" if tol is None else "% within 2\u03c3", transform=ax.transAxes, ha=align, va="top",
                fontsize=7, fontstyle="italic", color="0.25", zorder=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.4})
        for i, (k, e) in enumerate(errs.items()):
            s = look[k]
            value = f"{np.median(e):.3g}" if tol is None else f"{100 * float((e <= tol).mean()):.0f}%"
            ax.text(x, top_y - 0.085 * (i + 1), ("\u00b7\u00b7\u00b7 " if k == "protrig" else "\u2500\u2500 ") + value,
                    transform=ax.transAxes, ha=align, va="top", fontsize=7, color=s["color"], zorder=9,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.4})
        ax.axhline(1.0, color="0.35", lw=1.2, ls="--", alpha=0.7, zorder=7)
        if tol is not None:
            ax.axvline(tol, color="0.3", lw=0.8, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(floor, max(top, tol or 0.0) * 1.5)
        ax.set_ylim(0.0, 1.04)
        ax.set_xlabel(xlabel)
        if n % 3 == 0:
            ax.set_ylabel("Share of links")
    handles = [Line2D([0], [0], color=look[k]["color"], ls=look[k]["ls"], lw=look[k]["lw"], label=f"{label[k]}, n = {len(series[k])}")
               for k in label if k in series]
    handles.append(Line2D([0], [0], color="0.35", lw=1.2, ls="--", alpha=0.7, label="Ground truth (error 0)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=9,
               columnspacing=1.6, handlelength=2.4)
    path = str(out) + "_scatter"
    fig.savefig(path + ".pdf", dpi=args.dpi)
    fig.savefig(path + ".png", dpi=args.dpi)
    print(f"wrote {path}.pdf and {path}.png", flush=True)


def pool_chains(src, count, D):
    """count chains drawn with the test directories and filters of the manifest."""
    from types import SimpleNamespace

    pool_args = SimpleNamespace(**{**src["args"], "num": count, "pool_factor": 1, "chains": None})
    return [{"orig": c["file"], "stem": c["stem"], "residues": c["n"]} for c in shared().parse_candidates(pool_args, D)]


def mark_structures(args, records):
    """Write the residue window of every drawn link into the manifest of the structures figure, put its chains in the row order of the
    close-up, ray-trace the windows there and reassemble that figure, so that a red square on the ground truth shows where each close-up
    comes from and both figures read in the same order."""
    from types import SimpleNamespace

    S = shared()
    path = Path(args.manifest)
    saved = json.load(open(path))
    for c in saved["chains"]:
        c["marks"] = [{"first": r["link"] - r["flank"] + 2, "last": r["link"] + r["flank"] + 1, "tag": r["tag"]}
                      for r in records if r["stem"] == c["stem"]]
    order = list(dict.fromkeys(r["stem"] for r in records))
    saved["chains"] = [c for s in order for c in saved["chains"] if c["stem"] == s]
    json.dump(saved, open(path, "w"), indent=1)
    sargs = SimpleNamespace(**{**saved["args"], "pymol": args.pymol, "size": args.size, "dpi": args.dpi})
    S.run_pymol(sargs, path, masks_only=True)
    S.assemble(sargs, saved["chains"], Path(saved["outdir"]), Path(str(path)[: -len(".manifest.json")]), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "runs" / "fig_structures_pymol.manifest.json"),
                    help="manifest of fig_structures_pymol.py: chains, checkpoints and segmentation settings come from it; it also receives "
                         "the red-square marks of the links drawn")
    ap.add_argument("--ckpt", default=None, help="overrides the ProRate (free) checkpoint of the manifest")
    ap.add_argument("--valid-ckpt", default=None, help="overrides the ProRate (valid) checkpoint of the manifest")
    ap.add_argument("--segment-rate-weight", type=float, default=None, help="applies to ProRate (free) only")
    ap.add_argument("--zero-boundary-cost", action="store_true", help="applies to ProRate (free) only")
    ap.add_argument("--valid-link-weight", type=float, default=None,
                    help="link weight used to encode with the --valid-ckpt model instead of its own")
    ap.add_argument("--valid-link-tol", type=float, default=None,
                    help="link tolerance used to encode with the --valid-ckpt model instead of its own")
    ap.add_argument("--geobpe-decoded", default=None,
                    help="GeoBPE rebuild directory instead of the one of the manifest")
    ap.add_argument("--baseline-dir", default=str(ROOT / "runs" / "baselines"),
                    help="directory written by scripts/baselines.py: <name>_decoded/<chain stem>.npy")
    ap.add_argument("--num", type=int, default=6, help="first chains of the manifest to draw")
    ap.add_argument("--links", type=int, default=1,
                    help="links drawn per chain: the eligible ones at which GeoBPE is farthest from the ground truth")
    ap.add_argument("--protrig-max-rmsd", type=float, default=None,
                    help="a link is eligible only when both ProtRig models are within this per-atom RMSD (A) of the ground truth over the "
                         "residues shown, after superposing the whole chain; default: the value of the structures manifest, else 0.5")
    ap.add_argument("--geobpe-min-link-error", type=float, default=None,
                    help="a link is eligible only when the link error of GeoBPE is at least this many sigma; default: the value of the "
                         "structures manifest, else 2.0")
    ap.add_argument("--scatter-chains", type=int, default=100,
                    help="chains added to the scatter besides the ones drawn, taken with the test directories and filters of the "
                         "manifest; 0 = only the chains drawn")
    ap.add_argument("--flank", type=int, default=3, help="residues on each side of the link")
    ap.add_argument("--scatter-flank", type=int, default=8,
                    help="residues on each side of the link over which the RMSD and LDDT of the scatter are computed")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pymol", default="pymol", help="PyMOL executable")
    ap.add_argument("--size", type=int, default=1200, help="pixels per panel")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--figure-only", action="store_true", help="only reassemble the figures from the saved panel PNGs, manifest and scatter data")
    ap.add_argument("--render-only", action="store_true", help="re-run PyMOL on the saved PDBs, then reassemble the figures")
    ap.add_argument("--out", default=str(ROOT / "runs" / "fig_boundary_closeup"))
    args = ap.parse_args()

    out = Path(args.out)
    pdb_dir = Path(str(out) + "_pdb")
    manifest = out.with_suffix(".manifest.json")
    scatter_path = Path(str(out) + "_scatter.json")

    if args.figure_only or args.render_only:
        if not manifest.exists():
            raise SystemExit(f"{manifest} does not exist: run once without --figure-only/--render-only, with the same --out")
        saved = json.load(open(manifest))
        if args.render_only:
            run_pymol(args, manifest)
        assemble(args, saved["records"], pdb_dir, out, manifest)
        if not scatter_path.exists():
            raise SystemExit(f"{scatter_path} does not exist: run once without --figure-only/--render-only, with the same --out")
        assemble_scatter(args, json.load(open(scatter_path)), out)
        assemble_zoom(args, saved["records"], pdb_dir, out, Path(saved["args"]["manifest"]))
        return

    if shutil.which(args.pymol) is None:
        raise SystemExit(f"{args.pymol} not found: install PyMOL and pass its path with --pymol")
    src = json.load(open(args.manifest))
    for k in ("ckpt", "valid_ckpt", "segment_rate_weight", "batch_size", "device"):
        if getattr(args, k) is None:
            setattr(args, k, src["args"].get(k))
    args.zero_boundary_cost = args.zero_boundary_cost or bool(src["args"].get("zero_boundary_cost"))
    for k, v in (("protrig_max_rmsd", 0.5), ("geobpe_min_link_error", 2.0)):
        if getattr(args, k) is None:
            setattr(args, k, src["args"].get(k, v))

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import numpy as np
    from figures import link_values, omega_dev
    from prorate import data as D
    from prorate.metrics import PEPTIDE_LINK

    chains = list(src["chains"][: args.num])
    n_draw = len(chains)
    if args.scatter_chains:
        seen = {c["stem"] for c in chains}
        chains += [c for c in pool_chains(src, args.scatter_chains, D) if c["stem"] not in seen]
    print(f"{n_draw} chains drawn, {len(chains)} chains in the scatter", flush=True)

    decoded, keep = decode_all(args, chains, D, args.geobpe_decoded or src["args"]["geobpe_decoded"])

    stats = {m: junction_stats([decoded[m][ci] for ci in keep]) for m in KEYS[1:]}
    for m, s in stats.items():
        print(f"{SIDE[m]}, {s['n_junction']} links between fragments: beyond 2 sigma between fragments {s['beyond_2sigma_junction']}, "
              f"inside fragments {s['beyond_2sigma_interior']}, reference {s['beyond_2sigma_reference']}; omega beyond 10 deg between "
              f"fragments {s['omega_beyond_10_junction']}, inside fragments {s['omega_beyond_10_interior']}, reference "
              f"{s['omega_beyond_10_reference']}", flush=True)

    values = lambda x: np.column_stack([link_values(x), omega_dev(x)])
    flags = lambda v: [bool(abs(v[k] - PEPTIDE_LINK[k][0]) / PEPTIDE_LINK[k][1] > 2.0) for k in range(3)] + [bool(v[3] > 10.0)]
    tables = {ci: {"gt": values(decoded["protrig"][ci]["ref"]), **{m: values(decoded[m][ci]["x"]) for m in KEYS[1:]}} for ci in keep}

    scatter = scatter_points(decoded, tables, keep, args.scatter_flank)
    print("scatter links: " + ", ".join(f"{SIDE[m]} {len(p)}" for m, p in scatter.items()), flush=True)
    shared_links = sum(len(set(usable_links(decoded["protrig"][ci]["kind"], args.scatter_flank))
                           & set(usable_links(decoded["valid"][ci]["kind"], args.scatter_flank))) for ci in keep)
    print(f"links between fragments shared by both ProtRig models: {shared_links}", flush=True)
    with open(scatter_path, "w") as fh:
        json.dump({"flank": args.scatter_flank, "chains": len(keep),
                   "series": {m: np.round(p, 5).tolist() for m, p in scatter.items()}}, fh)

    ranked = sorted((t for ci in keep if ci < n_draw
                     for t in rank_links(decoded, ci, args.flank, args.links, args.protrig_max_rmsd, args.geobpe_min_link_error)),
                    reverse=True)
    jobs = [(ci, "protrig", i) for _, ci, i in ranked]
    missing = [chains[ci]["stem"] for ci in range(n_draw) if ci not in {j[0] for j in jobs}]
    print(f"{len(jobs)} links drawn, sorted by the displacement of GeoBPE from the ground truth over the residues shown, largest first; "
          f"chains of the manifest without an eligible link: {missing}", flush=True)

    pdb_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for ci, owner, i in jobs:
        c = chains[ci]
        ref = decoded["protrig"][ci]["ref"]
        rows = np.arange(i - args.flank + 1, i + args.flank + 1)
        tag = f"{c['stem']}_{SIDE[owner]}_{i + 1}"
        rec = {"tag": tag, "stem": c["stem"], "residues": c["residues"], "link": i, "source": owner, "flank": args.flank,
               "pdb": {}, "values": {}, "flags": {}, "kind": {}, "rmsd": {}, "lddt": {}}
        rec["pdb"]["gt"] = str(pdb_dir / f"{tag}_gt.pdb")
        write_pdb(ref, rows, np.zeros(len(rows)), rec["pdb"]["gt"])
        for key in KEYS[1:]:
            d = decoded[key][ci]
            aligned, rmsd = centered_fit(ref, d["x"], rows)
            lddt = fit_metrics(ref, d["x"], rows)[2]
            frag = np.concatenate([[0], np.cumsum(d["kind"] != 0)])
            rec["pdb"][key] = str(pdb_dir / f"{tag}_{key}.pdb")
            write_pdb(aligned, rows, frag[rows] % 2, rec["pdb"][key])
            rec["kind"][key] = KIND[int(d["kind"][i])] if key in ("protrig", "valid") else SIDE[key]
            rec["rmsd"][key] = rmsd
            rec["lddt"][key] = lddt
        for key in KEYS:
            rec["values"][key] = [float(v) for v in tables[ci][key][i]]
            rec["flags"][key] = flags(tables[ci][key][i])
            kind = "reference" if key == "gt" else rec["kind"][key]
            fit = "" if key == "gt" else f"  RMSD {rec['rmsd'][key]:.3f} A  LDDT {rec['lddt'][key]:.3f}"
            v = rec["values"][key]
            print(f"{tag:24s} {key:8s} {kind:18s} C-N {v[0]:.3f} A  CA-C-N {v[1]:.1f}  C-N-CA {v[2]:.1f}  omega dev {v[3]:.1f}{fit}",
                  flush=True)
        records.append(rec)
    if not records:
        raise SystemExit("no link to draw: raise --protrig-max-rmsd or --num, or lower --geobpe-min-link-error or --flank")
    json.dump({"outdir": str(pdb_dir), "args": vars(args), "stats": stats, "records": records}, open(manifest, "w"), indent=1)
    run_pymol(args, manifest)
    assemble(args, records, pdb_dir, out, manifest)
    assemble_scatter(args, json.load(open(scatter_path)), out)
    mark_structures(args, records)
    assemble_zoom(args, records, pdb_dir, out, Path(args.manifest))


if os.environ.get("CLOSEUP_RENDER") == "1":
    render()
elif __name__ == "__main__":
    main()