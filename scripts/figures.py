"""Compression results of ProtRig against GeoBPE: rate-distortion evaluation and the single results figure. Run from the repo root.

  python scripts/figures.py eval --ckpt <stage1.pt> --arm <name> --rate-weight <w> --test cameo=<dir> casp14=<dir> --out <eval.json>
  python scripts/figures.py geobpe --induce cameo=<dir> casp14=<dir> --out <json>
  python scripts/figures.py table --protrig runs/protrig_s8k_*/eval.json --geobpe runs/geobpe_paper_points.json --out logs/pareto_table.txt
  python scripts/figures.py panel --protrig runs/protrig_s8k_*/eval.json --geobpe runs/geobpe_paper_points.json --out runs/fig_panel

  eval    one ProtRig arm. BPR(T; D) = [L(T) + sum_x L(Enc(x))] / sum_x N(x) over D, the arm's training set (the chains
          recorded in the checkpoint), with the uniform per-token code of GeoBPE's BPE.capacity, and the distortion of
          every test chain decoded from its integer stream alone: backbone RMSD in ESM's form (ProteinChain.rmsd with
          only_compute_backbone_rmsd=True, per coordinate, the standard per-atom value divided by sqrt(3); this is
          GeoBPE's RMSD and the one on their axes; bb_rmsd_std is the standard form), CA RMSD, lDDT, TM-score and the
          validity of the decoded links. Test chains are not filtered: chain breaks are crossed by coded break
          transforms and pieces under 3 residues are sent raw, all charged in the rate. --geobpe-induce NAME=DIR also
          scores the chains GeoBPE's induce.py tokenized for that test set (NAME_geobpe_subset).
  geobpe  per-chain distortion and link geometry of induce.py output directories in the same forms, optionally
          paired with the per-chain results of an eval run.
  table   one table over eval outputs and GeoBPE baselines.
  panel   the single figure: a 4 x 2 grid of equal axes. Top: BPR against backbone RMSD and CA-LDDT (CAMEO, CASP14, ProtRig
          arms, GeoBPE and the other baselines), then RMSD and LDDT against chain length. Bottom: key, throughput, latency
          per chain, CPU cost. Reads the eval.json files, the per-chain JSONs and the JSON files of the speed benchmark
          protrig and, for GeoBPE, of the speed benchmark geobpe (only the run with mode "released"). Writes
          <out>.pdf, <out>.png and <speed-dir>/speed_table.txt.

          mkdir -p logs/compression/figures

python scripts/figures.py panel --protrig runs/protrig_s8k_{rw0p4_s0,rw0p2_s1,rw0p1_s0,rw0p06_s1,rw0p03_s1,rw0p015_s0,rw0p06_link3_s1}/eval.json --geobpe runs/geobpe_paper_points.json --out runs/fig_panel_seed1 > logs/compression/figures/panel_seed1.log 2>&1; tail -n 2 logs/compression/figures/panel_seed1.log
python scripts/analysis.py length --free runs/protrig_s8k_rw0p015_s0/eval_per_chain.json --valid runs/protrig_s8k_rw0p06_link3_s1/eval_per_chain.json --out runs/fig_robustness_length_seed1 > logs/analysis_length_seed1.log 2>&1; tail -n 2 logs/analysis_length_seed1.log

python scripts/figures.py panel --protrig runs/protrig_s8k_{rw0p4_s0,rw0p2_s1,rw0p1_s0,rw0p06_s1,rw0p03_s1,rw0p015_s0,rw0p06_link3_s1}/eval.json --geobpe runs/geobpe_paper_points.json --out runs/fig_panel_seed1 > logs/compression/figures/panel_seed1.log 2>&1; tail -n 2 logs/compression/figures/panel_seed1.log
"""
import argparse
import glob
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from esm.utils.structure.protein_chain import ProteinChain

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prorate import data as D
from prorate.checkpoint import load_protrig
from prorate.metrics import kabsch_rmsd, lddt, omega_report, peptide_validity

NAMES = ("cn_len_A", "ca_c_n_deg", "c_n_ca_deg")
METRIC_KEYS = ("bb_rmsd", "bb_rmsd_std", "ca_rmsd", "lddt_ca", "lddt_ca_repo", "tm")
BLUE, ORANGE, LIGHT = "#0072B2", "#D55E00", "#56B4E9"
ORDER = ("rw0p4_s0", "rw0p06_s0", "operating_point", "rw0p06_link3_s1", "rw0p015_s0")
ROW_KEYS = ("stream", "cameo", "casp14", "lddt", "lddt_casp14", "v_all", "v_bnd", "z_cn", "v_ref", "speed", "w_all", "w_bnd")


# ---- geometry of peptide links -----------------------------------------------------
def angle_deg(a, b, c):
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def link_values(X):
    """C-N length (A), CA-C-N and C-N-CA angles (degrees) of every link of X (n,3,3)."""
    C, N1, CA, CA1 = X[:-1, 2], X[1:, 0], X[:-1, 1], X[1:, 1]
    return np.stack([np.linalg.norm(N1 - C, axis=-1), angle_deg(CA, C, N1), angle_deg(C, N1, CA1)], 1)


def omega_dev(X):
    """Deviation from trans (degrees) of the peptide dihedral omega of every link of X (n,3,3)."""
    w = np.degrees(D._dihedral(X[:-1, 1], X[:-1, 2], X[1:, 0], X[1:, 1]))
    return 180.0 - np.abs(w)


def abs_stats(a):
    """Mean, 99th percentile and maximum of the absolute deviation on each link-geometry axis; a is (m, 3)."""
    out = {}
    for k, nm in enumerate(NAMES):
        out[f"{nm}_mean_abs"] = float(a[:, k].mean())
        out[f"{nm}_p99_abs"] = float(np.percentile(a[:, k], 99))
        out[f"{nm}_max_abs"] = float(a[:, k].max())
    return out


# ---- eval --------------------------------------------------------------------------
def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def description_bits(model):
    bits = model.description_bits()
    return {"L_V_floats": int(round(bits / 32.0)), "L_nets_floats": 0,
            "L_V_bits": bits, "L_full_bits": bits}


def load_layouts(files):
    """Chain layouts of the parsable files, each tagged with its file name, and the (name, reason) of the files that failed."""
    layouts, failed = [], []
    for f in files:
        try:
            lay = D.chain_layout(D.backbone_from_pdb(f))
            if not lay["items"]:
                raise ValueError("no resolved residue")
            lay["fname"] = f
            layouts.append(lay)
        except Exception as e:  # unparsable file: counted, never silent
            failed.append((os.path.basename(f), f"{type(e).__name__}: {e}"[:120]))
    return layouts, failed


def mean_metrics(rows):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in METRIC_KEYS}


@torch.no_grad()
def rate_on_chains(model, files, L, device, batch_size, min_piece, n_total=None):
    """Rate over every resolved residue of the chains in files, through the piece pipeline of the test
    sets: pieces of 3 or more residues are coded by the model, shorter ones sent raw, breaks charged.
    GeoBPE's capacity() tokenizes every residue of its training chains, so D is scored the same way.
    n_total is the residue count of D under the training pieces when files is a sample of D."""
    layouts, failed = load_layouts(files)
    if failed:
        print(f"  {len(failed)} files of D skipped: {failed[:3]}", flush=True)
    D.encode_chains(model, layouts, device, batch_size)
    bits = [D.chain_bits(model, lay) for lay in layouts]
    n_res = sum(b["residues"] for b in bits)
    n_tok = sum(b["tokens"] for b in bits)
    n_seg = sum(len(it["seg"]) for lay in layouts for it in lay["items"] if not it["raw"])
    stream = sum(b["total_bits"] for b in bits)
    piece_res = sum(it["n"] for lay in layouts for it in lay["items"] if min_piece <= it["n"] <= 512)
    n_all = n_total * n_res / piece_res if n_total else n_res
    tok_all = n_tok * n_all / n_res
    r = {"chains": len(layouts), "residues": n_res, "residues_full_D": n_all, "segments": n_seg,
         "residues_per_segment": n_res / max(n_seg, 1), "token_bits": stream,
         "break_bits": sum(b["break_bits"] for b in bits), "raw_bits": sum(b["raw_bits"] for b in bits),
         "breaks": sum(b["breaks"] for b in bits)}
    for name, den, den_all in (("per_res", n_res, n_all), ("per_tok", n_tok, tok_all)):
        r[f"bpr_{name}_tokens"] = stream / den
        r[f"bpr_{name}_L_V"] = L["L_V_bits"] / den_all + stream / den
        r[f"bpr_{name}_L_full"] = L["L_full_bits"] / den_all + stream / den
    return r


def tmscore(pred_chain, ref_chain, tmp):
    exe = shutil.which("TMscore")
    if exe is None:
        return float("nan")
    p, r = os.path.join(tmp, "pred.pdb"), os.path.join(tmp, "ref.pdb")
    pred_chain.to_pdb(p)
    ref_chain.to_pdb(r)
    txt = subprocess.run([exe, p, r], capture_output=True, text=True).stdout
    m = re.search(r"^TM-score\s*=\s*([0-9.]+)", txt, re.M)
    return float(m.group(1)) if m else float("nan")


def link_geometry(model, lay, pred, ref):
    """Decoded minus reference geometry of every peptide link inside a piece: C-N length (A),
    CA-C-N angle and C-N-CA angle (degrees). Returns (deltas (m, 3), is_boundary (m,), decoded values, reference
    values, decoded omega deviation, reference omega deviation), where a boundary is a link ProtRig closed with a
    6-DOF cell: the six values compensate pose, so they are not stereochemically constrained and these deviations
    are what to report for them."""
    idx, flag, off = [], [], 0
    for it in lay["items"]:
        n = it["n"]
        b = np.zeros(max(n - 1, 0), dtype=bool)
        if not it["raw"]:
            pos = 0
            for gid in it["seg"][:-1]:
                pos += model.codebook.length_of(gid)
                b[pos - 1] = True
        idx += list(range(off, off + n - 1))
        flag += b.tolist()
        off += n
    if not idx:
        return None
    i = np.asarray(idx)
    pl, rl = link_values(pred)[i], link_values(ref)[i]
    return pl - rl, np.asarray(flag), pl, rl, omega_dev(pred)[i], omega_dev(ref)[i]


def edge_fractions(model, joint_ids):
    """Per bond-geometry axis (theta_CACN, l_CN, theta_CNCA), the fraction of boundary cells at the first
    or last level of the grid. A search that exploits the range limits piles up here."""
    if not joint_ids:
        return {}
    idx = torch.cat(joint_ids).to(model.joints._basis.device)
    cell = model.joints.codes_from_index(idx) + model.joints._idx_shift.float()
    at_edge = (cell <= 0) | (cell >= (model.joints._levels - 1).float())
    return {n: float(at_edge[:, k].float().mean())
            for n, k in (("theta_cacn", 0), ("l_cn", 1), ("theta_cnca", 3))}


@torch.no_grad()
def evaluate_set(model, files, device, batch_size, tmp):
    """Every chain of a test set through the piece pipeline: rate itemized per chain,
    distortion over resolved residues, coverage counted."""
    layouts, failed = load_layouts(files)
    sync(device)
    t0 = time.perf_counter()
    D.encode_chains(model, layouts, device, batch_size)
    sync(device)
    encode_s = time.perf_counter() - t0
    rows, geo_d, geo_f, geo_p, geo_r, geo_op, geo_or, edge_ids = [], [], [], [], [], [], [], []
    for lay in layouts:
        D.encode_breaks(model, lay, device)        # encoder side: fits and rounds the break sextets
        pred = torch.from_numpy(D.decode_chain(model, lay, device))   # decoder side: no coordinates
        ref = torch.from_numpy(D.reference_coords(lay).astype(np.float64))
        assert pred.shape == ref.shape, (lay["fname"], pred.shape, ref.shape)
        bits = D.chain_bits(model, lay)
        edge_ids += [torch.tensor(it["joint"], dtype=torch.long)
                     for it in lay["items"] if not it["raw"] and it["joint"]]
        g = link_geometry(model, lay, pred.numpy(), ref.numpy())
        if g is not None:
            geo_d.append(g[0])
            geo_f.append(g[1])
            geo_p.append(g[2])
            geo_r.append(g[3])
            geo_op.append(g[4])
            geo_or.append(g[5])
        pc = ProteinChain.from_backbone_atom_coordinates(pred.numpy())
        rc = ProteinChain.from_backbone_atom_coordinates(ref.numpy())
        all_m = torch.ones(3 * pred.shape[0], dtype=torch.bool)
        ca_p, ca_r = pred[:, 1], ref[:, 1]
        m = torch.ones(ca_p.shape[0], dtype=torch.bool)
        rows.append({
            "chain": os.path.basename(lay["fname"]), "rows": lay["n_rows"],
            "residues": bits["residues"], "unresolved": lay["cuts"]["unresolved"],
            "breaks": bits["breaks"], "raw_residues": bits["raw_residues"],
            "cuts": lay["cuts"], **bits,
            "bb_rmsd": float(pc.rmsd(rc, only_compute_backbone_rmsd=True)),
            "bb_rmsd_std": kabsch_rmsd(pred.reshape(-1, 3), ref.reshape(-1, 3), all_m),
            "ca_rmsd": kabsch_rmsd(ca_p, ca_r, m),
            "lddt_ca": float(np.mean(pc.lddt_ca(rc))),
            "lddt_ca_repo": lddt(ca_p, ca_r, m),
            "tm": tmscore(pc, rc, tmp),
        })
    summary = mean_metrics(rows)
    n_res = sum(x["residues"] for x in rows)
    tot = {k: sum(x[k] for x in rows) for k in
           ("token_bits", "token_bits_coded", "total_bits_coded", "break_bits", "raw_bits",
            "total_bits", "tokens", "breaks",
            "raw_residues", "unresolved", "rows")}
    summary.update({
        "listed": len(files), "evaluated": len(rows), "failed_to_parse": failed,
        "chains_with_breaks": sum(x["breaks"] > 0 for x in rows),
        "chains_single_piece_no_unresolved": sum(x["breaks"] == 0 and x["unresolved"] == 0
                                                 for x in rows),
        "residues_scored": n_res, **tot,
        "rate_on_this_set_bits_per_res": tot["total_bits"] / max(n_res, 1),
        "rate_coded_secondary_bits_per_res": tot["total_bits_coded"] / max(n_res, 1),
        "rate_on_this_set_bits_per_tok": tot["total_bits"] / max(tot["tokens"], 1),
        "break_and_raw_share_of_rate": (tot["break_bits"] + tot["raw_bits"])
                                       / max(tot["total_bits"], 1e-9),
    })
    summary["encode_seconds"] = encode_s
    summary["encode_residues_per_s"] = n_res / max(encode_s, 1e-9)
    if geo_d:
        Dl, F = np.concatenate(geo_d), np.concatenate(geo_f)
        summary["link_geometry_decoded_minus_reference"] = {
            label: {"n_links": int(sel.sum()), **abs_stats(np.abs(Dl[sel]))}
            for label, sel in (("boundary", F), ("interior", ~F)) if sel.any()}
    if geo_p:
        summary["peptide_validity"] = peptide_validity(np.concatenate(geo_p), np.concatenate(geo_r),
                                                       np.concatenate(geo_f))
    if geo_op:
        summary["omega_planarity"] = omega_report(np.concatenate(geo_op), np.concatenate(geo_or),
                                                  np.concatenate(geo_f))
    summary["bond_axis_edge_fraction"] = edge_fractions(model, edge_ids)
    return summary, rows


def induce_stems(induce_dir):
    """Structure stems GeoBPE tokenized in a induce.py output directory."""
    stems = set()
    for f in glob.glob(os.path.join(induce_dir, "*.pkl")):
        if os.path.basename(f)[:-4].isdigit():
            with open(f, "rb") as fh:
                _, tok = pickle.load(fh)
            stems.add(Path(tok.fname).stem)
    if not stems:
        raise SystemExit(f"no per-structure pickles in {induce_dir}")
    return stems


def subset_summary(rows, stems):
    """Per-chain means and the rate over the chains whose stem is in stems."""
    sel = [r for r in rows if os.path.splitext(r["chain"])[0] in stems]
    n_res = sum(r["residues"] for r in sel)
    out = mean_metrics(sel) if sel else {}
    out.update({"evaluated": len(sel), "residues_scored": n_res,
                "rate_on_this_set_bits_per_res": sum(r["total_bits"] for r in sel) / max(n_res, 1)})
    return out


def run_eval(args):
    model = load_protrig(args.ckpt, args.device)
    if args.segment_rate_weight is not None:
        model.cfg.rate_weight = args.segment_rate_weight
    if args.zero_boundary_cost:
        model.boundary_cost.zero_()
    if args.link_weight is not None:
        model.cfg.link_weight = args.link_weight
    if args.link_tol is not None:
        model.cfg.link_tol = args.link_tol
    L = description_bits(model)
    print(f"{args.arm}: L(T) {L['L_V_floats']} floats ({L['L_V_bits']:.0f} bits)", flush=True)

    # D, exactly the training set of the arm.
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    files = [os.path.join(args.train_dir, os.path.basename(f)) for f in blob["train_files"]]
    d_src = f"checkpoint train_files ({len(files)})"
    min_piece = int(blob.get("args", {}).get("min_piece", 16))
    n_total = blob.get("train_residues")
    del blob
    if args.rate_chains and args.rate_chains < len(files):
        if n_total is None:
            raise SystemExit(f"{args.ckpt} has no train_residues, so --rate-chains cannot be used")
        rng = np.random.RandomState(0)
        files = sorted(rng.choice(files, args.rate_chains, replace=False).tolist())
        d_src += f"; stream rate estimated on {len(files)} random chains"
    else:
        n_total = None
    print(f"  D from {d_src}", flush=True)
    train_rate = rate_on_chains(model, files, L, args.device, args.batch_size, min_piece, n_total)
    result = {"arm": args.arm, "rate_weight": args.rate_weight, "ckpt": args.ckpt,
              "segment_rate_weight": float(model.cfg.rate_weight), "link_weight": float(model.cfg.link_weight), "link_tol": float(model.cfg.link_tol),
              "boundary_cost": [float(x) for x in model.boundary_cost],
              "D_source": d_src, "description_length": L, "train": train_rate}
    print(f"  D (train): {json.dumps(result['train'])}", flush=True)

    per_chain = {}
    restrict = dict(s.split("=", 1) for s in args.geobpe_induce)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        pool = sorted(np.random.RandomState(1).choice(files, min(100, len(files)), replace=False).tolist())
        train_summary, train_rows = evaluate_set(model, pool, args.device, args.batch_size, tmp)
        result["train_sample"] = train_summary
        per_chain["train_sample"] = train_rows
        print(f"  train sample: {train_summary['evaluated']} chains; "
              + " ".join(f"{k} {train_summary[k]:.3f}" for k in ("bb_rmsd", "lddt_ca", "tm")), flush=True)
        for spec in args.test:
            name, d = spec.split("=", 1)
            tfiles = sorted(glob.glob(os.path.join(d, "*.pdb")))
            summary, rows = evaluate_set(model, tfiles, args.device, args.batch_size, tmp)
            result[name] = summary
            summary["test_over_train_bb_rmsd"] = summary["bb_rmsd"] / train_summary["bb_rmsd"]
            print(f"  {name} test/train bb_rmsd ratio {summary['test_over_train_bb_rmsd']:.3f}", flush=True)
            per_chain[name] = rows
            if name in restrict:
                sub = subset_summary(rows, induce_stems(restrict[name]))
                result[name + "_geobpe_subset"] = sub
                print(f"  {name} on GeoBPE's {sub['evaluated']} chains: "
                      + " ".join(f"{k} {sub[k]:.3f}" for k in ("bb_rmsd", "lddt_ca") if k in sub), flush=True)
            print(f"  {name}: {summary['evaluated']}/{summary['listed']} chains "
                  f"({summary['chains_with_breaks']} with breaks, {summary['breaks']} breaks, "
                  f"{summary['unresolved']} unresolved residues, {summary['raw_residues']} raw); "
                  + " ".join(f"{k} {summary[k]:.3f}" for k in
                             METRIC_KEYS + ("rate_on_this_set_bits_per_res", "rate_coded_secondary_bits_per_res",
                                            "break_and_raw_share_of_rate")), flush=True)
            for lab, g in summary.get("link_geometry_decoded_minus_reference", {}).items():
                print(f"    {name} {lab} links ({g['n_links']}): |d C-N| {g['cn_len_A_mean_abs']:.3f} A "
                      f"(p99 {g['cn_len_A_p99_abs']:.3f}, max {g['cn_len_A_max_abs']:.3f}); "
                      f"|d CA-C-N| {g['ca_c_n_deg_mean_abs']:.2f} deg (p99 {g['ca_c_n_deg_p99_abs']:.2f}); "
                      f"|d C-N-CA| {g['c_n_ca_deg_mean_abs']:.2f} deg (p99 {g['c_n_ca_deg_p99_abs']:.2f})",
                      flush=True)
            print(f"    {name} bond-axis cells at grid edge: {summary['bond_axis_edge_fraction']}", flush=True)
            print(f"    {name} peptide validity: {json.dumps(summary.get('peptide_validity'))}", flush=True)
            print(f"    {name} omega planarity: {json.dumps(summary.get('omega_planarity'))}", flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.splitext(args.out)[0] + "_per_chain.json", "w") as f:
        json.dump(per_chain, f, indent=1)
    print(f"wrote {args.out}")


# ---- geobpe ------------------------------------------------------------------------
def score_directory(induce_dir):
    rows, deltas, skipped, links, omegas = {}, [], [], [], []
    for f in sorted(glob.glob(os.path.join(induce_dir, "*.pkl"))):
        if not os.path.basename(f)[:-4].isdigit():
            continue
        with open(f, "rb") as fh:
            stats, tok = pickle.load(fh)
        ref_chain = ProteinChain.from_pdb(tok.fname)
        ref = D.backbone_from_protein_chain(ref_chain)
        pred = np.asarray(tok.compute_coords(), dtype=np.float64).reshape(-1, 3, 3)
        stem = Path(tok.fname).stem
        if pred.shape != ref.shape or not np.isfinite(ref).all():
            skipped.append(stem)
            continue
        recon = ProteinChain.from_backbone_atom_coordinates(pred)
        ref_links = link_values(ref)
        pred_links = link_values(pred)
        bonded = ref_links[:, 0] <= D.CN_HI
        rows[stem] = {"bb_rmsd": float(recon.rmsd(ref_chain, only_compute_backbone_rmsd=True)),
                      "lddt_ca": float(np.mean(recon.lddt_ca(ref_chain))),
                      "stored_bb_rmsd": float(stats["rmsd"][-1]),
                      "stored_lddt": float(np.mean(stats["lddt"][-1])),
                      "residues": int(pred.shape[0]),
                      "gap_links": int((~bonded).sum())}
        deltas.append((pred_links - ref_links)[bonded])
        links.append((pred_links[bonded], ref_links[bonded]))
        omegas.append((omega_dev(pred)[bonded], omega_dev(ref)[bonded]))
    return rows, deltas, skipped, links, omegas


def summarize_directory(rows, deltas, skipped):
    clean = [r for r in rows.values() if r["gap_links"] == 0]
    out = {"chains": len(rows), "skipped": skipped,
           "chains_with_gap": len(rows) - len(clean),
           "gap_links": sum(r["gap_links"] for r in rows.values()),
           "bb_rmsd": float(np.mean([r["bb_rmsd"] for r in rows.values()])),
           "lddt_ca": float(np.mean([r["lddt_ca"] for r in rows.values()])),
           "bb_rmsd_no_gap": float(np.mean([r["bb_rmsd"] for r in clean])) if clean else None,
           "lddt_ca_no_gap": float(np.mean([r["lddt_ca"] for r in clean])) if clean else None,
           "max_abs_diff_vs_stored_rmsd": max(abs(r["bb_rmsd"] - r["stored_bb_rmsd"]) for r in rows.values()),
           "max_abs_diff_vs_stored_lddt": max(abs(r["lddt_ca"] - r["stored_lddt"]) for r in rows.values())}
    a = np.abs(np.concatenate(deltas))
    out["n_links_excluding_gaps"] = int(a.shape[0])
    out.update(abs_stats(a))
    return out


def pair_with_protrig(rows, protrig_rows):
    prot = {os.path.splitext(r["chain"])[0]: r for r in protrig_rows}
    common = sorted(set(rows) & set(prot))
    out = {"common_chains": len(common)}
    groups = {"all": common,
              "no_gap": [s for s in common if rows[s]["gap_links"] == 0],
              "gap": [s for s in common if rows[s]["gap_links"] > 0]}
    for label, sel in groups.items():
        if not sel:
            continue
        res = sum(prot[s]["residues"] for s in sel)
        out[label] = {"chains": len(sel),
                      "geobpe_bb_rmsd": float(np.mean([rows[s]["bb_rmsd"] for s in sel])),
                      "protrig_bb_rmsd": float(np.mean([prot[s]["bb_rmsd"] for s in sel])),
                      "geobpe_lddt_ca": float(np.mean([rows[s]["lddt_ca"] for s in sel])),
                      "protrig_lddt_ca": float(np.mean([prot[s]["lddt_ca"] for s in sel])),
                      "protrig_lower_rmsd_fraction": float(np.mean([prot[s]["bb_rmsd"] < rows[s]["bb_rmsd"] for s in sel])),
                      "protrig_stream_bits_per_res": sum(prot[s]["total_bits"] for s in sel) / max(res, 1)}
    return out


def run_geobpe(args):
    prot = json.load(open(args.protrig_per_chain)) if args.protrig_per_chain else {}
    result = {}
    for spec in args.induce:
        name, d = spec.split("=", 1)
        rows, deltas, skipped, links, omegas = score_directory(d)
        result[name] = summarize_directory(rows, deltas, skipped)
        result[name]["peptide_validity"] = peptide_validity(np.concatenate([p for p, _ in links]),
                                                            np.concatenate([r for _, r in links]))
        result[name]["omega_planarity"] = omega_report(np.concatenate([p for p, _ in omegas]),
                                                       np.concatenate([r for _, r in omegas]))
        if name in prot:
            result[name]["paired_with_protrig"] = pair_with_protrig(rows, prot[name])
        print(name, json.dumps(result[name], indent=1), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


# ---- points: eval outputs and paper baselines ---------------------------------------
def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(v, w=8, p=3):
    return ("-" if v is None else f"{v:.{p}f}").rjust(w)


def fmt_speed(v, p=1):
    return fmt(v, 0, p)


def protrig_row(path):
    P = json.load(open(path))
    sub = "cameo_geobpe_subset" if "cameo_geobpe_subset" in P else "cameo"
    sub_c = "casp14_geobpe_subset" if "casp14_geobpe_subset" in P else "casp14"
    pv = ("cameo", "peptide_validity")
    return {"method": "ProtRig (valid links)" if "natural" in P["arm"] else "ProtRig", "name": P["arm"], "bpr": get(P, "train", "bpr_per_res_L_V"),
            "stream": get(P, "train", "bpr_per_res_tokens"),
            "cameo": get(P, sub, "bb_rmsd"), "casp14": get(P, sub_c, "bb_rmsd"),
            "lddt": get(P, sub, "lddt_ca"), "lddt_casp14": get(P, sub_c, "lddt_ca"),
            "v_all": get(P, *pv, "decoded", "frac_any_beyond", "2.0"),
            "v_bnd": get(P, *pv, "decoded_boundary", "frac_any_beyond", "2.0"),
            "z_cn": (get(P, *pv, "decoded_boundary", "mean_abs_z") or [None])[0],
            "v_ref": get(P, *pv, "reference", "frac_any_beyond", "2.0"),
            "speed": get(P, "cameo", "encode_residues_per_s"),
            "w_all": get(P, "cameo", "omega_planarity", "decoded", "frac_beyond", "10.0"),
            "w_bnd": get(P, "cameo", "omega_planarity", "decoded_boundary", "frac_beyond", "10.0")}


def geobpe_rows(path):
    """Reference points from a JSON list written by hand from the paper: one object per point with name, method,
    bpr, cameo_rmsd, casp14_rmsd, cameo_lddt and casp14_lddt."""
    return [{**dict.fromkeys(ROW_KEYS), "method": p.get("method", "GeoBPE"), "name": p["name"], "bpr": p["bpr"],
             "cameo": p.get("cameo_rmsd"), "casp14": p.get("casp14_rmsd"),
             "lddt": p.get("cameo_lddt"), "lddt_casp14": p.get("casp14_lddt")} for p in json.load(open(path))]

def baseline_rows(directory):
    """Measured points of ProToken and AminoAseed from the baselines.py JSONs of the GPU runs (quality does not depend on the device):
    mean over every chain of each test set that the method decodes. The rate is the charged rate given to baselines.py."""
    out = []
    for f in sorted(glob.glob(os.path.join(directory, "*.json"))):
        r = json.load(open(f))
        if r["name"].endswith("_cpu1") or "latency" not in r:
            continue
        mean = lambda s, k: float(np.mean([x[k] for x in r["latency"] if x["set"] == s]))
        out.append({**dict.fromkeys(ROW_KEYS), "method": r["method"], "name": r["name"], "bpr": r["bpr"],
                    "cameo": mean("cameo", "bb_rmsd"), "casp14": mean("casp14", "bb_rmsd"),
                    "lddt": mean("cameo", "lddt_ca"), "lddt_casp14": mean("casp14", "lddt_ca")})
    return out


def collect_rows(args):
    measured = baseline_rows(args.baseline_dir) if getattr(args, "baseline_dir", None) else []
    have = {r["method"] for r in measured}
    paper = [r for s in args.geobpe for r in geobpe_rows(s) if r["method"] not in have]
    return paper + measured + [protrig_row(p) for p in args.protrig]


def aggregate_points(rows):
    """One point per (method, arm): the mean over seeds, with the s.d. when there is more than one seed."""
    keys = ("bpr", "cameo", "casp14", "lddt", "lddt_casp14")
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], re.sub(r"_s\d+$", "", r["name"])), []).append(r)
    out = []
    for (method, name), g in groups.items():
        rec = {"method": method, "name": name, "n": len(g)}
        for k in keys:
            v = [x[k] for x in g if x[k] is not None]
            rec[k] = float(np.mean(v)) if v else None
            rec[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out.append(rec)
    return out


def run_table(args):
    rows = collect_rows(args)
    head = (f"{'arm':28s} {'BPR':>8s} {'stream':>8s} {'CAMEO':>8s} {'CASP14':>8s} {'LDDT':>8s} "
            f"{'>2sig':>8s} {'>2sig_b':>8s} {'|zCN|_b':>8s} {'ref>2s':>8s} {'om>10':>8s} {'om>10_b':>8s} "
            f"{'res/s':>8s}")
    lines = [head]
    for r in rows:
        lines.append(f"{r['name']:28s} {fmt(r['bpr'])} {fmt(r['stream'])} {fmt(r['cameo'])} {fmt(r['casp14'])} "
                     f"{fmt(r['lddt'])} {fmt(r['v_all'])} {fmt(r['v_bnd'])} {fmt(r['z_cn'])} "
                     f"{fmt(r['v_ref'])} {fmt(r['w_all'])} {fmt(r['w_bnd'])} {fmt(r['speed'], 8, 0)}")
    text = "\n".join(lines)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")


# ---- speed data and table ------------------------------------------------------------
def read_dir(directory):
    out = {}
    for f in sorted(glob.glob(os.path.join(directory, "*.json"))):
        r = json.load(open(f))
        out[r["name"]] = r
    return out


def bpr_of(r):
    return r.get("bpr")


def label_of(name, r):
    """Readable name of a configuration: its method, its compression rate, plus what is special about it."""
    b = bpr_of(r)
    s = f"{b:.1f} bits/res" if b is not None else name
    if "link3" in name:
        s += " (valid)"
    if name.endswith("_cpu1"):
        s += ", 1 CPU thread"
    method = r.get("method") or ("GeoBPE" if name.startswith("geobpe") else "ProtRig")
    return f"{method}, {s}"


def fit_cost(r):
    if "ckpt" not in r:
        return None, None
    p = ROOT / Path(r["ckpt"]).parent / "metrics.jsonl"
    if not p.exists():
        return None, None
    rec = [json.loads(l) for l in open(p)]
    return rec[-1]["elapsed_s"], max(x.get("gpu_mem_gb", 0.0) for x in rec)


def largest_batch(r):
    return max(int(k) for k in r["throughput"])


def encode_rate(r, bs):
    t = r["throughput"].get(str(bs))
    return None if t is None else t["residues_per_s"]


def per_1000_residues(r, key, bs=1):
    t = r["throughput"].get(str(bs))
    return None if t is None else 1000.0 * t[key] / r["rows"]


def latency_stats(r):
    e = np.array([x["encode_ms"] for x in r["latency"]])
    d = np.array([x["decode_ms"] for x in r["latency"]])
    return float(np.median(e)), float(np.percentile(e, 95)), float(np.median(d)), float(np.percentile(d, 95))


def write_speed_table(path, entries):
    head = ["setting", "arm", "chains", "BPR", "enc b1", "enc b8", "enc b64", "decode", "enc ms med", "enc ms p95",
            "dec ms med", "dec ms p95", "wall s/1k (b1)", "cpu s/1k (b1)", "gpu MB", "ram MB", "fit min", "fit GB"]
    lines = []
    for name, r in entries:
        lm = latency_stats(r)
        fit_s, fit_gb = fit_cost(r)
        lines.append([label_of(name, r), name, str(r["chains"]), fmt_speed(bpr_of(r)), fmt_speed(encode_rate(r, 1), 0),
                      fmt_speed(encode_rate(r, 8), 0), fmt_speed(encode_rate(r, 64), 0), fmt_speed(r["decode"]["residues_per_s"], 0),
                      fmt_speed(lm[0], 0), fmt_speed(lm[1], 0), fmt_speed(lm[2], 1), fmt_speed(lm[3], 1),
                      fmt_speed(per_1000_residues(r, "wall_s"), 2), fmt_speed(per_1000_residues(r, "cpu_s"), 2),
                      fmt_speed(r["throughput"][str(largest_batch(r))].get("peak_gpu_mb"), 0),
                      fmt_speed(r.get("peak_ram_mb"), 0), fmt_speed(None if fit_s is None else fit_s / 60.0, 0), fmt_speed(fit_gb, 2)])
    widths = [max(len(head[i]), *(len(l[i]) for l in lines)) for i in range(len(head))]
    just = lambda c, w, i: c.ljust(w) if i < 2 else c.rjust(w)
    out = ["  ".join(just(h, w, i) for i, (h, w) in enumerate(zip(head, widths)))]
    out += ["  ".join(just(c, w, i) for i, (c, w) in enumerate(zip(l, widths))) for l in lines]
    Path(path).write_text("\n".join(out) + "\n")
    print("\n".join(out), flush=True)


def load_speed(args):
    prot = read_dir(args.speed_dir)
    geo_all = read_dir(args.geobpe_dir) if os.path.isdir(args.geobpe_dir) else {}
    geo = {n: r for n, r in geo_all.items() if r.get("mode", "released") == "released" and r.get("bpr") is not None}
    base_all = read_dir(args.baseline_dir) if os.path.isdir(args.baseline_dir) else {}
    base = {n: r for n, r in base_all.items() if r.get("bpr") is not None and n.endswith("_cpu1")}
    table_arms = [(n, prot[n]) for n in ORDER if n in prot]
    if not table_arms:
        raise SystemExit(f"no ProtRig JSON in {args.speed_dir}")
    cpu_arms = sorted(((n, r) for n, r in prot.items() if n.endswith("_cpu1")), key=lambda t: bpr_of(t[1]))
    arms = [(n, r) for n, r in table_arms if "link3" not in n]
    return {"prot": prot, "geo": geo, "base": base, "arms": arms, "cpu_arms": cpu_arms,
            "cpu1": prot.get("operating_point_cpu1")}


# ---- the figure ----------------------------------------------------------------------
FREE, VALID = "protrig_free", "protrig_valid"
SERIES = {
    FREE: dict(label="ProRate (free)", color=BLUE, marker="o", size=20, line=True, z=6),
    VALID: dict(label="ProRate (valid)", color=BLUE, marker="D", size=22, line=True, z=7),
    "GeoBPE": dict(label="GeoBPE", color=ORANGE, marker="o", size=20, line=False, z=5),
    "ProToken": dict(label="ProToken", color="#CC79A7", marker="s", size=20, line=False, z=5),
    "ESM3": dict(label="ESM3", color="#56E96A", marker="s", size=20, line=False, z=5),
    "VQ-VAE": dict(label="VQ-VAE", color="#E69F00", marker="v", size=20, line=False, z=4),
    "AminoASeed": dict(label="AminoASeed", color="#6A3D9A", marker="v", size=20, line=False, z=4),
}
DATASET_STYLE = {
    "cameo": dict(linestyle="-", face="filled", xmul=1.00),
    "casp14": dict(linestyle="--", face="white", xmul=1.035),
}


def series_of(r):
    """Series key of a point, or None for arms that are not drawn. The valid series is the link3 arm."""
    if r["method"].startswith("ProtRig"):
        if "link3" in r["name"]:
            return VALID
        if any(k in r["name"] for k in ("natural", "fixcn", "bits23", "link")):
            return None
        return FREE
    return r["method"]


def apply_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.grid": True, "grid.linewidth": 0.5, "grid.alpha": 0.22, "grid.linestyle": "-",
    })
    return plt


def draw_pareto(ax_rmsd, ax_lddt, points):
    from matplotlib import ticker

    for ax, metric_map in ((ax_rmsd, {"cameo": "cameo", "casp14": "casp14"}),
                           (ax_lddt, {"cameo": "lddt", "casp14": "lddt_casp14"})):
        for dname, metric_key in metric_map.items():
            ds = DATASET_STYLE[dname]
            for name, st in SERIES.items():
                pts = sorted((r for r in points if series_of(r) == name and r["bpr"] is not None
                              and r[metric_key] is not None), key=lambda r: r["bpr"])
                if not pts:
                    continue
                x = np.array([p["bpr"] for p in pts]) * ds["xmul"]
                y = np.array([p[metric_key] for p in pts])
                xe = np.array([p["bpr_sd"] for p in pts]) * ds["xmul"]
                ye = np.array([p[metric_key + "_sd"] for p in pts])
                if st["line"] and len(pts) > 1:
                    ax.plot(x, y, color=st["color"], lw=1.0, linestyle=ds["linestyle"], alpha=0.65, zorder=st["z"] - 1)
                if (xe > 0).any() or (ye > 0).any():
                    ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=st["color"], elinewidth=0.7,
                                capsize=1.5, capthick=0.7, alpha=0.85, zorder=st["z"])
                face = st["color"] if ds["face"] == "filled" else "white"
                ax.scatter(x, y, s=st["size"], marker=st["marker"], facecolor=face, edgecolor=st["color"],
                           linewidth=1.2, zorder=st["z"] + 1)
        ax.set_xscale("log")
        ax.set_xlim(6, 4000)
        ax.set_xticks([10, 100, 1000])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("Bits per residue (BPR)")
    ax_rmsd.set_title("RMSD against BPR", pad=10, fontweight="bold")
    ax_lddt.set_title("CA-LDDT against BPR", pad=10, fontweight="bold")
    ax_rmsd.set_yscale("log")
    ax_rmsd.set_ylim(0.045, 8)
    ax_rmsd.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax_rmsd.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_rmsd.yaxis.set_minor_locator(ticker.NullLocator())
    ax_rmsd.set_ylabel("Backbone RMSD (Å)")
    ax_lddt.set_ylim(0.42, 1.02)
    ax_lddt.set_ylabel("CA-LDDT")


def per_chain(path, geobpe=False):
    """{(set, stem): (length, bb_rmsd, lddt_ca)} from an eval_per_chain.json of ProtRig or a bench_speed.py geobpe or baselines.py JSON.
    The key includes the test set, because a chain name can occur in both CAMEO and CASP14."""
    P = json.load(open(path))
    rows = [(r["set"], r) for r in P["latency"]] if geobpe else [(k, r) for k in ("cameo", "casp14") for r in P[k]]
    return {(s, os.path.splitext(r["chain"])[0]): (r["rows"], r["bb_rmsd"], r["lddt_ca"]) for s, r in rows if "bb_rmsd" in r}


#def pearson(x, y):
#    """Pearson r with a 95% interval from the Fisher transform."""
#    r, n = float(np.corrcoef(x, y)[0, 1]), len(x)
#    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)#
#    return r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def draw_length(ax_rmsd, ax_lddt, args):
    import matplotlib.patheffects as pe

    series = [(per_chain(args.length_free), "o", True, BLUE, "-"),
              (per_chain(args.length_valid), "D", False, BLUE, "-"),
              (per_chain(args.length_geobpe, geobpe=True), "o", True, ORANGE, "-")]
    for key, path in (("ProToken", args.length_protoken), ("AminoASeed", args.length_aminoaseed)):
        series.append((per_chain(path, geobpe=True), SERIES[key]["marker"], True, SERIES[key]["color"], "-"))
    stems = sorted(set.intersection(*(set(d) for d, *_ in series)))
    stems = [s for s in stems if series[0][0][s][0] <= args.max_length]
    if len(stems) < 20:
        raise SystemExit(f"only {len(stems)} chains in common: check the per-chain files")
    length = np.array([series[0][0][s][0] for s in stems], dtype=float)
    edges = np.quantile(length, np.linspace(0, 1, 6))
    bins = np.clip(np.searchsorted(edges, length, side="right") - 1, 0, 4)
    print(f"{len(stems)} chains in common, length {length.min():.0f} to {length.max():.0f}", flush=True)
    for ax, k, ylabel in ((ax_rmsd, 1, "Backbone RMSD (Å)"), (ax_lddt, 2, "CA-LDDT")):
        zorders = (8, 7, 5, 3, 4)   # ProRate (free), ProRate (valid) on top; then AminoAseed, ProToken, GeoBPE below
        for (d, marker, filled, color, ls), z in zip(series, zorders):
            y = np.array([d[s][k] for s in stems], dtype=float)
            xm = [float(np.median(length[bins == b])) for b in range(5)]
            ym = [float(y[bins == b].mean()) for b in range(5)]
            ys = [float(y[bins == b].std()) for b in range(5)]
            eb = ax.errorbar(xm, ym, yerr=ys, color=color, ls=ls, lw=1.2, capsize=2, capthick=1.2, marker=marker,
                             ms=4, zorder=z)
            for art in (eb[0], *eb[1], *eb[2]):
                art.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
        ax.set_xlabel("Chain length (residues)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 250, 500])
    ax_rmsd.set_yscale("log")
    ax_rmsd.set_ylim(0.04, 10)
    ax_lddt.set_ylim(top=1.01)
    ax_rmsd.set_title("RMSD against length", pad=10, fontweight="bold")
    ax_lddt.set_title("CA-LDDT against length", pad=10, fontweight="bold")

    

def draw_speed(ax_a, ax_b, ax_c, S, latency_arm):
    from matplotlib import ticker
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D

    prot, geo, base, arms, cpu_arms, cpu1 = S["prot"], S["geo"], S["base"], S["arms"], S["cpu_arms"], S["cpu1"]
    tint = lambda c, f=0.6: tuple(f + (1 - f) * v for v in to_rgb(c))
    on_gpu = lambda r: str(r.get("device", "cpu")).startswith("cuda")
    rivals = [(ORANGE, "s", r) for r in geo.values()]
    rivals += [(SERIES[r["method"]]["color"], SERIES[r["method"]]["marker"], r) for r in base.values()]

    # throughput against compression rate; marker fill and line style give the operation
    ax = ax_a
    pts = sorted(((bpr_of(r), r) for _, r in arms if bpr_of(r) is not None), key=lambda t: t[0])

    def draw(points, marker, color, fill, value, line, size, z):
        xy = [(b, value(r)) for b, r in points if value(r) is not None]
        if not xy:
            return
        xs, ys = zip(*xy)
        if line:
            ax.plot(xs, ys, color=color, lw=0.8, alpha=0.6, ls=line)
        face = {"solid": color, "tint": tint(color), "open": "white"}[fill]
        ax.scatter(xs, ys, s=size, marker=marker, facecolor=face, edgecolor=color, linewidth=1.0, zorder=z)

    operations = (("decode", "solid", lambda r: r["decode"]["residues_per_s"], ":"),
                  ("encode, batch 64", "tint", lambda r: encode_rate(r, 64), "-"),
                  ("encode, batch 1", "open", lambda r: encode_rate(r, 1), "--"))
    for label, fill, value, line in operations:
        draw(pts, "o", BLUE, fill, value, line, 18, 3)
        for color, marker, r in rivals:
            draw([(r["bpr"], r)], marker, color, fill, value, None, 12, 4)
    ax.set_xscale("log")
    ax.set_xlim(6, 1000)
    ax.set_xticks([10, 100, 1000])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yscale("log")
    ax.set_ylim(15, 4e4)
    ax.set_xlabel("Bits per residue")
    ax.set_ylabel("Residues per second")
    ax.set_title("Throughput", pad=10, fontweight="bold")
    key = [Line2D([0], [0], color="0.35", ls=line, lw=0.9, marker="o", ms=4,
                  mfc={"solid": "0.35", "tint": "0.8", "open": "white"}[fill], mec="0.35", label=label)
           for label, fill, _, line in operations]
    ax.legend(handles=key, loc="upper right", frameon=False, fontsize=6,
              handlelength=2.0, labelspacing=0.25)

    # per-chain latency against chain length; median per length bin, band = 5th to 95th percentile (encode only)
    ax = ax_b
    lat_arm = prot.get(latency_arm)
    geo_lat = next((r for r in geo.values() if r.get("latency")), None)
    series = []
    if lat_arm and lat_arm.get("latency"):
        series.append(("ProRate GPU", lat_arm, BLUE))
    if cpu1 and cpu1.get("latency"):
        series.append(("ProRate CPU", cpu1, LIGHT))
    if geo_lat:
        series.append(("GeoBPE CPU", geo_lat, ORANGE))
    for r in base.values():
        if r.get("latency"):
            series.append((f"{r['method']} {'GPU' if on_gpu(r) else 'CPU'}", r, SERIES[r["method"]]["color"]))
    all_n = np.concatenate([[x["rows"] for x in r["latency"]] for _, r, _ in series]).astype(float)
    edges = np.geomspace(all_n.min(), all_n.max() * 1.001, 10)

    def binned(r, key):
        n = np.array([x["rows"] for x in r["latency"]], dtype=float)
        t = np.array([x[key] for x in r["latency"]], dtype=float)
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (n >= lo) & (n < hi)
            if m.sum() >= 5:
                out.append((np.sqrt(lo * hi), np.median(t[m]), np.percentile(t[m], 5), np.percentile(t[m], 95)))
        return np.array(out)

    tops, rights = [], []
    for label, r, color in series:
        for key_, line in (("encode_ms", "-"), ("decode_ms", "--")):
            b = binned(r, key_)
            if len(b) == 0:
                continue
            ax.plot(b[:, 0], b[:, 1], color=color, ls=line, lw=1.1)
            rights.append(b[-1, 0])
            if key_ == "encode_ms":
                ax.fill_between(b[:, 0], b[:, 2], b[:, 3], color=color, alpha=0.15, lw=0)
                tops.append(b[:, 3].max())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([50, 100, 200, 500])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlim(30, 500)
    ax.set_ylim(top=max(tops) * 60)
    handles = [Line2D([0], [0], color=c, lw=1.2, label=l) for l, _, c in series]
    handles += [Line2D([0], [0], color="0.4", lw=1.1, ls="-", label="encode"),
                Line2D([0], [0], color="0.4", lw=1.1, ls="--", label="decode")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6, handlelength=1.3, labelspacing=0.3,
              ncol=2, columnspacing=0.8)
    ax.set_xlabel("Chain length (residues)")
    ax.set_ylabel("Time per chain (ms)")
    ax.set_title("Latency per chain", pad=10, fontweight="bold")

    # CPU time per 1000 residues at batch 1: one group per hardware, one row per compression rate
    ax = ax_c
    gpu_rows = [(bpr_of(r), BLUE, "o", r) for _, r in arms]
    cpu_rows = [(bpr_of(r), BLUE, "o", r) for _, r in cpu_arms]
    for color, marker, r in rivals:
        (gpu_rows if on_gpu(r) else cpu_rows).append((r["bpr"], color, marker, r))
    ticks, labels, heads, vals, y = [], [], [], [], 0
    for title, items in (("GPU", gpu_rows), ("CPU", cpu_rows)):
        items = [(b, color, marker, per_1000_residues(r, "cpu_s")) for b, color, marker, r in items if b is not None]
        items = sorted((t for t in items if t[3] is not None), key=lambda t: t[0])
        if not items:
            continue
        heads.append(len(ticks))
        ticks.append(y)
        labels.append(title)
        y += 1
        for b, color, marker, v in items:
            ax.scatter([v], [y], s=26, marker=marker, facecolor="white", edgecolor=color, linewidth=1.2, zorder=3)
            ax.annotate(f"{v:.2f}" if v < 10 else f"{v:.0f}", (v, y), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=6, color="0.25")
            ticks.append(y)
            labels.append(f"{b:.1f} bits/res" if b < 100 else f"{b:.0f} bits/res")
            vals.append(v)
            y += 1
        y += 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.2)
    for i in heads:
        ax.get_yticklabels()[i].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(min(vals) * 0.6, max(vals) * 5)
    ax.set_xlabel("Seconds per 1,000 residues")
    ax.set_title("Time to encode", pad=10, fontweight="bold")


def draw_key(ax, points):
    from matplotlib.lines import Line2D

    drawn = {series_of(r) for r in points if r["bpr"] is not None}
    ax.axis("off")
    methods = [Line2D([0], [0], marker=st["marker"], linestyle="None", color="none", markerfacecolor=st["color"],
                      markeredgecolor=st["color"], markersize=1.2 * np.sqrt(st["size"]), label=st["label"])
               for name, st in SERIES.items() if name in drawn]
    spacer = Line2D([0], [0], linestyle="None", marker="None", label=" ")
    datasets = [Line2D([0], [0], marker="o", linestyle="-", color="0.4", markerfacecolor="0.4", markeredgecolor="0.4",
                       markersize=6, label="CAMEO"),
                Line2D([0], [0], marker="o", linestyle="--", color="0.4", markerfacecolor="white",
                       markeredgecolor="0.4", markersize=6, label="CASP14")]
    ax.legend(handles=methods + [spacer] + datasets, loc="upper left", bbox_to_anchor=(0.0, 1.2), frameon=False,
              fontsize=11, handletextpad=0.7, labelspacing=0.45, handlelength=2.2, borderpad=0.2)


def run_panel(args):
    plt = apply_style()
    points = aggregate_points(collect_rows(args))
    speed = load_speed(args)

    AW, AH, PITCH, LEFT, BOTTOM, GAP = 1.65, 1.53, 2.45, 0.65, 0.6, 0.95
    W = LEFT + 3 * PITCH + AW + 0.1
    H = BOTTOM + 2 * AH + GAP + 0.45
    fig = plt.figure(figsize=(W, H))

    def add(col, row):
        bottom = BOTTOM + (AH + GAP if row == 0 else 0.0)
        return fig.add_axes([(LEFT + col * PITCH) / W, bottom / H, AW / W, AH / H])

    top = [add(c, 0) for c in range(4)]
    ax_key, ax_a, ax_b, ax_c = add(0, 1), add(1, 1), add(2, 1), add(3, 1)
    for ax in top + [ax_a, ax_b, ax_c]:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    draw_pareto(top[0], top[1], points)
    draw_length(top[2], top[3], args)
    draw_speed(ax_a, ax_b, ax_c, speed, args.latency_arm)
    draw_key(ax_key, points)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}", flush=True)


def add_point_arguments(p):
    p.add_argument("--protrig", nargs="+", default=[], help="eval.json files of ProtRig arms; seeds are averaged by arm name")
    p.add_argument("--geobpe", nargs="*", default=[], help="JSON lists of reference points written from the paper")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--arm", required=True)
    e.add_argument("--rate-weight", type=float, required=True,
                   help="the arm's --rate-weight, recorded for the sweep plot")
    e.add_argument("--segment-rate-weight", type=float, default=None,
                   help="segment with this rate weight instead of the checkpoint's; the codebook is unchanged")
    e.add_argument("--zero-boundary-cost", action="store_true",
                   help="zero the fitted boundary cost E(L) before segmenting, as scripts/tables.py score does")
    e.add_argument("--link-weight", type=float, default=None,
                   help="encode with this link weight instead of the checkpoint's; codebook and segmentation are unchanged")
    e.add_argument("--link-tol", type=float, default=None,
                   help="encode with this hard limit (Engh and Huber sigma) on the departure of every boundary cell from the true link instead of the checkpoint's; 0 = off")
    e.add_argument("--pretrain-dir", default=None,
                   help="only for checkpoints without train_files: the arm's --pretrain-dir")
    e.add_argument("--max-chains", "--overfit", dest="max_chains", type=int, default=None,
                   help="only for checkpoints without train_files: the arm's --max-chains "
                        "(or its older --overfit); D is sorted(glob)[:N], 0 = all")
    e.add_argument("--rate-chains", type=int, default=0,
                   help="estimate the stream rate on this many random chains of D; 0 = all of D")
    e.add_argument("--train-dir", default=str(ROOT / "release" / "data" / "train"),
                   help="directory of the training chains named in the checkpoint")
    e.add_argument("--test", nargs="+", required=True, metavar="NAME=DIR")
    e.add_argument("--geobpe-induce", nargs="*", default=[], metavar="NAME=DIR",
                   help="per test set, GeoBPE's induce.py output directory; adds NAME_geobpe_subset")
    e.add_argument("--out", required=True)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--device", default="cuda")

    g = sub.add_parser("geobpe")
    g.add_argument("--induce", nargs="+", required=True, metavar="NAME=DIR")
    g.add_argument("--protrig-per-chain", default=None, help="the *_per_chain.json of an eval run")
    g.add_argument("--out", required=True)

    t = sub.add_parser("table")
    add_point_arguments(t)
    t.add_argument("--out", default=str(ROOT / "logs" / "pareto_table.txt"))

    n = sub.add_parser("panel")
    add_point_arguments(n)
    n.add_argument("--length-free", default=str(ROOT / "runs" / "protrig_s8k_rw0p015_s0" / "eval_per_chain.json"))
    n.add_argument("--length-valid", default=str(ROOT / "runs" / "protrig_s8k_rw0p06_link3_s1" / "eval_per_chain.json"))
    n.add_argument("--length-geobpe", default=str(ROOT / "runs" / "geobpe" / "quality" / "geobpe_K600.json"))
    n.add_argument("--max-length", type=int, default=1000)
    n.add_argument("--speed-dir", default=str(ROOT / "runs" / "speed"))
    n.add_argument("--geobpe-dir", default=str(ROOT / "runs" / "speed" / "geobpe"))
    n.add_argument("--length-protoken", default=str(ROOT / "runs" / "baselines" / "protoken.json"))
    n.add_argument("--length-aminoaseed", default=str(ROOT / "runs" / "baselines" / "aminoaseed.json"))
    n.add_argument("--baseline-dir", default=str(ROOT / "runs" / "baselines"))
    n.add_argument("--latency-arm", default="operating_point")
    n.add_argument("--out", default=str(ROOT / "runs" / "fig_panel"))

    args = ap.parse_args()
    {"eval": run_eval, "geobpe": run_geobpe, "table": run_table, "panel": run_panel}[args.cmd](args)


if __name__ == "__main__":
    main()