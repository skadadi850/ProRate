"""Fit a ProtRig codec: alternate exact-DP segmentation with k-medoids codebook fitting.

Each round (1) segments every fit piece exactly under the current codebook, (2) refits the
codebooks as k-medoids on the chosen segments (warm-started), (3) refits the boundary range
and rate table on exact boundary parameters and recalibrates the boundary-error term, and
(4) scores held-out pieces that were never used for fitting. There is no network and no
gradient step. Per-round metrics and wall-clock timings go to metrics.jsonl.

Objective the DP minimizes, per residue:
    fragment SSE + boundary error E(L) + rate_weight * bits
with E(L) = a * L**p measured from the closed-loop decode (see fit_boundary_cost) and
bits the uniform per-token code the evaluation reports.

The fitted set D is whole chains: --fit-chains N takes N non-held-out chains (0 = all) and uses every
piece of them. stage1.pt records exactly those files and their residue count, so the evaluation
amortizes the description length L(T) over the data the codebooks were fitted on.

Everything runs in fp32/fp64. Do not wrap any of it in autocast or enable TF32: the
geometry (forward kinematics, Kabsch, closed-loop boundaries) is precision-sensitive.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from prorate import decoder as dec
from prorate.data import ProtRigDataset, collate
from prorate.decoder import place_fragments, segment_bounds
from prorate.checkpoint import load_protrig
from prorate.model import ProtRig, ProtRigConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-dir", default="data/vqvae_pretrain/train")
    ap.add_argument("--file-list", default=None,
                    help="text file of PDB paths, one per line, used INSTEAD of globbing "
                         "--pretrain-dir. Order is kept; --max-chains takes the first N")
    ap.add_argument("--max-chains", type=int, default=0,
                    help="train on the first N chains of the sorted glob / file list; 0 = all")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--resume", default=None, help="ckpt.pt from an earlier run of this arm")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="stop when the convergence objective changes by less than this "
                         "fraction between rounds (held-out J when there is a validation set)")
    ap.add_argument("--fit-chains", type=int, default=4000,
                    help="chains (every piece of each) segmented and refitted each round; 0 = all")
    ap.add_argument("--boundary-chains", type=int, default=400,
                    help="pieces used to fit the boundary range, rate table and boundary cost")
    ap.add_argument("--val-chains", type=int, default=200,
                    help="chains held out of every fit and scored each round; capped at a fifth of "
                         "the chains; 0 = none")
    ap.add_argument("--val-by", choices=("chain", "cluster"), default="chain",
                    help="chain: random chains. cluster: whole connected components of (PDB entry, "
                         "sequence cluster id), so no validation chain shares an entry or a cluster "
                         "with a fit chain; needs --cluster-json")
    ap.add_argument("--cluster-tsv", default=None,
                    help="MMseqs2 cluster table (representative, member); used instead of the cluster ids in --cluster-json")
    ap.add_argument("--cluster-json", default=None,
                    help="StructTokenBench filtered_chain_data_cache*.json (keys <pdb>_<chain>, field cluster_id)")
    ap.add_argument("--boundary-cost", choices=("fit", "off"), default="fit",
                    help="fit: the DP charges the measured boundary error E(L); off: it does not")
    ap.add_argument("--fix-cn", action="store_true",
                    help="fix l_CN at the ideal peptide bond length at every boundary; not coded, not charged")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--boundary-batch", type=int, default=128,
                    help="chains per batch in the boundary fit, the calibration and the held-out pass; results do not depend on it")
    ap.add_argument("--link-weight", type=float, default=0.0,
                    help="boundary search objective: squared angstroms of placement error per squared sigma (Engh and Huber) of "
                         "departure of C-N length, CA-C-N, C-N-CA and omega from the true link; 0 = unconstrained; > 0 turns the search on")
    ap.add_argument("--link-tol", type=float, default=0.0,
                    help="hard limit, in Engh and Huber sigma, on the departure of a boundary cell from the true link (C-N length, CA-C-N, C-N-CA, omega) in the encoder; 0 = off; > 0 turns the boundary search on")
    ap.add_argument("--boundary-search", action="store_true",
                    help="encoder: choose each boundary cell by coordinate descent on the next fragment's placement error")
    ap.add_argument("--natural-boundary", action="store_true",
                    help="limit theta_CACN, l_CN and theta_CNCA at every boundary to the 0.5-99.5 percentile range "
                         "of the same quantities over the fit chains' links")
    ap.add_argument("--natural-q", type=float, default=0.5,
                    help="with --natural-boundary: the box is the q to 100-q percentile range")
    ap.add_argument("--forced-calibration", action="store_true",
                    help="fit the boundary cost E(L) on forced uniform segmentations of every length 1..l_max")
    ap.add_argument("--calibration-pieces", type=int, default=40,
                    help="pieces used for the forced calibration (cost grows with l_max times this)")
    ap.add_argument("--max-items", type=int, default=8000,
                    help="segments per length used by k-medoids (cost is quadratic in this)")
    ap.add_argument("--min-piece", type=int, default=16)
    ap.add_argument("--prune-from", type=int, default=0,
                    help="from this round on, drop every code whose SSE saving does not pay its L(T) bits at the arm's rate weight; 0 = never")
    ap.add_argument("--rate-weight", type=float, default=None,
                    help="REQUIRED. Squared angstroms of backbone SSE per bit")
    ap.add_argument("--l-max", type=int, default=16)
    ap.add_argument("--codebook-sizes", default=None,
                    help="per-length sizes as L:K pairs, L may be a range, e.g. "
                         "'1:64,2:64,3-6:256,7-16:128'; every length up to --l-max covered")
    ap.add_argument("--fsq-levels", default=None,
                    help="six boundary levels (theta_CACN, l_CN, psi, theta_CNCA, omega, phi), "
                         "e.g. '16,16,64,16,32,64'; joint_bits becomes sum(log2(levels))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.rate_weight is None:
        raise SystemExit("--rate-weight is required and is squared angstroms per bit")
    if args.val_by == "cluster" and not (args.cluster_json or args.cluster_tsv):
        raise SystemExit("--val-by cluster needs --cluster-tsv or --cluster-json")
    return args


def parse_codebook_sizes(spec, l_max):
    sizes = {}
    for part in spec.split(","):
        lens, k = part.split(":")
        lo, _, hi = lens.partition("-")
        for L in range(int(lo), int(hi or lo) + 1):
            if L in sizes:
                raise SystemExit(f"--codebook-sizes: length {L} given twice")
            sizes[L] = int(k)
    missing = [L for L in range(1, l_max + 1) if L not in sizes]
    extra = [L for L in sizes if not 1 <= L <= l_max]
    if missing or extra:
        raise SystemExit(f"--codebook-sizes: missing lengths {missing}, out of range {extra}")
    return tuple(sorted(sizes.items()))


def quant_overrides(args):
    kw = {}
    if args.codebook_sizes:
        kw["codebook_sizes"] = parse_codebook_sizes(args.codebook_sizes, args.l_max)
    if args.fsq_levels:
        levels = tuple(int(x) for x in args.fsq_levels.split(","))
        if len(levels) != 6:
            raise SystemExit(f"--fsq-levels: the boundary quantizer has 6 axes; got {len(levels)}")
        kw["fsq_levels"] = levels
    if args.fix_cn:
        kw["fix_cn"] = True
    if args.natural_boundary:
        kw["natural_boundary"] = True
    if args.boundary_search:
        kw["boundary_search"] = True
    if args.link_weight:
        kw["link_weight"] = args.link_weight
    if args.link_tol:
        kw["link_tol"] = args.link_tol
    return kw


def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


# ------------------------------------------------------------------------ bookkeeping
def code_version():
    return {}


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def append_metric(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def save_ckpt(path, model, cfg, args, rnd, J):
    blob = {"model": model.state_dict(), "cfg": cfg, "args": vars(args), "round": rnd,
            "J": J, "code": code_version()}
    torch.save(blob, path + ".tmp")
    os.replace(path + ".tmp", path)


def training_files(args):
    if args.file_list:
        with open(args.file_list) as f:
            files = [l.strip() for l in f if l.strip()]
        missing = [p for p in files if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"--file-list {args.file_list}: {len(missing)} of {len(files)} "
                             f"paths do not exist, e.g. {missing[:3]}")
        src = f"file list {os.path.abspath(args.file_list)}"
    else:
        files = sorted(glob.glob(os.path.join(args.pretrain_dir, "*.pdb")))
        src = f"glob {os.path.abspath(args.pretrain_dir)}/*.pdb"
    n_listed = len(files)
    if args.max_chains:
        files = files[: args.max_chains]
    print(f"training chains from {src}: {n_listed} listed"
          + (f", first {len(files)} kept by --max-chains" if args.max_chains else ", all kept"),
          flush=True)
    return files


def split_val(train, n_val, seed):
    """Hold out random whole chains: every piece of a chain goes to the same side. Returns
    (val item ids, other item ids, files of the other side)."""
    fnames = list(dict.fromkeys(r["fname"] for r in train.items))
    k = min(n_val, len(fnames) // 5)
    order = np.random.RandomState(seed + 1000).permutation(len(fnames))
    val_files = {fnames[i] for i in order[:k]}
    val_ids = [i for i, r in enumerate(train.items) if r["fname"] in val_files]
    rest_ids = [i for i, r in enumerate(train.items) if r["fname"] not in val_files]
    return val_ids, rest_ids, [f for f in fnames if f not in val_files]


def split_val_cluster(train, n_val, seed, cluster_json=None, cluster_tsv=None):
    """Hold out whole connected components of (PDB entry, sequence cluster): chains are joined when they
    share an entry or a cluster. Clusters come from an MMseqs2 table (representative, member) when given,
    else from the cluster_id field of the StructTokenBench json, where -1 joins nothing. Components larger
    than 1% of the chains are never held out, so the validation set leans toward small components.
    Returns (val item ids, other item ids, files of the other side)."""
    cluster_of = {}
    if cluster_tsv:
        with open(cluster_tsv) as fh:
            for line in fh:
                rep, member = line.split()[:2]
                cluster_of[member] = rep
    else:
        for k, v in json.load(open(cluster_json)).items():
            cid = v.get("cluster_id", -1)
            if cid is not None and cid >= 0:
                cluster_of[k] = str(cid)
    fnames = list(dict.fromkeys(r["fname"] for r in train.items))
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_unclustered = 0
    for f in fnames:
        k = stem(f)
        union("f:" + k, "e:" + k.split("_")[0])
        c = cluster_of.get(k)
        if c is None:
            n_unclustered += 1
        else:
            union("f:" + k, "c:" + c)
    comps = defaultdict(list)
    for f in fnames:
        comps[find("f:" + stem(f))].append(f)
    target = min(n_val, len(fnames) // 5)
    cap = max(1, len(fnames) // 100)
    keys = list(comps)
    val_files = set()
    for i in np.random.RandomState(seed + 1000).permutation(len(keys)):
        c = comps[keys[i]]
        if len(c) <= cap and len(val_files) + len(c) <= target:
            val_files.update(c)
        if len(val_files) >= target:
            break
    print(f"cluster split: {len(keys)} components over {len(fnames)} chains (largest "
          f"{max(map(len, comps.values()))}); {n_unclustered} chains without a cluster; "
          f"{len(val_files)} validation chains", flush=True)
    val_ids = [i for i, r in enumerate(train.items) if r["fname"] in val_files]
    rest_ids = [i for i, r in enumerate(train.items) if r["fname"] not in val_files]
    return val_ids, rest_ids, [f for f in fnames if f not in val_files]


def natural_ranges(train, ids, q=0.5):
    """q to 100-q percentile range over every link of the given pieces of theta_CACN, l_CN (tail),
    theta_CNCA (joint) and omega as a deviation from trans, wrap(omega - pi). Returns (lo, hi), each a list
    of four, in the axis order of BoundaryQuantizer.limit."""
    tail = np.concatenate([train.items[i]["tail"][:, :2] for i in ids])
    joint = np.concatenate([train.items[i]["joint"][:, :2] for i in ids])
    dev = joint[:, 1] - np.pi
    dev = np.arctan2(np.sin(dev), np.cos(dev))
    cols = [tail[:, 0], tail[:, 1], joint[:, 0], dev]
    return [float(np.percentile(c, q)) for c in cols], [float(np.percentile(c, 100.0 - q)) for c in cols]


# ------------------------------------------------------------------------ fitting passes
@torch.no_grad()
def collect_boundary_targets(model, loader, device):
    out = []
    for batch in loader:
        out.append(model.boundary_targets({k: v.to(device) for k, v in batch.items()}))
    return torch.cat(out, 0)


@torch.no_grad()
def segment_pass(model, loader, device, floor_frac: float = 0.25, n_cand: int = 8):
    """Segment every piece exactly under the current codebook. Returns the chosen segments'
    prototype vectors per length (the k-medoids input, topped up from random candidate
    spans for lengths the segmentation barely uses) and the realized model objective."""
    cb, cfg = model.codebook, model.cfg
    lengths = list(range(1, cfg.l_max + 1))
    E = model.boundary_energy(torch.device(device), torch.float32)
    samples, cand = {}, {}
    D = E_tot = bits = 0.0
    n_res = n_seg = n_batches = 0
    t0 = time.time()
    total_batches = len(loader)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        head, tail, joint, mask = batch["head"], batch["tail"], batch["joint"], batch["mask"]
        b, _, sse = model.segment(head, tail, joint, mask)
        for L, rec in model.gather_prototypes(head, tail, joint, b, mask).items():
            samples.setdefault(L, []).append(rec["g"])
            bi = torch.tensor([x[0] for x in rec["loc"]], device=device)
            st = torch.tensor([x[1] for x in rec["loc"]], device=device)
            D += float(sse[bi, st, L - 1].sum())
            E_tot += float(E[L - 1]) * int((st > 0).sum())   # a piece's first segment pays none
        for L, g in model.sample_candidates(batch, lengths, n_cand).items():
            cand.setdefault(L, []).append(g)
        n_res += int(mask.sum())
        k_seg = int(b.sum())
        n_seg += k_seg
        # uniform code: every segment pays its id, every segment but a piece's first pays a boundary
        bits += k_seg * float(np.log2(cb.total)) + (k_seg - mask.shape[0]) * cfg.joint_bits
        n_batches += 1
        if n_batches % 50 == 0:
            print(f"    segment pass {n_batches}/{total_batches} batches, "
                  f"{n_res / (time.time() - t0):.0f} residues/s", flush=True)
    samples = {L: torch.cat(v, 0) for L, v in samples.items()}
    chosen = dict(samples)   # the segments the DP picked, before any top-up
    cand = {L: torch.cat(v, 0) for L, v in cand.items()}
    topped = []
    for L in lengths:
        floor = max(1, int(floor_frac * cb.sizes[L]))
        have = samples[L].shape[0] if L in samples else 0
        if have < floor and L in cand:
            pick = torch.randperm(cand[L].shape[0], device=cand[L].device)[: floor - have]
            samples[L] = torch.cat([samples[L], cand[L][pick]], 0) if L in samples else cand[L][pick]
            topped.append(L)
    stats = {"sse_per_res": D / n_res, "e_per_res": E_tot / n_res, "bits_per_res": bits / n_res,
             "seg_len": n_res / n_seg, "rmsd_proxy": (D / (3.0 * n_res)) ** 0.5,
             "topped": topped, "n_res": n_res}
    stats["w_bits_per_res"] = cfg.rate_weight * stats["bits_per_res"]
    stats["J_per_res"] = stats["sse_per_res"] + stats["e_per_res"] + stats["w_bits_per_res"]
    return samples, stats, chosen


# ------------------------------------------------------------------------ boundary cost
def excess_samples(out, batch):
    """(L, oracle SSE, excess SSE) for every segment that has an entry boundary. oracle = the
    fragment alone at its best rigid pose; excess = its SSE in the decoded chain (one global
    Kabsch alignment of the chain) minus the oracle, so it is the pose error the boundary
    quantizer adds."""
    m, tgt = batch["mask"], batch["coords"]
    aligned, _ = dec.kabsch_align(tgt, out["pred"], m.repeat_interleave(3, dim=1))
    placed = place_fragments(out["head_q"], out["tail_q"], out["joint_q"], out["b"], m, tgt)
    se_c = ((aligned - tgt) ** 2).sum(-1).cpu().numpy()
    se_o = ((placed - tgt) ** 2).sum(-1).cpu().numpy()
    rows = []
    for bi, s, e in segment_bounds(out["b"], m):
        if s == 0:
            continue
        sl = slice(3 * s, 3 * (e + 1))
        o, c = float(se_o[bi, sl].sum()), float(se_c[bi, sl].sum())
        rows.append((e - s + 1, o, c - o))
    return rows


def fit_boundary_cost(rows, min_n: int = 5):
    """Weighted log-log fit of mean excess against L: E(L) = a * L**p, p clipped to [0, 3].
    Returns (a, p, r2, {L: [mean, n]}) or None if too few lengths have enough segments."""
    by = {}
    for L, _, x in rows:
        by.setdefault(L, []).append(x)
    table = {int(L): [float(np.mean(v)), len(v)] for L, v in sorted(by.items())}
    Ls = [L for L, v in sorted(by.items()) if len(v) >= min_n and np.mean(v) > 0]
    if len(Ls) < 3:
        return None
    x = np.log(np.array(Ls, float))
    y = np.log(np.array([np.mean(by[L]) for L in Ls]))
    w = np.array([len(by[L]) for L in Ls], float)
    A = np.stack([np.ones_like(x), x], 1)
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    p = float(np.clip(coef[1], 0.0, 3.0))
    log_a = float(np.average(y - p * x, weights=w))
    pred = log_a + p * x
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - np.average(y, weights=w)) ** 2).sum())
    return float(np.exp(log_a)), p, 1.0 - ss_res / max(ss_tot, 1e-12), table


def uniform_starts(mask, L):
    """Segment-start mask (B,n) with a segment every L residues from the start of each piece."""
    pos = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    return (pos % L == 0) & mask


@torch.no_grad()
def forced_excess(model, loader, dev):
    """Boundary excess samples for every segment length 1..l_max, each measured on a forced uniform
    segmentation, so the fit of E(L) does not depend on the segmentation E(L) itself produces."""
    rows = []
    for L in range(1, model.cfg.l_max + 1):
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(batch, b=uniform_starts(batch["mask"], L))
            rows += excess_samples(out, batch)
    return rows


@torch.no_grad()
def boundary_pass(model, loader, dev, cal_loader=None):
    """Closed-loop decode of the boundary pieces: the integer boundary codes (for the rate table) and the
    per-segment excess samples (for the boundary cost). With cal_loader the excess comes from forced
    segmentations of every length on those pieces."""
    codes, rows = [], []
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = model(batch)
        codes.append(out["joint_codes"][out["bj"]])
        if cal_loader is None:
            rows += excess_samples(out, batch)
    if cal_loader is not None:
        rows = forced_excess(model, cal_loader, dev)
    return torch.cat(codes, 0), rows


def calibrate(model, rows, mode, relax: float = 0.5):
    """Relaxed update of the boundary cost E(L) = a * L**p. The excess is measured under a segmentation that
    itself depends on E, so the raw fit alternates between two regimes. The stored (a, p) moves a fraction
    `relax` of the way to the fit: a in log space, p linearly. The first calibration (a == 0) is taken as is."""
    if mode != "fit":
        return None
    fit = fit_boundary_cost(rows)
    if fit is None:
        return None
    a_fit, p_fit, r2, table = fit
    a_old, p_old = (float(x) for x in model.boundary_cost)
    a, p = a_fit, p_fit
    if a_old > 0.0:
        a = float(np.exp((1.0 - relax) * np.log(a_old) + relax * np.log(a_fit)))
        p = (1.0 - relax) * p_old + relax * p_fit
    model.boundary_cost.copy_(torch.tensor([a, p]))
    return {"E_r2": r2, "excess_L": table, "E_a_fit": a_fit, "E_p_fit": p_fit}


# ------------------------------------------------------------------------ held-out scoring
@torch.no_grad()
def val_pass(model, loader, dev):
    """Decode held-out pieces for real (closed-loop boundaries) and report the true
    objective: decoded SSE per residue (global Kabsch per piece) + rate_weight * bits."""
    cfg, cb = model.cfg, model.codebook
    sse = bits = 0.0
    n_res = n_seg = 0
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = model(batch)
        m = batch["mask"]
        _, rmsd = dec.kabsch_align(batch["coords"], out["pred"], m.repeat_interleave(3, dim=1))
        sse += float((3.0 * m.sum(1).float() * rmsd ** 2).sum())
        k_seg = sum(rec["idx"].numel() for rec in out["groups"].values())
        bits += k_seg * float(np.log2(cb.total)) + (k_seg - m.shape[0]) * cfg.joint_bits
        n_res += int(m.sum())
        n_seg += k_seg
    return {"sse_val_per_res": sse / n_res, "bits_val": bits / n_res, "seg_len_val": n_res / n_seg,
            "rmsd_val": (sse / (3.0 * n_res)) ** 0.5,
            "J_val": sse / n_res + cfg.rate_weight * bits / n_res}


# ------------------------------------------------------------------------ main
def main():
    t_start = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    dev = args.device
    cfg_path = os.path.join(args.save_dir, "config.json" if not args.resume else "config.resume.json")
    run_record = {"args": vars(args), "code": code_version(), "started": time.ctime()}
    write_json(cfg_path, run_record)
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    if not args.resume and os.path.exists(metrics_path):
        os.replace(metrics_path, metrics_path + ".prev")

    if args.resume:
        model = load_protrig(args.resume, dev)    # the checkpoint's own config governs a resumed run
    else:
        model = ProtRig(ProtRigConfig(l_max=args.l_max, rate_weight=args.rate_weight, **quant_overrides(args))).to(dev)
    cfg = model.cfg
    print(f"codebook sizes {model.codebook.sizes} total {model.codebook.total}; boundary levels "
          f"{cfg.fsq_levels} uniform-code {cfg.joint_bits:.2f} bits (fix_cn {cfg.fix_cn}); rate_weight "
          f"{cfg.rate_weight} A^2/bit; boundary cost {args.boundary_cost}", flush=True)

    files = training_files(args)
    t0 = time.time()
    train = ProtRigDataset(files, min_len=args.min_piece)
    t_load = time.time() - t0
    print(f"{len(train)} pieces from {len(files)} chains in {t_load:.0f}s; dropped {train.dropped}",
          flush=True)
    if len(train) == 0:
        raise SystemExit("no usable structures")

    if args.val_by == "cluster":
        val_ids, rest_ids, fit_files = split_val_cluster(train, args.val_chains, args.seed,
                                                         args.cluster_json, args.cluster_tsv)
    else:
        val_ids, rest_ids, fit_files = split_val(train, args.val_chains, args.seed)
    # the fitted set D: whole chains and every piece of each
    order = np.random.RandomState(args.seed).permutation(len(fit_files))
    chosen = [fit_files[i] for i in order[: args.fit_chains]] if args.fit_chains else list(fit_files)
    chosen_set = set(chosen)
    fit_ids = [i for i in rest_ids if train.items[i]["fname"] in chosen_set]
    perm = [fit_ids[i] for i in np.random.RandomState(args.seed).permutation(len(fit_ids))]
    train_residues = int(sum(train.items[i]["n"] for i in fit_ids))
    nat = natural_ranges(train, fit_ids, args.natural_q) if cfg.natural_boundary else None
    if nat:
        print("natural boundary range: theta_CACN [%.3f, %.3f]  l_CN [%.3f, %.3f]  theta_CNCA [%.3f, %.3f]  "
              "omega-from-trans [%.3f, %.3f]" % tuple(v for pair in zip(*nat) for v in pair), flush=True)
    print(f"split: {len(fit_ids)} fit pieces from {len(chosen)} chains ({train_residues} residues), "
          f"{len(val_ids)} held-out pieces", flush=True)
    run_record.update(n_listed=len(files), n_pieces=len(train), n_val_pieces=len(val_ids),
                      dropped=train.dropped, train_files=chosen, train_residues=train_residues)
    write_json(cfg_path, run_record)

    def loader_of(ids, batch_size=None):
        items = sorted((train.items[i] for i in ids), key=lambda r: r["n"])
        return DataLoader(items, batch_size=batch_size or args.batch_size, shuffle=False, collate_fn=collate)

    fit_loader = loader_of(perm)
    bnd_loader = loader_of(perm[-args.boundary_chains:], args.boundary_batch)
    cal_loader = loader_of(perm[-args.calibration_pieces:], args.boundary_batch) if args.forced_calibration else None
    val_loader = loader_of(val_ids, args.boundary_batch) if val_ids else None

    def fit_boundary():
        """Boundary range, rate table and boundary cost under the current codebook."""
        model.joints.fit(collect_boundary_targets(model, bnd_loader, dev))
        if nat:
            model.joints.limit(*nat)
        codes, rows = boundary_pass(model, bnd_loader, dev, cal_loader)
        model.joints.fit_rate(codes)
        return calibrate(model, rows, args.boundary_cost)

    rnd, prev = 0, None
    if args.resume:
        blob = torch.load(args.resume, map_location=dev, weights_only=False)
        rnd, prev = int(blob["round"]), blob.get("J")
        del blob
        print(f"RESUMED from {args.resume} at round {rnd}", flush=True)
    else:
        t0 = time.time()
        lengths = list(range(1, args.l_max + 1))
        per = -(-args.max_items // len(fit_loader)) + 1
        cand = {}
        for batch in fit_loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            for L, g in model.sample_candidates(batch, lengths, per).items():
                cand.setdefault(L, []).append(g)
        sync(dev)
        t_cand = time.time() - t0
        t0 = time.time()
        model.codebook.init_from_samples({L: torch.cat(v, 0) for L, v in cand.items()},
                                         max_items=args.max_items, seed=args.seed)
        sync(dev)
        t_cb = time.time() - t0
        t0 = time.time()
        fit_boundary()
        sync(dev)
        t_bnd = time.time() - t0
        a, p = (float(x) for x in model.boundary_cost)
        print(f"init: candidates {t_cand:.1f}s, codebook {t_cb:.1f}s, boundary fit {t_bnd:.1f}s; "
              f"E(L) = {a:.3f} * L^{p:.2f}", flush=True)
        append_metric(metrics_path, {"round": 0, "t_load": t_load, "t_cand": t_cand,
                                     "t_codebook": t_cb, "t_boundary_fit": t_bnd,
                                     "n_pieces": len(train), "E_a": a, "E_p": p,
                                     "elapsed_s": time.time() - t_start})

    ckpt_path = os.path.join(args.save_dir, "ckpt.pt")
    while rnd < args.rounds:
        t_round = time.time()
        sync(dev)
        t0 = time.time()
        samples, st, picked = segment_pass(model, fit_loader, dev)
        sync(dev)
        t_segment = time.time() - t0

        t0 = time.time()
        model.codebook.init_from_samples(samples, max_items=args.max_items,
                                         seed=args.seed + rnd + 1, warm=True)
        if args.prune_from and rnd + 1 >= args.prune_from:
            model.codebook.prune(picked, cfg.rate_weight)
            cfg.codebook_sizes = tuple(sorted(model.codebook.sizes.items()))
        sync(dev)
        t_codebook = time.time() - t0

        t0 = time.time()
        cal = fit_boundary()
        sync(dev)
        t_bfit = time.time() - t0

        t0 = time.time()
        val = val_pass(model, val_loader, dev) if val_loader is not None else {}
        sync(dev)
        t_val = time.time() - t0

        rnd += 1
        t_total = time.time() - t_round
        a, p = (float(x) for x in model.boundary_cost)
        rec = {"round": rnd, "J_per_res": st["J_per_res"], "sse_per_res": st["sse_per_res"],
               "e_per_res": st["e_per_res"], "w_bits_per_res": st["w_bits_per_res"],
               "bits_per_res": st["bits_per_res"], "seg_len": st["seg_len"],
               "rmsd_proxy": st["rmsd_proxy"], "boundary_table_bits": float(model.joints.mean_bits),
               "topped": st["topped"], "E_a": a, "E_p": p, **(cal or {}), **val,
               "t_segment": t_segment, "t_codebook": t_codebook, "t_boundary_fit": t_bfit,
               "t_val": t_val, "t_round": t_total, "n_res": st["n_res"],
               "res_per_s": st["n_res"] / max(t_segment, 1e-9),
               "elapsed_s": time.time() - t_start,
               "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 2 ** 30
                              if str(dev).startswith("cuda") else 0.0)}
        rec["codebook_total"] = model.codebook.total
        print(f"         codebook {model.codebook.total} codes, L(T) {model.description_bits() / 8e6:.2f} MB", flush=True)
        append_metric(metrics_path, rec)
        print(f"round {rnd:3d} J/res {st['J_per_res']:.4f} = frag {st['sse_per_res']:.4f} + "
              f"boundary {st['e_per_res']:.4f} + w*bits {st['w_bits_per_res']:.4f} | seg_len "
              f"{st['seg_len']:.2f} bits/res {st['bits_per_res']:.2f} | E(L) = {a:.3f}*L^{p:.2f}"
              + (f" (r2 {cal['E_r2']:.2f})" if cal else "") + f" | topped {st['topped']}", flush=True)
        if val:
            print(f"         held-out: J {val['J_val']:.4f} (decoded SSE/res {val['sse_val_per_res']:.4f}) "
                  f"rmsd {val['rmsd_val']:.3f} seg_len {val['seg_len_val']:.2f} "
                  f"bits/res {val['bits_val']:.2f}", flush=True)
        print(f"         time {t_total:.0f}s = segment {t_segment:.0f} ({rec['res_per_s']:.0f} res/s) "
              f"+ codebook {t_codebook:.0f} + boundary fit {t_bfit:.0f} + held-out {t_val:.0f}", flush=True)
        J_conv = val["J_val"] if val else st["J_per_res"]
        save_ckpt(ckpt_path, model, cfg, args, rnd, J_conv)
        done = prev is not None and abs(prev - J_conv) <= args.tol * prev
        prev = J_conv
        if done:
            print(f"converged: objective changed by <= {args.tol:.0e} (relative)", flush=True)
            break

    torch.save({"model": model.state_dict(), "cfg": cfg, "args": vars(args),
                "train_files": chosen, "train_residues": train_residues, "train_pieces": len(fit_ids),
                "code": code_version()},
               os.path.join(args.save_dir, "stage1.pt"))
    print(f"done in {(time.time() - t_start) / 60:.1f} min; wrote stage1.pt", flush=True)


if __name__ == "__main__":
    main()
