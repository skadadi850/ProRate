"""Backbone parsing, chain pieces, datasets, batching, break coding and rate accounting.

A chain is a sequence of pieces. Residues missing N, CA or C are unresolved and get no geometry. A piece
is a maximal run of resolved residues joined by peptide bonds. The next piece is placed by one rigid
transform (six float32 values) fitted closed loop. Pieces shorter than MIN_PIECE are sent raw."""
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from prorate.decoder import FLOAT_BITS, HEAD_DEFAULT, JOINT_DEFAULT, TAIL_DEFAULT, assemble

CN_LO, CN_HI = 1.0, 2.0          # C-N distance outside this range is a break
THETA_LO, THETA_HI = 0.5, 2.9    # bond angle outside this range is implausible
MIN_PIECE = 3


# ---- numpy geometry ---------------------------------------------------------------
def _angle(a, b, c):
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))


def _kabsch_transform(P, Q):
    """R, t with P @ R.T + t the least-squares fit of P (m,3) onto Q (m,3)."""
    pc, qc = P.mean(0), Q.mean(0)
    u, _, vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return R, qc - pc @ R.T


def _rotmat(rv):
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.eye(3)
    k = rv / ang
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def _rotvec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-8:  # angle near pi: take the axis from the symmetric part
        _, v = np.linalg.eigh((R + np.eye(3)) / 2.0)
        return v[:, -1] * ang
    return axis / n * ang


# ---- backbone readers -------------------------------------------------------------
def backbone_from_protein_chain(pc) -> np.ndarray:
    """(n,3,3) N, CA, C of an ESM ProteinChain, NaN where missing. Row i is pc.residue_index[i]."""
    from esm.utils import residue_constants as RC
    idx = [RC.atom_order[a] for a in ("N", "CA", "C")]
    bb = np.array(pc.atom37_positions[:, idx, :], dtype=np.float64)
    bb[~np.asarray(pc.atom37_mask[:, idx], dtype=bool)] = np.nan
    return bb


def backbone_from_pdb(fname: str) -> np.ndarray:
    from esm.utils.structure.protein_chain import ProteinChain
    return backbone_from_protein_chain(ProteinChain.from_pdb(fname))


# ---- pieces -----------------------------------------------------------------------
def link_internals(prev, nxt):
    """The six link values placing residue `nxt` (3,3) after `prev` (3,3), and the head of `nxt`."""
    (N0, CA0, C0), (N1, CA1, C1) = prev, nxt
    tail = np.array([_angle(CA0, C0, N1), np.linalg.norm(N1 - C0), _dihedral(N0, CA0, C0, N1)])
    joint = np.array([_angle(C0, N1, CA1), _dihedral(CA0, C0, N1, CA1), _dihedral(C0, N1, CA1, C1)])
    head = np.array([_angle(N1, CA1, C1), np.linalg.norm(CA1 - N1), np.linalg.norm(C1 - CA1)])
    return tail, joint, head


def piece_record(bb: np.ndarray) -> Dict:
    """Internal coordinates of one piece. head[0] is the first residue's measured geometry."""
    n = bb.shape[0]
    head = np.tile(HEAD_DEFAULT, (n, 1))
    tail = np.tile(TAIL_DEFAULT, (max(n - 1, 0), 1))
    joint = np.tile(JOINT_DEFAULT, (max(n - 1, 0), 1))
    if n:
        N0, CA0, C0 = bb[0]
        head[0] = [_angle(N0, CA0, C0), np.linalg.norm(N0 - CA0), np.linalg.norm(C0 - CA0)]
    for i in range(n - 1):
        tail[i], joint[i], head[i + 1] = link_internals(bb[i], bb[i + 1])
    return {"head": head.astype(np.float32), "tail": tail.astype(np.float32),
            "joint": joint.astype(np.float32), "coords": bb.reshape(-1, 3).astype(np.float32), "n": n}


def split_pieces(bb: np.ndarray) -> Dict:
    """Pieces as (start, end) residue rows with end inclusive, the resolved mask, and cut counts."""
    n = bb.shape[0]
    resolved = np.isfinite(bb).all(axis=(1, 2))
    tau = np.full(n, np.nan)
    tau[resolved] = _angle(bb[resolved, 0], bb[resolved, 1], bb[resolved, 2])
    bad_tau = resolved & ~((tau >= THETA_LO) & (tau <= THETA_HI))
    bonded = np.zeros(max(n - 1, 0), dtype=bool)
    cuts = {"unresolved": int((~resolved).sum()), "cn_distance": 0, "angle": 0}
    for i in range(n - 1):
        if not (resolved[i] and resolved[i + 1]):
            continue
        t, j, _ = link_internals(bb[i], bb[i + 1])
        if not (CN_LO <= t[1] <= CN_HI):
            cuts["cn_distance"] += 1
        elif not (THETA_LO <= t[0] <= THETA_HI and THETA_LO <= j[0] <= THETA_HI) or bad_tau[i] or bad_tau[i + 1]:
            cuts["angle"] += 1
        else:
            bonded[i] = True
    pieces, s = [], None
    for i in range(n):
        if not resolved[i]:
            if s is not None:
                pieces.append((s, i - 1))
                s = None
            continue
        if s is None:
            s = i
        if i == n - 1 or not bonded[i]:
            pieces.append((s, i))
            s = None
    return {"pieces": pieces, "resolved": resolved, "cuts": cuts}


def chain_layout(bb: np.ndarray) -> Dict:
    sp = split_pieces(bb)
    items = []
    for s, e in sp["pieces"]:
        rec = piece_record(bb[s : e + 1])
        rec.update(start=s, end=e, raw=(e - s + 1) < MIN_PIECE)
        items.append(rec)
    return {"items": items, "resolved": sp["resolved"], "cuts": sp["cuts"], "n_rows": bb.shape[0]}


# ---- encoding, decoding, rate -----------------------------------------------------
@torch.no_grad()
def encode_chains(model, layouts: List[Dict], device, batch_size=8):
    """Tokenize every model piece. Fills item['seg'] and item['joint']. Pieces are batched by length."""
    work = sorted((it for lay in layouts for it in lay["items"] if not it["raw"]), key=lambda it: it["n"])
    for i in range(0, len(work), batch_size):
        chunk = work[i : i + batch_size]
        out = model({k: v.to(device) for k, v in collate(chunk).items()})
        seg_ids, joint_ids = model.token_stream(out)
        for it, s, j in zip(chunk, seg_ids, joint_ids):
            assert len(j) == len(s) - 1
            it["seg"], it["joint"] = s, j
    return layouts


def _piece_frame(model, it, device):
    if it["raw"]:
        h, t, j = (torch.from_numpy(it[x]).double()[None] for x in ("head", "tail", "joint"))
    else:
        h, t, j = model.rebuild_internals_from_indices([it["seg"]], [it["joint"]], device)
        h, t, j = h.double().cpu(), t.double().cpu(), j.double().cpu()
    return assemble(h, t, j)[0].numpy()


@torch.no_grad()
def encode_breaks(model, layout: Dict, device):
    """Encoder side. One float32 (rotation vector, translation) sextet per piece after the first, fitted
    closed loop: the truth so far is carried into the decoded frame by a Kabsch fit, each decoded piece is
    fitted onto its carried truth, and later pieces are fitted against what the decoder will build.
    Call after encode_chains and before decode_chain."""
    placed, breaks = [], []
    for k, it in enumerate(layout["items"]):
        own = _piece_frame(model, it, device)
        if k == 0:
            placed.append(own)
            continue
        truth = np.concatenate([p["coords"] for p in layout["items"][:k]]).astype(np.float64)
        R1, s1 = _kabsch_transform(truth, np.concatenate(placed))
        R2, s2 = _kabsch_transform(own, it["coords"].astype(np.float64) @ R1.T + s1)
        sent = np.concatenate([_rotvec(R2), s2]).astype(np.float32)
        breaks.append(sent)
        sent = sent.astype(np.float64)
        placed.append(own @ _rotmat(sent[:3]).T + sent[3:])
    layout["breaks"] = breaks


@torch.no_grad()
def decode_chain(model, layout: Dict, device) -> np.ndarray:
    """Decoder side: (n_resolved,3,3) from the integers, raw floats and break sextets only."""
    placed = []
    for k, it in enumerate(layout["items"]):
        own = _piece_frame(model, it, device)
        if k > 0:
            br = layout["breaks"][k - 1].astype(np.float64)
            own = own @ _rotmat(br[:3]).T + br[3:]
        placed.append(own)
    return np.concatenate(placed).reshape(-1, 3, 3)


def reference_coords(layout: Dict) -> np.ndarray:
    return np.concatenate([it["coords"].reshape(-1, 3, 3) for it in layout["items"]])


def chain_bits(model, layout: Dict) -> Dict:
    """Stream bits of one chain under the uniform per-token code. L(T) is not included: it is
    model.description_bits, added once over the evaluation set.

    Model pieces: log2(vocabulary) per segment and joint_bits per boundary. Breaks: six float32 values
    plus ceil(log2(n_rows + 1)) bits for the position, plus one bit per piece for raw or model. Raw pieces:
    9m - 6 float32 values for m residues. token_bits_coded charges boundary cells their fitted per-axis
    entropy instead of the uniform code, as a secondary figure."""
    items = layout["items"]
    n_breaks = max(len(items) - 1, 0)
    dev = model.codebook.codes(1).device
    log_v = math.log2(model.codebook.total)
    seg_bits = j_uniform = j_coded = 0.0
    for it in items:
        if it["raw"]:
            continue
        seg_bits += len(it["seg"]) * log_v
        nb = len(it["joint"])
        j_uniform += nb * float(model.cfg.joint_bits)
        if nb:
            codes = model.joints.codes_from_index(torch.tensor(it["joint"], device=dev))
            j_coded += float(model.joints.cost(codes).sum())
    tok, tok_coded = seg_bits + j_uniform, seg_bits + j_coded
    brk = n_breaks * (6 * FLOAT_BITS + math.ceil(math.log2(layout["n_rows"] + 1))) + (len(items) if n_breaks else 0.0)
    raw = sum((9 * it["n"] - 6) * FLOAT_BITS for it in items if it["raw"])
    n_tokens = sum(len(it["seg"]) if not it["raw"] else it["n"] for it in items) + n_breaks
    return {"token_bits": tok, "token_bits_coded": tok_coded, "break_bits": brk, "raw_bits": raw,
            "total_bits": tok + brk + raw, "total_bits_coded": tok_coded + brk + raw, "breaks": n_breaks,
            "raw_residues": sum(it["n"] for it in items if it["raw"]),
            "residues": sum(it["n"] for it in items), "tokens": n_tokens}


# ---- dataset and batching ---------------------------------------------------------
def _load_pieces(job):
    """Worker: one PDB -> (path, pieces, error text). pieces is None if the file is unreadable."""
    f = job[0]
    try:
        layout = chain_layout(backbone_from_pdb(f))
    except Exception as e:
        return f, None, f"{type(e).__name__}: {e}"[:200]
    keep = ("head", "tail", "joint", "coords", "n", "raw")
    return f, [{k: it[k] for k in keep} for it in layout["items"]], None


class ProtRigDataset(Dataset):
    """Pieces of at least min_len and at most max_len residues from a list of PDB files. Others are
    counted in `dropped`, and the distinct read errors are kept in `errors`. Parsing runs in spawned
    processes (default: allocated CPUs minus one, at most 12)."""

    def __init__(self, pdbs: Sequence[str], min_len: int = 16, max_len: int = 512, workers: Optional[int] = None):
        self.min_len, self.max_len = min_len, max_len
        self.items: List[Dict] = []
        self.dropped = {"unreadable": 0, "short": 0, "long": 0}
        self.errors: Dict[str, str] = {}
        n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        w = workers or max(1, min(12, n_cpu - 1))
        jobs = [(f,) for f in pdbs]
        if w > 1 and len(jobs) > 64:
            with ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn")) as ex:
                results = list(ex.map(_load_pieces, jobs, chunksize=16))
        else:
            results = [_load_pieces(j) for j in jobs]
        for f, pieces, err in results:
            if pieces is None:
                self.dropped["unreadable"] += 1
                self.errors.setdefault(err, f)
                continue
            for it in pieces:
                if it["raw"] or it["n"] < min_len:
                    self.dropped["short"] += 1
                elif it["n"] > max_len:
                    self.dropped["long"] += 1
                else:
                    it["fname"] = f
                    self.items.append(it)
        for err, f in list(self.errors.items())[:3]:
            print(f"unreadable, e.g. {f}: {err}", flush=True)
        if jobs and self.dropped["unreadable"] == len(jobs):
            raise RuntimeError(f"all {len(jobs)} files are unreadable; first error: {next(iter(self.errors))}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        return self.items[i]


def collate(batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    n_max = max(r["n"] for r in batch)
    head = torch.tensor(np.tile(HEAD_DEFAULT, (B, n_max, 1)), dtype=torch.float32)
    tail = torch.tensor(np.tile(TAIL_DEFAULT, (B, n_max - 1, 1)), dtype=torch.float32)
    joint = torch.tensor(np.tile(JOINT_DEFAULT, (B, n_max - 1, 1)), dtype=torch.float32)
    coords = torch.zeros(B, 3 * n_max, 3)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for i, r in enumerate(batch):
        n = r["n"]
        head[i, :n] = torch.from_numpy(r["head"])
        tail[i, : n - 1] = torch.from_numpy(r["tail"])
        joint[i, : n - 1] = torch.from_numpy(r["joint"])
        coords[i, : 3 * n] = torch.from_numpy(r["coords"])
        mask[i, :n] = True
    return {"head": head, "tail": tail, "joint": joint, "coords": coords, "mask": mask}
