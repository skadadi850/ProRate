"""ProtRig: a parameter-free structure codec. Segmentation is the exact minimizer of
sum over segments of [3 L RMSD(span, nearest code)^2 + boundary cost + rate_weight * bits]."""
from dataclasses import dataclass, replace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from prorate.decoder import (
    FLOAT_BITS,
    LINK_SIGMA,
    assemble,
    fragment_local,
    params_from_placement,
    params_from_triads,
    place,
    rigid_fit,
    rigid_fit_batch,
    seed_triad,
    segment_bounds,
    triad_frame,
    wrap,
)
from prorate.quantizer import (
    BoundaryQuantizer,
    FragmentCodebook,
    proto_dim,
    proto_join,
    proto_split,
    proto_to_coords,
    rmsd_matrix,
)


@dataclass
class ProtRigConfig:
    l_max: int = 16
    fsq_levels: tuple = (16, 16, 64, 16, 32, 64)   # theta_CACN, l_CN, psi, theta_CNCA, omega, phi
    joint_bits: float = 29.0                       # overwritten in __post_init__
    rate_weight: float = 0.2                       # squared angstroms of backbone SSE per bit
    codebook_sizes: tuple = None                   # ((L, K), ...) or None for the default sizes
    fix_cn: bool = False                           # l_CN fixed at the ideal peptide bond: not coded, not charged
    natural_boundary: bool = False                 # theta_CACN, l_CN, theta_CNCA at boundaries limited to the range of interior links
    boundary_search: bool = False                  # encoder picks each boundary cell by minimizing the next fragment's placement error
    link_weight: float = 0.0                       # squared angstroms of placement error per squared sigma of departure of a boundary cell from the true link (C-N length, CA-C-N, C-N-CA, omega); > 0 turns the boundary search on
    link_tol: float = 0.0                          # hard limit, in Engh and Huber sigma, on the departure of a boundary cell from the true link (C-N length, CA-C-N, C-N-CA, omega); 0 = off; > 0 turns the boundary search on

    def __post_init__(self):
        bits = np.log2(np.asarray(self.fsq_levels, dtype=float))
        self.joint_bits = float(bits.sum() - (bits[1] if self.fix_cn else 0.0))


def dp_segment(scores, mask):
    """Maximum-score segmentation by semi-Markov DP. scores (B,n,K): scores[b,s,k] is span [s, s+k];
    -inf marks an invalid span. Returns the segment-start mask (B,n) and the path score (B,)."""
    B, n, K = scores.shape
    lens = mask.long().sum(1).tolist()
    sc = scores.detach().double().cpu().numpy()
    b = np.zeros((B, n), dtype=bool)
    total = np.zeros(B)
    for bi in range(B):
        ni, scb = lens[bi], sc[bi]
        best = np.full(ni + 1, -np.inf)
        back = np.zeros(ni + 1, dtype=np.int64)
        best[0] = 0.0
        for e in range(1, ni + 1):
            k = np.arange(min(K, e))
            s = e - 1 - k
            v = best[s] + scb[s, k]
            j = int(np.argmax(v))
            best[e], back[e] = v[j], s[j]
        e = ni
        while e > 0:
            e = int(back[e])
            b[bi, e] = True
        total[bi] = best[ni]
    return (torch.from_numpy(b).to(scores.device),
            torch.from_numpy(total).to(device=scores.device, dtype=scores.dtype))


class ProtRig(nn.Module):
    def __init__(self, cfg: ProtRigConfig = None):
        super().__init__()
        cfg = replace(cfg) if cfg is not None else ProtRigConfig()   # re-runs __post_init__ on unpickled configs
        self.cfg = cfg
        self.codebook = FragmentCodebook(
            l_max=cfg.l_max, sizes=dict(cfg.codebook_sizes) if cfg.codebook_sizes else None)
        self.joints = BoundaryQuantizer(levels=cfg.fsq_levels, fix_cn=cfg.fix_cn)
        # [a, p]: expected extra SSE a * L**p that quantizing a segment's entry boundary adds. Fitted by the trainer.
        self.register_buffer("boundary_cost", torch.tensor([0.0, 1.0]))

    def description_bits(self):
        """L(T): the codebook entries and the boundary grid (axis_lo, axis_hi, axis_shift), as float32."""
        n = sum(self.codebook.sizes[L] * proto_dim(L) for L in range(1, self.cfg.l_max + 1))
        return float(n + 3 * self.joints.dim) * FLOAT_BITS

    # ---- segmentation -------------------------------------------------------------
    @torch.no_grad()
    def span_costs(self, head, tail, joint, mask):
        """Backbone SSE (3 L RMSD^2 to the nearest code) of every span, (B,n,l_max); entry [b,s,k]
        is span [s, s+k]. Invalid spans are +inf."""
        B, n = mask.shape
        sse = head.new_full((B, n, self.cfg.l_max), float("inf"))
        for L in range(1, min(self.cfg.l_max, n) + 1):
            S = n - L + 1
            ok = mask.unfold(1, L, 1).all(-1)
            if not bool(ok.any()):
                continue
            h = head.unfold(1, L, 1).permute(0, 1, 3, 2).reshape(B, S, 3 * L)
            if L > 1:
                t = tail.unfold(1, L - 1, 1).permute(0, 1, 3, 2).reshape(B, S, 3 * (L - 1))
                j = joint.unfold(1, L - 1, 1).permute(0, 1, 3, 2).reshape(B, S, 3 * (L - 1))
            else:
                t = j = head.new_zeros(B, S, 0)
            g = torch.cat([h, t, j], -1)[ok]
            d = rmsd_matrix(proto_to_coords(g, L), self.codebook.coords(L)).min(1).values
            cell = head.new_full((B, S), float("inf"))
            cell[ok] = 3.0 * L * d * d
            sse[:, :S, L - 1] = cell
        return sse

    def rate_cost(self, device, dtype):
        """Bits per segment: log2 of the vocabulary plus the boundary cell. Constant in L."""
        bits = float(np.log2(self.codebook.total)) + self.cfg.joint_bits
        return torch.full((self.cfg.l_max,), bits, device=device, dtype=dtype)

    def boundary_energy(self, device, dtype):
        L = torch.arange(1, self.cfg.l_max + 1, device=device, dtype=dtype)
        a = self.boundary_cost[0].to(device=device, dtype=dtype)
        p = self.boundary_cost[1].to(device=device, dtype=dtype)
        return a * L ** p

    @torch.no_grad()
    def segment(self, head, tail, joint, mask):
        """Returns the segment-start mask b, the path cost (B,) and the per-span fragment SSE."""
        sse = self.span_costs(head, tail, joint, mask)
        E = self.boundary_energy(sse.device, sse.dtype)
        cost = sse + E
        cost[:, 0, :] -= E          # the first segment of a piece has no entry boundary
        cost = cost + self.cfg.rate_weight * self.rate_cost(sse.device, sse.dtype)
        b, total = dp_segment(-cost, mask)
        return b, -total, sse

    # ---- gather and scatter -------------------------------------------------------
    def gather_prototypes(self, head, tail, joint, b, mask):
        """Raw prototype vectors grouped by segment length, with the (chain, start, end) of each member."""
        groups: Dict[int, Dict] = {}
        for bi, s, e in segment_bounds(b, mask):
            g = proto_join(head[bi, s : e + 1].unsqueeze(0), tail[bi, s:e].unsqueeze(0),
                           joint[bi, s:e].unsqueeze(0))[0]
            rec = groups.setdefault(e - s + 1, {"g": [], "loc": []})
            rec["g"].append(g)
            rec["loc"].append((bi, s, e))
        for rec in groups.values():
            rec["g"] = torch.stack(rec["g"])
        return groups

    def scatter_prototypes(self, groups, head, tail, joint):
        """Write the chosen codes back as segment internals. Boundary rows are left to quantize_boundaries."""
        head_q, tail_q, joint_q = head.clone(), tail.clone(), joint.clone()
        for L, rec in groups.items():
            h, t, j = proto_split(rec["c"], L)
            for k, (bi, s, e) in enumerate(rec["loc"]):
                head_q[bi, s : e + 1] = h[k]
                if L > 1:
                    tail_q[bi, s:e] = t[k]
                    joint_q[bi, s:e] = j[k]
        return head_q, tail_q, joint_q

    @torch.no_grad()
    def sample_candidates(self, batch, lengths, n_per_length: int):
        """Prototype vectors of random spans of the given lengths, for seeding and topping up codebooks."""
        head, tail, joint, mask = batch["head"], batch["tail"], batch["joint"], batch["mask"]
        lens = mask.long().sum(1)
        out = {}
        for L in lengths:
            eligible = torch.nonzero(lens >= L).flatten()
            if eligible.numel() == 0:
                continue
            gs = []
            for _ in range(n_per_length):
                bi = int(eligible[torch.randint(0, eligible.numel(), (1,))])
                s = int(torch.randint(0, int(lens[bi]) - L + 1, (1,)))
                e = s + L - 1
                gs.append(proto_join(head[bi, s : e + 1].unsqueeze(0), tail[bi, s:e].unsqueeze(0),
                                     joint[bi, s:e].unsqueeze(0))[0])
            out[L] = torch.stack(gs, 0)
        return out

    # ---- boundaries ---------------------------------------------------------------
    @torch.no_grad()
    def _search_cells_batch(self, a, bb, c, y, hd, tgt, w, codes, codes_true, truth, table, levels, shift, sweeps: int = 3):
        J = self.joints
        B = a.shape[0]
        lam = float(self.cfg.link_weight)
        tol = float(self.cfg.link_tol)
        pen = [0, 1, 3, 4]
        sig = torch.as_tensor(LINK_SIGMA[pen], dtype=torch.float64)
        ref = truth[:, pen]

        def values(ix):
            return torch.stack([table[k][ix[..., k]] for k in range(6)], -1)

        def cost(p):
            m = p.shape[1]
            a_, b_, c_ = (v[:, None, :].expand(B, m, 3) for v in (a, bb, c))
            n = place(a_, b_, c_, p[..., 0], p[..., 1], p[..., 2])
            ca = place(b_, c_, n, p[..., 3], hd[:, 1:2].expand(B, m), p[..., 4])
            cc = place(c_, n, ca, hd[:, 0:1].expand(B, m), hd[:, 2:3].expand(B, m), p[..., 5])
            x = ca[:, :, None, :] + torch.einsum("baj,bmij->bmai", y, triad_frame(n, ca, cc))
            sse = (((x - tgt[:, None]) ** 2).sum(-1) * w[:, None]).sum(-1)
            if lam == 0.0 and tol == 0.0:
                return sse
            d = p[..., pen] - ref[:, None, :]
            d[..., 3] = wrap(d[..., 3])
            z = d / sig
            out = sse + lam * (z ** 2).sum(-1)
            if tol > 0.0:
                zt = z[..., [k for k in range(4) if not (J.fix_cn and k == 1)]]
                out = out + 1e6 * (zt.abs() - tol).clamp(min=0.0).sum(-1)
            return out

        idx = (codes + shift).round().long()
        idx_true = (codes_true + shift).round().long()
        c_free = cost(values(idx)[:, None])[:, 0]
        c_true = cost(values(idx_true)[:, None])[:, 0]
        take = c_true < c_free
        idx = torch.where(take[:, None], idx_true, idx)
        best = torch.where(take, c_true, c_free)
        rows = torch.arange(B)
        axes = [k for k in (2, 4, 5, 0, 3, 1) if not (J.fix_cn and k == 1)]
        for _ in range(sweeps):
            moved = False
            for k in axes:
                cand = idx[:, None, :].repeat(1, levels[k], 1)
                cand[:, :, k] = torch.arange(levels[k])[None, :]
                cs = cost(values(cand))
                j = torch.argmin(cs, dim=1)
                v = cs[rows, j]
                better = v < best - 1e-9
                idx = torch.where(better[:, None], cand[rows, j], idx)
                best = torch.where(better, v, best)
                moved = moved or bool(better.any())
            if not moved:
                break
        return idx

    @torch.no_grad()
    def quantize_boundaries(self, head_q, tail_q, joint_q, b, mask, target, sweeps: int = 3):
        """Closed-loop boundaries, encoder side only, batched over chains. Boundary k of every chain is placed at once.
        Each boundary is fitted against the previous fragment as actually placed, quantization error included, and
        aims at the true pose of the next fragment, so a rounding error is corrected by the next boundary instead of
        inherited downstream. float64 on the CPU from float32 grid values, so it reproduces the decoder's placement.
        Returns (tail, joint, codes)."""
        dev, dt = tail_q.device, tail_q.dtype
        head, tail, joint, tgt = (x.double().cpu() for x in (head_q, tail_q, joint_q, target))
        tail, joint = tail.clone(), joint.clone()
        J = self.joints
        jdev = J.axis_lo.device
        levels = [int(v) for v in J._levels]
        shift = J._idx_shift.float().cpu()
        table = []
        for k in range(6):
            cells = torch.zeros(levels[k], 6)
            cells[:, k] = torch.arange(levels[k], dtype=torch.float32) - shift[k]
            table.append(J._from_grid(cells.to(jdev)).double().cpu()[:, k])
        B, nm1 = tail.shape[:2]
        codes_out = tail.new_zeros(B, nm1, 6)
        frags, n_frag = {}, [0] * B
        for bi, s, e in segment_bounds(b, mask):
            frags.setdefault(e - s + 1, []).append((bi, n_frag[bi], s, e))
            n_frag[bi] += 1
        n_steps, A = max(n_frag), 3 * max(frags)
        P_y, P_tgt = tgt.new_zeros(B, n_steps, A, 3), tgt.new_zeros(B, n_steps, A, 3)
        P_w = tgt.new_zeros(B, n_steps, A)
        P_nx, P_first, P_last, P_rn = (tgt.new_zeros(B, n_steps, 3, 3) for _ in range(4))
        P_tn, P_hd = tgt.new_zeros(B, n_steps, 3), tgt.new_zeros(B, n_steps, 3)
        P_s = torch.zeros(B, n_steps, dtype=torch.long)
        for L, lst in frags.items():
            bi_t, j_t = torch.tensor([x[0] for x in lst]), torch.tensor([x[1] for x in lst])
            h = torch.stack([head[x[0], x[2] : x[3] + 1] for x in lst])
            t = torch.stack([tail[x[0], x[2] : x[3]] for x in lst])
            jt = torch.stack([joint[x[0], x[2] : x[3]] for x in lst])
            tg = torch.stack([tgt[x[0], 3 * x[2] : 3 * (x[3] + 1)] for x in lst])
            loc = assemble(h, t, jt)
            rn, tn = rigid_fit_batch(loc, tg)
            frame = triad_frame(loc[:, 0], loc[:, 1], loc[:, 2])
            P_y[bi_t, j_t, : 3 * L] = (loc - loc[:, 1:2]) @ frame
            P_tgt[bi_t, j_t, : 3 * L] = tg
            P_w[bi_t, j_t, : 3 * L] = 1.0
            P_nx[bi_t, j_t] = loc[:, :3] @ rn.transpose(-1, -2) + tn[:, None, :]
            P_first[bi_t, j_t] = loc[:, :3]
            P_last[bi_t, j_t] = loc[:, -3:]
            P_rn[bi_t, j_t], P_tn[bi_t, j_t] = rn, tn
            P_hd[bi_t, j_t] = h[:, 0]
            P_s[bi_t, j_t] = torch.tensor([x[2] for x in lst])
        n_frag_t = torch.tensor(n_frag)
        cur = P_last[:, 0] @ P_rn[:, 0].transpose(-1, -2) + P_tn[:, 0, None, :]   # last triad of fragment 0 on its target
        for k in range(1, n_steps):
            rows = torch.nonzero(n_frag_t > k).flatten()
            a, bb, c = cur[rows, 0], cur[rows, 1], cur[rows, 2]
            nx = P_nx[rows, k]
            tl, jt = params_from_triads(a, bb, c, nx[:, 0], nx[:, 1], nx[:, 2])
            _, cd = J(torch.cat([tl, jt], -1).float().to(jdev))
            cd = cd.cpu()
            hd = P_hd[rows, k]
            link = P_s[rows, k] - 1
            if self.cfg.boundary_search or self.cfg.link_weight > 0.0 or self.cfg.link_tol > 0.0:
                truth = torch.cat([tail[rows, link], joint[rows, link]], -1)
                _, cr = J(truth.float().to(jdev))
                idx = self._search_cells_batch(a, bb, c, P_y[rows, k], hd, P_tgt[rows, k], P_w[rows, k], cd, cr.cpu(),
                                               truth, table, levels, shift, sweeps)
            else:
                idx = (cd + shift).round().long()
            pq = torch.stack([table[m][idx[:, m]] for m in range(6)], -1)
            n_ = place(a, bb, c, pq[:, 0], pq[:, 1], pq[:, 2])
            ca = place(bb, c, n_, pq[:, 3], hd[:, 1], pq[:, 4])
            cc = place(c, n_, ca, hd[:, 0], hd[:, 2], pq[:, 5])
            rp, tp = rigid_fit_batch(P_first[rows, k], torch.stack([n_, ca, cc], dim=-2))
            cur[rows] = P_last[rows, k] @ rp.transpose(-1, -2) + tp[:, None, :]
            tail[rows, link], joint[rows, link] = pq[:, :3], pq[:, 3:]
            codes_out[rows, link] = idx.double() - shift.double()
        return tail.to(device=dev, dtype=dt), joint.to(device=dev, dtype=dt), codes_out.to(device=dev, dtype=dt)
    
    @torch.no_grad()
    def boundary_targets(self, batch):
        """Exact boundary values under the current codebook and segmentation: what BoundaryQuantizer.fit
        must be given. They do not depend on the boundary quantization, so the fit is not circular."""
        head, tail, joint, mask = batch["head"], batch["tail"], batch["joint"], batch["mask"]
        b, _, _ = self.segment(head, tail, joint, mask)
        groups = self.gather_prototypes(head, tail, joint, b, mask)
        for L, rec in groups.items():
            rec["c"] = self.codebook.quantize(rec["g"], L)[1]
        head_q, tail_q, joint_q = self.scatter_prototypes(groups, head, tail, joint)
        t6, j6 = params_from_placement(head_q, tail_q, joint_q, b, mask, batch["coords"])
        return torch.cat([t6, j6], -1)[b[:, 1:] & mask[:, 1:]]

    # ---- encode and decode --------------------------------------------------------
    @torch.no_grad()
    def forward(self, batch, b=None):
        """Encode and decode one batch. b (B,n) bool, if given, fixes the segment starts instead of the DP."""
        if "coords" not in batch:
            raise RuntimeError("closed-loop boundaries need batch['coords']")
        head, tail, joint, mask = batch["head"], batch["tail"], batch["joint"], batch["mask"]
        if b is None:
            b, _, _ = self.segment(head, tail, joint, mask)
        groups = self.gather_prototypes(head, tail, joint, b, mask)
        for L, rec in groups.items():
            rec["idx"], rec["c"] = self.codebook.quantize(rec["g"], L)
        head_q, tail_q, joint_q = self.scatter_prototypes(groups, head, tail, joint)
        bj = b[:, 1:] & mask[:, 1:]   # link i is a boundary iff residue i+1 opens a segment
        tail_q, joint_q, joint_codes = self.quantize_boundaries(head_q, tail_q, joint_q, b, mask, batch["coords"])
        return {"pred": assemble(head_q, tail_q, joint_q), "b": b, "mask": mask, "groups": groups,
                "joint_codes": joint_codes, "bj": bj, "head_q": head_q, "tail_q": tail_q, "joint_q": joint_q}

    
    @torch.no_grad()
    def token_stream(self, out):
        """Per chain: the global segment ids in order, and the boundary cell indices in order."""
        B = out["mask"].shape[0]
        seg_ids = [[] for _ in range(B)]
        for L, rec in out["groups"].items():
            gids = (rec["idx"] + self.codebook.offsets[L]).tolist()
            for (bi, s, e), g in zip(rec["loc"], gids):
                seg_ids[bi].append((s, int(g)))
        seg_ids = [[g for _, g in sorted(v)] for v in seg_ids]
        j_idx = self.joints.index(out["joint_codes"])
        joint_ids = [j_idx[bi][out["bj"][bi]].tolist() for bi in range(B)]
        return seg_ids, joint_ids

    @torch.no_grad()
    def rebuild_internals_from_indices(self, seg_ids, joint_ids, device=None, dtype=torch.float32):
        """Integers -> (head, tail, joint). A segment of L residues contributes L head rows and L-1
        interior rows; every segment but the last then contributes one boundary row from its cell.
        Chains of different lengths cannot be stacked and raise."""
        dev = device if device is not None else self.codebook.codes(1).device
        heads, tails, joints = [], [], []
        for L_ids, J_ids in zip(seg_ids, joint_ids):
            if len(J_ids) != len(L_ids) - 1:
                raise ValueError(f"{len(L_ids)} segments need {len(L_ids) - 1} boundary cells, got {len(J_ids)}")
            gam = self.joints.params_from_index(torch.tensor(J_ids, dtype=torch.long, device=dev))
            h, t, j = [], [], []
            for m, gid in enumerate(L_ids):
                L, local = self.codebook.length_of(gid), self.codebook.local_of(gid)
                hh, tt, jj = proto_split(self.codebook.codes(L)[local : local + 1], L)
                h.append(hh[0])
                t.append(tt[0])
                j.append(jj[0])
                if m < len(L_ids) - 1:
                    t.append(gam[m : m + 1, :3])
                    j.append(gam[m : m + 1, 3:])
            heads.append(torch.cat(h))
            tails.append(torch.cat(t))
            joints.append(torch.cat(j))
        sizes = {x.shape[0] for x in heads}
        if len(sizes) != 1:
            raise ValueError(f"chains of different lengths {sorted(sizes)}; decode them one at a time")
        return (torch.stack(heads).to(dtype), torch.stack(tails).to(dtype), torch.stack(joints).to(dtype))
