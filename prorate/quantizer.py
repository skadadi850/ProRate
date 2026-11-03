"""Per-length fragment codebooks (k-medoids on squared Kabsch RMSD) and the six-axis boundary quantizer."""
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from prorate.decoder import B_C_N, HEAD_DEFAULT, JOINT_DEFAULT, TAIL_DEFAULT, assemble, wrap


def proto_dim(L: int) -> int:
    return 9 * L - 6


def proto_split(v, L: int):
    """(M, 9L-6) -> head (M,L,3), interior tail (M,L-1,3), interior joint (M,L-1,3). The boundary
    row after the last residue belongs to the boundary token, not to the code."""
    M = v.shape[0]
    return (v[:, : 3 * L].reshape(M, L, 3),
            v[:, 3 * L : 6 * L - 3].reshape(M, L - 1, 3),
            v[:, 6 * L - 3 :].reshape(M, L - 1, 3))


def proto_join(head, tail, joint):
    M = head.shape[0]
    return torch.cat([head.reshape(M, -1), tail.reshape(M, -1), joint.reshape(M, -1)], dim=-1)


def proto_to_coords(v, L: int):
    return assemble(*proto_split(v, L))


def angular_mask(L: int, device) -> torch.Tensor:
    """True for the periodic entries of a prototype: psi, omega and phi."""
    head = torch.zeros(3 * L, device=device)
    tail = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(L - 1)
    joint = torch.tensor([0.0, 1.0, 1.0], device=device).repeat(L - 1)
    return torch.cat([head, tail, joint]).bool()


def ideal_prototype(L: int, device, dtype):
    head = torch.tensor(HEAD_DEFAULT, device=device, dtype=dtype).repeat(L, 1)
    tail = torch.tensor(TAIL_DEFAULT, device=device, dtype=dtype).repeat(L - 1, 1)
    joint = torch.tensor(JOINT_DEFAULT, device=device, dtype=dtype).repeat(L - 1, 1)
    return proto_join(head.unsqueeze(0), tail.unsqueeze(0), joint.unsqueeze(0))[0]


def _det3(H):
    return (H[..., 0, 0] * (H[..., 1, 1] * H[..., 2, 2] - H[..., 1, 2] * H[..., 2, 1])
            - H[..., 0, 1] * (H[..., 1, 0] * H[..., 2, 2] - H[..., 1, 2] * H[..., 2, 0])
            + H[..., 0, 2] * (H[..., 1, 0] * H[..., 2, 1] - H[..., 1, 1] * H[..., 2, 0]))


def _sym3_eigvals(A):
    """Eigenvalues (largest, middle, smallest) of symmetric 3x3 matrices in closed form."""
    a00, a11, a22 = A[..., 0, 0], A[..., 1, 1], A[..., 2, 2]
    a01, a02, a12 = A[..., 0, 1], A[..., 0, 2], A[..., 1, 2]
    q = (a00 + a11 + a22) / 3.0
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * (a01 ** 2 + a02 ** 2 + a12 ** 2)
    p = torch.sqrt((p2 / 6.0).clamp(min=1e-300))
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    det_b = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
             + b02 * (b01 * b12 - b11 * b02))
    phi = torch.acos((det_b / 2.0).clamp(-1.0, 1.0)) / 3.0
    e1 = q + 2.0 * p * torch.cos(phi)
    e3 = q + 2.0 * p * torch.cos(phi + 2.0 * math.pi / 3.0)
    return e1, 3.0 * q - e1 - e3, e3


def rmsd_matrix(X, Y, max_pairs: int = 4_000_000):
    """Kabsch RMSD between every pair, X (M,P,3) against Y (K,P,3) -> (M,K). The 3x3 step runs in
    float64 from closed-form eigenvalues of H^T H, chunked over M. Keep it out of TF32 and autocast:
    RMSD^2 is a difference of near-equal large terms."""
    P, K = X.shape[1], Y.shape[0]
    Xc, Yc = X - X.mean(1, keepdim=True), Y - Y.mean(1, keepdim=True)
    ny = (Yc ** 2).sum((1, 2)).unsqueeze(0)
    Yd = Yc.double()
    chunk = max(1, max_pairs // K)
    out = []
    for i in range(0, Xc.shape[0], chunk):
        xb = Xc[i : i + chunk]
        H = torch.einsum("mpi,kpj->mkij", xb.double(), Yd)
        e1, e2, e3 = _sym3_eigvals(H.transpose(-1, -2) @ H)
        s1, s2, s3 = (torch.sqrt(e.clamp(min=0.0)) for e in (e1, e2, e3))
        tr = (s1 + s2 + torch.sign(_det3(H)) * s3).to(X.dtype)
        nx = (xb ** 2).sum((1, 2)).unsqueeze(1)
        out.append(torch.sqrt(((nx + ny - 2.0 * tr) / P).clamp(min=1e-12)))
    return torch.cat(out, dim=0)


def k_medoids_matrix(D, k: int, iters: int = 20, seed: int = 0, init=None):
    """k-medoids on a precomputed (N,N) distance matrix. Returns medoid indices and assignments."""
    N = D.shape[0]
    k = min(k, N)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if init is not None:
        med = init[:k].to(D.device)
        if med.numel() < k:
            med = torch.cat([med, torch.randperm(N, generator=g)[: k - med.numel()].to(D.device)])
    else:
        med = torch.randperm(N, generator=g)[:k].to(D.device)
    assign = torch.zeros(N, dtype=torch.long, device=D.device)
    for _ in range(iters):
        assign = torch.argmin(D[:, med], dim=1)
        onehot = torch.zeros(N, k, device=D.device, dtype=D.dtype)
        onehot.scatter_(1, assign.unsqueeze(1), 1.0)
        within = D @ onehot                      # [i, j]: summed distance from i to the members of cluster j
        within = torch.where(onehot.bool(), within, torch.full_like(within, float("inf")))
        new = torch.argmin(within, dim=0)
        empty = torch.isinf(within.amin(dim=0))
        if bool(empty.any()):
            new[empty] = torch.randint(N, (int(empty.sum()),), generator=g).to(D.device)
        if torch.equal(new, med):
            break
        med = new
    return med, assign


class FragmentCodebook(nn.Module):
    """One codebook per segment length. Entries are internal-coordinate prototypes of observed fragments."""

    def __init__(self, l_max: int = 16, sizes: Dict[int, int] = None):
        super().__init__()
        self.sizes = sizes or {L: (64 if L <= 2 else 256 if L <= 6 else 128) for L in range(1, l_max + 1)}
        self.l_max = l_max
        self.offsets, off = {}, 0
        for L in range(1, l_max + 1):
            self.register_buffer(f"code_{L}", torch.zeros(self.sizes[L], proto_dim(L)))
            self.register_buffer(f"ang_{L}", angular_mask(L, "cpu"))
            self.offsets[L], off = off, off + self.sizes[L]
        self.total = off
        self._coord_cache = {}
        self.register_buffer("_init_flag", torch.zeros((), dtype=torch.bool))

    @property
    def initialized(self) -> bool:
        return bool(self._init_flag)

    @initialized.setter
    def initialized(self, value: bool):
        self._init_flag.fill_(bool(value))

    def codes(self, L):
        return getattr(self, f"code_{L}")

    def coords(self, L):
        """Coordinates of the length-L codes, rebuilt only when the code buffer changes."""
        code = self.codes(L)
        key = (code._version, code.data_ptr())
        hit = self._coord_cache.get(L)
        if hit is not None and hit[0] == key and hit[1].device == code.device and hit[1].dtype == code.dtype:
            return hit[1]
        c = proto_to_coords(code, L)
        self._coord_cache[L] = (key, c)
        return c

    @torch.no_grad()
    def init_from_samples(self, samples: Dict[int, torch.Tensor], max_items: int = 8000,
                          seed: int = 0, warm: bool = False):
        """k-medoids per length on squared RMSD, so entries are observed fragments and the objective
        matches the segmentation cost. warm=True starts from the samples nearest the current codes.
        Lengths without samples keep their codes once initialized; before that they are filled with
        perturbed ideal geometry."""
        done = set()
        for L, g in samples.items():
            if L not in self.offsets or g.numel() == 0:
                continue
            if g.shape[0] > max_items:
                g = g[torch.randperm(g.shape[0], device=g.device)[:max_items]]
            K = min(self.sizes[L], g.shape[0])
            X = proto_to_coords(g, L)
            D = rmsd_matrix(X, X).pow_(2)
            init = rmsd_matrix(X, self.coords(L)).argmin(0) if (warm and self.initialized) else None
            idx, _ = k_medoids_matrix(D, K, seed=seed, init=init)
            code = self.codes(L)
            code[:K] = g[idx]
            if K < code.shape[0]:
                code[K:] = g[torch.randint(g.shape[0], (code.shape[0] - K,), device=g.device)]
            done.add(L)
            del D
        if not self.initialized:
            unseen = [L for L in range(1, self.l_max + 1) if L not in done]
            for L in unseen:
                code = self.codes(L)
                code.copy_(ideal_prototype(L, code.device, code.dtype).unsqueeze(0).expand_as(code))
                code.add_(0.15 * torch.randn_like(code) * getattr(self, f"ang_{L}").to(code.dtype))
            if unseen:
                print(f"codebook: lengths {unseen} unseen at init, filled from ideal geometry")
        self.initialized = True

    @torch.no_grad()
    def quantize(self, g, L):
        """Nearest code by Kabsch RMSD. Returns (index, code row)."""
        if not self.initialized:
            raise RuntimeError("codebook is uninitialized: call init_from_samples first")
        d = rmsd_matrix(proto_to_coords(g, L), self.coords(L))
        if not torch.isfinite(d).all():
            raise RuntimeError(f"non-finite RMSD at L={L}: the codebook holds a degenerate entry")
        idx = torch.argmin(d, dim=1)
        return idx, self.codes(L)[idx]

    @torch.no_grad()
    def prune(self, chosen: Dict[int, torch.Tensor], exchange: float, max_members: int = 40000) -> Dict[int, Dict]:
        """MDL pruning. chosen[L] (M, 9L-6) holds the prototypes of the segments the segmentation picked over the
        fitted set. A code stays only if the squared angstroms its members would lose by moving to the next-best
        remaining code, over exchange (squared angstroms per bit), exceed its 32 * (9L - 6) bits of L(T). Codes with
        no member go first, the rest one at a time, worst first. Every length keeps at least one code. Returns
        per length the sizes before and after, the member count and the squared angstroms added."""
        report = {}
        for L in range(1, self.l_max + 1):
            code = self.codes(L)
            K = code.shape[0]
            g = chosen.get(L)
            if g is None or g.shape[0] == 0:
                keep = torch.zeros(1, dtype=torch.long, device=code.device)
                report[L] = {"before": K, "after": 1, "members": 0, "sse_added": 0.0}
            else:
                n_members, scale = g.shape[0], 1.0
                if n_members > max_members:
                    g = g[torch.randperm(n_members, device=g.device)[:max_members]]
                    scale = n_members / max_members
                sq = 3.0 * L * rmsd_matrix(proto_to_coords(g, L), self.coords(L)) ** 2
                base = sq.min(1).values
                cols = torch.nonzero(torch.bincount(sq.argmin(1), minlength=K) > 0).flatten()
                sq = sq[:, cols]
                alive = torch.ones(cols.numel(), dtype=torch.bool, device=sq.device)
                price = 32.0 * proto_dim(L) * exchange
                while int(alive.sum()) > 1:
                    two = sq.masked_fill(~alive[None, :], float("inf")).topk(2, dim=1, largest=False)
                    gain = torch.zeros(alive.numel(), device=sq.device)
                    gain.scatter_add_(0, two.indices[:, 0], two.values[:, 1] - two.values[:, 0])
                    gain = (gain * scale).masked_fill(~alive, float("inf"))
                    j = int(gain.argmin())
                    if float(gain[j]) >= price:
                        break
                    alive[j] = False
                keep = cols[alive]
                added = float((sq[:, alive].min(1).values - base).sum()) * scale
                report[L] = {"before": K, "after": int(keep.numel()), "members": n_members, "sse_added": added}
            setattr(self, f"code_{L}", code[keep].clone())
            self.sizes[L] = int(keep.numel())
        off = 0
        for L in range(1, self.l_max + 1):
            self.offsets[L] = off
            off += self.sizes[L]
        self.total = off
        self._coord_cache = {}
        return report
    
    def length_of(self, gid: int) -> int:
        for L in range(1, self.l_max + 1):
            if self.offsets[L] <= gid < self.offsets[L] + self.sizes[L]:
                return L
        raise ValueError(f"global id {gid} outside vocabulary of size {self.total}")

    def local_of(self, gid: int) -> int:
        return gid - self.offsets[self.length_of(gid)]


class BoundaryQuantizer(nn.Module):
    """Uniform grid over the six boundary values (theta_CACN, l_CN, psi, theta_CNCA, omega, phi).

    With the next fragment's head fixed, these six place the next fragment at any pose relative to the
    previous triad. Each axis is an affine map of its fitted range to [0, 1] with `levels[k]` cells.
    Periodic axes (psi, omega, phi) are cut on the circle at the middle of the largest empty arc of the
    fit sample (fit). The rate is the uniform code, sum(log2 levels); fit_rate and cost give a fitted
    per-axis entropy as a reporting quantity only."""

    def __init__(self, levels=(16, 16, 64, 16, 32, 64), fix_cn=False):
        super().__init__()
        lv = [int(x) for x in levels]
        assert len(lv) == 6 and min(lv) >= 2, "six axes with at least two levels each"
        self.dim = 6
        self.fix_cn = bool(fix_cn)
        self.register_buffer("_levels", torch.tensor(lv, dtype=torch.long))
        self.register_buffer("_basis", torch.cumprod(torch.tensor([1] + lv[:-1], dtype=torch.long), dim=0))
        self.register_buffer("_idx_shift", self._levels // 2)
        self.register_buffer("axis_lo", torch.full((6,), -float(np.pi)))
        self.register_buffer("axis_hi", torch.full((6,), float(np.pi)))
        shift = torch.zeros(6)
        shift[4] = float(np.pi)
        self.register_buffer("axis_shift", shift)
        periodic = torch.zeros(6, dtype=torch.bool)
        periodic[[2, 4, 5]] = True
        self.register_buffer("axis_periodic", periodic)
        bits = torch.full((6, max(lv)), 64.0)
        for k, n_k in enumerate(lv):
            bits[k, :n_k] = 0.0 if (self.fix_cn and k == 1) else float(np.log2(n_k))
        self.register_buffer("axis_bits", bits)
        total = sum(np.log2(n) for k, n in enumerate(lv) if not (self.fix_cn and k == 1))
        self.register_buffer("mean_bits", torch.tensor(float(total)))

    @property
    def levels(self):
        return self._levels.float()

    def centered(self, x):
        xs = x - self.axis_shift
        return torch.where(self.axis_periodic, wrap(xs), xs)

    @torch.no_grad()
    def fit(self, x, pad: float = 0.02):
        """Set the cut and the range of every axis from exact boundary values x (N,6), never from
        closed-loop values, which depend on the range. centered() cuts a periodic axis at
        axis_shift + pi, so the shift is the middle of the largest empty arc minus pi."""
        if x.shape[0] > 1:
            shift = self.axis_shift.clone()
            for k in range(self.dim):
                if bool(self.axis_periodic[k]):
                    a = torch.sort(wrap(x[:, k]))[0]
                    gaps = torch.cat([a[1:] - a[:-1], a[:1] + 2.0 * np.pi - a[-1:]])
                    j = int(torch.argmax(gaps))
                    shift[k] = a[j] + 0.5 * gaps[j] - np.pi
            self.axis_shift.copy_(shift)
        xs = self.centered(x)
        lo, hi = xs.amin(0), xs.amax(0)
        span = (hi - lo).clamp(min=1e-6)
        self.axis_lo.copy_(lo - pad * span)
        self.axis_hi.copy_(hi + pad * span)

    @torch.no_grad()
    def limit(self, lo, hi, axes=(0, 1, 3, 4)):
        """Set the range of theta_CACN, l_CN, theta_CNCA and omega to [lo, hi]. For omega the range is a
        deviation from trans in radians: the cut moves to the cis side (axis_shift = pi), so the centered
        value is wrap(omega - pi). Call after fit, which re-cuts periodic axes."""
        for k, a, b in zip(axes, lo, hi):
            self.axis_lo[k] = float(a)
            self.axis_hi[k] = float(b)
            if k == 4:
                self.axis_shift[4] = float(np.pi)

    @torch.no_grad()
    def fit_rate(self, codes, alpha: float = 0.5):
        """Per-axis marginal entropy table from observed integer codes (N, 6)."""
        bits = torch.full_like(self.axis_bits, 64.0)
        total = 0.0
        for k in range(self.dim):
            n_k = int(self._levels[k])
            if self.fix_cn and k == 1:
                bits[k, :n_k] = 0.0
                continue
            idx = (codes[:, k] + self._idx_shift[k]).round().long().clamp(0, n_k - 1)
            p = torch.bincount(idx, minlength=n_k).float() + alpha
            p = p / p.sum()
            bits[k, :n_k] = -torch.log2(p)
            total += float(-(p * torch.log2(p)).sum())
        self.axis_bits.copy_(bits)
        self.mean_bits.fill_(total)

    def cost(self, codes):
        """Bits an entropy coder with the fitted table charges per boundary; (...,6) -> (...)."""
        idx = (codes + self._idx_shift.float()).round().long()
        idx = torch.minimum(idx.clamp(min=0), self._levels - 1)
        return torch.stack([self.axis_bits[k][idx[..., k]] for k in range(self.dim)], -1).sum(-1)

    def warp(self, x):
        return (self.centered(x) - self.axis_lo) / (self.axis_hi - self.axis_lo)

    def unwarp(self, u):
        x = self.axis_lo + u * (self.axis_hi - self.axis_lo) + self.axis_shift
        return torch.where(self.axis_periodic, wrap(x), x)

    def _to_grid(self, x):
        return self.warp(x).clamp(0.0, 1.0) * (self.levels - 1) - self._idx_shift.float()

    def _from_grid(self, codes):
        u = (codes + self._idx_shift.float()) / (self.levels - 1)
        x = self.unwarp(u.clamp(0.0, 1.0))
        if self.fix_cn:
            x = torch.cat([x[..., :1], torch.full_like(x[..., 1:2], B_C_N), x[..., 2:]], dim=-1)
        return x

    def forward(self, x):
        """Nearest cell. Returns (dequantized values, integer codes). The values equal what
        params_from_index rebuilds from the codes, so encoder and decoder agree exactly."""
        codes = torch.round(self._to_grid(x))
        return self._from_grid(codes), codes

    def index(self, codes):
        shifted = (codes + self._idx_shift.float()).round().long()
        shifted = torch.minimum(shifted.clamp(min=0), self._levels - 1)
        return (shifted * self._basis).sum(-1)

    def codes_from_index(self, idx):
        return ((idx.unsqueeze(-1) // self._basis) % self._levels).float() - self._idx_shift.float()

    def params_from_index(self, idx):
        return self._from_grid(self.codes_from_index(idx))
