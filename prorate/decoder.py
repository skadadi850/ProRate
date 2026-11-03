"""Forward kinematics, rigid fitting and boundary placement. No trained parameters."""
import math

import numpy as np
import torch

FLOAT_BITS = 32.0

# Engh and Huber ideal backbone geometry.
B_N_CA, B_CA_C, B_C_N = 1.458, 1.525, 1.329
HEAD_DEFAULT = np.array([np.radians(111.0), B_N_CA, B_CA_C], dtype=np.float64)   # theta_NCAC, l_NCA, l_CAC
TAIL_DEFAULT = np.array([np.radians(116.2), B_C_N, np.pi], dtype=np.float64)     # theta_CACN, l_CN, psi
JOINT_DEFAULT = np.array([np.radians(121.7), np.pi, -np.pi / 3.0], dtype=np.float64)  # theta_CNCA, omega, phi

# Engh and Huber sigma of the boundary quantities by cell axis (theta_CACN, l_CN, psi, theta_CNCA, omega, phi). Omega: 5 degrees, so that
# 2 sigma is the 10 degree validity cut. psi and phi are free torsions and carry no penalty (0).
LINK_SIGMA = np.array([np.radians(2.0), 0.014, 0.0, np.radians(1.8), np.radians(5.0), 0.0], dtype=np.float64)

def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def place(a, b, c, angle, length, torsion):
    """Atom d with bond angle (b, c, d), bond length |cd| and torsion (a, b, c, d). Broadcasts over
    leading dimensions. Angle and length are clamped away from the degenerate values."""
    angle = angle.clamp(1e-2, math.pi - 1e-2)
    length = length.clamp(min=0.5)
    bc = unit(c - b)
    n = unit(torch.cross(b - a, bc, dim=-1))
    m = torch.cross(n, bc, dim=-1)
    x = -length * torch.cos(angle)
    y = length * torch.cos(torsion) * torch.sin(angle)
    z = length * torch.sin(torsion) * torch.sin(angle)
    return c + x.unsqueeze(-1) * bc + y.unsqueeze(-1) * m + z.unsqueeze(-1) * n


def seed_triad(head0):
    """First residue (N, CA, C) from (theta_NCAC, l_NCA, l_CAC): CA at the origin, C on +x, N in the xy-plane."""
    th, l_nca, l_cac = head0.unbind(-1)
    th = th.clamp(1e-2, math.pi - 1e-2)
    zero = torch.zeros_like(th)
    ca = torch.stack([zero, zero, zero], dim=-1)
    c = torch.stack([l_cac, zero, zero], dim=-1)
    n = torch.stack([l_nca * torch.cos(th), l_nca * torch.sin(th), zero], dim=-1)
    return n, ca, c


def assemble(head, tail, joint):
    """head (B,n,3), tail (B,n-1,3), joint (B,n-1,3) -> coords (B,3n,3) in N, CA, C order.

    Link i places residue i+1: N from (theta_CACN, l_CN, psi) = tail[i], CA from
    (theta_CNCA, l_NCA, omega) with l_NCA from head[i+1] and the rest from joint[i], C from
    (theta_NCAC, l_CAC, phi) with the first two from head[i+1] and phi from joint[i]."""
    out = list(seed_triad(head[:, 0]))
    for i in range(head.shape[1] - 1):
        th_cacn, l_cn, psi = tail[:, i].unbind(-1)
        th_cnca, omega, phi = joint[:, i].unbind(-1)
        th_ncac, l_nca, l_cac = head[:, i + 1].unbind(-1)
        n = place(out[-3], out[-2], out[-1], th_cacn, l_cn, psi)
        ca = place(out[-2], out[-1], n, th_cnca, l_nca, omega)
        c = place(out[-1], n, ca, th_ncac, l_cac, phi)
        out += [n, ca, c]
    return torch.stack(out, dim=1)


def fragment_local(head, tail, joint, bi, s, e):
    """Coordinates (3L, 3) of residues s..e of chain bi from their internals, in a local frame."""
    return assemble(head[bi, s : e + 1].unsqueeze(0), tail[bi, s:e].unsqueeze(0),
                    joint[bi, s:e].unsqueeze(0))[0]


def triad_frame(n, ca, c):
    """Right-handed frames (..., 3, 3) with columns x, y, z at CA, from triads (N, CA, C)."""
    x = unit(c - ca)
    z = unit(torch.cross(x, unit(n - ca), dim=-1))
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def rigid_fit(src, dst):
    """R, t with dst ~ src @ R.T + t in the least-squares sense. float64 SVD on the CPU."""
    dev, dt = src.device, src.dtype
    s, d = src.double().cpu(), dst.double().cpu()
    sc, dc = s.mean(0), d.mean(0)
    u, _, vh = torch.linalg.svd((s - sc).T @ (d - dc))
    sign = torch.sign(torch.det(vh.T @ u.T))
    one = torch.ones((), dtype=sign.dtype)
    r = vh.T @ torch.diag(torch.stack([one, one, sign])) @ u.T
    return r.to(device=dev, dtype=dt), (dc - r @ sc).to(device=dev, dtype=dt)


def rigid_fit_batch(src, dst):
    """rigid_fit for a batch. src, dst (B,m,3) -> R (B,3,3), t (B,3) with dst ~ src @ R^T + t. float64 SVD on the
    CPU; the results stay float64 on the CPU."""
    s, d = src.double().cpu(), dst.double().cpu()
    sc, dc = s.mean(1, keepdim=True), d.mean(1, keepdim=True)
    u, _, vh = torch.linalg.svd(torch.einsum("bmi,bmj->bij", s - sc, d - dc))
    sign = torch.sign(torch.det(vh.transpose(-1, -2) @ u.transpose(-1, -2)))
    one = torch.ones_like(sign)
    r = vh.transpose(-1, -2) @ torch.diag_embed(torch.stack([one, one, sign], dim=-1)) @ u.transpose(-1, -2)
    t = dc[:, 0] - torch.einsum("bij,bj->bi", r, sc[:, 0])
    return r, t


def kabsch_align(P, Q, mask=None):
    """Align Q onto P. P, Q (B,m,3), mask (B,m). Returns the aligned Q and the per-sample RMSD."""
    if mask is None:
        mask = torch.ones(P.shape[:2], dtype=torch.bool, device=P.device)
    dev, dt = P.device, P.dtype
    P, Q, w = P.double().cpu(), Q.double().cpu(), mask.cpu().unsqueeze(-1).double()
    cnt = w.sum(1).clamp(min=1.0)
    Pc = (P * w).sum(1, keepdim=True) / cnt.unsqueeze(1)
    Qc = (Q * w).sum(1, keepdim=True) / cnt.unsqueeze(1)
    Pd, Qd = (P - Pc) * w, (Q - Qc) * w
    u, _, vh = torch.linalg.svd(torch.einsum("bpi,bpj->bij", Pd, Qd))
    sign = torch.sign(torch.det(u @ vh))
    ones = torch.ones_like(sign)
    r = u @ torch.diag_embed(torch.stack([ones, ones, sign], dim=-1)) @ vh
    rot = torch.einsum("bij,bpj->bpi", r, Qd)
    rmsd = torch.sqrt(((Pd - rot * w) ** 2).sum((1, 2)) / cnt.squeeze(-1)).clamp(min=1e-6)
    return (rot + Pc).to(device=dev, dtype=dt), rmsd.to(device=dev, dtype=dt)


def angle(a, b, c):
    u, v = unit(a - b), unit(c - b)
    return torch.arccos((u * v).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))


def dihedral(a, b, c, d):
    b0, b1, b2 = a - b, unit(c - b), d - c
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    return torch.atan2((torch.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def params_from_triads(a, b, c, n, ca, cc):
    """Previous triad (a, b, c) and next triad (n, ca, cc) -> the boundary tail (theta_CACN, l_CN, psi)
    and joint (theta_CNCA, omega, phi). The next head is not returned: it belongs to the code."""
    tail = torch.stack([angle(b, c, n), (n - c).norm(dim=-1), dihedral(a, b, c, n)], dim=-1)
    joint = torch.stack([angle(c, n, ca), dihedral(b, c, n, ca), dihedral(c, n, ca, cc)], dim=-1)
    return tail, joint


@torch.no_grad()
def segment_bounds(b, mask):
    """[(chain, start, end), ...] for every segment, in chain order. b marks segment starts."""
    out = []
    lens = mask.long().sum(1).tolist()
    for bi in range(b.shape[0]):
        ni = lens[bi]
        if ni == 0:
            continue
        starts = torch.nonzero(b[bi, :ni]).flatten().tolist()
        if not starts or starts[0] != 0:
            starts = [0] + starts
        for m, s in enumerate(starts):
            out.append((bi, s, starts[m + 1] - 1 if m + 1 < len(starts) else ni - 1))
    return out


@torch.no_grad()
def place_fragments(head_q, tail_q, joint_q, b, mask, target):
    """Every quantized fragment fitted rigidly onto its own target span: the distortion floor of
    any boundary parameterization."""
    pred = target.clone()
    for bi, s, e in segment_bounds(b, mask):
        loc = fragment_local(head_q, tail_q, joint_q, bi, s, e)
        r, t = rigid_fit(loc, target[bi, 3 * s : 3 * (e + 1)])
        pred[bi, 3 * s : 3 * (e + 1)] = loc @ r.T + t
    return pred


@torch.no_grad()
def params_from_placement(head_q, tail_q, joint_q, b, mask, target):
    """Tail and joint with the boundary rows replaced by the exact values that reproduce
    place_fragments through assemble."""
    placed = place_fragments(head_q, tail_q, joint_q, b, mask, target)
    t_out, j_out = tail_q.clone(), joint_q.clone()
    segs = segment_bounds(b, mask)
    for k in range(1, len(segs)):
        bi, s, _ = segs[k]
        pb, _, pe = segs[k - 1]
        if pb != bi:
            continue
        a, nx = placed[bi, 3 * pe : 3 * pe + 3], placed[bi, 3 * s : 3 * s + 3]
        tl, jt = params_from_triads(a[0:1], a[1:2], a[2:3], nx[0:1], nx[1:2], nx[2:3])
        t_out[bi, s - 1], j_out[bi, s - 1] = tl[0], jt[0]
    return t_out, j_out
