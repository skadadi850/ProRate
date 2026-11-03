"""Distortion and link-geometry reports. Nothing here is fitted."""
import numpy as np
import torch


def kabsch_rmsd(pred, ref, mask):
    p, r = pred[mask].double(), ref[mask].double()
    if p.shape[0] < 3:
        return float("nan")
    pc, rc = p - p.mean(0, keepdim=True), r - r.mean(0, keepdim=True)
    u, _, vt = torch.linalg.svd(pc.T @ rc)
    d = torch.sign(torch.det(u @ vt))
    one = torch.ones((), device=p.device, dtype=p.dtype)
    rot = u @ torch.diag(torch.stack([one, one, d])) @ vt
    return float(torch.sqrt((((pc @ rot) - rc) ** 2).sum(-1).mean()))


def lddt(pred, ref, mask, cutoff=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """Mean per-residue lDDT over residues with a neighbour inside the cutoff in the reference.
    Pass CA coordinates for CA-lDDT."""
    p, r = pred[mask].double(), ref[mask].double()
    n = p.shape[0]
    if n < 2:
        return float("nan")
    dr, dp = torch.cdist(r, r), torch.cdist(p, p)
    valid = (dr < cutoff) & ~torch.eye(n, dtype=torch.bool, device=p.device)
    cnt = valid.sum(1)
    has = cnt > 0
    if not bool(has.any()):
        return float("nan")
    err = (dp - dr).abs()
    kept = sum(((err < t) & valid).sum(1) for t in thresholds).double()
    return float((kept / (len(thresholds) * cnt.clamp(min=1)))[has].mean())


# Engh and Huber peptide-link ideals as (mean, sigma): C-N length (A), CA-C-N angle, C-N-CA angle (degrees).
PEPTIDE_LINK = ((1.329, 0.014), (116.2, 2.0), (121.7, 1.8))


def peptide_validity(pred_links, ref_links, is_boundary=None, z_cuts=(2.0, 3.0)):
    """pred_links, ref_links (m,3): C-N length, CA-C-N and C-N-CA angle of the same links in the decoded and
    the reference structure. Per set of links: the fraction with any quantity beyond each z cut against the
    Engh and Huber ideals, and the mean |z| per quantity. Natural structures are not ideal, so the reference
    row is the baseline the decoded rows are read against."""
    mu = np.array([m for m, _ in PEPTIDE_LINK])
    sd = np.array([s for _, s in PEPTIDE_LINK])

    def stats(v):
        z = np.abs((v - mu) / sd)
        return {"n_links": int(v.shape[0]),
                "frac_any_beyond": {str(c): float((z > c).any(1).mean()) for c in z_cuts},
                "mean_abs_z": [float(x) for x in z.mean(0)]}

    out = {"reference": stats(ref_links), "decoded": stats(pred_links)}
    if is_boundary is not None:
        for label, sel in (("decoded_boundary", is_boundary), ("decoded_interior", ~is_boundary)):
            if sel.any():
                out[label] = stats(pred_links[sel])
    return out


def omega_report(pred_dev, ref_dev, is_boundary=None, cuts=(10.0, 30.0)):
    """pred_dev, ref_dev (m,): deviation of the peptide dihedral omega from trans (180 - |omega|, degrees)
    of the same links in the decoded and the reference structure. Mean, p99 and the fraction beyond each
    cut for the reference, all decoded links and, with is_boundary, decoded boundary and interior links."""
    def stats(v):
        return {"n_links": int(v.shape[0]), "mean_deg": float(v.mean()),
                "p99_deg": float(np.percentile(v, 99)),
                "frac_beyond": {str(c): float((v > c).mean()) for c in cuts}}

    out = {"reference": stats(ref_dev), "decoded": stats(pred_dev)}
    if is_boundary is not None:
        for label, sel in (("decoded_boundary", is_boundary), ("decoded_interior", ~is_boundary)):
            if sel.any():
                out[label] = stats(pred_dev[sel])
    return out
