"""The three loss terms: L_CE (S_L), L_NCL (S_L, marginal), L_TCI (S_U, conditional).

L_TCI enforces class-conditional independence via the rank-1 3rd-moment identity:
under CI, T_{a,b,c}/Sigma_{a,b} = kappa*q_c is constant over (a,b) for fixed c, so
its variance over pairs is 0. See docs/FUSE_TCI_spec.md.
"""

from __future__ import annotations

from ..heads.models import _decorr_loss


def ce_loss(logits, y, class_weight=None):
    """logits: (B, H, A, C). Sum of cross-entropy over every head and answer."""
    import torch

    B, H, A, C = logits.shape
    crit = torch.nn.CrossEntropyLoss(weight=class_weight)
    return sum(crit(logits[:, h, a, :], y) for h in range(H) for a in range(A))


def ncl_loss(v):
    """v: (B, H, A). Marginal decorrelation of each head's A answers, summed over heads."""
    return sum(_decorr_loss(v[:, h, :]) for h in range(v.shape[1]))


def tci_loss(v, eps: float):
    """v: (B, m) verifier scores in [0,1] on S_U. Returns the TCI violation.

    For each c, Var over pairs (a,b), a<b, both != c, of  T_{a,b,c} / Sigma_{a,b}.
    """
    import torch

    B, m = v.shape
    u = 2.0 * v - 1.0                                   # map [0,1] -> [-1,1]
    uc = u - u.mean(dim=0, keepdim=True)                # center over S_U

    Sigma = (uc.t() @ uc) / B                           # (m, m) pairwise covariance
    T = torch.einsum("xa,xb,xc->abc", uc, uc, uc) / B   # (m, m, m) 3rd moments

    sign = torch.sign(Sigma)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    denom = sign * Sigma.abs().clamp_min(eps)           # eps-floor |Sigma_ab|
    R = T / denom.unsqueeze(-1)                         # R[a,b,c] = T_{a,b,c}/Sigma_{a,b}

    iu = torch.triu_indices(m, m, offset=1, device=v.device)   # pairs a<b, (2, P)
    a, b = iu[0], iu[1]
    Rp = R[a, b, :]                                     # (P, m): each pair's ratio per c
    cs = torch.arange(m, device=v.device)
    valid = (a.unsqueeze(1) != cs) & (b.unsqueeze(1) != cs)     # (P, m): exclude c in {a,b}
    valid = valid.to(Rp.dtype)

    cnt = valid.sum(dim=0).clamp_min(1.0)               # pairs per c
    mean = (Rp * valid).sum(dim=0) / cnt
    var = ((Rp - mean.unsqueeze(0)) ** 2 * valid).sum(dim=0) / cnt
    return var.sum()
