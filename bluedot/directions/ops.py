"""Pure tensor ops on directions. No model/dataset deps."""

import torch
from torch import Tensor

_EPS = 1e-8


def normalize(v: Tensor, eps: float = _EPS) -> Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def cosine_similarity(a: Tensor, b: Tensor, eps: float = _EPS) -> Tensor:
    return (normalize(a, eps) * normalize(b, eps)).sum(dim=-1)


def project(v: Tensor, onto: Tensor, eps: float = _EPS) -> Tensor:
    """Scalar-project `v` onto `onto` and return the projection vector."""
    u = normalize(onto, eps)
    coeff = (v * u).sum(dim=-1, keepdim=True)
    return coeff * u


def orthogonalize(
    v: Tensor,
    against: Tensor,
    lam: float = 1.0,
    normalize_out: bool = True,
    eps: float = _EPS,
) -> Tensor:
    """
    Remove `lam *` the component of `v` along `against`.
    Use lam=1.0 for full orthogonalization.
    """
    out = v - lam * project(v, against, eps)
    if normalize_out:
        out = normalize(out, eps)
    return out
