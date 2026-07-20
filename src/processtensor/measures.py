"""Functions of quantum objects, agnostic to the specific object type.

Every quantity here is defined for any leg-based quantum tensor—a state,
a channel, or a process tensor—through its Choi representation. The same
function therefore computes, depending only on the object passed in, e.g.
the entanglement negativity of a bipartite density matrix or the temporal
negativity of a process tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .utils import TOL, relative_entropy, trace_norm, von_neumann_entropy

if TYPE_CHECKING:
    from .qtensor import QTensor


# -- Information measures ---------------------------------------------------


def purity(x: QTensor) -> float:
    """Purity ``Tr[Y^2]`` of the trace-normalized Choi matrix."""
    rho = x.choi / x.trace
    return float(np.trace(rho @ rho).real)


def entropy(x: QTensor) -> float:
    """Von Neumann entropy of the trace-normalized Choi matrix."""
    return von_neumann_entropy(x.choi / x.trace)


def negativity(x: QTensor, legs: list[int]) -> float:
    """Entanglement negativity across the bipartition ``legs`` | rest.

    Computed as ``(||Y^T_A||_1 - 1) / 2`` on the trace-normalized Choi
    matrix, where the partial transpose acts on the given legs. For a
    multipartite state this is the usual (spatial) entanglement negativity;
    for a process tensor it detects temporal entanglement.
    """
    pt_choi = x.partial_transpose(legs).choi
    return (trace_norm(pt_choi) / x.trace - 1) / 2


def mutual_information(x: QTensor, partition: list[list[int]]) -> float:
    """Generalized quantum mutual information over a leg partition.

    The quantum relative entropy ``S(Y || Y_0 (x) ... (x) Y_m)`` between
    the trace-normalized Choi matrix and the product of its marginals on
    each block of ``partition``. Blocks must be contiguous, in leg order,
    and cover all legs.
    """
    flat = [leg for block in partition for leg in block]
    if flat != list(range(x.nlegs)):
        raise ValueError("Partition must be contiguous, ordered, and cover all legs.")
    rho = x.choi / x.trace
    sigma = np.array([[1.0]], dtype=complex)
    for block in partition:
        marginal = x.partial_trace(list(block))
        sigma = np.kron(sigma, marginal.choi / marginal.trace)
    return relative_entropy(rho, sigma)


# -- Operator Schmidt decomposition across a cut ----------------------------


def schmidt_values(x: QTensor, cut: int) -> np.ndarray:
    """Operator Schmidt coefficients across the cut before leg ``cut``.

    Singular values of the leg tensor matricized into (legs before the cut)
    x (legs after the cut). These are the coefficients of the operator
    Schmidt decomposition ``Y = sum_i s_i A_i (x) B_i`` of the Choi matrix,
    with orthonormal operator bases on either side of the cut. Note this is
    the *operator-space* decomposition: a pure state whose state vector has
    Schmidt rank r yields r^2 operator Schmidt values.
    """
    if not 0 < cut < x.nlegs:
        raise ValueError(f"Cut must be between 1 and {x.nlegs - 1}.")
    rows = int(np.prod(x.data.shape[:cut]))
    return np.linalg.svd(x.data.reshape(rows, -1), compute_uv=False)


def schmidt_rank(x: QTensor, cut: int, rtol: float = TOL) -> int:
    """Operator Schmidt rank across the cut before leg ``cut``.

    The number of Schmidt values above ``rtol`` times the largest. For a
    process tensor this equals the minimal bond dimension of a matrix
    product operator representation at that cut (at most ``dE^2`` for a
    process built from an environment of dimension ``dE``).
    """
    s = schmidt_values(x, cut)
    return int(np.sum(s > rtol * s[0]))


def bond_entropy(x: QTensor, cut: int) -> float:
    """Entropy of the normalized squared Schmidt values across a cut.

    Also known as the operator entanglement entropy: with
    ``p_i = s_i^2 / sum_j s_j^2``, returns ``-sum_i p_i ln p_i``. It bounds
    how compressible the object is across the cut - zero if and only if
    the tensor factorizes there.
    """
    s = schmidt_values(x, cut)
    p = s**2 / np.sum(s**2)
    p = p[p > TOL]
    return float(-np.sum(p * np.log(p)))
