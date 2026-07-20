"""Representation conversions for quantum objects.

A quantum object admits several equivalent representations. This module
converts between the package's canonical storage format—the fused-leg
Liouville tensor—and other representations, starting with the Choi
matrix. It is the intended home for further conversions (Kraus,
Stinespring, chi-matrix, superoperator) as they are added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .qtensor import QTensor


def choi_matrix(x: QTensor) -> np.ndarray:
    """Choi matrix of a quantum tensor, on the product of its legs' spaces.

    Rows collect the ket index of every leg (in leg order), columns the bra
    index. For a state this is simply the density matrix; for a channel or
    process tensor the (unnormalized) Choi operator.
    """
    pairs = [d for dim in x.dims for d in (dim, dim)]
    t = x.data.reshape(pairs)  # (bra_0, ket_0, bra_1, ket_1, ...)
    kets = list(range(1, 2 * x.nlegs, 2))
    bras = list(range(0, 2 * x.nlegs, 2))
    D = int(np.prod(x.dims))
    return t.transpose(kets + bras).reshape(D, D)


def choi_to_tensor(matrix: np.ndarray, dims: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`choi_matrix`: fused-leg data from a Choi matrix."""
    matrix = np.asarray(matrix, dtype=complex)
    n = len(dims)
    t = matrix.reshape(tuple(dims) + tuple(dims))  # (ket_0, ..., bra_0, ...)
    perm = [ax for leg in range(n) for ax in (n + leg, leg)]
    return t.transpose(perm).reshape([d * d for d in dims])
