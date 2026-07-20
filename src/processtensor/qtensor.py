"""Base class for leg-based quantum tensors."""

from __future__ import annotations

import numpy as np

from . import representations
from .utils import TOL, is_hermitian, is_psd


class QTensor:
    """A quantum object stored as a tensor of fused Liouville legs.

    ``data`` carries one fused index per physical wire, of dimension
    ``d_j^2`` where ``d_j`` is the Hilbert space dimension of wire ``j``
    (recorded in ``dims``). Each fused index combines a bra-ket pair with
    the bra index major (reorder-fuse-transpose form).

    Subclasses fix the physical meaning and ordering of the legs:
    ``QuantumState`` (one leg per subsystem), ``QuantumChannel``
    ``(in, out)``, and ``ProcessTensor`` ``(in_0, out_0, ..., in_k, out_k)``.

    Information measures (entropy, negativity, mutual information) live in
    :mod:`processtensor.measures` as functions of any quantum tensor.
    """

    def __init__(self, data: np.ndarray, dims: tuple[int, ...] | None = None):
        data = np.asarray(data, dtype=complex)
        if dims is None:
            dims = tuple(round(np.sqrt(s)) for s in data.shape)
        dims = tuple(int(d) for d in dims)
        if data.shape != tuple(d * d for d in dims):
            raise ValueError(f"Data shape {data.shape} incompatible with Hilbert dims {dims}.")
        self.data = data
        self.dims = dims

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dims={self.dims})"

    @property
    def nlegs(self) -> int:
        return len(self.dims)

    # -- Choi representation ------------------------------------------------

    @property
    def choi(self) -> np.ndarray:
        """Choi matrix over the tensor product of the legs' Hilbert spaces.

        For a state this is the density matrix; for a channel the
        (unnormalized) Choi operator on ``H_in (x) H_out``. See
        :func:`processtensor.representations.choi_matrix`.
        """
        return representations.choi_matrix(self)

    @classmethod
    def from_choi(cls, matrix: np.ndarray, dims: tuple[int, ...]) -> QTensor:
        """Inverse of :attr:`choi`: build the leg tensor from a Choi matrix."""
        dims = tuple(int(d) for d in dims)
        return cls(representations.choi_to_tensor(matrix, dims), dims)

    @property
    def trace(self) -> float:
        """Trace of the Choi matrix."""
        return float(np.trace(self.choi).real)

    # -- Leg operations -----------------------------------------------------

    def partial_trace(self, keep: list[int]) -> QTensor:
        """Trace out all legs except those in ``keep`` (order preserved)."""
        keep = sorted(keep)
        data = self.data
        for ax in reversed([i for i in range(self.nlegs) if i not in keep]):
            tvec = np.eye(self.dims[ax]).reshape(-1)
            data = np.tensordot(data, tvec, axes=([ax], [0]))
        return QTensor(data, tuple(self.dims[i] for i in keep))

    def partial_transpose(self, legs: list[int]) -> QTensor:
        """Transpose (swap bra and ket) the given legs."""
        pairs = [x for d in self.dims for x in (d, d)]
        t = self.data.reshape(pairs)
        perm = list(range(2 * self.nlegs))
        for leg in legs:
            perm[2 * leg], perm[2 * leg + 1] = perm[2 * leg + 1], perm[2 * leg]
        data = t.transpose(perm).reshape(self.data.shape)
        return QTensor(data, self.dims)

    def link_product(self, other: QTensor, legs: list[int], other_legs: list[int]) -> QTensor:
        """Link product with ``other`` over the given leg pairs.

        See :func:`processtensor.operations.link_product`.
        """
        from .operations import link_product

        return link_product(self, other, legs, other_legs)

    # -- Validity -----------------------------------------------------------

    def is_hermitian(self, atol: float = TOL) -> bool:
        return is_hermitian(self.choi, atol)

    def is_cp(self, atol: float = TOL) -> bool:
        """Complete positivity: the Choi matrix is positive semidefinite."""
        return is_psd(self.choi, atol)
