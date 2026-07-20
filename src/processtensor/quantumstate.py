"""Quantum states as vectorized density matrices."""

from __future__ import annotations

import numpy as np

from . import measures
from .qtensor import QTensor
from .utils import TOL


class QuantumState(QTensor):
    """A (possibly multipartite) density matrix, one fused leg per subsystem."""

    @classmethod
    def from_matrix(cls, rho: np.ndarray, dims: tuple[int, ...] | None = None) -> QuantumState:
        """Build from a density matrix, optionally split into subsystems."""
        rho = np.asarray(rho, dtype=complex)
        if dims is None:
            dims = (rho.shape[0],)
        return cls.from_choi(rho, dims)

    @classmethod
    def from_pure(cls, psi: np.ndarray, dims: tuple[int, ...] | None = None) -> QuantumState:
        """Build from a pure state vector."""
        psi = np.asarray(psi, dtype=complex)
        return cls.from_matrix(np.outer(psi, psi.conj()), dims)

    @property
    def density_matrix(self) -> np.ndarray:
        return self.choi

    def is_valid(self, atol: float = TOL) -> bool:
        """Hermitian, positive semidefinite, and unit trace."""
        return self.is_cp(atol) and bool(np.isclose(self.trace, 1.0, atol=atol))

    def purity(self) -> float:
        return measures.purity(self)

    def entropy(self) -> float:
        """Von Neumann entropy of the density matrix."""
        return measures.entropy(self)

    def negativity(self, cut: int | list[int] | None = None) -> float:
        """Entanglement negativity across a bipartition of the subsystems.

        ``cut`` is either the number of leading subsystems in the first
        block (default: half), or an explicit list of subsystem indices.
        """
        cut = self.nlegs // 2 if cut is None else cut
        if isinstance(cut, int):
            if not 0 < cut < self.nlegs:
                raise ValueError(f"Cut must be between 1 and {self.nlegs - 1}.")
            cut = list(range(cut))
        return measures.negativity(self, cut)

    def schmidt_rank(self, cut: int | None = None) -> int:
        """Operator Schmidt rank across the first ``cut`` subsystems | rest
        (default: half). A pure state with state-vector Schmidt rank r has
        operator Schmidt rank r^2."""
        cut = self.nlegs // 2 if cut is None else cut
        return measures.schmidt_rank(self, cut)

    def bond_entropy(self, cut: int | None = None) -> float:
        """Operator entanglement entropy across the first ``cut`` subsystems
        | rest (default: half)."""
        cut = self.nlegs // 2 if cut is None else cut
        return measures.bond_entropy(self, cut)

    def mutual_information(self, partition: list[list[int]] | None = None) -> float:
        """Quantum mutual information over a partition of the subsystems
        (default: every subsystem its own block)."""
        if partition is None:
            partition = [[j] for j in range(self.nlegs)]
        return measures.mutual_information(self, partition)
