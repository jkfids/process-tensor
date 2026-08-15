"""Quantum channels (and general CP maps) as two-leg superoperator tensors."""

from __future__ import annotations

import numpy as np

from .qtensor import QTensor
from .quantumstate import QuantumState
from .utils import (
    TOL,
    rft_unitary,
    superop_from_kraus,
    superop_from_unitary,
    vec,
)


class QuantumChannel(QTensor):
    """A CP map stored as its superoperator with legs ``(in, out)``.

    ``data[a, b]`` is the superoperator element mapping fused input index
    ``a`` to fused output index ``b`` (reorder-fuse-transpose form, with
    time flowing input to output). Trace-decreasing CP maps are allowed, so this
    class doubles as the instrument type for ``ProcessTensor.apply``.
    """

    def __init__(self, data: np.ndarray, dims: tuple[int, int] | None = None):
        super().__init__(data, dims)
        if self.nlegs != 2:
            raise ValueError(f"A channel has legs (in, out); got {self.nlegs} legs.")

    @property
    def superoperator(self) -> np.ndarray:
        """Liouville superoperator ``S`` with ``vec(E[rho]) = S vec(rho)``."""
        return self.data.T

    @classmethod
    def from_superoperator(cls, S: np.ndarray) -> QuantumChannel:
        return cls(np.asarray(S, dtype=complex).T)

    @classmethod
    def from_unitary(cls, U: np.ndarray) -> QuantumChannel:
        """Closed-system unitary channel ``rho -> U rho U^dag``."""
        return cls.from_superoperator(superop_from_unitary(U))

    @classmethod
    def from_kraus(cls, kraus: list[np.ndarray]) -> QuantumChannel:
        """Channel ``rho -> sum_i K_i rho K_i^dag``."""
        return cls.from_superoperator(superop_from_kraus(kraus))

    @classmethod
    def from_stinespring(cls, U: np.ndarray, env_state: np.ndarray) -> QuantumChannel:
        """Open-system channel ``rho -> Tr_E[U (rho (x) sigma_E) U^dag]``.

        ``U`` acts on ``H_S (x) H_E`` and ``sigma_E`` is the initial
        environment density matrix.
        """
        env_state = np.asarray(env_state, dtype=complex)
        dE = env_state.shape[0]
        dS = U.shape[0] // dE
        W = rft_unitary(U, dS, dE)  # (env_in, sys_in, env_out, sys_out)
        data = np.einsum("m,manb,n->ab", vec(env_state), W, vec(np.eye(dE)))
        return cls(data)

    def apply(self, state: QuantumState | np.ndarray) -> QuantumState:
        """Apply the map to a state (``QuantumState`` or density matrix)."""
        rho = state.density_matrix if isinstance(state, QuantumState) else state
        out = vec(rho) @ self.data
        return QuantumState(out, (self.dims[1],))

    def compose(self, other: QuantumChannel) -> QuantumChannel:
        """Sequential composition ``self o other`` (``other`` acts first)."""
        return QuantumChannel(other.data @ self.data)

    def __matmul__(self, other: QuantumChannel) -> QuantumChannel:
        return self.compose(other)

    def is_tp(self, atol: float = TOL) -> bool:
        """Trace preservation: ``Tr_out[Choi] = I_in``."""
        marginal = self.partial_trace([0]).choi
        return np.allclose(marginal, np.eye(self.dims[0]), rtol=0, atol=atol)

    def is_cptp(self, atol: float = TOL) -> bool:
        return self.is_cp(atol) and self.is_tp(atol)
