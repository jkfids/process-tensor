"""Process tensors: multi-time quantum processes as leg tensors."""

from __future__ import annotations

import numpy as np

from . import measures
from .qtensor import QTensor
from .quantumchannel import QuantumChannel
from .quantumstate import QuantumState
from .utils import TOL, rft_unitary, vec


class ProcessTensor(QTensor):
    """A k-slot process tensor with legs ``(in_0, out_0, ..., in_k, out_k)``.

    Fundamentally a multilinear map taking an initial state and a sequence
    of k instruments (CP maps) to the output state; represented here by its
    Choi operator, a positive semidefinite operator on
    ``H_in_0 (x) H_out_0 (x) ... (x) H_in_k (x) H_out_k`` obeying the causal
    (containment) constraints. The stored leg tensor is the
    reorder-fuse-transpose form, with each fused leg a Liouville index of
    dimension ``d^2``.
    """

    def __init__(self, data: np.ndarray, dims: tuple[int, ...] | None = None):
        super().__init__(data, dims)
        if self.nlegs % 2 != 0:
            raise ValueError("A process tensor has an (in, out) leg pair per step.")

    @property
    def steps(self) -> int:
        """Number of time steps (k + 1 for a k-slot process)."""
        return self.nlegs // 2

    @property
    def k(self) -> int:
        """Number of instrument slots."""
        return self.steps - 1

    # -- Construction -------------------------------------------------------

    @classmethod
    def from_stinespring(cls, unitaries: list[np.ndarray], env_state: np.ndarray) -> ProcessTensor:
        """Build a k-slot process tensor from its Stinespring dilation.

        The initial environment vector is threaded through the RFT tensors
        of the k+1 joint unitaries (each on ``H_S (x) H_E``) and capped with
        the environment trace vector, leaving open system legs
        ``(in_0, out_0, ..., in_k, out_k)``.
        """
        env_state = np.asarray(env_state, dtype=complex)
        dE = env_state.shape[0]
        dS = unitaries[0].shape[0] // dE
        T = vec(env_state)  # open env bond, contracted through the chain
        for U in unitaries:
            W = rft_unitary(U, dS, dE)  # (env_in, sys_in, env_out, sys_out)
            T = np.einsum("...m,manb->...abn", T, W)
        T = np.tensordot(T, vec(np.eye(dE)), axes=([-1], [0]))
        return cls(T, (dS,) * (2 * len(unitaries)))

    # -- Action -------------------------------------------------------------

    def apply(
        self,
        state: QuantumState | np.ndarray,
        instruments: list[QuantumChannel] | None = None,
    ) -> QuantumState:
        """Contract an initial state and k instruments to the output state.

        Implements the multilinear action of the process tensor as a comb
        contraction: the initial state enters ``in_0``, instrument ``j``
        connects ``out_j`` to ``in_{j+1}``, and the open ``out_k`` leg is
        returned.
        """
        instruments = [] if instruments is None else instruments
        if len(instruments) != self.k:
            raise ValueError(f"Expected {self.k} instruments, got {len(instruments)}.")
        rho = state.density_matrix if isinstance(state, QuantumState) else state
        t = np.tensordot(self.data, vec(rho), axes=([0], [0]))
        for A in instruments:
            t = np.tensordot(A.data, t, axes=([0], [0]))  # (A_out, in_j+1, ...)
            t = np.trace(t, axis1=0, axis2=1)
        return QuantumState(t, (self.dims[-1],))

    # -- Validity -----------------------------------------------------------

    def is_causal(self, atol: float = TOL) -> bool:
        """Check the causal (containment) constraints.

        ``Tr_out_j[Y_0:j] = Y_0:j-1 (x) I_in_j`` for all j, and
        ``Tr_out_0[Y_0:0] = I_in_0``.
        """
        pt = QTensor(self.data, self.dims)
        for j in range(self.k, 0, -1):
            d_in = pt.dims[2 * j]
            Y = pt.partial_trace(list(range(2 * j + 1)))  # drop out_j
            Z = pt.partial_trace(list(range(2 * j)))  # drop in_j and out_j
            if not np.allclose(Y.choi, np.kron(Z.choi / d_in, np.eye(d_in)), rtol=0, atol=atol):
                return False
            pt = QTensor(Z.data / d_in, Z.dims)
        marginal = pt.partial_trace([0]).choi
        return np.allclose(marginal, np.eye(pt.dims[0]), rtol=0, atol=atol)

    def is_valid(self, atol: float = TOL) -> bool:
        """Complete positivity plus causal ordering."""
        return self.is_cp(atol) and self.is_causal(atol)

    # -- Temporal correlation measures --------------------------------------

    def markov_product(self) -> ProcessTensor:
        """The product of single-time-step marginals ``(x)_j Tr_{~j}[Y]``.

        This is the closest fully Markovian (time-factorized) process,
        rescaled to the trace of the original Choi matrix.
        """
        m = np.array([[1.0]], dtype=complex)
        for j in range(self.steps):
            marginal = self.partial_trace([2 * j, 2 * j + 1])
            m = np.kron(m, marginal.choi / marginal.trace)
        return ProcessTensor.from_choi(m * self.trace, self.dims)

    def gqmi(self) -> float:
        """Generalized quantum mutual information.

        ``S(Y || Y_Markov)`` with trace-normalized Choi matrices.
        """
        return measures.mutual_information(self, [[2 * j, 2 * j + 1] for j in range(self.steps)])

    def temporal_negativity(self, cut: int | None = None) -> float:
        """Entanglement negativity across a temporal bipartition.

        ``cut`` is the number of leading time steps in the earlier block
        (default: half). Nonzero negativity witnesses temporal entanglement,
        i.e. genuinely quantum temporal correlations.
        """
        cut = self.steps // 2 if cut is None else cut
        if not 0 < cut < self.steps:
            raise ValueError(f"Cut must be between 1 and {self.steps - 1}.")
        return measures.negativity(self, list(range(2 * cut, self.nlegs)))

    def schmidt_rank(self, cut: int | None = None) -> int:
        """Operator Schmidt rank across a temporal cut.

        ``cut`` is the number of leading time steps (default: half). Equals
        the minimal bond dimension of an MPO representation at that cut—at
        most ``dE^2`` for a process built from an environment of dimension
        ``dE``.
        """
        cut = self.steps // 2 if cut is None else cut
        if not 0 < cut < self.steps:
            raise ValueError(f"Cut must be between 1 and {self.steps - 1}.")
        return measures.schmidt_rank(self, 2 * cut)

    def bond_entropy(self, cut: int | None = None) -> float:
        """Operator entanglement (bond) entropy across a temporal cut.

        ``cut`` is the number of leading time steps (default: half).
        """
        cut = self.steps // 2 if cut is None else cut
        if not 0 < cut < self.steps:
            raise ValueError(f"Cut must be between 1 and {self.steps - 1}.")
        return measures.bond_entropy(self, 2 * cut)
