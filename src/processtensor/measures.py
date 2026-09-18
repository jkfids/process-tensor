"""Functions of quantum objects, agnostic to the specific object type.

Every quantity here is defined for any leg-based quantum tensor—a state,
a channel, or a process tensor—through its Choi representation. The same
function therefore computes, depending only on the object passed in, e.g.
the entanglement negativity of a bipartite density matrix or the temporal
negativity of a process tensor.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from .utils import SPECTRAL_TOL, TOL, pauli, relative_entropy, trace_norm, von_neumann_entropy

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


# -- Magic -----------------------------------------------------------------


def stabilizer_renyi2(
    x: QTensor,
    *,
    restarts: int = 8,
    maxiter: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Stabilizer Rényi-2 entropy of the trace-normalized Choi matrix.

    For a pure n-qubit state, returns ``-ln Q(psi)``, where
    ``Q(psi) = sum_P <psi|P|psi>**4 / 2**n`` and P ranges over all strings
    in ``{I, X, Y, Z}^n``. For mixed states, the roof extension is
    ``-ln(max sum_a p_a Q(psi_a))``, over all pure-state decompositions of
    ``rho = x.choi / x.trace``. All legs must be qubits.

    Mixed inputs emit a ``RuntimeWarning`` and use local optimization over
    rank-squared ensemble members. The result is a numerical upper bound,
    subject to floating-point error and the ``SPECTRAL_TOL`` support cutoff.

    ``restarts`` counts the spectral start plus random starts; ``maxiter``
    bounds iterations per start. Pass a seeded generator for reproducible
    searches. Raises ``RuntimeError`` if every optimization fails.

    The roof extension follows Leone and Bittel, Phys. Rev. A 110,
    L040403 (2024), Definition 5 (arXiv:2404.11652).
    """
    if not x.dims or any(d != 2 for d in x.dims):
        raise ValueError("All legs must be qubits.")
    if not isinstance(restarts, int) or restarts < 1:
        raise ValueError("restarts must be a positive integer.")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")

    factor = _choi_factor(x)
    if factor.shape[1] == 1:
        return _pure_stabilizer_renyi2(factor[:, 0])
    warnings.warn(
        "Mixed Choi state: optimizing stabilizer Rényi-2 entropy; "
        "the result is a numerical upper bound.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _mixed_stabilizer_renyi2(factor, restarts, maxiter, rng)


def _choi_factor(x: QTensor) -> np.ndarray:
    """Spectral factor ``A`` of the trace-normalized Choi matrix.

    Eigenvalues at or below ``SPECTRAL_TOL`` are discarded and the remainder
    renormalized, giving ``rho = A A^dag`` on the retained support.
    """
    rho = x.choi
    trace = np.trace(rho)
    if not np.all(np.isfinite(rho)) or not np.isfinite(trace) or trace.real <= 0:
        raise ValueError("The Choi matrix must be finite with positive trace.")
    rho = rho / trace.real
    if not np.allclose(rho, rho.conj().T, rtol=0, atol=TOL):
        raise ValueError("The Choi matrix must be Hermitian.")
    eigenvalues, eigenvectors = np.linalg.eigh((rho + rho.conj().T) / 2)
    if eigenvalues[0] < -TOL:
        raise ValueError("The Choi matrix must be positive semidefinite.")
    keep = eigenvalues > SPECTRAL_TOL
    weights = eigenvalues[keep]
    return eigenvectors[:, keep] * np.sqrt(weights / np.sum(weights))


def _pure_stabilizer_renyi2(psi: np.ndarray) -> float:
    """Stabilizer Rényi-2 entropy of a normalized qubit state vector."""
    n = psi.size.bit_length() - 1
    moment = sum(
        np.vdot(psi, pauli("".join(label)) @ psi).real ** 4 for label in product("IXYZ", repeat=n)
    )
    return float(-np.log(min(moment / psi.size, 1.0)))


def _mixed_stabilizer_renyi2(
    factor: np.ndarray, restarts: int, maxiter: int, rng: np.random.Generator | None
) -> float:
    """Roof extension from ensembles ``B = A U``, with ``U U^dag = I``."""
    dim, rank = factor.shape
    n = dim.bit_length() - 1

    # Identity comes first, giving the ensemble probabilities.
    paulis = np.array(
        [factor.conj().T @ pauli("".join(label)) @ factor for label in product("IXYZ", repeat=n)]
    )
    spectral = np.eye(rank, rank * rank, dtype=complex)
    best = _ensemble_stabilizer_purity(spectral, paulis, dim)
    rng = np.random.default_rng() if rng is None else rng
    upper = np.triu_indices(rank, k=1)

    def unpack(parameters: np.ndarray) -> np.ndarray:
        real, imag = parameters.reshape(2, rank, rank * rank)
        return real + 1j * imag

    def objective(parameters: np.ndarray) -> float:
        return -_ensemble_stabilizer_purity(unpack(parameters), paulis, dim)

    def constraint(parameters: np.ndarray) -> np.ndarray:
        u = unpack(parameters)
        residual = u @ u.conj().T - np.eye(rank)
        return np.concatenate(
            (residual.diagonal().real, residual[upper].real, residual[upper].imag)
        )

    succeeded = False
    for start in range(restarts):
        u = spectral
        if start:
            u = _orthonormal_rows(
                rng.normal(size=spectral.shape) + 1j * rng.normal(size=spectral.shape)
            )
        parameters = np.concatenate((u.real.ravel(), u.imag.ravel()))
        result = optimize.minimize(
            objective,
            parameters,
            method="SLSQP",
            constraints={"type": "eq", "fun": constraint},
            options={"maxiter": maxiter, "ftol": 1e-10},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        if np.max(np.abs(constraint(result.x))) > TOL:
            continue
        # Remove residual feasibility error before accepting an upper bound.
        u = _orthonormal_rows(unpack(result.x))
        candidate = _ensemble_stabilizer_purity(u, paulis, dim)
        if np.isfinite(candidate):
            best = max(best, candidate)
            succeeded = True
    if not succeeded:
        raise RuntimeError("Stabilizer Rényi-2 optimization failed for every start.")
    return float(-np.log(min(best, 1.0)))


def _orthonormal_rows(matrix: np.ndarray) -> np.ndarray:
    """Nearest matrix with orthonormal rows."""
    left, _, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right


def _ensemble_stabilizer_purity(u: np.ndarray, paulis: np.ndarray, dim: int) -> float:
    """Average stabilizer purity for the ensemble ``A @ u``.

    ``paulis`` contains ``A^dag P A``, with identity first. Zero-weight
    members contribute zero; the others use normalized Pauli expectations.
    """
    moments = np.einsum("ia,pij,ja->pa", u.conj(), paulis, u).real
    probabilities = moments[0]
    nonzero = probabilities > 0
    expectations = moments[:, nonzero] / probabilities[nonzero]
    return float(np.sum(probabilities[nonzero] * np.sum(expectations**4, axis=0)) / dim)


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
    how compressible the object is across the cut—zero if and only if
    the tensor factorizes there.
    """
    s = schmidt_values(x, cut)
    p = s**2 / np.sum(s**2)
    p = p[p > SPECTRAL_TOL]
    return float(-np.sum(p * np.log(p)))
