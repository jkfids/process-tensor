"""Shared numerical primitives.

Conventions (fixed throughout the package):

- Column-stacking vectorization: ``vec(rho)[j*d + i] = rho[i, j]``, so the
  superoperator of the unitary conjugation ``rho -> U rho U^dag`` is
  ``kron(conj(U), U)``.
- A *fused* (Liouville) index combines a bra-ket pair with the bra index
  major: ``alpha = (bra, ket)``. Unfusing a fused leg of dimension d^2 via
  ``reshape(d, d)`` therefore yields axes ``(bra, ket)``.
- Joint system-environment operators act on ``H_S (x) H_E`` (system major).
"""

from __future__ import annotations

import numpy as np

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

TOL = 1e-8


def pauli(label: str) -> np.ndarray:
    """Pauli matrix, or Kronecker-product Pauli string, for a label.

    ``pauli("X")`` is the single-qubit X matrix; a multi-character label
    such as ``pauli("XZ")`` returns the tensor product ``X (x) Z``.
    """
    label = label.upper()
    if not label or any(c not in _PAULI for c in label):
        raise ValueError(f"Invalid Pauli label {label!r}: use characters I, X, Y, Z.")
    out = _PAULI[label[0]]
    for c in label[1:]:
        out = np.kron(out, _PAULI[c])
    return out


def vec(rho: np.ndarray) -> np.ndarray:
    """Column-stacking vectorization of a matrix."""
    return np.asarray(rho).T.reshape(-1)


def unvec(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`vec`."""
    d = round(np.sqrt(v.size))
    return v.reshape(d, d).T


def dagger(a: np.ndarray) -> np.ndarray:
    """Conjugate transpose."""
    return a.conj().T


def superop_from_unitary(U: np.ndarray) -> np.ndarray:
    """Liouville superoperator of ``rho -> U rho U^dag``."""
    return np.kron(U.conj(), U)


def superop_from_kraus(kraus: list[np.ndarray]) -> np.ndarray:
    """Liouville superoperator of ``rho -> sum_i K_i rho K_i^dag``."""
    return sum(np.kron(K.conj(), K) for K in kraus)


def rft_unitary(U: np.ndarray, dS: int, dE: int) -> np.ndarray:
    """Reorder-fuse-transpose tensor of a system-environment unitary.

    Given ``U`` on ``H_S (x) H_E``, returns the 4-leg tensor
    ``W[env_in, sys_in, env_out, sys_out]`` of the superoperator
    ``conj(U) (x) U`` with each leg a fused Liouville index.
    """
    u = U.reshape(dS, dE, dS, dE)  # (s_out, e_out, s_in, e_in)
    w = np.einsum("SEsi,TFtj->ijstEFST", u.conj(), u)
    # axes: (e_in', e_in, s_in', s_in, e_out', e_out, s_out', s_out)
    return w.reshape(dE * dE, dS * dS, dE * dE, dS * dS)


def is_hermitian(a: np.ndarray, atol: float = TOL) -> bool:
    return np.allclose(a, dagger(a), atol=atol)


def is_psd(a: np.ndarray, atol: float = TOL) -> bool:
    """Whether a Hermitian matrix is positive semidefinite."""
    return is_hermitian(a, atol) and np.linalg.eigvalsh(a).min() > -atol


def trace_norm(a: np.ndarray) -> float:
    """Trace norm of a Hermitian matrix (sum of absolute eigenvalues)."""
    return float(np.abs(np.linalg.eigvalsh(a)).sum())


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy ``-Tr[rho ln rho]`` of a density matrix."""
    p = np.linalg.eigvalsh(rho)
    p = p[p > TOL]
    return float(-np.sum(p * np.log(p)))


def relative_entropy(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Quantum relative entropy ``S(rho || sigma)`` of density matrices.

    Returns ``inf`` when the support of ``rho`` is not contained in the
    support of ``sigma``.
    """
    s, V = np.linalg.eigh(sigma)
    # Weight of rho in each eigenvector of sigma.
    w = np.einsum("ij,ji->i", dagger(V) @ rho, V).real
    if np.any(w[s <= TOL] > TOL):
        return np.inf
    keep = s > TOL
    tr_rho_log_sigma = np.sum(w[keep] * np.log(s[keep]))
    return float(-von_neumann_entropy(rho) - tr_rho_log_sigma)


def unitary_from_hamiltonian(H: np.ndarray) -> np.ndarray:
    """Unitary ``exp(-iH)`` of a Hermitian ``H`` (via eigendecomposition)."""
    lam, V = np.linalg.eigh(H)
    return V @ np.diag(np.exp(-1j * lam)) @ dagger(V)


def random_unitary(d: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Haar-random unitary via the QR decomposition of a Ginibre matrix."""
    rng = np.random.default_rng() if rng is None else rng
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def maximally_mixed(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex) / d


def bell_state() -> np.ndarray:
    """Density matrix of the two-qubit Bell state ``(|00> + |11>)/sqrt(2)``."""
    psi = np.zeros(4, dtype=complex)
    psi[[0, 3]] = 1 / np.sqrt(2)
    return np.outer(psi, psi.conj())
