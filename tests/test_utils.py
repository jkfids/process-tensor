import numpy as np
import pytest

from processtensor import (
    maximally_mixed,
    random_unitary,
    relative_entropy,
    unitary_from_hamiltonian,
    unvec,
    vec,
    von_neumann_entropy,
)
from processtensor.utils import pauli, superop_from_kraus, superop_from_unitary

RNG = np.random.default_rng(42)


def random_density_matrix(d: int) -> np.ndarray:
    a = RNG.normal(size=(d, d)) + 1j * RNG.normal(size=(d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def test_vec_unvec_roundtrip():
    rho = random_density_matrix(3)
    assert np.allclose(unvec(vec(rho)), rho)


def test_vec_column_stacking():
    m = np.arange(4).reshape(2, 2)
    assert np.array_equal(vec(m), [0, 2, 1, 3])


def test_superop_from_unitary_acts_by_conjugation():
    U = random_unitary(3, RNG)
    rho = random_density_matrix(3)
    assert np.allclose(unvec(superop_from_unitary(U) @ vec(rho)), U @ rho @ U.conj().T)


def test_random_unitary_is_unitary():
    U = random_unitary(4, RNG)
    assert np.allclose(U @ U.conj().T, np.eye(4))


def test_unitary_from_hamiltonian():
    theta = 0.7
    U = unitary_from_hamiltonian(theta * pauli("X"))
    expected = np.cos(theta) * pauli("I") - 1j * np.sin(theta) * pauli("X")
    assert np.allclose(U, expected)


def test_pauli_string_is_kron_of_singles():
    assert np.allclose(pauli("XZ"), np.kron(pauli("X"), pauli("Z")))
    assert np.allclose(pauli("x"), pauli("X"))


def test_superop_from_empty_kraus_raises():
    """An empty Kraus list is not the zero map; summing it would return 0."""
    with pytest.raises(ValueError):
        superop_from_kraus([])


def test_pauli_invalid_label_raises():
    with pytest.raises(ValueError):
        pauli("Q")
    with pytest.raises(ValueError):
        pauli("")


def test_entropy_of_maximally_mixed():
    assert von_neumann_entropy(maximally_mixed(4)) == pytest.approx(np.log(4))


def test_relative_entropy_self_is_zero():
    rho = random_density_matrix(3)
    assert relative_entropy(rho, rho) == pytest.approx(0.0, abs=1e-9)


def test_relative_entropy_support_mismatch_is_inf():
    pure = np.diag([1.0, 0.0]).astype(complex)
    assert relative_entropy(maximally_mixed(2), pure) == np.inf


def test_relative_entropy_tolerates_tiny_eigenvalues():
    """Genuine small eigenvalues of sigma are support, not numerical zeros.

    A cutoff at the matrix-comparison tolerance 1e-8 would call this a support
    mismatch and return inf, though the relative entropy is finite.
    """
    eps = 1e-10  # below TOL, above SPECTRAL_TOL
    sigma = np.diag([1 - eps, eps]).astype(complex)
    rho = np.diag([0.5, 0.5]).astype(complex)
    expected = 0.5 * np.log(0.5 / (1 - eps)) + 0.5 * np.log(0.5 / eps)
    assert relative_entropy(rho, sigma) == pytest.approx(expected)
