import numpy as np
import pytest

from processtensor import QuantumState, bell_state, maximally_mixed, negativity


def test_density_matrix_roundtrip():
    rho = bell_state()
    state = QuantumState.from_matrix(rho, (2, 2))
    assert state.dims == (2, 2)
    assert np.allclose(state.density_matrix, rho)


def test_bell_state_measures():
    state = QuantumState.from_matrix(bell_state(), (2, 2))
    assert state.is_valid()
    assert state.purity() == pytest.approx(1.0)
    assert state.entropy() == pytest.approx(0.0, abs=1e-9)
    # Marginal of a Bell pair is maximally mixed.
    marginal = state.partial_trace([0])
    assert np.allclose(marginal.choi, maximally_mixed(2))
    assert state.mutual_information([[0], [1]]) == pytest.approx(2 * np.log(2))
    assert state.negativity([1]) == pytest.approx(0.5)


def test_product_state_is_uncorrelated():
    rho = np.kron(np.diag([0.2, 0.8]), np.diag([0.5, 0.5])).astype(complex)
    state = QuantumState.from_matrix(rho, (2, 2))
    assert state.mutual_information([[0], [1]]) == pytest.approx(0.0, abs=1e-9)
    assert state.negativity([1]) == pytest.approx(0.0, abs=1e-9)


def test_negativity_cut_forms_agree():
    state = QuantumState.from_matrix(bell_state(), (2, 2))
    # Default (half cut), integer cut, explicit subsystem list, and the
    # object-agnostic function all compute the same bipartition.
    assert state.negativity() == pytest.approx(0.5)
    assert state.negativity(1) == pytest.approx(0.5)
    assert state.negativity([0]) == pytest.approx(0.5)
    assert negativity(state, [0]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        state.negativity(2)


def test_operator_schmidt_rank_and_bond_entropy():
    # A pure state with state-vector Schmidt rank r has operator Schmidt
    # rank r^2 across the same cut.
    bell = QuantumState.from_matrix(bell_state(), (2, 2))
    assert bell.schmidt_rank() == 4
    assert bell.bond_entropy() == pytest.approx(np.log(4))
    product = QuantumState.from_matrix(
        np.kron(np.diag([0.2, 0.8]), np.diag([0.5, 0.5])).astype(complex), (2, 2)
    )
    assert product.schmidt_rank() == 1
    assert product.bond_entropy() == pytest.approx(0.0, abs=1e-9)
    # Explicit cut agrees with the default and the standalone function.
    from processtensor import bond_entropy

    assert bell.bond_entropy(1) == pytest.approx(bond_entropy(bell, 1))


def test_from_pure():
    psi = np.array([1, 1j]) / np.sqrt(2)
    state = QuantumState.from_pure(psi)
    assert state.is_valid()
    assert np.allclose(state.density_matrix, np.outer(psi, psi.conj()))


def test_invalid_states():
    not_psd = np.diag([1.5, -0.5]).astype(complex)
    assert not QuantumState.from_matrix(not_psd).is_valid()
    wrong_trace = np.eye(2, dtype=complex)
    assert not QuantumState.from_matrix(wrong_trace).is_valid()
