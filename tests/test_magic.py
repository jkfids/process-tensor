import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from processtensor import (
    ProcessTensor,
    QuantumChannel,
    QuantumState,
    bell_state,
    pauli,
    stabilizer_renyi2,
)
from processtensor.measures import _choi_factor, _ensemble_stabilizer_purity, _orthonormal_rows

RNG = np.random.default_rng(42)
T_STATE = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
FREE_MIXTURE = np.array([[0.75, 0.25], [0.25, 0.25]])  # (|0><0| + |+><+|) / 2


@pytest.mark.filterwarnings("error")
def test_pure_state_values():
    ground = QuantumState.from_pure(np.array([1, 0]))
    bell = QuantumState.from_matrix(bell_state(), (2, 2))
    assert stabilizer_renyi2(ground) == pytest.approx(0, abs=1e-9)
    assert stabilizer_renyi2(bell) == pytest.approx(0, abs=1e-9)
    assert stabilizer_renyi2(QuantumState.from_pure(T_STATE)) == pytest.approx(np.log(4 / 3))


def test_clifford_invariance_and_additivity():
    clifford = (pauli("X") + pauli("Z")) / np.sqrt(2) @ np.diag([1, 1j])
    rotated = QuantumState.from_pure(clifford @ T_STATE)
    pair = QuantumState.from_pure(np.kron(T_STATE, T_STATE), (2, 2))
    assert stabilizer_renyi2(rotated) == pytest.approx(np.log(4 / 3))
    assert stabilizer_renyi2(pair) == pytest.approx(2 * np.log(4 / 3))


def test_free_mixed_states():
    for rho in (np.eye(2) / 2, FREE_MIXTURE):
        with pytest.warns(RuntimeWarning, match="numerical upper bound"):
            value = stabilizer_renyi2(QuantumState.from_matrix(rho), rng=RNG)
        assert value == pytest.approx(0, abs=1e-7)


def test_mixed_magic_and_reproducibility():
    """The spectral ensemble consists of T-type states with M2 = ln(4/3)."""
    rho = (pauli("I") + 0.6 * pauli("X") + 0.6 * pauli("Y")) / 2
    state = QuantumState.from_matrix(rho)
    with pytest.warns(RuntimeWarning):
        first = stabilizer_renyi2(state, restarts=3, rng=np.random.default_rng(5))
        second = stabilizer_renyi2(state, restarts=3, rng=np.random.default_rng(5))
    assert 0 < first <= np.log(4 / 3) + 1e-9
    assert first == pytest.approx(second, abs=1e-9)


def test_channel_and_process_normalization():
    channel = QuantumChannel.from_unitary(np.diag([1, np.exp(1j * np.pi / 4)]))
    state = QuantumState.from_matrix(channel.choi / 2, (2, 2))
    process = ProcessTensor.from_choi(3 * channel.choi, (2, 2))
    for x in (channel, state, process):
        assert stabilizer_renyi2(x) == pytest.approx(np.log(4 / 3))


def test_swap_memory_process():
    swap = np.eye(4)[[0, 2, 1, 3]]
    process = ProcessTensor.from_stinespring([swap, swap], np.ones((2, 2)) / 2)
    with pytest.warns(RuntimeWarning):
        value = stabilizer_renyi2(process, restarts=2, rng=RNG)
    assert value == pytest.approx(0, abs=1e-7)


def test_ensemble_decomposition():
    rho = np.array([[0.65, 0.1 + 0.2j], [0.1 - 0.2j, 0.35]])
    a = _choi_factor(QuantumState.from_matrix(rho))
    u = _orthonormal_rows(RNG.normal(size=(2, 4)) + 1j * RNG.normal(size=(2, 4)))
    u = np.pad(u, ((0, 0), (0, 1)))  # Include a zero-weight member.
    b = a @ u
    assert np.allclose(b @ b.conj().T, rho)
    assert np.allclose(_orthonormal_rows(u), u)
    paulis = np.array([a.conj().T @ pauli(label) @ a for label in "IXYZ"])
    expected = 0.0
    for vector in b.T:
        weight = np.vdot(vector, vector).real
        if weight > 0:
            state = QuantumState.from_pure(vector / np.sqrt(weight))
            expected += weight * np.exp(-stabilizer_renyi2(state))
    assert _ensemble_stabilizer_purity(u, paulis, 2) == pytest.approx(expected)


def test_invalid_states():
    for rho in (np.zeros((2, 2)), np.diag([np.nan, 1]), [[1, 1], [0, 0]], np.diag([1.1, -0.1])):
        with pytest.raises(ValueError):
            stabilizer_renyi2(QuantumState.from_matrix(rho))
    for d in (3, 4):
        with pytest.raises(ValueError):
            stabilizer_renyi2(QuantumState.from_matrix(np.eye(d) / d))


def test_invalid_optimizer_controls():
    state = QuantumState.from_pure(T_STATE)
    for kwargs in ({"restarts": 0}, {"maxiter": -1}, {"restarts": 1.5}):
        with pytest.raises(ValueError):
            stabilizer_renyi2(state, **kwargs)


def test_spectral_roundoff():
    state = QuantumState.from_matrix(np.diag([1 + 1e-13, -1e-13]))
    assert stabilizer_renyi2(state) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("success, scale", [(False, 1), (True, 2)])
def test_failed_or_infeasible_optimization_raises(monkeypatch, success, scale):
    def minimize(fun, parameters, **kwargs):
        return OptimizeResult(success=success, x=parameters * scale)

    monkeypatch.setattr("scipy.optimize.minimize", minimize)
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError):
        stabilizer_renyi2(QuantumState.from_matrix(FREE_MIXTURE), restarts=1)
