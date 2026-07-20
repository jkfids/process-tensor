import numpy as np

from processtensor import QuantumChannel, maximally_mixed, pauli, random_unitary

RNG = np.random.default_rng(42)

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def depolarizing(p: float) -> QuantumChannel:
    kraus = [np.sqrt(1 - 3 * p / 4) * pauli("I")] + [np.sqrt(p / 4) * pauli(s) for s in "XYZ"]
    return QuantumChannel.from_kraus(kraus)


def amplitude_damping(gamma: float) -> QuantumChannel:
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return QuantumChannel.from_kraus([K0, K1])


def test_kraus_channels_are_cptp():
    assert depolarizing(0.3).is_cptp()
    assert amplitude_damping(0.4).is_cptp()


def test_depolarizing_apply():
    ch = depolarizing(1.0)
    rho = np.array([[0.9, 0.3], [0.3, 0.1]], dtype=complex)
    assert np.allclose(ch.apply(rho).density_matrix, maximally_mixed(2))


def test_amplitude_damping_apply():
    ch = amplitude_damping(1.0)
    excited = np.diag([0.0, 1.0]).astype(complex)
    ground = np.diag([1.0, 0.0]).astype(complex)
    assert np.allclose(ch.apply(excited).density_matrix, ground)


def test_unitary_channel():
    U = random_unitary(2, RNG)
    ch = QuantumChannel.from_unitary(U)
    rho = np.array([[0.7, 0.2j], [-0.2j, 0.3]], dtype=complex)
    assert ch.is_cptp()
    assert np.allclose(ch.apply(rho).density_matrix, U @ rho @ U.conj().T)
    # The Choi matrix of a unitary channel is a maximally entangled state.
    assert np.linalg.matrix_rank(ch.choi) == 1


def test_stinespring_replace_channel_orientation():
    """Regression test for the time-reversal bug in the old implementation.

    SWAP with environment |0><0| implements the replace channel, whose Choi
    matrix is I (x) |0><0| on H_in (x) H_out. The time-reversed process
    would give |0><0| (x) I instead.
    """
    sigma = np.diag([1.0, 0.0]).astype(complex)
    ch = QuantumChannel.from_stinespring(SWAP, sigma)
    assert ch.is_cptp()
    assert np.allclose(ch.choi, np.kron(np.eye(2), sigma))
    assert not np.allclose(ch.choi, np.kron(sigma, np.eye(2)))


def test_stinespring_matches_kraus():
    # CNOT (system control) with env |0><0| fully dephases the system:
    # the environment records the system's Z basis state.
    U = np.eye(4, dtype=complex)[:, [0, 1, 3, 2]]
    ch = QuantumChannel.from_stinespring(U, np.diag([1.0, 0.0]))
    kraus_equiv = QuantumChannel.from_kraus(
        [np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)]
    )
    assert ch.is_cptp()
    assert np.allclose(ch.choi, kraus_equiv.choi)


def test_compose():
    U, V = random_unitary(2, RNG), random_unitary(2, RNG)
    composed = QuantumChannel.from_unitary(V) @ QuantumChannel.from_unitary(U)
    expected = QuantumChannel.from_unitary(V @ U)
    assert np.allclose(composed.superoperator, expected.superoperator)
