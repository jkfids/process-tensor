import numpy as np
import pytest

from processtensor import (
    ProcessTensor,
    QuantumChannel,
    pauli,
    random_unitary,
    unitary_from_hamiltonian,
)

RNG = np.random.default_rng(42)

PLUS = np.ones((2, 2), dtype=complex) / 2  # |+><+| environment state
GROUND = np.diag([1.0, 0.0]).astype(complex)


def heisenberg(theta: float) -> np.ndarray:
    """H_J = -theta/2 (XX + YY + ZZ); SWAP-like at pi/2."""
    H = -theta / 2 * sum(pauli(s + s) for s in "XYZ")
    return unitary_from_hamiltonian(H)


def controlled_x(theta: float) -> np.ndarray:
    """H_RX = -theta/2 (X (x) I - X (x) Z); CNOT-like at pi/2."""
    H = -theta / 2 * (pauli("XI") - pauli("XZ"))
    return unitary_from_hamiltonian(H)


def random_process(k: int, dS: int = 2, dE: int = 2) -> ProcessTensor:
    unitaries = [random_unitary(dS * dE, RNG) for _ in range(k + 1)]
    sigma = np.diag(np.eye(dE)[0]).astype(complex)
    return ProcessTensor.from_stinespring(unitaries, sigma)


def test_zero_slot_process_equals_channel():
    U = random_unitary(4, RNG)
    pt = ProcessTensor.from_stinespring([U], GROUND)
    ch = QuantumChannel.from_stinespring(U, GROUND)
    assert pt.k == 0
    assert np.allclose(pt.choi, ch.choi)


def test_stinespring_processes_are_valid():
    for k in (0, 1, 2):
        pt = random_process(k)
        assert pt.is_cp()
        assert pt.is_causal()
        assert pt.is_valid()
        assert pt.trace == pytest.approx(2 ** (k + 1))


def test_random_psd_matrix_is_not_causal():
    a = RNG.normal(size=(16, 16)) + 1j * RNG.normal(size=(16, 16))
    choi = a @ a.conj().T
    choi *= 4 / np.trace(choi).real
    pt = ProcessTensor.from_choi(choi, (2, 2, 2, 2))
    assert pt.is_cp()
    assert not pt.is_causal()


def test_non_interacting_process_is_markovian():
    U = np.kron(random_unitary(2, RNG), random_unitary(2, RNG))
    pt = ProcessTensor.from_stinespring([U, U], GROUND)
    assert pt.gqmi() == pytest.approx(0.0, abs=1e-9)
    assert pt.temporal_negativity() == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(pt.choi, pt.markov_product().choi)


def test_heisenberg_swap_process():
    """At theta = pi/2 the SE unitary is a SWAP: the process is a perfect
    quantum memory, with maximal GQMI 2 ln 2 and temporal negativity 1/2."""
    U = heisenberg(np.pi / 2)
    pt = ProcessTensor.from_stinespring([U, U], PLUS)
    assert pt.is_valid()
    assert pt.gqmi() == pytest.approx(2 * np.log(2))
    assert pt.temporal_negativity() == pytest.approx(0.5)


def test_controlled_x_process():
    """At theta = pi/2 (CNOT-like, env |+>) the process is a perfectly
    correlated bit-flip: GQMI ln 2 but zero temporal entanglement
    (classical memory only)."""
    U = controlled_x(np.pi / 2)
    pt = ProcessTensor.from_stinespring([U, U], PLUS)
    assert pt.is_valid()
    assert pt.gqmi() == pytest.approx(np.log(2))
    assert pt.temporal_negativity() == pytest.approx(0.0, abs=1e-9)


def test_schmidt_rank_is_mpo_bond_dimension():
    """The operator Schmidt rank across a temporal cut equals the minimal
    MPO bond dimension there: at most dE^2, saturated by the SWAP process
    (with maximal bond entropy ln dE^2), and 1 for a Markovian process."""
    swap = ProcessTensor.from_stinespring([heisenberg(np.pi / 2)] * 2, PLUS)
    assert swap.schmidt_rank() == 4
    assert swap.bond_entropy() == pytest.approx(np.log(4))

    U = np.kron(random_unitary(2, RNG), random_unitary(2, RNG))
    markov = ProcessTensor.from_stinespring([U, U], GROUND)
    assert markov.schmidt_rank() == 1
    assert markov.bond_entropy() == pytest.approx(0.0, abs=1e-9)

    two_slot = random_process(2)
    assert two_slot.schmidt_rank(1) <= 4
    assert two_slot.schmidt_rank(2) <= 4
    with pytest.raises(ValueError):
        two_slot.schmidt_rank(3)


def test_apply_identity_instrument_composes_channels():
    """Feeding the identity instrument through a 1-slot process reproduces
    the composition of the two dilated channels only when the environment
    carries no memory (non-interacting steps)."""
    U = np.kron(random_unitary(2, RNG), random_unitary(2, RNG))
    pt = ProcessTensor.from_stinespring([U, U], GROUND)
    rho = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
    identity = QuantumChannel.from_unitary(np.eye(2, dtype=complex))
    out = pt.apply(rho, [identity])
    ch = QuantumChannel.from_stinespring(U, GROUND)
    expected = ch.apply(ch.apply(rho))
    assert np.allclose(out.density_matrix, expected.density_matrix)


def test_apply_swap_process_delays_input():
    """The SWAP process stores the input in the environment and releases it
    one step later: the final output equals the initial state regardless of
    the (trace-preserving) instrument applied in between."""
    U = heisenberg(np.pi / 2)
    pt = ProcessTensor.from_stinespring([U, U], PLUS)
    rho = np.array([[0.8, 0.1j], [-0.1j, 0.2]], dtype=complex)
    depolarize = QuantumChannel.from_kraus([0.5 * pauli(s) for s in "IXYZ"])
    out = pt.apply(rho, [depolarize])
    assert np.allclose(out.density_matrix, rho)


def test_output_state_is_valid():
    pt = random_process(1)
    rho = np.diag([0.3, 0.7]).astype(complex)
    noise = QuantumChannel.from_kraus([np.sqrt(0.9) * pauli("I"), np.sqrt(0.1) * pauli("X")])
    out = pt.apply(rho, [noise])
    assert out.is_valid()
