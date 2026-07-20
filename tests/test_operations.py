import numpy as np
import pytest

from processtensor import (
    ProcessTensor,
    QTensor,
    QuantumChannel,
    QuantumState,
    link_product,
    pauli,
    random_unitary,
)
from processtensor.utils import rft_unitary

RNG = np.random.default_rng(7)


def se_tensor(U: np.ndarray, dS: int, dE: int) -> QTensor:
    """System-environment unitary as a 4-leg (env_in, sys_in, env_out, sys_out)
    quantum tensor."""
    return QTensor(rft_unitary(U, dS, dE), (dE, dS, dE, dS))


def test_link_of_channels_is_composition():
    ch1 = QuantumChannel.from_unitary(random_unitary(2, RNG))
    ch2 = QuantumChannel.from_kraus([np.sqrt(0.8) * pauli("I"), np.sqrt(0.2) * pauli("Z")])
    linked = link_product(ch1, ch2, [1], [0])  # out of ch1 <-> in of ch2
    assert np.allclose(linked.choi, ch2.compose(ch1).choi)


def test_link_of_state_and_channel_is_application():
    state = QuantumState.from_pure(np.array([1, 1j]) / np.sqrt(2))
    ch = QuantumChannel.from_unitary(random_unitary(2, RNG))
    linked = link_product(state, ch, [0], [0])
    assert np.allclose(linked.choi, ch.apply(state).density_matrix)


def test_link_reconstructs_stinespring_process():
    """Chain-linking the SE-unitary tensors through their environment legs
    (env state in, trace cap out) reproduces ``from_stinespring``."""
    dS = dE = 2
    U1, U2 = random_unitary(dS * dE, RNG), random_unitary(dS * dE, RNG)
    sigma = QuantumState.from_pure(np.array([1, 0]))  # environment |0>

    t = link_product(sigma, se_tensor(U1, dS, dE), [0], [0])  # (in0, env, out0)
    t = link_product(t, se_tensor(U2, dS, dE), [1], [0])  # (in0, out0, in1, env, out1)
    t = t.partial_trace([0, 1, 2, 4])  # trace the final environment leg

    pt = ProcessTensor.from_stinespring([U1, U2], np.diag([1.0, 0.0]))
    assert np.allclose(t.data, pt.data)


def test_link_method_matches_function():
    ch1 = QuantumChannel.from_unitary(random_unitary(2, RNG))
    ch2 = QuantumChannel.from_unitary(random_unitary(2, RNG))
    assert np.allclose(ch1.link_product(ch2, [1], [0]).data, link_product(ch1, ch2, [1], [0]).data)


def test_link_dimension_mismatch_raises():
    qubit = QuantumState.from_pure(np.array([1, 0]))
    qutrit_channel = QuantumChannel.from_unitary(random_unitary(3, RNG))
    with pytest.raises(ValueError):
        link_product(qubit, qutrit_channel, [0], [0])
    with pytest.raises(ValueError):
        link_product(qubit, qutrit_channel, [0], [0, 1])
