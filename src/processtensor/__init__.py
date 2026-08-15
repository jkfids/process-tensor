"""Minimal numerical process tensors.

A small, readable implementation of process tensors and related quantum
objects (states, channels) in their leg-based Liouville tensor form, with
validity checks (complete positivity, causal ordering) and object-agnostic
correlation measures (entropy, negativity, mutual information).
"""

from .__about__ import __version__
from .measures import (
    bond_entropy,
    entropy,
    mutual_information,
    negativity,
    purity,
    schmidt_rank,
    schmidt_values,
)
from .operations import link_product
from .processtensor import ProcessTensor
from .qtensor import QTensor
from .quantumchannel import QuantumChannel
from .quantumstate import QuantumState
from .representations import choi_matrix
from .utils import (
    bell_state,
    maximally_mixed,
    pauli,
    random_unitary,
    relative_entropy,
    unitary_from_hamiltonian,
    unvec,
    vec,
    von_neumann_entropy,
)

__all__ = [  # noqa: RUF022 - grouped by DESIGN.md category, not isort order
    "__version__",
    "QTensor",
    "QuantumState",
    "QuantumChannel",
    "ProcessTensor",
    "choi_matrix",
    "link_product",
    "entropy",
    "negativity",
    "mutual_information",
    "purity",
    "schmidt_values",
    "schmidt_rank",
    "bond_entropy",
    "pauli",
    "bell_state",
    "maximally_mixed",
    "random_unitary",
    "relative_entropy",
    "unitary_from_hamiltonian",
    "vec",
    "unvec",
    "von_neumann_entropy",
]
