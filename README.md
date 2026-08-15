# processtensor

[![CI](https://github.com/jkfids/process-tensor/actions/workflows/ci.yml/badge.svg)](https://github.com/jkfids/process-tensor/actions/workflows/ci.yml)

A minimal numerical implementation of **process tensors**—the general
description of multi-time quantum processes with non-Markovian memory—together
with the quantum states and channels they act on.

A $k$-slot process tensor is a multilinear map taking an initial system state
$\rho_\mathrm{in}$ and a sequence of $k$ *instruments* (CP maps representing
operations performed on the system at intermediate times) to the output state:

```math
\mathcal{T}_{0:k}[\rho_\mathrm{in}, (\mathcal{A}_0, \dots, \mathcal{A}_{k-1})]
= \rho_\mathrm{out}.
```

Like a quantum channel, it admits equivalent representations; this package
works with its Choi representation $\Upsilon_{0:k}$, a positive semidefinite
operator on the in/out spaces of every time step obeying causal (containment)
constraints. Any process tensor arises physically from joint
system-environment dynamics (a multi-time Stinespring dilation), which is how
process tensors are constructed here.

## Features

The package divides into quantum *objects*, their *representations* and
*operations*, and *measures*, following the philosophy laid out in
[DESIGN.md](DESIGN.md).

**Objects**

- `QuantumState` – density matrices with subsystem structure: validity
  checks, purity, marginals.
- `QuantumChannel` – CP maps as superoperators: construction from Kraus
  operators, unitaries, or Stinespring dilations
  $\mathrm{Tr}_E[U(\rho \otimes \sigma_E)U^\dagger]$; application to states,
  composition, CP/TP checks.
- `ProcessTensor` – $k$-slot process tensors built from a sequence of
  system-environment unitaries: `apply(state, instruments)` implements the
  multilinear action, `is_valid()` checks complete positivity and the causal
  containment constraints, and `markov_product()` gives the closest
  time-factorized (Markovian) process.

**Representations and operations**

- Every object exposes its Choi matrix (`.choi` / `from_choi`); further
  representations (Kraus, Stinespring, chi-matrix) will live in
  `representations`.
- `link_product` – the composition of quantum objects in the Choi picture
  (Chiribella–D'Ariano–Perinotti); feeding a state through a channel,
  composing channels, and building process tensors from system-environment
  tensors are all special cases.

**Measures**

Object-agnostic functionals in `measures`, with convenience methods on the
classes—the same function computes the spatial version for a state and the
temporal version for a process tensor:

- `entropy`, `purity`, `mutual_information` – von Neumann entropy, purity,
  and generalized quantum mutual information of any object; for a process
  tensor, `gqmi()` computes $S(\Upsilon \Vert \Upsilon_\mathrm{Markov})$, the
  relative entropy to the product of single-time-step marginals.
- `negativity` – entanglement negativity across a cut: spatial entanglement
  of a state, or temporal entanglement (`temporal_negativity()`) witnessing
  genuinely quantum memory in a process.
- `schmidt_rank` / `bond_entropy` – operator Schmidt rank and entanglement
  entropy across a cut; for a process tensor the rank equals the minimal MPO
  bond dimension at that temporal cut.

## Installation

```bash
git clone https://github.com/jkfids/process-tensor
cd process-tensor
pip install -e .            # or `pip install -e '.[dev,examples]'`
```

The only runtime dependency is numpy.

## Quickstart

Build the 1-slot process tensor of a qubit exchanging with a single
environment qubit under a Heisenberg interaction:

```python
import numpy as np
from processtensor import ProcessTensor, QuantumChannel, pauli, unitary_from_hamiltonian

# SWAP-like system-environment unitary: U = exp(-iH), H = -π/4 (XX + YY + ZZ)
H = -np.pi / 4 * (pauli("XX") + pauli("YY") + pauli("ZZ"))
U = unitary_from_hamiltonian(H)
plus = np.ones((2, 2)) / 2  # environment in |+⟩⟨+|

pt = ProcessTensor.from_stinespring([U, U], plus)

pt.is_valid()  # True  (completely positive + causally ordered)
pt.gqmi()  # 1.386... = 2 ln 2 (maximal temporal correlations)
pt.temporal_negativity()  # 0.5   (temporal entanglement: quantum memory)
pt.schmidt_rank()  # 4     (MPO bond dimension, saturating dE² = 4)

# Act on an input state with an instrument in the intermediate slot:
rho_in = np.diag([1.0, 0.0])
instrument = QuantumChannel.from_unitary(pauli("X"))
rho_out = pt.apply(rho_in, [instrument])
rho_out.density_matrix  # == rho_in: the SWAP process returns the stored input
```

For a guided tour—states, channels, process tensors, the link product, and
temporal correlation measures—see
[`docs/getting_started.ipynb`](docs/getting_started.ipynb).

## Conventions

Every object stores a tensor with one fused Liouville index (dimension
`d²`, bra index major) per physical wire, with column-stacking
vectorization and chronological leg order. The full set of mathematical
and style conventions is fixed in [DESIGN.md](DESIGN.md), together with
the design philosophy of the package.

## Development

```bash
pip install -e '.[dev]'
pre-commit install   # lint/format hooks (ruff check + ruff format) on commit
pytest               # unit tests
ruff check .         # lint
```

Tests, linting, and formatting run in CI on Python 3.12–3.14. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## License

MIT—see [LICENSE](LICENSE).
