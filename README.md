# processtensor

[![CI](https://github.com/jkfids/process-tensor/actions/workflows/ci.yml/badge.svg)](https://github.com/jkfids/process-tensor/actions/workflows/ci.yml)

A minimal numerical implementation of **process tensors**—the general
description of multi-time quantum processes with non-Markovian memory—together
with the quantum states and channels they act on.

A `k`-slot process tensor is a multilinear map taking an initial system state
`ρ_in` and a sequence of `k` *instruments* (CP maps representing operations
performed on the system at intermediate times) to the output state:

```math
\mathcal{T}_{0:k}[\rho_\mathrm{in}, (\mathcal{A}_0, \dots, \mathcal{A}_{k-1})]
= \rho_\mathrm{out}.
```

Every such map arises from unitary system-environment dynamics interleaved
with the instruments—a multi-time Stinespring dilation—which is how processes
are built here:

```math
\rho_\mathrm{out} = \mathrm{Tr}_E\big[\mathcal{U}_k\,
(\mathcal{A}_{k-1} \otimes \mathcal{I}_E)\, \mathcal{U}_{k-1} \cdots
\mathcal{U}_1\, (\mathcal{A}_0 \otimes \mathcal{I}_E)\, \mathcal{U}_0\,
(\rho_\mathrm{in} \otimes \sigma_E)\big],
```

where each `U_j` is a joint unitary on `H_S ⊗ H_E`, entering as the
conjugation `ρ ↦ U_j ρ U_j†`, and `σ_E` is the initial environment state.
Like a channel, a process tensor admits several equivalent representations.
Objects are stored here as fused-leg Liouville tensors, from which the others
follow—among them the Choi operator `Υ_0:k`, positive semidefinite and subject
to causal (containment) constraints.

## Features

The package divides into quantum *objects*, their *representations* and
*operations*, and *measures*, following the philosophy laid out in
[DESIGN.md](DESIGN.md).

**Objects**

- `QuantumState` – density matrices with subsystem structure.
- `QuantumChannel` – CP maps as superoperators, built from Kraus operators,
  unitaries, or a Stinespring dilation. Trace-decreasing maps are allowed,
  so this is also the instrument type.
- `ProcessTensor` – `k`-slot processes from `k+1` joint unitaries, with
  `apply(state, instruments)` for the multilinear action, `is_valid()` for
  complete positivity and the causal constraints, and `markov_product()`
  for the closest time-factorized process.

**Representations and operations**

- `representations` – conversions out of the stored fused-leg tensor,
  presently the Choi matrix (`.choi` / `from_choi`); Kraus, Stinespring, and
  chi-matrix forms will follow.
- `link_product` – composition of objects (Chiribella–D'Ariano–Perinotti),
  of which feeding a state through a channel, composing channels, and
  contracting a process with its instruments are special cases.

**Measures**

Object-agnostic functionals in `measures`, with convenience methods on the
classes: one function gives the spatial quantity for a state and the temporal
one for a process tensor.

- `entropy`, `purity`, `mutual_information` – von Neumann entropy, purity,
  and generalized quantum mutual information; for a process tensor `gqmi()`
  is `S(Υ ‖ Υ_Markov)`, the relative entropy to
  the product of single-time-step marginals.
- `negativity` – spatial entanglement of a state, or temporal entanglement
  (`temporal_negativity()`) witnessing genuinely quantum memory.
- `schmidt_rank`, `bond_entropy` – operator Schmidt data across a cut; for a
  process the rank is the minimal MPO bond dimension there.

## Installation

```bash
git clone https://github.com/jkfids/process-tensor
cd process-tensor
pip install -e .  # or `pip install -e '.[dev,examples]'`
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

pt.is_valid()  # True (completely positive + causally ordered)
pt.gqmi()  # 1.386... = 2 ln 2 (maximal temporal correlations)
pt.temporal_negativity()  # 0.5 (temporal entanglement: quantum memory)
pt.schmidt_rank()  # 4 (MPO bond dimension, saturating dE² = 4)

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
pre-commit install  # hooks run on commit
pytest  # unit tests
mypy  # type check
pre-commit run --all-files  # lint and format, as CI runs them
```

Tests, type checking, linting, and formatting run in CI on Python 3.12–3.14.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## License

MIT—see [LICENSE](LICENSE).
