# Design notes

This document records the philosophy and conventions of `processtensor`. It
is the canonical reference for both; the README carries only a summary.

## Purpose

`processtensor` is a minimal numerical implementation of the process tensor
formalism: multi-time quantum processes together with the states and
channels they act on. The package is intended as a small, readable, and
reusable core—partly pedagogical in aim—rather than a comprehensive
framework; a comprehensive package is planned as a separate project. Its
only runtime dependency is numpy. When a design choice trades generality or
performance against readability, readability wins.

## Philosophy

Much of quantum information theory reduces to three kinds of things, and
the package is organized around exactly these:

1. **Quantum objects.** States, channels, and process tensors. Maps
   between objects are themselves quantum objects: a channel is a map on
   states, and a process tensor is a higher-order map taking an initial
   state and a sequence of instruments to an output state. All objects
   share one storage format (Sec. "Mathematical conventions") and one base
   class.

2. **Representations and operations.** Every object admits equivalent
   representations—Choi, superoperator, and in due course Kraus,
   Stinespring, and chi-matrix forms—between which one converts without
   changing the object. Objects compose through operations, canonically
   the link product, of which state-through-channel application, channel
   composition, and process tensor contraction are special cases.

3. **Measures.** Functionals that take a quantum object and return a
   useful quantity: entropy, negativity, mutual information, operator
   Schmidt data. A measure is agnostic to the specific object; the same
   negativity function yields the spatial entanglement of a bipartite
   state or the temporal entanglement of a process tensor, depending only
   on what it is handed.

## Architecture

The module layout mirrors the philosophy one-to-one:

- `qtensor.py`, `quantumstate.py`, `quantumchannel.py`,
  `processtensor.py` – the objects. `QTensor` holds structural code only
  (data, dimensions, leg operations, validity checks); subclasses fix the
  physical meaning of the legs and add thin convenience methods that
  delegate to the function modules.
- `representations.py` – conversions between the canonical storage format
  and other representations.
- `operations.py` – compositions of objects, presently the link product.
- `measures.py` – scalar functionals of any quantum object.
- `utils.py` – numerical primitives (vectorization, Pauli matrices,
  entropies of plain matrices).

New code goes where its category dictates: a new quantity belongs in
`measures.py`, a new conversion in `representations.py`, a new composition
in `operations.py`. Classes never carry quantity or conversion logic;
they hold structure and delegation.

## Mathematical conventions

Leg ordering is the principal correctness risk in this codebase (an
earlier implementation silently built processes time-reversed), so the
conventions are fixed here once and tested against analytically known
values.

- **Vectorization** is column-stacking: `vec(ρ)[j·d + i] = ρ[i, j]`. The
  superoperator of `ρ → UρU†` is `conj(U) ⊗ U`.
- **Fused legs.** Every object stores a tensor with one fused Liouville
  index per physical wire, of dimension `d²` with the bra index major:
  unfusing a leg by `reshape(d, d)` yields axes `(bra, ket)`.
- **Leg order.** States carry one leg per subsystem; channels legs
  `(in, out)`; a k-slot process tensor legs `(in₀, out₀, …, in_k, out_k)`.
  Time increases with leg index, and inputs precede outputs within a time
  step.
- **Subsystem order** is big-endian: subsystem 0 is the leftmost Kronecker
  factor, i.e. the most significant block of the composite index. Thus
  `pauli("XZ") = X ⊗ Z` acts with X on subsystem 0.
- **System-environment order.** Joint operators act on `H_S ⊗ H_E`
  (system major).
- **Superoperator storage.** `data[a, b]` maps fused input index `a` to
  fused output index `b`; this is the transpose of the conventional
  superoperator `S`, and application reads `vec(ρ) @ data`.
- **Choi normalization.** Choi matrices are unnormalized, with
  `Tr Υ = d_S^(k+1)` for a valid process; measures normalize to unit
  trace internally.
- **Tolerance.** Numerical comparisons default to `TOL = 1e-8`. Deciding
  whether an eigenvalue is zero is a separate question with a separate
  constant, `SPECTRAL_TOL = 1e-12`: a valid density matrix can carry
  genuine eigenvalues far below the comparison tolerance, and rounding
  those to zero would misreport the support of a spectrum.

## Language and style

- US spelling. Em dashes appear only in prose and are set without
  surrounding spaces; lists and comments use spaced en dashes as
  separators.
- Code lines are at most 100 characters, docstring and comment lines at
  most 80 (both enforced by ruff).
- Docstrings are concise prose stating the mathematics and the applicable
  conventions, using RST markup (double backticks, `:func:` references)
  and no Args/Returns boilerplate; module docstrings state the module's
  scope. Every public member carries one, opening with a one-line summary
  followed by a blank line (ruff `D`, pep257 convention).
- Comments record only constraints the code cannot express itself.
- Functions are annotated; the package ships `py.typed` and is checked
  with mypy. Arrays are annotated as bare `np.ndarray`.
- All text is self-contained: definitions are stated in place, never
  deferred to a paper. Citations of the literature (as for the link
  product) may supplement a definition but must not substitute for it.
  Prefer the terminology of the literature (link product, Choi
  representation, generalized quantum mutual information).
