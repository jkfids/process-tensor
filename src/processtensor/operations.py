"""Operations that combine quantum objects.

The central operation is the *link product* of Chiribella, D'Ariano, and
Perinotti (arXiv:0904.4483), which composes quantum networks in the Choi
picture: for Choi operators sharing a space X,

    Y_1 * Y_2 = Tr_X[(Y_1 (x) I)(I (x) Y_2^T_X)].

Composing a state with a channel, two channels in sequence, or a process
tensor with its instruments are all special cases.
"""

from __future__ import annotations

import numpy as np

from .qtensor import QTensor


def link_product(x: QTensor, y: QTensor, legs_x: list[int], legs_y: list[int]) -> QTensor:
    """Link product of two quantum tensors over the given leg pairs.

    Contracts leg ``legs_x[i]`` of ``x`` with leg ``legs_y[i]`` of ``y``
    for each pair; the result carries the remaining legs of ``x`` followed
    by the remaining legs of ``y``.

    In the fused-leg convention this is a plain tensor contraction: the
    partial transpose in the Choi-picture definition (see module docstring)
    is absorbed by the leg fusion, since a fused leg already pairs the bra
    and ket indices that the transpose would swap.
    """
    if len(legs_x) != len(legs_y):
        raise ValueError("legs_x and legs_y must pair up one-to-one.")
    for lx, ly in zip(legs_x, legs_y, strict=True):
        if x.dims[lx] != y.dims[ly]:
            raise ValueError(
                f"Cannot link leg {lx} (dim {x.dims[lx]}) of {x!r} with "
                f"leg {ly} (dim {y.dims[ly]}) of {y!r}."
            )
    data = np.tensordot(x.data, y.data, axes=(legs_x, legs_y))
    dims = tuple(d for i, d in enumerate(x.dims) if i not in legs_x) + tuple(
        d for i, d in enumerate(y.dims) if i not in legs_y
    )
    return QTensor(data, dims)
