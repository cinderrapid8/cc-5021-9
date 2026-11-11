"""Curve encoding: turn selected key values into the container's value form.

This module owns quantization only. It does not select keys, measure errors,
or know about acceptance tolerances.
"""

from .format import PROP_ROTATION
from .qmath import q_norm


def quantize(values, bits):
    """Per-component affine quantization over the key set's own range.

    Returns (mins, maxs, integer codes). Range-zero components encode as 0 so
    decoding returns the exact minimum.
    """
    comps = len(values[0])
    mins = [min(v[c] for v in values) for c in range(comps)]
    maxs = [max(v[c] for v in values) for c in range(comps)]
    levels = (1 << bits) - 1
    out = []
    for v in values:
        qv = []
        for c in range(comps):
            rng = maxs[c] - mins[c]
            qv.append(0 if rng <= 0.0 else int(round((v[c] - mins[c]) / rng * levels)))
        out.append(qv)
    return mins, maxs, out


def const_value(value, prop):
    """Value stored for a constant track (rotation is unit-normalized)."""
    return q_norm(value) if prop == PROP_ROTATION else list(value)
