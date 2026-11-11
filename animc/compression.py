"""Compression module interface.

This composes three independent pieces:

  * ``error_metrics.ErrorBudget``  - the only source of acceptance tolerances
  * ``keypoints.select_keypoints`` - which authored keys survive
  * ``curve_codec.quantize``       - how surviving values are encoded

The interface requires both ``budget`` and ``bits``; no default tolerances are
defined here, so a caller cannot silently compress with an unstated accuracy.

Normalization that was historically folded into the encoder (quaternion
hemisphere signs and the exact loop seam) is performed by explicitly named
steps in this module. It is a data-shaping policy, kept separate from keypoint
selection and from error measurement, and it never mutates the budget.
"""

from . import curve_codec, keypoints
from .error_metrics import ErrorBudget
from .format import (
    ENC_CONST,
    ENC_KEYS,
    FLAG_LOOPS,
    MAGIC,
    MAX_NAME_LEN,
    PROP_NAMES,
    PROP_ROTATION,
    VERSION,
    Writer,
    VALID_BITS,
)
from .qmath import q_dot, q_neg, q_norm


def normalize_hemisphere(values):
    """Flip quaternion signs so consecutive keys share a hemisphere.

    q and -q are the same rotation; choosing the sign closest to the previous
    key keeps slerp on the short path.
    """
    out = [list(values[0])]
    for v in values[1:]:
        prev = out[-1]
        out.append(list(v) if q_dot(prev, v) >= 0.0 else q_neg(v))
    return out


def snap_loop_seam(times, values, prop, budget, loops, duration_ticks):
    """Explicit loop policy: if the two authored ends already fit the budget,
    store them exactly equal so the decoded loop seam is deterministic.

    This uses the budget for the decision but does not alter it; it is applied
    before selection and applies to the value list only.
    """
    if not (loops and len(times) >= 2
            and times[0] == 0 and times[-1] == duration_ticks):
        return values
    if budget.within(prop, values[0], values[-1]):
        values[-1] = list(values[0])
    return values


def compress(anim, budget, bits):
    """Encode a SourceAnim into .acmp bytes.

    Returns (data, stats) with per-track before/after key counts. ``budget``
    must be an ErrorBudget and ``bits`` a supported width; both are required
    explicitly.
    """
    if not isinstance(budget, ErrorBudget):
        raise TypeError("compress requires an explicit ErrorBudget")
    if bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}")

    w = Writer()
    w.raw(MAGIC)
    w.u16(VERSION)
    w.u16(FLAG_LOOPS if anim.loops else 0)
    w.u32(anim.ticks_per_second)
    w.u32(anim.duration_ticks)
    w.u16(len(anim.joints))
    w.u16(0)  # reserved
    w.u32(len(anim.tracks))

    for j in anim.joints:
        nb = j.name.encode("utf-8")
        if not (1 <= len(nb) <= MAX_NAME_LEN):
            raise ValueError(f"joint name too long: {j.name!r}")
        w.u8(len(nb))
        w.raw(nb)
        w.i16(j.parent)
        w.f32s(j.bind_t)
        w.f32s(q_norm(j.bind_r))
        w.f32s(j.bind_s)

    stats = []
    for tr in anim.tracks:
        prop = tr.prop
        times = list(tr.times)
        values = [list(v) for v in tr.values]

        if prop == PROP_ROTATION:
            values = normalize_hemisphere(values)

        values = snap_loop_seam(
            times, values, prop, budget, anim.loops, anim.duration_ticks)

        kept = keypoints.select_keypoints(times, values, prop, budget)
        ktimes = [times[i] for i in kept]
        kvalues = [values[i] for i in kept]

        # A track whose kept values all fit the budget around the first value
        # collapses to one const key (deterministic encoding).
        if len(ktimes) > 1 and keypoints.collapse_to_constant(kvalues, prop, budget):
            ktimes, kvalues = [ktimes[0]], [kvalues[0]]

        stats.append({
            "joint": anim.joints[tr.joint].name,
            "property": PROP_NAMES[prop],
            "keys_before": len(times),
            "keys_after": len(ktimes),
        })

        w.u16(tr.joint)
        w.u8(prop)
        if len(ktimes) == 1:
            w.u8(ENC_CONST)
            w.f32s(curve_codec.const_value(kvalues[0], prop))
            continue

        w.u8(ENC_KEYS)
        w.u8(bits)
        w.u8(0)  # reserved
        w.u32(len(ktimes))
        mins, maxs, qvals = curve_codec.quantize(kvalues, bits)
        w.f32s(mins)
        w.f32s(maxs)
        for t in ktimes:
            w.u32(t)
        if bits == 8:
            for qv in qvals:
                w.raw(bytes(qv))
        else:
            for qv in qvals:
                for q in qv:
                    w.u16(q)

    return w.bytes(), stats
