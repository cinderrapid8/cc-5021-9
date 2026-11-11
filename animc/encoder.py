"""Encoder: keyframe reduction + quantization + .acmp packaging."""

from . import format as fmt
from .format import (
    ENC_CONST,
    ENC_KEYS,
    FLAG_LOOPS,
    MAGIC,
    PROP_COMPONENTS,
    PROP_ROTATION,
    VERSION,
    Writer,
)
from .qmath import q_angle_deg, q_dot, q_neg, q_norm, q_slerp, lerp, v_dist


def _hemisphere_fix(values):
    """Flip quaternion signs so consecutive keys stay in the same hemisphere
    (q and -q are the same rotation; pick the sign closest to the previous
    key so interpolation takes the short path)."""
    out = [list(values[0])]
    for v in values[1:]:
        prev = out[-1]
        out.append(list(v) if q_dot(prev, v) >= 0.0 else q_neg(v))
    return out


def _interp_error(prop, a, b, f, ref, tol):
    if prop == PROP_ROTATION:
        return q_angle_deg(q_slerp(a, b, f), ref) > tol
    return v_dist(lerp(a, b, f), ref) > tol


def _segment_ok(times, values, i, j, prop, tol):
    """True if every original key strictly between i and j is within tol of
    the i..j interpolation."""
    span = times[j] - times[i]
    a, b = values[i], values[j]
    for k in range(i + 1, j):
        f = (times[k] - times[i]) / span
        if _interp_error(prop, a, b, f, values[k], tol):
            return False
    return True


def reduce_track(times, values, prop, tol):
    """Greedy maximal-segment decimation. Guarantees that at every original
    key time the reduced curve stays within tol of the original. First and
    last keys are always kept (deterministic loop seam)."""
    n = len(times)
    if n <= 2:
        return list(range(n))
    kept = [0]
    i = 0
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not _segment_ok(times, values, i, j, prop, tol):
            j -= 1
        kept.append(j)
        i = j
    return kept


def _quantize(values, bits):
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


def encode(anim, pos_tol=0.01, rot_tol_deg=0.25, scale_tol=0.005, bits=16):
    """Encode a SourceAnim into .acmp bytes.

    Returns (data, stats) where stats has per-track before/after key counts.
    """
    if bits not in fmt.VALID_BITS:
        raise ValueError(f"bits must be one of {fmt.VALID_BITS}")
    tols = {0: pos_tol, 1: rot_tol_deg, 2: scale_tol}

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
        if not (1 <= len(nb) <= fmt.MAX_NAME_LEN):
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
        comps = PROP_COMPONENTS[prop]
        times = list(tr.times)
        values = [list(v) for v in tr.values]

        if prop == PROP_ROTATION:
            values = _hemisphere_fix(values)

        # Deterministic loop seam: if the clip loops and the two ends are
        # already within tolerance, snap the last key exactly to the first.
        if anim.loops and len(times) >= 2 and times[0] == 0 and times[-1] == anim.duration_ticks:
            close = (
                q_angle_deg(values[0], values[-1]) <= rot_tol_deg
                if prop == PROP_ROTATION
                else v_dist(values[0], values[-1]) <= tols[prop]
            )
            if close:
                values[-1] = list(values[0])

        kept = reduce_track(times, values, prop, tols[prop])
        ktimes = [times[i] for i in kept]
        kvalues = [values[i] for i in kept]

        # Constant track: every kept key within tolerance of the first ->
        # collapse to a single const value (deterministic encoding).
        if len(ktimes) > 1:
            if prop == PROP_ROTATION:
                const = all(q_angle_deg(kvalues[0], v) <= tols[prop] for v in kvalues[1:])
            else:
                const = all(v_dist(kvalues[0], v) <= tols[prop] for v in kvalues[1:])
            if const:
                ktimes, kvalues = [ktimes[0]], [kvalues[0]]

        stats.append({
            "joint": anim.joints[tr.joint].name,
            "property": fmt.PROP_NAMES[prop],
            "keys_before": len(times),
            "keys_after": len(ktimes),
        })

        w.u16(tr.joint)
        w.u8(prop)
        if len(ktimes) == 1:
            # Constant / very short track: deterministic const encoding.
            w.u8(ENC_CONST)
            v = q_norm(kvalues[0]) if prop == PROP_ROTATION else kvalues[0]
            w.f32s(v)
            continue

        w.u8(ENC_KEYS)
        w.u8(bits)
        w.u8(0)  # reserved
        w.u32(len(ktimes))
        mins, maxs, qvals = _quantize(kvalues, bits)
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


def raw_f32_size(anim):
    """Size of a canonical uncompressed f32 dump, used as the honest
    baseline for the compression ratio (no ZIP trickery)."""
    size = 4 + 2 + 2 + 4 + 4 + 2 + 2 + 4  # same global header
    for j in anim.joints:
        size += 1 + len(j.name.encode("utf-8")) + 2 + 40
    for tr in anim.tracks:
        comps = PROP_COMPONENTS[tr.prop]
        size += 2 + 1 + 1 + 4  # joint, property, reserved, key_count
        size += len(tr.times) * 4 + len(tr.times) * comps * 4
    return size
