"""Unified curve and pose sampling.

A single interpolation implementation serves both the editing preview (source
JSON keys) and decompression verification (decoded, dequantized keys). Callers
adapt their stored values into a :class:`Curve` view; this module owns
bisection, endpoint clamping, lerp/slerp selection and rotation normalization
rules. Keeping it in one place means preview and verification cannot drift.

``Curve.value_at`` must return values in sampling space: rotations unit-length
normalized, other properties unchanged.
"""

from bisect import bisect_right

from .format import ENC_CONST, PROP_ROTATION
from .qmath import lerp, q_norm, q_slerp


class Curve:
    """Read-only view over strictly increasing key times."""

    __slots__ = ("prop", "times", "_read")

    def __init__(self, prop, times, read):
        self.prop = prop
        self.times = times
        self._read = read

    def value_at(self, index):
        return self._read(index)

    def sample(self, t_ticks):
        times = self.times
        if t_ticks <= times[0]:
            return list(self.value_at(0))
        if t_ticks >= times[-1]:
            return list(self.value_at(len(times) - 1))
        i = bisect_right(times, t_ticks) - 1
        f = (t_ticks - times[i]) / (times[i + 1] - times[i])
        a = self.value_at(i)
        b = self.value_at(i + 1)
        if self.prop == PROP_ROTATION:
            return q_slerp(a, b, f)
        return lerp(a, b, f)


class ConstantCurve:
    """Single-value curve used by const-encoded tracks."""

    __slots__ = ("prop", "value")

    def __init__(self, prop, value):
        self.prop = prop
        self.value = value

    def sample(self, t_ticks):
        return list(self.value)


def sample_keys(times, values, prop, t_ticks):
    """Sample authored key values directly through the shared sampler."""
    return Curve(prop, times, values.__getitem__).sample(t_ticks)


def source_curve(track):
    """Curve view over a source module Track (values already normalized)."""
    return Curve(track.prop, track.times, track.values.__getitem__)


def decoded_curve(track):
    """Curve view over a decoder module Track, normalizing rotation on read."""
    if track.encoding == ENC_CONST:
        return ConstantCurve(track.prop, track.const_value)
    if track.prop == PROP_ROTATION:
        return Curve(track.prop, track.times, lambda i: q_norm(track.dequant(i)))
    return Curve(track.prop, track.times, track.dequant)


def resolve_time(t_ticks, loops, duration_ticks):
    """Apply the playback clock rule: wrap for looping clips, clamp otherwise."""
    if loops and duration_ticks > 0:
        return t_ticks % duration_ticks
    return min(max(t_ticks, 0.0), float(duration_ticks))


def sample_pose(joints, curve_for, t_ticks):
    """Full local pose (t, r, s) per joint.

    ``curve_for(joint_index, prop)`` returns a Curve/ConstantCurve or None.
    Time resolution is the caller's responsibility so preview (no wrapping)
    and playback (wraps/clamps) can share this body without changing behavior.
    """
    pose = []
    for i, j in enumerate(joints):
        t = list(j.bind_t)
        r = list(j.bind_r)
        s = list(j.bind_s)
        c = curve_for(i, 0)
        if c is not None:
            t = c.sample(t_ticks)
        c = curve_for(i, 1)
        if c is not None:
            r = c.sample(t_ticks)
        c = curve_for(i, 2)
        if c is not None:
            s = c.sample(t_ticks)
        pose.append((t, r, s))
    return pose
