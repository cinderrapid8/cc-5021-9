"""Decoder: strict .acmp parsing + playback sampling.

This path is independent from the encoder: it only sees the byte stream and
validates every field before use. Any corrupt length or illegal index raises
FormatError before any pose is produced.
"""

from bisect import bisect_right

from . import format as fmt
from .format import (
    ENC_CONST,
    ENC_KEYS,
    FLAG_LOOPS,
    MAGIC,
    PROP_COMPONENTS,
    PROP_ROTATION,
    VALID_BITS,
    VERSION,
    FormatError,
    Reader,
)
from .qmath import lerp, q_norm, q_slerp


class Joint:
    __slots__ = ("name", "parent", "bind_t", "bind_r", "bind_s")

    def __init__(self, name, parent, bind_t, bind_r, bind_s):
        self.name = name
        self.parent = parent
        self.bind_t = bind_t
        self.bind_r = bind_r
        self.bind_s = bind_s


class Track:
    __slots__ = ("joint", "prop", "encoding", "const_value", "bits",
                 "key_count", "mins", "maxs", "times", "data")

    def __init__(self):
        self.joint = -1
        self.prop = -1
        self.encoding = -1
        self.const_value = None
        self.bits = 0
        self.key_count = 0
        self.mins = None
        self.maxs = None
        self.times = None
        self.data = None

    def dequant(self, key_idx):
        comps = PROP_COMPONENTS[self.prop]
        levels = (1 << self.bits) - 1
        out = []
        base = key_idx * comps
        for c in range(comps):
            q = self.data[base + c]
            rng = self.maxs[c] - self.mins[c]
            out.append(self.mins[c] if rng <= 0.0 else self.mins[c] + q / levels * rng)
        return out


class DecodedAnim:
    def __init__(self):
        self.ticks_per_second = 0
        self.duration_ticks = 0
        self.loops = False
        self.joints = []
        self.tracks = []
        self.track_map = {}

    def duration_seconds(self):
        return self.duration_ticks / self.ticks_per_second


def decode(data):
    """Parse .acmp bytes into a DecodedAnim. Raises FormatError on any
    corruption; never returns a partially-valid animation."""
    r = Reader(data)
    if r.take(4) != MAGIC:
        raise FormatError("bad magic")
    if r.u16() != VERSION:
        raise FormatError("unsupported version")
    flags = r.u16()
    if flags & ~FLAG_LOOPS:
        raise FormatError(f"unknown flags {flags:#x}")

    anim = DecodedAnim()
    anim.loops = bool(flags & FLAG_LOOPS)
    anim.ticks_per_second = r.u32()
    if not (1 <= anim.ticks_per_second <= 100000):
        raise FormatError(f"invalid ticks_per_second {anim.ticks_per_second}")
    anim.duration_ticks = r.u32()
    if anim.duration_ticks == 0:
        raise FormatError("zero duration")

    joint_count = r.u16()
    if not (1 <= joint_count <= fmt.MAX_JOINTS):
        raise FormatError(f"invalid joint_count {joint_count}")
    if r.u16() != 0:
        raise FormatError("reserved header field must be 0")
    track_count = r.u32()
    if track_count > fmt.MAX_TRACKS:
        raise FormatError(f"invalid track_count {track_count}")

    for i in range(joint_count):
        name_len = r.u8()
        if not (1 <= name_len <= fmt.MAX_NAME_LEN):
            raise FormatError(f"joint {i}: invalid name length {name_len}")
        try:
            name = r.take(name_len).decode("utf-8")
        except UnicodeDecodeError:
            raise FormatError(f"joint {i}: name is not valid utf-8") from None
        parent = r.i16()
        if parent >= i or parent < -1:
            raise FormatError(f"joint {name}: invalid parent index {parent}")
        bind_t = r.f32s(3)
        bind_r = r.f32s(4)
        bind_s = r.f32s(3)
        if sum(x * x for x in bind_r) < 1e-12:
            raise FormatError(f"joint {name}: degenerate bind rotation")
        anim.joints.append(Joint(name, parent, bind_t, q_norm(bind_r), bind_s))

    seen = set()
    for _ in range(track_count):
        tr = Track()
        tr.joint = r.u16()
        if tr.joint >= joint_count:
            raise FormatError(f"track references invalid joint index {tr.joint}")
        tr.prop = r.u8()
        if tr.prop not in PROP_COMPONENTS:
            raise FormatError(f"invalid property id {tr.prop}")
        if (tr.joint, tr.prop) in seen:
            raise FormatError(f"duplicate track for joint {tr.joint} property {tr.prop}")
        seen.add((tr.joint, tr.prop))
        tr.encoding = r.u8()
        comps = PROP_COMPONENTS[tr.prop]

        if tr.encoding == ENC_CONST:
            tr.const_value = r.f32s(comps)
            if tr.prop == PROP_ROTATION:
                tr.const_value = q_norm(tr.const_value)
        elif tr.encoding == ENC_KEYS:
            tr.bits = r.u8()
            if tr.bits not in VALID_BITS:
                raise FormatError(f"invalid bits per component {tr.bits}")
            if r.u8() != 0:
                raise FormatError("reserved track field must be 0")
            tr.key_count = r.u32()
            if not (2 <= tr.key_count <= fmt.MAX_KEYS_PER_TRACK):
                raise FormatError(f"invalid key_count {tr.key_count}")
            # Reject corrupt lengths up front, before reading any payload.
            bytes_per_key = 4 + comps * (tr.bits // 8)
            need = tr.key_count * bytes_per_key + 2 * comps * 4
            if need > r.remaining():
                raise FormatError(
                    f"track payload overruns file: need {need} bytes, have {r.remaining()}"
                )
            tr.mins = r.f32s(comps)
            tr.maxs = r.f32s(comps)
            tr.times = [r.u32() for _ in range(tr.key_count)]
            for k in range(1, tr.key_count):
                if tr.times[k] <= tr.times[k - 1]:
                    raise FormatError("key times not strictly increasing")
            if tr.times[-1] > anim.duration_ticks:
                raise FormatError("key time past clip duration")
            nbytes = tr.key_count * comps * (tr.bits // 8)
            raw = r.take(nbytes)
            if tr.bits == 8:
                tr.data = list(raw)
            else:
                tr.data = [raw[i] | (raw[i + 1] << 8) for i in range(0, nbytes, 2)]
        else:
            raise FormatError(f"invalid encoding {tr.encoding}")
        anim.tracks.append(tr)

    if not r.at_end():
        raise FormatError(f"{r.remaining()} trailing bytes after last track")

    anim.track_map = {(t.joint, t.prop): t for t in anim.tracks}
    return anim


def track_value_at(tr, t_ticks):
    if tr.encoding == ENC_CONST:
        return list(tr.const_value)
    times = tr.times
    if t_ticks <= times[0]:
        v = tr.dequant(0)
        return q_norm(v) if tr.prop == PROP_ROTATION else v
    if t_ticks >= times[-1]:
        v = tr.dequant(tr.key_count - 1)
        return q_norm(v) if tr.prop == PROP_ROTATION else v
    i = bisect_right(times, t_ticks) - 1
    f = (t_ticks - times[i]) / (times[i + 1] - times[i])
    a = tr.dequant(i)
    b = tr.dequant(i + 1)
    if tr.prop == PROP_ROTATION:
        return q_slerp(q_norm(a), q_norm(b), f)
    return lerp(a, b, f)


def sample_pose(anim, t_ticks):
    """Full local pose (t, r, s) per joint at time t (ticks)."""
    if anim.loops and anim.duration_ticks > 0:
        t_ticks = t_ticks % anim.duration_ticks
    else:
        t_ticks = min(max(t_ticks, 0.0), float(anim.duration_ticks))
    pose = []
    for i, j in enumerate(anim.joints):
        t, r, s = list(j.bind_t), list(j.bind_r), list(j.bind_s)
        tr = anim.track_map.get((i, 0))
        if tr:
            t = track_value_at(tr, t_ticks)
        tr = anim.track_map.get((i, 1))
        if tr:
            r = track_value_at(tr, t_ticks)
        tr = anim.track_map.get((i, 2))
        if tr:
            s = track_value_at(tr, t_ticks)
        pose.append((t, r, s))
    return pose
