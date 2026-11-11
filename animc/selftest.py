"""Self-test: roundtrip correctness, rotation hemisphere handling, loop seam
exactness, const/short-track encodings, and rejection of corrupt files
(bad lengths, illegal indices) before any pose is produced."""

import struct

from . import samplegen
from .decoder import decode, sample_pose, track_value_at
from .encoder import encode
from .format import (
    ENC_CONST,
    ENC_KEYS,
    FormatError,
    PROP_ROTATION,
)
from .qmath import q_angle_deg, q_norm
from .source import parse_source


def _expect_format_error(data, what, failures):
    try:
        decode(data)
    except FormatError:
        return
    except Exception as e:  # wrong exception type still counts as failure
        failures.append(f"{what}: raised {type(e).__name__} instead of FormatError")
        return
    failures.append(f"{what}: corrupt file was ACCEPTED")


def _find_track_offset(data, joint_count, track_idx):
    """Walk the container with the same layout rules to locate a track's
    header offset (test-only helper)."""
    pos = 4 + 2 + 2 + 4 + 4 + 2 + 2 + 4
    for _ in range(joint_count):
        name_len = data[pos]
        pos += 1 + name_len + 2 + 40
    for _ in range(track_idx):
        pos += 4  # joint, prop, encoding
        enc = data[pos - 1]
        if enc == ENC_CONST:
            prop = data[pos - 2]
            pos += 4 * (4 if prop == PROP_ROTATION else 3)
        else:
            prop = data[pos - 2]
            comps = 4 if prop == PROP_ROTATION else 3
            bits = data[pos]
            kc = struct.unpack_from("<I", data, pos + 2)[0]
            pos += 2 + 4 + 8 * comps + kc * 4 + kc * comps * (bits // 8)
    return pos


def run():
    failures = []

    def check(cond, what):
        if not cond:
            failures.append(what)

    clips = {g().get("name"): g() for g in samplegen.GENERATORS}

    # --- roundtrip + const/short-track encodings -----------------------------
    for name, doc in clips.items():
        src = parse_source(doc)
        data, stats = encode(src)
        dec = decode(data)
        check(len(dec.joints) == len(src.joints), f"{name}: joint count mismatch")
        check(len(dec.tracks) == len(src.tracks), f"{name}: track count mismatch")
        # pose sampling must work across the whole clip
        for k in range(0, src.duration_ticks + 1, 7):
            pose = sample_pose(dec, float(k))
            check(len(pose) == len(dec.joints), f"{name}: bad pose at {k}")

    src = parse_source(clips["slow_head_turn"])
    data, _ = encode(src)
    dec = decode(data)
    enc_by_prop = {}
    for tr in dec.tracks:
        jname = dec.joints[tr.joint].name
        enc_by_prop[(jname, tr.prop)] = tr.encoding
    check(enc_by_prop[("spine", 0)] == ENC_CONST, "dense constant track not encoded as const")
    check(enc_by_prop[("neck", 0)] == ENC_CONST, "single-key track not encoded as const")
    check(enc_by_prop[("head", 2)] == ENC_CONST, "constant scale track not encoded as const")
    check(enc_by_prop[("head", 1)] == ENC_KEYS, "head rotation should be key-encoded")

    # const value fidelity
    for tr in dec.tracks:
        if dec.joints[tr.joint].name == "neck" and tr.prop == 0:
            v = track_value_at(tr, 0.0)
            check(abs(v[1] - 0.35) < 1e-6, "const track value mismatch")

    # --- loop seam: decoded first frame == decoded last frame exactly -------
    check(dec.loops, "slow_head_turn should be marked looping")
    for tr in dec.tracks:
        a = track_value_at(tr, 0.0)
        b = track_value_at(tr, float(dec.duration_ticks))
        check(a == b, f"loop seam not exact on joint {dec.joints[tr.joint].name} prop {tr.prop}")

    # --- hemisphere handling on the sign-flipped half-turn clip -------------
    src = parse_source(clips["half_turn"])
    data, _ = encode(src)
    dec = decode(data)
    prev = None
    worst_step = 0.0
    for k in range(src.duration_ticks + 1):
        pose = sample_pose(dec, float(k))
        q = pose[1][3] if False else pose[1][1]  # hips rotation
        if prev is not None:
            worst_step = max(worst_step, q_angle_deg(prev, q))
        prev = q
    check(worst_step < 30.0, f"decoded rotation discontinuous: {worst_step:.2f} deg/frame step")
    # and the decoded end pose must match the authored end pose (~178 deg)
    pose0 = sample_pose(dec, 0.0)[1][1]
    pose1 = sample_pose(dec, float(src.duration_ticks))[1][1]
    check(170.0 < q_angle_deg(pose0, pose1) <= 180.0,
          "half-turn amplitude lost after decode")

    # --- corrupt files must be rejected --------------------------------------
    good = data
    _expect_format_error(b"\x00" * 4 + good[4:], "bad magic", failures)
    _expect_format_error(good[:10], "truncated header", failures)
    _expect_format_error(good[: len(good) // 2], "truncated payload", failures)
    _expect_format_error(good + b"\x00", "trailing garbage", failures)

    n_joints = len(dec.joints)
    # header: magic(4) version(2) flags(2) tps(4) duration(4) joint_count(2)
    #         reserved(2) track_count(4) -> joints begin at offset 24
    bad = bytearray(good)
    struct.pack_into("<H", bad, 16, 5000)
    _expect_format_error(bytes(bad), "joint_count overflow", failures)
    # track_count absurd
    bad = bytearray(good)
    struct.pack_into("<I", bad, 20, 999999)
    _expect_format_error(bytes(bad), "track_count overflow", failures)

    # locate first key-encoded track and corrupt its fields
    t_idx = next(i for i, tr in enumerate(dec.tracks) if tr.encoding == ENC_KEYS)
    off = _find_track_offset(good, n_joints, t_idx)

    bad = bytearray(good)
    struct.pack_into("<H", bad, off, n_joints + 7)  # joint index out of range
    _expect_format_error(bytes(bad), "illegal joint index", failures)

    bad = bytearray(good)
    bad[off + 2] = 9  # property id
    _expect_format_error(bytes(bad), "illegal property id", failures)

    bad = bytearray(good)
    bad[off + 3] = 9  # encoding
    _expect_format_error(bytes(bad), "illegal encoding id", failures)

    bad = bytearray(good)
    bad[off + 4] = 12  # bits
    _expect_format_error(bytes(bad), "illegal bits", failures)

    bad = bytearray(good)
    struct.pack_into("<I", bad, off + 6, 10 ** 8)  # key_count absurd
    _expect_format_error(bytes(bad), "corrupt key_count length", failures)

    # non-monotonic key times: find times array and swap first two
    tr = dec.tracks[t_idx]
    comps = 4 if tr.prop == PROP_ROTATION else 3
    times_off = off + 4 + 2 + 4 + 8 * comps
    bad = bytearray(good)
    t0 = struct.unpack_from("<I", good, times_off)[0]
    t1 = struct.unpack_from("<I", good, times_off + 4)[0]
    struct.pack_into("<I", bad, times_off, t1)
    struct.pack_into("<I", bad, times_off + 4, t0)
    _expect_format_error(bytes(bad), "non-monotonic key times", failures)

    # corrupt a parent index inside the joint table (joints start at 24)
    bad = bytearray(good)
    joff = 24
    name_len = good[joff]
    parent_off = joff + 1 + name_len
    struct.pack_into("<h", bad, parent_off, 0)  # joint 0 parented to itself
    _expect_format_error(bytes(bad), "illegal parent index", failures)

    # --- 8-bit quantization path also works ----------------------------------
    data8, _ = encode(parse_source(clips["fast_wave"]), bits=8)
    dec8 = decode(data8)
    check(all(t.bits == 8 or t.encoding == ENC_CONST for t in dec8.tracks),
          "8-bit encode did not produce 8-bit tracks")
    sample_pose(dec8, 30.0)

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("selftest: all checks passed")
    return 0
