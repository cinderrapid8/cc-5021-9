"""Procedurally generated validation clips:

  slow_head_turn - slow sinusoidal head yaw, looping; includes a constant
                   multi-key track and a single-key (very short) track.
  fast_wave      - fast 3.5 Hz arm wave, dense 60 fps keys.
  long_chain     - 12-joint chain, +-3 deg per joint; tiny root errors
                   accumulate into large tip deviation.
  half_turn      - near-180-degree turns with deliberately sign-flipped
                   authored quaternions (q == -q) to test hemisphere handling.
"""

import json
import math
import random

from .qmath import q_from_axis_angle


def _joint(name, parent, t, r=None, s=None):
    return {
        "name": name,
        "parent": parent,
        "bind_t": list(t),
        "bind_r": list(r) if r else [0.0, 0.0, 0.0, 1.0],
        "bind_s": list(s) if s else [1.0, 1.0, 1.0],
    }


def _track(joint, prop, times, values):
    return {"joint": joint, "property": prop,
            "times": list(times),
            "values": [[round(c, 6) for c in v] for v in values]}


def _yaw(deg):
    return q_from_axis_angle([0.0, 1.0, 0.0], math.radians(deg))


def _roll(deg):
    return q_from_axis_angle([0.0, 0.0, 1.0], math.radians(deg))


def slow_head_turn():
    tps, dur = 30, 120  # 4 s, loops
    times = list(range(dur + 1))
    joints = [
        _joint("root", -1, (0, 0, 0)),
        _joint("spine", 0, (0, 0.45, 0)),
        _joint("neck", 1, (0, 0.35, 0)),
        _joint("head", 2, (0, 0.22, 0)),
        _joint("head_tip", 3, (0, 0.2, 0.16)),  # off the yaw axis so head
                                                # rotation moves it in world space
    ]
    tracks = [
        _track("head", "rotation", times,
               [_yaw(35.0 * math.sin(2 * math.pi * k / dur)) for k in times]),
        _track("neck", "rotation", times,
               [_yaw(8.0 * math.sin(2 * math.pi * k / dur + 0.5)) for k in times]),
        # constant track authored densely -> must collapse to const encoding
        _track("spine", "translation", times, [[0.0, 0.45, 0.0] for _ in times]),
        # very short track: a single key
        _track("neck", "translation", [0], [[0.0, 0.35, 0.0]]),
        # constant scale track
        _track("head", "scale", times, [[1.0, 1.0, 1.0] for _ in times]),
    ]
    return {"name": "slow_head_turn", "ticks_per_second": tps,
            "duration_ticks": dur, "loops": True,
            "joints": joints, "tracks": tracks, "end_joints": ["head_tip"]}


def fast_wave():
    tps, dur = 60, 120  # 2 s
    times = list(range(dur + 1))
    joints = [
        _joint("root", -1, (0, 0, 0)),
        _joint("shoulder", 0, (0, 1.5, 0)),
        _joint("elbow", 1, (0.30, 0, 0)),
        _joint("wrist", 2, (0.28, 0, 0)),
        _joint("finger_tip", 3, (0.12, 0, 0)),
    ]
    hz = 3.5
    tracks = [
        _track("shoulder", "rotation", times,
               [_roll(-20.0 + 50.0 * k / dur) for k in times]),
        _track("elbow", "rotation", times,
               [_roll(30.0 + 40.0 * math.sin(2 * math.pi * hz * k / tps)) for k in times]),
        _track("wrist", "rotation", times,
               [_roll(15.0 * math.sin(2 * math.pi * hz * k / tps + 1.2)) for k in times]),
    ]
    return {"name": "fast_wave", "ticks_per_second": tps,
            "duration_ticks": dur, "loops": False,
            "joints": joints, "tracks": tracks, "end_joints": ["finger_tip"]}


def long_chain():
    tps, dur = 30, 150  # 5 s, loops
    n = 12
    times = list(range(dur + 1))
    joints = [_joint("j0", -1, (0, 0, 0))]
    for i in range(1, n):
        joints.append(_joint(f"j{i}", i - 1, (0, 0.2, 0)))
    tracks = []
    for i in range(n):
        tracks.append(_track(
            f"j{i}", "rotation", times,
            [_roll(3.0 * math.sin(2 * math.pi * k / dur + i * 0.35)) for k in times]))
    return {"name": "long_chain", "ticks_per_second": tps,
            "duration_ticks": dur, "loops": True,
            "joints": joints, "tracks": tracks, "end_joints": [f"j{n - 1}"]}


def half_turn():
    tps, dur = 30, 60  # 2 s
    times = list(range(dur + 1))
    rng = random.Random(20261111)

    def smooth(x):
        return x * x * (3.0 - 2.0 * x)

    def messy_yaw(deg_at):
        vals = []
        for k in times:
            q = _yaw(deg_at(smooth(k / dur)))
            if rng.random() < 0.5:  # simulate messy DCC export: q == -q flips
                q = [-c for c in q]
            vals.append(q)
        return vals

    joints = [
        _joint("root", -1, (0, 0, 0)),
        _joint("hips", 0, (0, 1.0, 0)),
        _joint("torso", 1, (0, 0.4, 0)),
        _joint("reach_tip", 2, (0.35, 0.5, 0)),  # off the yaw axis
    ]
    tracks = [
        _track("hips", "rotation", times, messy_yaw(lambda s: 178.0 * s)),
        # counter-rotation that does NOT cancel the hips, so the tip sweeps
        _track("torso", "rotation", times, messy_yaw(lambda s: -35.0 * s)),
    ]
    return {"name": "half_turn", "ticks_per_second": tps,
            "duration_ticks": dur, "loops": False,
            "joints": joints, "tracks": tracks, "end_joints": ["reach_tip"]}


GENERATORS = [slow_head_turn, fast_wave, long_chain, half_turn]


def write_samples(out_dir):
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for gen in GENERATORS:
        doc = gen()
        path = os.path.join(out_dir, doc["name"] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
        paths.append(path)
    return paths
