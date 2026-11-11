#!/usr/bin/env python3
"""Differential evidence for the compression-module refactor.

Freezes the pre-refactor external behavior of curve compression and checks the
post-refactor implementation against it. Equivalence is bounded by:

  * exact .acmp bytes (sha256 + length) for the authored samples at 16 and
    8 bits, and against the committed assets in ``out/``;
  * the reconstructed trajectory at every tick (local pose + FK world points),
    digested over the whole clip;
  * rotation wraparound (half_turn) and narrow peaks (fast_wave) as the
    targeted before/after probes;
  * the legacy ``encoder.encode`` adapter staying byte-identical to the new
    explicit ``compression.compress`` interface.

Run from the repository root:

    python3 tests/verify_equivalence.py
"""

import hashlib
import inspect
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from animc import sampling
from animc.compression import compress
from animc.decoder import decode, sample_pose, track_value_at
from animc.encoder import encode
from animc.error_metrics import ROTATION, TRANSLATION, ErrorBudget
from animc.evaluate import fk_world
from animc.qmath import m_point, q_angle_deg
from animc.samplegen import GENERATORS
from animc.source import parse_source, sample_pose as source_pose
from animc.source import sample_track_values

BUDGET = ErrorBudget(pos=0.01, rot_deg=0.25, scale=0.005)

# Captured from the pre-refactor implementation on 2026-11-11.
FROZEN = {
    "fast_wave": {
        "bits16": {"bytes": 3049, "keys_before": 363, "keys_after": 221,
                   "sha256": "61ea0c17cf942e77d4a221b6700358b198183fa4cd7716d15ef684122269143c",
                   "trajectory_sha256": "06dc9852711b76c1ac295588edb9aaee7b883ee31ef5012f911df967b104e649",
                   "trajectory_ticks": 121},
        "bits8": {"bytes": 2165, "keys_before": 363, "keys_after": 221,
                  "sha256": "33eb294c36c6084e95f7d12c99e588677528fb0173e2a5f7faa4065b20a3cd13",
                  "trajectory_sha256": "9191caec7936245dd3dbde5590b0b7ec158346e07de4425b684c97e6ea9349e1",
                  "trajectory_ticks": 121},
        "max_rotation_step_deg": 0.41666637011839397,
        "max_rotation_step_tick": 71,
        "narrow_peaks": [
            {"deg": 69.99996073944412, "tick": 90},
            {"deg": 69.94509540854405, "tick": 73},
            {"deg": 69.94509540854405, "tick": 107},
            {"deg": 69.78126529315546, "tick": 4},
            {"deg": 69.78126529315546, "tick": 56},
            {"deg": 69.50729837833842, "tick": 21},
        ],
    },
    "half_turn": {
        "bits16": {"bytes": 614, "keys_before": 122, "keys_after": 26,
                   "sha256": "daef4bd71ec88652024711483295ae07118459dccd1f6a9f1e7be915f284c3ec",
                   "trajectory_sha256": "4b43c51b79d9fa0d0a1e604d16cfac138d3fa08f733e3e08c0cca334f7ca8b74",
                   "trajectory_ticks": 61},
        "bits8": {"bytes": 510, "keys_before": 122, "keys_after": 26,
                  "sha256": "cf8dbbab797b597c2cd1293b89e7f677fe497947b8e0b287f3b2d89f853d0d82",
                  "trajectory_sha256": "dafbaad6b2634661a3d1f50daf6f959d1e7cbe40f9e5f8e32ac4aa2b178c1063",
                  "trajectory_ticks": 61},
        "max_rotation_step_deg": 4.398907570468877,
        "max_rotation_step_tick": 28,
        "narrow_peaks": [],
    },
    "long_chain": {
        "bits16": {"bytes": 2126, "keys_before": 1812, "keys_after": 88,
                   "sha256": "022ca803d31011678cb49f41b3376d8712ea239d45647cc49dd1e906e4e0f279",
                   "trajectory_sha256": "2ff88b9bc4790a2086388515be2f3cd8844b4bc6d49ac1420ddc500e0411a792",
                   "trajectory_ticks": 151},
        "bits8": {"bytes": 1774, "keys_before": 1812, "keys_after": 88,
                  "sha256": "8c565fe87772726b12ff64fd7b7686ee4b3b336401555b77e38991d2ebfbe553",
                  "trajectory_sha256": "65b977a9a5193370b0f1255782770f81dd9f77898f608f0f9d5a2c5417deb3f9",
                  "trajectory_ticks": 151},
        "max_rotation_step_deg": 0.1235297689672385,
        "max_rotation_step_tick": 143,
        "narrow_peaks": [
            {"deg": 2.866692546350842, "tick": 22},
            {"deg": 2.839067396493163, "tick": 112},
        ],
    },
    "slow_head_turn": {
        "bits16": {"bytes": 816, "keys_before": 485, "keys_after": 38,
                   "sha256": "50fe6fcb71a3f9764ae1f0c9148c25b9443a6b4cedce0b6c0bb9131bc617ac9b",
                   "trajectory_sha256": "782d94646e54a7bdac09d58ae98c48111da1da4781b9484ada3fad3a3888dbfd",
                   "trajectory_ticks": 121},
        "bits8": {"bytes": 676, "keys_before": 485, "keys_after": 38,
                  "sha256": "791b403e201b924afbc285ccf23245ef81eca6c47e6f298cddedb0c04b66e628",
                  "trajectory_sha256": "b300883eff107fcbf07461c777c97ecfa5af82cdb21e11b102ba0670bf1b1722",
                  "trajectory_ticks": 121},
        "max_rotation_step_deg": 0.0,
        "max_rotation_step_tick": 0,
        "narrow_peaks": [
            {"deg": 7.997766874861176, "tick": 20},
            {"deg": 7.928843566569233, "tick": 83},
        ],
    },
}

GEN_BY_NAME = {g.__name__: g for g in GENERATORS}


def _canon(v):
    return "%.17g" % float(v)


def _trajectory_digest(dec, duration_ticks):
    rows = []
    for t in range(duration_ticks + 1):
        pose = sample_pose(dec, float(t))
        worlds = fk_world(dec.joints, pose)
        rows.append({
            "t": t,
            "pose": [[[_canon(c) for c in comp] for comp in joint] for joint in pose],
            "world": [[_canon(c) for c in m_point(worlds[i], [0.0, 0.0, 0.0])]
                      for i in range(len(dec.joints))],
        })
    blob = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest(), len(rows)


def _rotation_scan(dec, duration_ticks):
    max_step = 0.0
    max_tick = 0
    prev = None
    for t in range(duration_ticks + 1):
        pose = sample_pose(dec, float(t))
        rot = pose[1][1] if len(pose) > 1 else pose[0][1]
        if prev is not None:
            step = q_angle_deg(prev, rot)
            if step > max_step:
                max_step, max_tick = step, t
        prev = rot
    return max_step, max_tick


def _narrow_peaks(dec, duration_ticks):
    tr = dec.tracks[1] if len(dec.tracks) > 1 else dec.tracks[0]
    angles = [q_angle_deg(track_value_at(tr, float(t)), [0.0, 0.0, 0.0, 1.0])
              for t in range(duration_ticks + 1)]
    peaks = [{"deg": angles[i], "tick": i}
             for i in range(1, len(angles) - 1)
             if angles[i] > angles[i - 1] and angles[i] > angles[i + 1]
             and angles[i] > 0.5]
    return sorted(peaks, key=lambda p: -p["deg"])[:6]


def main():
    failures = []

    def check(cond, what, detail=""):
        if not cond:
            failures.append(f"{what} {detail}".strip())

    for name, frozen in sorted(FROZEN.items()):
        src = parse_source(GEN_BY_NAME[name]())
        rec = {"clip": name}
        dec16 = None
        for bits, key in ((16, "bits16"), (8, "bits8")):
            data, stats = compress(src, BUDGET, bits)
            legacy, _ = encode(src, bits=bits)
            check(legacy == data, f"{name}/{key}: legacy adapter diverged")
            check(len(data) == frozen[key]["bytes"],
                  f"{name}/{key}: byte length", f"{len(data)} != {frozen[key]['bytes']}")
            check(hashlib.sha256(data).hexdigest() == frozen[key]["sha256"],
                  f"{name}/{key}: byte sha256")
            check(sum(s["keys_before"] for s in stats) == frozen[key]["keys_before"],
                  f"{name}/{key}: keys_before")
            check(sum(s["keys_after"] for s in stats) == frozen[key]["keys_after"],
                  f"{name}/{key}: keys_after")

            dec = decode(data)
            if bits == 16:
                dec16 = dec
            digest, ticks = _trajectory_digest(dec, src.duration_ticks)
            check(ticks == frozen[key]["trajectory_ticks"],
                  f"{name}/{key}: trajectory tick count")
            check(digest == frozen[key]["trajectory_sha256"],
                  f"{name}/{key}: reconstructed trajectory digest")
            rec[key] = (len(data), digest[:16])

            # Committed pre-refactor asset must still decode to the same trace.
            asset = ROOT / "out" / f"{name}.acmp"
            if asset.exists():
                committed = asset.read_bytes()
                if bits == 16:
                    check(committed == data, f"{name}: committed asset bytes changed")
                old_dec = decode(committed)
                old_digest, _ = _trajectory_digest(old_dec, src.duration_ticks)
                if bits == 16:
                    check(old_digest == frozen[key]["trajectory_sha256"],
                          f"{name}: committed asset decode digest")

        step, tick = _rotation_scan(dec16, src.duration_ticks)
        check(step == frozen["max_rotation_step_deg"],
              f"{name}: max rotation step", f"{step!r} != {frozen['max_rotation_step_deg']!r}")
        check(tick == frozen["max_rotation_step_tick"], f"{name}: max rotation step tick")
        check(_narrow_peaks(dec16, src.duration_ticks) == frozen["narrow_peaks"],
              f"{name}: narrow peak positions")
        rec["wrap"] = (round(step, 4), tick)
        print(f"{name}: 16-bit {rec['bits16'][0]}B traj {rec['bits16'][1]}, "
              f"8-bit {rec['bits8'][0]}B traj {rec['bits8'][1]}, "
              f"wrap {rec['wrap'][0]}deg@{rec['wrap'][1]}, "
              f"peaks {len(frozen['narrow_peaks'])}")

    # Rotation wraparound must remain a genuine near-180-degree turn.
    half = parse_source(GEN_BY_NAME["half_turn"]())
    dec = decode(compress(half, BUDGET, 16)[0])
    end_angle = q_angle_deg(sample_pose(dec, 0.0)[1][1],
                            sample_pose(dec, float(half.duration_ticks))[1][1])
    check(170.0 < end_angle <= 180.0, "half_turn: wraparound amplitude lost",
          f"{end_angle:.3f} deg")

    # Error spaces are distinct and rotation is sign-insensitive.
    check(ROTATION.distance([0, 0, 0, 1], [0, 0, 0, -1]) == 0.0,
          "rotation: q and -q not equivalent")
    check(abs(TRANSLATION.distance([0, 0, 0], [0.01, 0, 0]) - 0.01) < 1e-12,
          "translation: Euclidean distance wrong")
    check(ROTATION.units == "degrees" and TRANSLATION.units == "units",
          "error spaces do not carry distinct units")
    check(BUDGET.for_property(1) == (ROTATION, 0.25), "rotation budget space/tolerance")
    check(BUDGET.for_property(0) == (TRANSLATION, 0.01), "translation budget space/tolerance")

    # The compression interface must not hide a missing argument behind a default.
    for fn in (compress, ErrorBudget):
        sig = inspect.signature(fn)
        check(all(p.default is inspect.Parameter.empty for p in sig.parameters.values()),
              f"{fn.__name__}: has a masking default", str(sig))
    try:
        compress(half, None, 16)
    except TypeError:
        pass
    else:
        failures.append("compress accepted a missing ErrorBudget")
    try:
        compress(half, BUDGET)
    except TypeError:
        pass
    else:
        failures.append("compress accepted a missing bits width")

    # Preview (source) and verification (decoded) both go through one sampler.
    src = parse_source(GEN_BY_NAME["fast_wave"]())
    tr = src.tracks[0]
    dec = decode(compress(src, BUDGET, 16)[0])
    for t in (0.0, 1.5, 17.0, 60.5, float(src.duration_ticks)):
        check(sample_track_values(tr.times, tr.values, tr.prop, t)
              == sampling.sample_keys(tr.times, tr.values, tr.prop, t),
              "source sampling bypasses shared sampler", f"t={t}")
        check(track_value_at(dec.tracks[0], t)
              == sampling.decoded_curve(dec.tracks[0]).sample(t),
              "decoded sampling bypasses shared sampler", f"t={t}")
        direct = sampling.sample_pose(
            src.joints,
            lambda i, p: sampling.source_curve(src.track_map[(i, p)])
            if (i, p) in src.track_map else None,
            t)
        check(source_pose(src, t) == direct, "source pose bypasses shared sampler", f"t={t}")
        resolved = sampling.resolve_time(t, dec.loops, dec.duration_ticks)
        direct_dec = sampling.sample_pose(
            dec.joints,
            lambda i, p: sampling.decoded_curve(dec.track_map[(i, p)])
            if (i, p) in dec.track_map else None,
            resolved)
        check(sample_pose(dec, t) == direct_dec,
              "decoded pose bypasses shared sampler", f"t={t}")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        print(f"verification: {len(failures)} failure(s)")
        return 1
    print("verification: PASS (bytes, trajectories, wrap, peaks, interface, sampling)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
