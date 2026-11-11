"""Command line interface.

  python -m animc make-samples --out samples
  python -m animc report --src samples --out out [--pos-tol 0.01]
      [--rot-tol 0.25] [--scale-tol 0.005] [--bits 16] [--eval-hz 240]
  python -m animc encode --in clip.json --out clip.acmp
  python -m animc inspect --in clip.acmp
  python -m animc selftest
"""

import argparse
import json
import os
import sys
import time

from . import samplegen, selftest
from .decoder import decode, sample_pose
from .encoder import encode, raw_f32_size
from .evaluate import evaluate
from .format import FormatError, PROP_NAMES
from .source import SourceError, load_source


def _cmd_make_samples(args):
    paths = samplegen.write_samples(args.out)
    for p in paths:
        print(f"wrote {p}")
    return 0


def _time_decode(data, repeats=30):
    """Decode + resample the whole clip at 60 Hz; report mean milliseconds."""
    dec = decode(data)  # parse once outside the loop is cheating: parse inside
    t0 = time.perf_counter()
    for _ in range(repeats):
        d = decode(data)
        frames = max(1, int(d.duration_seconds() * 60))
        for f in range(frames + 1):
            sample_pose(d, f / 60.0 * d.ticks_per_second)
    return (time.perf_counter() - t0) / repeats * 1000.0


def _report_one(src_path, out_dir, args):
    src = load_source(src_path)
    data, stats = encode(src, pos_tol=args.pos_tol, rot_tol_deg=args.rot_tol,
                         scale_tol=args.scale_tol, bits=args.bits)
    dec = decode(data)
    metrics = evaluate(src, dec, eval_hz=args.eval_hz)
    decode_ms = _time_decode(data)

    raw = raw_f32_size(src)
    base = os.path.splitext(os.path.basename(src_path))[0]
    acmp_path = os.path.join(out_dir, base + ".acmp")
    with open(acmp_path, "wb") as f:
        f.write(data)

    keys_before = sum(s["keys_before"] for s in stats)
    keys_after = sum(s["keys_after"] for s in stats)
    return {
        "name": src.name,
        "source_json_bytes": os.path.getsize(src_path),
        "raw_f32_bytes": raw,
        "acmp_bytes": len(data),
        "ratio_vs_raw_f32": raw / len(data),
        "keys_before": keys_before,
        "keys_after": keys_after,
        "tracks": len(stats),
        "decode_ms_full_clip": decode_ms,
        "local_max_error": metrics["local_max"],
        "local_worst": metrics["local_worst"],
        "end_joints": metrics["end_joints"],
        "acmp_path": acmp_path,
        "source_path": src_path,
    }


def _cmd_report(args):
    os.makedirs(args.out, exist_ok=True)
    srcs = sorted(
        os.path.join(args.src, p) for p in os.listdir(args.src) if p.endswith(".json"))
    if not srcs:
        print(f"no .json clips in {args.src}", file=sys.stderr)
        return 1

    rows = []
    for p in srcs:
        try:
            rows.append(_report_one(p, args.out, args))
        except (SourceError, FormatError, ValueError) as e:
            print(f"{p}: ERROR {e}", file=sys.stderr)
            return 1

    hdr = f"{'clip':<16} {'raw f32':>9} {'acmp':>9} {'ratio':>6} {'keys':>11} {'decode ms':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<16} {r['raw_f32_bytes']:>9} {r['acmp_bytes']:>9} "
              f"{r['ratio_vs_raw_f32']:>5.1f}x "
              f"{r['keys_before']:>5}->{r['keys_after']:<5} {r['decode_ms_full_clip']:>9.3f}")

    print("\nmax local error (translation units / rotation deg / scale units):")
    for r in rows:
        lm = r["local_max_error"]
        print(f"  {r['name']:<16} pos {lm['translation']:.5f}  "
              f"rot {lm['rotation']:.4f} deg  scale {lm['scale']:.6f}")

    print("\nend-joint world error after FK propagation:")
    for r in rows:
        for e in r["end_joints"]:
            print(f"  {r['name']:<16} {e['joint']:<12} "
                  f"max pos {e['max_pos_error']:.5f} (@{e['max_pos_error_time_s']:.2f}s)  "
                  f"max rot {e['max_rot_error_deg']:.4f} deg (@{e['max_rot_error_time_s']:.2f}s)")

    report = {"clips": [{k: v for k, v in r.items() if k not in ("acmp_path", "source_path")}
                        for r in rows],
              "tolerances": {"pos": args.pos_tol, "rot_deg": args.rot_tol,
                             "scale": args.scale_tol},
              "bits": args.bits}
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    manifest = [{"name": r["name"],
                 "source": os.path.relpath(r["source_path"], args.out),
                 "compressed": os.path.basename(r["acmp_path"])} for r in rows]
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nwrote {os.path.join(args.out, 'report.json')} and manifest.json")
    return 0


def _cmd_encode(args):
    src = load_source(getattr(args, "in"))
    data, stats = encode(src, pos_tol=args.pos_tol, rot_tol_deg=args.rot_tol,
                         scale_tol=args.scale_tol, bits=args.bits)
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"{args.out}: {len(data)} bytes "
          f"({sum(s['keys_before'] for s in stats)} -> "
          f"{sum(s['keys_after'] for s in stats)} keys)")
    return 0


def _cmd_inspect(args):
    with open(getattr(args, "in"), "rb") as f:
        data = f.read()
    try:
        dec = decode(data)
    except FormatError as e:
        print(f"REJECTED: {e}", file=sys.stderr)
        return 1
    print(f"valid .acmp: {len(data)} bytes")
    print(f"  time unit: 1/{dec.ticks_per_second} s, duration "
          f"{dec.duration_ticks} ticks ({dec.duration_seconds():.3f} s), loops={dec.loops}")
    print(f"  joints ({len(dec.joints)}):")
    for i, j in enumerate(dec.joints):
        print(f"    [{i}] {j.name} parent={j.parent}")
    print(f"  tracks ({len(dec.tracks)}):")
    for tr in dec.tracks:
        jname = dec.joints[tr.joint].name
        if tr.encoding == 0:
            print(f"    {jname}.{PROP_NAMES[tr.prop]}: const {tr.const_value}")
        else:
            print(f"    {jname}.{PROP_NAMES[tr.prop]}: {tr.key_count} keys, "
                  f"{tr.bits}-bit, range {tr.mins} .. {tr.maxs}")
    return 0


def _cmd_selftest(_args):
    return selftest.run()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="animc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-samples", help="generate validation clips")
    p.add_argument("--out", default="samples")
    p.set_defaults(fn=_cmd_make_samples)

    p = sub.add_parser("report", help="encode all clips, evaluate, write report+manifest")
    p.add_argument("--src", default="samples")
    p.add_argument("--out", default="out")
    _add_encode_opts(p)
    p.add_argument("--eval-hz", type=float, default=240.0)
    p.set_defaults(fn=_cmd_report)

    p = sub.add_parser("encode", help="encode one clip")
    p.add_argument("--in", required=True)
    p.add_argument("--out", required=True)
    _add_encode_opts(p)
    p.set_defaults(fn=_cmd_encode)

    p = sub.add_parser("inspect", help="validate and dump an .acmp file")
    p.add_argument("--in", required=True)
    p.set_defaults(fn=_cmd_inspect)

    p = sub.add_parser("selftest", help="run corruption/roundtrip self-tests")
    p.set_defaults(fn=_cmd_selftest)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (SourceError, FormatError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _add_encode_opts(p):
    p.add_argument("--pos-tol", type=float, default=0.01,
                   help="translation tolerance (world units)")
    p.add_argument("--rot-tol", type=float, default=0.25,
                   help="rotation tolerance (degrees)")
    p.add_argument("--scale-tol", type=float, default=0.005,
                   help="scale tolerance (unitless)")
    p.add_argument("--bits", type=int, default=16, choices=(8, 16),
                   help="quantization bits per component")


if __name__ == "__main__":
    sys.exit(main())
