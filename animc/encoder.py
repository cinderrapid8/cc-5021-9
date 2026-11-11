"""Legacy compression entry point (compatibility adapter).

The implementation now lives in :mod:`animc.compression`. This module preserves
the historical ``encode(anim, pos_tol=..., rot_tol_deg=..., scale_tol=..., bits=...)``
signature and defaults so existing callers and already-shipped assets remain
bit-for-bit reproducible. New code should call ``compression.compress`` with an
explicit ``ErrorBudget``.
"""

from .compression import compress
from .error_metrics import ErrorBudget
from .format import PROP_COMPONENTS


def encode(anim, pos_tol=0.01, rot_tol_deg=0.25, scale_tol=0.005, bits=16):
    """Encode a SourceAnim into .acmp bytes.

    Returns (data, stats) where stats has per-track before/after key counts.
    """
    budget = ErrorBudget(pos=pos_tol, rot_deg=rot_tol_deg, scale=scale_tol)
    return compress(anim, budget, bits)


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
