"""Keypoint selection: deterministic greedy maximal-segment decimation.

Selection is independent of both curve encoding and final verification. It
receives an explicit ErrorBudget, measures interpolation error at authored key
times in the correct property-specific space, and never rewrites input values
or tolerance values.
"""

from .error_metrics import ERROR_SPACES
from .format import PROP_ROTATION
from .qmath import lerp, q_slerp


def interpolate(prop, a, b, f):
    """Interpolate between two kept values in the property's natural space."""
    if prop == PROP_ROTATION:
        return q_slerp(a, b, f)
    return lerp(a, b, f)


def interpolation_error(prop, a, b, f, ref):
    """Distance between the i..j interpolation and an authored reference."""
    return ERROR_SPACES[prop].distance(interpolate(prop, a, b, f), ref)


def segment_within_budget(times, values, i, j, prop, budget):
    """True if every authored key strictly between i and j stays inside the
    budget when interpolated from the kept endpoints i..j."""
    space, tol = budget.for_property(prop)
    span = times[j] - times[i]
    a, b = values[i], values[j]
    for k in range(i + 1, j):
        f = (times[k] - times[i]) / span
        if not space.within(interpolate(prop, a, b, f), values[k], tol):
            return False
    return True


def select_keypoints(times, values, prop, budget):
    """Greedy maximal-segment decimation.

    Guarantees that at every authored key time the interpolated curve stays
    within the budget of the authored value. The first and last keys are
    always kept for a deterministic loop seam.
    """
    n = len(times)
    if n <= 2:
        return list(range(n))
    kept = [0]
    i = 0
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not segment_within_budget(times, values, i, j, prop, budget):
            j -= 1
        kept.append(j)
        i = j
    return kept


def collapse_to_constant(values, prop, budget):
    """True when every value is within budget of the first value, i.e. the
    whole track can be encoded as one constant without exceeding acceptance."""
    space, tol = budget.for_property(prop)
    return all(space.within(values[0], v, tol) for v in values[1:])
