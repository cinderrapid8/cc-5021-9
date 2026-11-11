"""Property-specific error spaces and an explicit error budget.

Rotation and translation/scale errors do not share a metric: rotation error is
a geodesic angle in degrees (invariant under q == -q and wrapped at 180 deg),
while translation and scale errors are Euclidean distances in their own units.
A single scalar tolerance applied to both would silently conflate the spaces,
so a budget carries one tolerance per space and consumers ask the budget for
the space they are measuring.

There are deliberately no defaults here. A compression call must name its
budget explicitly; a missing or mistyped tolerance raises instead of being
papered over by a fallback value.
"""

import math

from .format import PROP_ROTATION, PROP_SCALE, PROP_TRANSLATION
from .qmath import q_angle_deg, v_dist


class ErrorSpace:
    """A metric with an explicit unit.

    Subclasses implement ``distance``; ``within`` is the acceptance predicate
    in that space. Tolerance values are always interpreted in ``units``.
    """

    __slots__ = ("name", "units")

    def __init__(self, name, units):
        self.name = name
        self.units = units

    def distance(self, a, b):
        raise NotImplementedError

    def within(self, a, b, tol):
        return self.distance(a, b) <= tol


class RotationErrorSpace(ErrorSpace):
    """Angular distance in degrees. q_angle_deg uses |dot| so q and -q are the
    same rotation and the metric wraps at 180 degrees."""

    __slots__ = ()

    def __init__(self):
        super().__init__("rotation", "degrees")

    def distance(self, a, b):
        return q_angle_deg(a, b)


class EuclideanErrorSpace(ErrorSpace):
    """Euclidean distance in the property's own units."""

    __slots__ = ()

    def distance(self, a, b):
        return v_dist(a, b)


ROTATION = RotationErrorSpace()
TRANSLATION = EuclideanErrorSpace("translation", "units")
SCALE = EuclideanErrorSpace("scale", "unitless")

ERROR_SPACES = {
    PROP_TRANSLATION: TRANSLATION,
    PROP_ROTATION: ROTATION,
    PROP_SCALE: SCALE,
}


class ErrorBudget:
    """Explicit acceptance tolerances, one per error space.

    Every field is required and validated at construction. ``for_property``
    returns the (space, tolerance) pair so callers cannot apply a rotation
    tolerance to a translation (or vice versa).
    """

    __slots__ = ("pos", "rot_deg", "scale")

    def __init__(self, pos, rot_deg, scale):
        self.pos = _checked_tolerance("pos", pos)
        self.rot_deg = _checked_tolerance("rot_deg", rot_deg)
        self.scale = _checked_tolerance("scale", scale)

    def for_property(self, prop):
        if prop == PROP_TRANSLATION:
            return TRANSLATION, self.pos
        if prop == PROP_ROTATION:
            return ROTATION, self.rot_deg
        if prop == PROP_SCALE:
            return SCALE, self.scale
        raise ValueError(f"unknown property id {prop}")

    def distance(self, prop, a, b):
        return ERROR_SPACES[prop].distance(a, b)

    def within(self, prop, a, b):
        space, tol = self.for_property(prop)
        return space.within(a, b, tol)

    def __repr__(self):
        return f"ErrorBudget(pos={self.pos!r}, rot_deg={self.rot_deg!r}, scale={self.scale!r})"


def _checked_tolerance(name, value):
    if value is None:
        raise TypeError(f"{name} tolerance is required; no default is assumed")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} tolerance must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} tolerance must be finite and non-negative, got {value!r}")
    return value
