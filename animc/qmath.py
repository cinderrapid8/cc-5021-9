"""Quaternion / matrix helpers (pure stdlib).

Quaternion layout: [x, y, z, w]. Matrices are row-major 4x4, column-vector
convention: p' = M * p, world = parent_world * local.
"""

import math


def q_norm(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def q_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def q_neg(q):
    return [-q[0], -q[1], -q[2], -q[3]]


def q_slerp(a, b, t):
    """Slerp with hemisphere fix: q and -q are the same rotation, always
    take the short path."""
    d = q_dot(a, b)
    bb = b
    if d < 0.0:
        d = -d
        bb = q_neg(b)
    if d > 0.9995:
        return q_norm([a[i] + (bb[i] - a[i]) * t for i in range(4)])
    d = min(1.0, d)
    theta = math.acos(d)
    s = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / s
    wb = math.sin(t * theta) / s
    return [a[i] * wa + bb[i] * wb for i in range(4)]


def q_angle_deg(a, b):
    """Absolute rotation angle between two quaternions, in degrees."""
    d = abs(q_dot(q_norm(a), q_norm(b)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def q_from_axis_angle(axis, angle_rad):
    n = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    h = angle_rad * 0.5
    s = math.sin(h) / n
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)]


def lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]


def trs_mat4(t, q, s):
    x, y, z, w = q_norm(q)
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    r00, r01, r02 = 1.0 - (yy + zz), xy - wz, xz + wy
    r10, r11, r12 = xy + wz, 1.0 - (xx + zz), yz - wx
    r20, r21, r22 = xz - wy, yz + wx, 1.0 - (xx + yy)
    sx, sy, sz = s
    return [
        r00 * sx, r01 * sy, r02 * sz, t[0],
        r10 * sx, r11 * sy, r12 * sz, t[1],
        r20 * sx, r21 * sy, r22 * sz, t[2],
        0.0, 0.0, 0.0, 1.0,
    ]


def m_mul(a, b):
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            out[r * 4 + c] = (
                a[r * 4 + 0] * b[0 * 4 + c]
                + a[r * 4 + 1] * b[1 * 4 + c]
                + a[r * 4 + 2] * b[2 * 4 + c]
                + a[r * 4 + 3] * b[3 * 4 + c]
            )
    return out


def m_point(m, p):
    return [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    ]


def v_dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))
