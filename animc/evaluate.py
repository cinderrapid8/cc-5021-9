"""Error evaluation: local per-track error plus world-space error of end
joints after hierarchical (FK) propagation, so a small root rotation error
shows up as the large tip deviation it actually causes."""

from . import decoder as dec_mod
from . import source as src_mod
from .error_metrics import ERROR_SPACES
from .format import PROP_NAMES
from .qmath import m_mul, m_point, q_angle_deg, trs_mat4, v_dist


def fk_world(joints, pose):
    """World matrix per joint, parents before children (format guarantees
    parent index < child index)."""
    worlds = []
    for i, j in enumerate(joints):
        t, r, s = pose[i]
        local = trs_mat4(t, r, s)
        worlds.append(local if j.parent < 0 else m_mul(worlds[j.parent], local))
    return worlds


def _eval_times(anim_src, hz):
    step = anim_src.ticks_per_second / hz
    n = int(anim_src.duration_ticks / step) + 1
    times = [i * step for i in range(n)]
    if times[-1] < anim_src.duration_ticks:
        times.append(float(anim_src.duration_ticks))
    return times


def evaluate(src, dec, eval_hz=240.0, end_joints=None):
    """Compare source animation against decoded animation.

    Returns a dict with per-property local maxima and per-end-joint world
    maxima (position in source units, rotation in degrees).
    """
    times = _eval_times(src, eval_hz)
    ends = end_joints if end_joints is not None else src.end_joints

    local_max = {0: 0.0, 1: 0.0, 2: 0.0}
    local_worst = {0: None, 1: None, 2: None}
    end_max = {e: {"pos": 0.0, "pos_time": 0.0, "rot": 0.0, "rot_time": 0.0}
               for e in ends}

    src_tracks = src.track_map
    for t_ticks in times:
        pose_src = src_mod.sample_pose(src, t_ticks)
        pose_dec = dec_mod.sample_pose(dec, t_ticks)

        # Local error per encoded track.
        for tr in dec.tracks:
            strack = src_tracks.get((tr.joint, tr.prop))
            if strack is None:
                continue
            a = src_mod.sample_track_values(strack.times, strack.values, tr.prop, t_ticks)
            b = dec_mod.track_value_at(tr, t_ticks)
            err = ERROR_SPACES[tr.prop].distance(a, b)
            if err > local_max[tr.prop]:
                local_max[tr.prop] = err
                local_worst[tr.prop] = (
                    src.joints[tr.joint].name, PROP_NAMES[tr.prop], t_ticks / src.ticks_per_second)

        # World-space error at end joints after FK propagation.
        ws = fk_world(src.joints, pose_src)
        wd = fk_world(dec.joints, pose_dec)
        for e in ends:
            ps = m_point(ws[e], [0.0, 0.0, 0.0])
            pd = m_point(wd[e], [0.0, 0.0, 0.0])
            dpos = v_dist(ps, pd)
            drot = q_angle_deg(
                _mat_rot_quat(ws[e]), _mat_rot_quat(wd[e]))
            rec = end_max[e]
            if dpos > rec["pos"]:
                rec["pos"] = dpos
                rec["pos_time"] = t_ticks / src.ticks_per_second
            if drot > rec["rot"]:
                rec["rot"] = drot
                rec["rot_time"] = t_ticks / src.ticks_per_second

    return {
        "local_max": {PROP_NAMES[p]: local_max[p] for p in local_max},
        "local_worst": {PROP_NAMES[p]: local_worst[p] for p in local_worst},
        "end_joints": [
            {"joint": src.joints[e].name,
             "max_pos_error": end_max[e]["pos"],
             "max_pos_error_time_s": end_max[e]["pos_time"],
             "max_rot_error_deg": end_max[e]["rot"],
             "max_rot_error_time_s": end_max[e]["rot_time"]}
            for e in ends
        ],
    }


def _mat_rot_quat(m):
    """Extract rotation as quaternion from a world matrix (uniform-ish scale
    assumption is fine for error reporting)."""
    import math

    sx = math.sqrt(m[0] ** 2 + m[4] ** 2 + m[8] ** 2) or 1.0
    sy = math.sqrt(m[1] ** 2 + m[5] ** 2 + m[9] ** 2) or 1.0
    sz = math.sqrt(m[2] ** 2 + m[6] ** 2 + m[10] ** 2) or 1.0
    r = [m[0] / sx, m[1] / sy, m[2] / sz,
         m[4] / sx, m[5] / sy, m[6] / sz,
         m[8] / sx, m[9] / sy, m[10] / sz]
    tr = r[0] + r[4] + r[8]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [(r[5] - r[7]) / s, (r[6] - r[2]) / s, (r[1] - r[3]) / s, 0.25 * s]
    if r[0] > r[4] and r[0] > r[8]:
        s = math.sqrt(1.0 + r[0] - r[4] - r[8]) * 2
        return [0.25 * s, (r[3] + r[1]) / s, (r[6] + r[2]) / s, (r[5] - r[7]) / s]
    if r[4] > r[8]:
        s = math.sqrt(1.0 + r[4] - r[0] - r[8]) * 2
        return [(r[3] + r[1]) / s, 0.25 * s, (r[5] + r[7]) / s, (r[6] - r[2]) / s]
    s = math.sqrt(1.0 + r[8] - r[0] - r[4]) * 2
    return [(r[6] + r[2]) / s, (r[5] + r[7]) / s, 0.25 * s, (r[1] - r[3]) / s]
