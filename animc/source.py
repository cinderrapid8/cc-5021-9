"""Source (uncompressed) animation: JSON authoring format + sampling.

JSON schema:
{
  "name": "clip name",
  "ticks_per_second": 30,
  "duration_ticks": 120,
  "loops": true,
  "joints": [{"name": "root", "parent": -1,
              "bind_t": [0,0,0], "bind_r": [0,0,0,1], "bind_s": [1,1,1]}, ...],
  "tracks": [{"joint": "head", "property": "rotation",
              "times": [0, 1, ...], "values": [[x,y,z,w], ...]}, ...],
  "end_joints": ["head_tip"]        # optional; defaults to leaf joints
}
"""

import json
from bisect import bisect_right

from .format import NAME_TO_PROP, PROP_COMPONENTS, PROP_ROTATION
from .qmath import lerp, q_norm, q_slerp


class SourceError(Exception):
    pass


class Joint:
    __slots__ = ("name", "parent", "bind_t", "bind_r", "bind_s")

    def __init__(self, name, parent, bind_t, bind_r, bind_s):
        self.name = name
        self.parent = parent
        self.bind_t = bind_t
        self.bind_r = bind_r
        self.bind_s = bind_s


class Track:
    __slots__ = ("joint", "prop", "times", "values")

    def __init__(self, joint, prop, times, values):
        self.joint = joint
        self.prop = prop
        self.times = times
        self.values = values


class SourceAnim:
    def __init__(self, name, ticks_per_second, duration_ticks, loops, joints, tracks, end_joints):
        self.name = name
        self.ticks_per_second = ticks_per_second
        self.duration_ticks = duration_ticks
        self.loops = loops
        self.joints = joints
        self.tracks = tracks
        self.end_joints = end_joints
        self.track_map = {(t.joint, t.prop): t for t in tracks}

    def leaf_joints(self):
        has_child = [False] * len(self.joints)
        for j in self.joints:
            if j.parent >= 0:
                has_child[j.parent] = True
        return [i for i, c in enumerate(has_child) if not c]


def _check_vec(v, n, what):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise SourceError(f"{what}: expected {n} components")
    return [float(x) for x in v]


def load_source(path):
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return parse_source(doc)


def parse_source(doc):
    try:
        name = str(doc.get("name", "clip"))
        tps = int(doc["ticks_per_second"])
        duration = int(doc["duration_ticks"])
        if tps <= 0 or tps > 100000:
            raise SourceError("ticks_per_second out of range")
        if duration <= 0:
            raise SourceError("duration_ticks must be positive")
        loops = bool(doc.get("loops", False))

        joints = []
        for i, jd in enumerate(doc["joints"]):
            jname = str(jd["name"])
            parent = int(jd["parent"])
            if parent >= i or parent < -1:
                raise SourceError(f"joint {jname}: invalid parent {parent}")
            joints.append(Joint(
                jname, parent,
                _check_vec(jd.get("bind_t", [0, 0, 0]), 3, f"{jname}.bind_t"),
                q_norm(_check_vec(jd.get("bind_r", [0, 0, 0, 1]), 4, f"{jname}.bind_r")),
                _check_vec(jd.get("bind_s", [1, 1, 1]), 3, f"{jname}.bind_s"),
            ))
        if not joints:
            raise SourceError("no joints")
        names = {}
        for j in joints:
            if j.name in names:
                raise SourceError(f"duplicate joint name {j.name}")
            names[j.name] = True

        tracks = []
        seen = set()
        for td in doc.get("tracks", []):
            jname = str(td["joint"])
            if jname not in names:
                raise SourceError(f"track references unknown joint {jname}")
            joint_names = [j.name for j in joints]
            jidx = joint_names.index(jname)
            prop = NAME_TO_PROP[str(td["property"])]
            if (jidx, prop) in seen:
                raise SourceError(f"duplicate track {jname}.{td['property']}")
            seen.add((jidx, prop))
            times = [int(t) for t in td["times"]]
            ncomp = PROP_COMPONENTS[prop]
            values = [_check_vec(v, ncomp, f"{jname}.{td['property']}") for v in td["values"]]
            if len(times) != len(values):
                raise SourceError(f"{jname}.{td['property']}: times/values length mismatch")
            if not times:
                raise SourceError(f"{jname}.{td['property']}: empty track")
            if any(times[k] >= times[k + 1] for k in range(len(times) - 1)):
                raise SourceError(f"{jname}.{td['property']}: times not strictly increasing")
            if times[-1] > duration:
                raise SourceError(f"{jname}.{td['property']}: key past duration")
            if prop == PROP_ROTATION:
                values = [q_norm(v) for v in values]
            tracks.append(Track(jidx, prop, times, values))

        end_joints = doc.get("end_joints")
        anim = SourceAnim(name, tps, duration, loops, joints, tracks, [])
        if end_joints is None:
            anim.end_joints = anim.leaf_joints()
        else:
            anim.end_joints = []
            for jn in end_joints:
                if jn not in names:
                    raise SourceError(f"end_joints: unknown joint {jn}")
                anim.end_joints.append([j.name for j in joints].index(jn))
        return anim
    except KeyError as e:
        raise SourceError(f"missing field {e}") from None


def sample_track_values(times, values, prop, t_ticks):
    """Sample one track at time t (ticks). Lerp for translation/scale,
    slerp with hemisphere fix for rotation. Clamps at both ends."""
    if t_ticks <= times[0]:
        return list(values[0])
    if t_ticks >= times[-1]:
        return list(values[-1])
    i = bisect_right(times, t_ticks) - 1
    t0, t1 = times[i], times[i + 1]
    f = (t_ticks - t0) / (t1 - t0)
    a, b = values[i], values[i + 1]
    if prop == PROP_ROTATION:
        return q_slerp(a, b, f)
    return lerp(a, b, f)


def sample_pose(anim, t_ticks):
    """Full local pose (t, r, s) per joint at time t (ticks)."""
    pose = []
    for i, j in enumerate(anim.joints):
        t = list(j.bind_t)
        r = list(j.bind_r)
        s = list(j.bind_s)
        tr = anim.track_map.get((i, 0))
        if tr:
            t = sample_track_values(tr.times, tr.values, 0, t_ticks)
        tr = anim.track_map.get((i, 1))
        if tr:
            r = sample_track_values(tr.times, tr.values, 1, t_ticks)
        tr = anim.track_map.get((i, 2))
        if tr:
            s = sample_track_values(tr.times, tr.values, 2, t_ticks)
        pose.append((t, r, s))
    return pose
