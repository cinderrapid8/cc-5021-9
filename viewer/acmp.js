/* acmp.js - independent .acmp decoder + pose sampler (browser and Node).
 * Mirrors the Python decoder's strict validation: any corrupt length or
 * illegal index throws FormatError before any pose is produced. */
'use strict';

const ACMP_MAGIC = 0x504d4341; // 'ACMP' as little-endian u32
const ACMP_VERSION = 1;
const PROP_COMPONENTS = [3, 4, 3]; // translation, rotation, scale
const PROP_ROTATION = 1;
const ENC_CONST = 0;
const ENC_KEYS = 1;
const FLAG_LOOPS = 1;
const MAX_JOINTS = 512;
const MAX_TRACKS = 4096;
const MAX_KEYS_PER_TRACK = 1000000;
const MAX_NAME_LEN = 64;

class FormatError extends Error {}

class Cursor {
  constructor(buf) {
    if (buf instanceof ArrayBuffer) this.view = new DataView(buf);
    else this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    this.len = this.view.byteLength;
    this.pos = 0;
  }
  remaining() { return this.len - this.pos; }
  need(n) {
    if (this.pos + n > this.len) {
      throw new FormatError(`unexpected end of data at offset ${this.pos} (need ${n}, have ${this.remaining()})`);
    }
  }
  u8() { this.need(1); const v = this.view.getUint8(this.pos); this.pos += 1; return v; }
  u16() { this.need(2); const v = this.view.getUint16(this.pos, true); this.pos += 2; return v; }
  i16() { this.need(2); const v = this.view.getInt16(this.pos, true); this.pos += 2; return v; }
  u32() { this.need(4); const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  f32() { this.need(4); const v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; }
  f32s(n) { const out = []; for (let i = 0; i < n; i++) out.push(this.f32()); return out; }
  bytes(n) { this.need(n); const out = new Uint8Array(this.view.buffer, this.view.byteOffset + this.pos, n); this.pos += n; return out; }
}

function qNorm(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (n < 1e-12) return [0, 0, 0, 1];
  return q.map(c => c / n);
}
function qDot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }
function qSlerp(a, b, t) {
  let d = qDot(a, b), bb = b;
  if (d < 0) { d = -d; bb = b.map(c => -c); }
  if (d > 0.9995) return qNorm(a.map((c, i) => c + (bb[i] - c) * t));
  d = Math.min(1, d);
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return a.map((c, i) => c * wa + bb[i] * wb);
}
function lerp(a, b, t) { return a.map((c, i) => c + (b[i] - c) * t); }

function decodeAcmp(buf) {
  const r = new Cursor(buf);
  if (r.u32() !== ACMP_MAGIC) throw new FormatError('bad magic');
  if (r.u16() !== ACMP_VERSION) throw new FormatError('unsupported version');
  const flags = r.u16();
  if (flags & ~FLAG_LOOPS) throw new FormatError(`unknown flags ${flags}`);

  const anim = {
    loops: !!(flags & FLAG_LOOPS),
    ticksPerSecond: r.u32(),
    durationTicks: 0,
    joints: [],
    tracks: [],
    trackMap: new Map(),
  };
  if (anim.ticksPerSecond < 1 || anim.ticksPerSecond > 100000) {
    throw new FormatError(`invalid ticks_per_second ${anim.ticksPerSecond}`);
  }
  anim.durationTicks = r.u32();
  if (anim.durationTicks === 0) throw new FormatError('zero duration');

  const jointCount = r.u16();
  if (jointCount < 1 || jointCount > MAX_JOINTS) throw new FormatError(`invalid joint_count ${jointCount}`);
  if (r.u16() !== 0) throw new FormatError('reserved header field must be 0');
  const trackCount = r.u32();
  if (trackCount > MAX_TRACKS) throw new FormatError(`invalid track_count ${trackCount}`);

  const utf8 = new TextDecoder('utf-8', { fatal: true });
  for (let i = 0; i < jointCount; i++) {
    const nameLen = r.u8();
    if (nameLen < 1 || nameLen > MAX_NAME_LEN) throw new FormatError(`joint ${i}: invalid name length ${nameLen}`);
    let name;
    try { name = utf8.decode(r.bytes(nameLen)); }
    catch { throw new FormatError(`joint ${i}: name is not valid utf-8`); }
    const parent = r.i16();
    if (parent >= i || parent < -1) throw new FormatError(`joint ${name}: invalid parent index ${parent}`);
    const bindT = r.f32s(3), bindR = r.f32s(4), bindS = r.f32s(3);
    if (bindR.reduce((s, x) => s + x * x, 0) < 1e-12) throw new FormatError(`joint ${name}: degenerate bind rotation`);
    anim.joints.push({ name, parent, bindT, bindR: qNorm(bindR), bindS });
  }

  const seen = new Set();
  for (let k = 0; k < trackCount; k++) {
    const tr = {};
    tr.joint = r.u16();
    if (tr.joint >= jointCount) throw new FormatError(`track references invalid joint index ${tr.joint}`);
    tr.prop = r.u8();
    if (tr.prop > 2) throw new FormatError(`invalid property id ${tr.prop}`);
    const ident = tr.joint * 8 + tr.prop;
    if (seen.has(ident)) throw new FormatError(`duplicate track for joint ${tr.joint} property ${tr.prop}`);
    seen.add(ident);
    tr.encoding = r.u8();
    const comps = PROP_COMPONENTS[tr.prop];

    if (tr.encoding === ENC_CONST) {
      tr.constValue = r.f32s(comps);
      if (tr.prop === PROP_ROTATION) tr.constValue = qNorm(tr.constValue);
    } else if (tr.encoding === ENC_KEYS) {
      tr.bits = r.u8();
      if (tr.bits !== 8 && tr.bits !== 16) throw new FormatError(`invalid bits per component ${tr.bits}`);
      if (r.u8() !== 0) throw new FormatError('reserved track field must be 0');
      tr.keyCount = r.u32();
      if (tr.keyCount < 2 || tr.keyCount > MAX_KEYS_PER_TRACK) throw new FormatError(`invalid key_count ${tr.keyCount}`);
      // Reject corrupt lengths up front, before reading any payload.
      const bytesPerKey = 4 + comps * (tr.bits / 8);
      const need = tr.keyCount * bytesPerKey + 2 * comps * 4;
      if (need > r.remaining()) {
        throw new FormatError(`track payload overruns file: need ${need} bytes, have ${r.remaining()}`);
      }
      tr.mins = r.f32s(comps);
      tr.maxs = r.f32s(comps);
      tr.times = [];
      for (let i = 0; i < tr.keyCount; i++) tr.times.push(r.u32());
      for (let i = 1; i < tr.keyCount; i++) {
        if (tr.times[i] <= tr.times[i - 1]) throw new FormatError('key times not strictly increasing');
      }
      if (tr.times[tr.keyCount - 1] > anim.durationTicks) throw new FormatError('key time past clip duration');
      const n = tr.keyCount * comps;
      tr.data = new (tr.bits === 8 ? Uint8Array : Uint16Array)(n);
      for (let i = 0; i < n; i++) tr.data[i] = tr.bits === 8 ? r.u8() : r.u16();
    } else {
      throw new FormatError(`invalid encoding ${tr.encoding}`);
    }
    anim.tracks.push(tr);
  }

  if (r.remaining() !== 0) throw new FormatError(`${r.remaining()} trailing bytes after last track`);
  for (const tr of anim.tracks) anim.trackMap.set(tr.joint * 8 + tr.prop, tr);
  return anim;
}

function dequant(tr, keyIdx) {
  const comps = PROP_COMPONENTS[tr.prop];
  const levels = (1 << tr.bits) - 1;
  const out = [];
  const base = keyIdx * comps;
  for (let c = 0; c < comps; c++) {
    const q = tr.data[base + c];
    const rng = tr.maxs[c] - tr.mins[c];
    out.push(rng <= 0 ? tr.mins[c] : tr.mins[c] + (q / levels) * rng);
  }
  return out;
}

function trackValueAt(tr, tTicks) {
  if (tr.encoding === ENC_CONST) return tr.constValue.slice();
  const times = tr.times;
  const norm = v => (tr.prop === PROP_ROTATION ? qNorm(v) : v);
  if (tTicks <= times[0]) return norm(dequant(tr, 0));
  if (tTicks >= times[times.length - 1]) return norm(dequant(tr, tr.keyCount - 1));
  let lo = 0, hi = times.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= tTicks) lo = mid; else hi = mid;
  }
  const f = (tTicks - times[lo]) / (times[lo + 1] - times[lo]);
  const a = dequant(tr, lo), b = dequant(tr, lo + 1);
  if (tr.prop === PROP_ROTATION) return qSlerp(qNorm(a), qNorm(b), f);
  return lerp(a, b, f);
}

function samplePose(anim, tTicks) {
  let t = tTicks;
  if (anim.loops && anim.durationTicks > 0) t = ((tTicks % anim.durationTicks) + anim.durationTicks) % anim.durationTicks;
  else t = Math.min(Math.max(tTicks, 0), anim.durationTicks);
  return anim.joints.map((j, i) => {
    let tr;
    let pt = j.bindT.slice(), pr = j.bindR.slice(), ps = j.bindS.slice();
    if ((tr = anim.trackMap.get(i * 8 + 0))) pt = trackValueAt(tr, t);
    if ((tr = anim.trackMap.get(i * 8 + 1))) pr = trackValueAt(tr, t);
    if ((tr = anim.trackMap.get(i * 8 + 2))) ps = trackValueAt(tr, t);
    return [pt, pr, ps];
  });
}

function trsMat4(t, q, s) {
  const [x, y, z, w] = qNorm(q);
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const r00 = 1 - (yy + zz), r01 = xy - wz, r02 = xz + wy;
  const r10 = xy + wz, r11 = 1 - (xx + zz), r12 = yz - wx;
  const r20 = xz - wy, r21 = yz + wx, r22 = 1 - (xx + yy);
  return [
    r00 * s[0], r01 * s[1], r02 * s[2], t[0],
    r10 * s[0], r11 * s[1], r12 * s[2], t[1],
    r20 * s[0], r21 * s[1], r22 * s[2], t[2],
    0, 0, 0, 1,
  ];
}

function mMul(a, b) {
  const out = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      out[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
    }
  }
  return out;
}

function fkWorld(joints, pose) {
  const worlds = [];
  for (let i = 0; i < joints.length; i++) {
    const local = trsMat4(pose[i][0], pose[i][1], pose[i][2]);
    worlds.push(joints[i].parent < 0 ? local : mMul(worlds[joints[i].parent], local));
  }
  return worlds;
}

function worldPos(m) { return [m[3], m[7], m[11]]; }

// ---- source (uncompressed JSON) sampling: lerp/slerp between authored keys
function srcTrackValue(tr, tTicks) {
  const times = tr.times, vals = tr.values;
  if (tTicks <= times[0]) return vals[0].slice();
  if (tTicks >= times[times.length - 1]) return vals[vals.length - 1].slice();
  let lo = 0, hi = times.length - 1;
  while (lo + 1 < hi) { const m = (lo + hi) >> 1; if (times[m] <= tTicks) lo = m; else hi = m; }
  const f = (tTicks - times[lo]) / (times[lo + 1] - times[lo]);
  if (tr.property === 'rotation') return qSlerp(qNorm(vals[lo]), qNorm(vals[lo + 1]), f);
  return vals[lo].map((c, i) => c + (vals[lo + 1][i] - c) * f);
}

function sampleSourcePose(src, tTicks) {
  return src.joints.map(j => {
    let pt = j.bind_t.slice(), pr = j.bind_r.slice(), ps = j.bind_s.slice();
    for (const tr of src.tracks) {
      if (tr.joint !== j.name) continue;
      if (tr.property === 'translation') pt = srcTrackValue(tr, tTicks);
      else if (tr.property === 'rotation') pr = srcTrackValue(tr, tTicks);
      else if (tr.property === 'scale') ps = srcTrackValue(tr, tTicks);
    }
    return [pt, pr, ps];
  });
}

const api = { FormatError, decodeAcmp, trackValueAt, samplePose, fkWorld, worldPos, qSlerp, qNorm,
              sampleSourcePose };
if (typeof module !== 'undefined' && module.exports) module.exports = api;
if (typeof window !== 'undefined') window.acmp = api;

// Node CLI: node acmp.js file.acmp [ticks ...] -> decode summary + tip positions
if (typeof require !== 'undefined' && typeof process !== 'undefined' && require.main === module) {
  const fs = require('fs');
  const file = process.argv[2];
  try {
    const anim = decodeAcmp(fs.readFileSync(file));
    console.log(`valid .acmp: ${anim.joints.length} joints, ${anim.tracks.length} tracks, ` +
      `${anim.durationTicks} ticks @ ${anim.ticksPerSecond} tps, loops=${anim.loops}`);
    const leaves = anim.joints.map((j, i) => i).filter(i => !anim.joints.some(j => j.parent === i));
    const probeTicks = process.argv.slice(3).map(Number);
    const times = probeTicks.length ? probeTicks : [0, anim.durationTicks / 3, anim.durationTicks * 2 / 3, anim.durationTicks];
    for (const t of times) {
      const worlds = fkWorld(anim.joints, samplePose(anim, t));
      const tips = leaves.map(i => worldPos(worlds[i]).map(v => v.toFixed(6)).join(','));
      console.log(`t=${t} tips=[${tips.join(' | ')}]`);
    }
  } catch (e) {
    if (e instanceof FormatError) { console.error(`REJECTED: ${e.message}`); process.exit(1); }
    throw e;
  }
}
