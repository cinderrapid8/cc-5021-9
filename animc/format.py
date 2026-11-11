"""Binary container format for compressed animation (.acmp).

Layout (all integers little-endian):

  u8[4]  magic 'ACMP'
  u16    version (=1)
  u16    flags (bit0: loops)
  u32    ticks_per_second      -- time unit
  u32    duration_ticks
  u16    joint_count
  u16    reserved (0)
  u32    track_count
  joints x joint_count:
    u8     name_len (1..64)
    u8[]   name (utf-8)
    i16    parent index (-1 = root, must be < own index)
    f32x3  bind translation
    f32x4  bind rotation (quat xyzw)
    f32x3  bind scale
  tracks x track_count:
    u16    joint_index
    u8     property (0=translation, 1=rotation, 2=scale)
    u8     encoding (0=const, 1=quantized keys)
    encoding 0 (const):
      f32 x components  value
    encoding 1 (keys):
      u8     bits per component (8 or 16)
      u8     reserved (0)
      u32    key_count (>=2)
      f32 x components  quant range min
      f32 x components  quant range max
      u32 x key_count   key times (ticks, strictly increasing, <= duration)
      data: key_count * components * (bits/8) bytes, unsigned LE

Component counts: translation=3, rotation=4, scale=3.
"""

import struct

MAGIC = b"ACMP"
VERSION = 1

PROP_TRANSLATION = 0
PROP_ROTATION = 1
PROP_SCALE = 2
PROP_COMPONENTS = {PROP_TRANSLATION: 3, PROP_ROTATION: 4, PROP_SCALE: 3}
PROP_NAMES = {PROP_TRANSLATION: "translation", PROP_ROTATION: "rotation", PROP_SCALE: "scale"}
NAME_TO_PROP = {v: k for k, v in PROP_NAMES.items()}

ENC_CONST = 0
ENC_KEYS = 1

FLAG_LOOPS = 1

MAX_JOINTS = 512
MAX_TRACKS = 4096
MAX_KEYS_PER_TRACK = 1_000_000
MAX_NAME_LEN = 64
VALID_BITS = (8, 16)


class FormatError(Exception):
    """Raised when a compressed file is corrupt or violates the spec."""


class Reader:
    """Bounds-checked cursor. Every read validates against the remaining
    length so a corrupt length field can never cause an out-of-bounds read."""

    __slots__ = ("data", "pos")

    def __init__(self, data):
        self.data = bytes(data)
        self.pos = 0

    def remaining(self):
        return len(self.data) - self.pos

    def take(self, n):
        if n < 0 or self.pos + n > len(self.data):
            raise FormatError(
                f"unexpected end of data at offset {self.pos} (need {n}, have {self.remaining()})"
            )
        b = self.data[self.pos : self.pos + n]
        self.pos += n
        return b

    def u8(self):
        return self.take(1)[0]

    def u16(self):
        return struct.unpack("<H", self.take(2))[0]

    def i16(self):
        return struct.unpack("<h", self.take(2))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def f32(self):
        return struct.unpack("<f", self.take(4))[0]

    def f32s(self, n):
        return list(struct.unpack(f"<{n}f", self.take(4 * n)))

    def at_end(self):
        return self.pos == len(self.data)


class Writer:
    __slots__ = ("parts",)

    def __init__(self):
        self.parts = []

    def u8(self, v):
        self.parts.append(struct.pack("<B", v))

    def u16(self, v):
        self.parts.append(struct.pack("<H", v))

    def i16(self, v):
        self.parts.append(struct.pack("<h", v))

    def u32(self, v):
        self.parts.append(struct.pack("<I", v))

    def f32(self, v):
        self.parts.append(struct.pack("<f", v))

    def f32s(self, vs):
        self.parts.append(struct.pack(f"<{len(vs)}f", *vs))

    def raw(self, b):
        self.parts.append(bytes(b))

    def bytes(self):
        return b"".join(self.parts)
