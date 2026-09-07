"""Minimal pure-Python GGUF v3 writer. No dependencies beyond numpy.

Deliberately hand-written rather than pulling in the `gguf` package: it is the
exact mirror of src/gguf.c, so writing with this and reading with tc-inspect
round-trips the format through two independent implementations. A disagreement
between them is a bug one of them can be blamed for.

ggml stores dims fastest-varying first, which is the reverse of a C-contiguous
numpy shape. add_tensor() reverses for you — pass the numpy shape.
"""
import struct
import numpy as np

MAGIC = 0x46554747
VERSION = 3
ALIGNMENT = 32

# metadata value types — must match gguf_type_t in src/gguf.h
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

# ggml tensor types — must match ggml_type_t in src/gguf.h
GGML_F32, GGML_F16 = 0, 1


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


class GGUFWriter:
    def __init__(self, path):
        self.path = path
        self.kv = []        # (key, encoded_value_bytes)
        self.tensors = []   # (name, np.ndarray, ggml_type)

    # ---- metadata -------------------------------------------------------
    def add_string(self, key, val):
        self.kv.append((key, struct.pack("<I", T_STRING) + _str(val)))

    def add_u32(self, key, val):
        self.kv.append((key, struct.pack("<II", T_UINT32, val)))

    def add_f32(self, key, val):
        self.kv.append((key, struct.pack("<I", T_FLOAT32) + struct.pack("<f", val)))

    def add_bool(self, key, val):
        self.kv.append((key, struct.pack("<IB", T_BOOL, 1 if val else 0)))

    def add_f32_array(self, key, vals):
        body = struct.pack("<IQ", T_FLOAT32, len(vals))
        body += b"".join(struct.pack("<f", float(v)) for v in vals)
        self.kv.append((key, struct.pack("<I", T_ARRAY) + body))

    def add_string_array(self, key, vals):
        body = struct.pack("<IQ", T_STRING, len(vals))
        body += b"".join(_str(v) for v in vals)
        self.kv.append((key, struct.pack("<I", T_ARRAY) + body))

    # ---- tensors --------------------------------------------------------
    def add_tensor(self, name, arr, dtype="f32"):
        """arr: numpy array. Stored C-contiguous; dims written reversed for ggml."""
        if dtype == "f32":
            arr, ty = np.ascontiguousarray(arr, dtype="<f4"), GGML_F32
        elif dtype == "f16":
            arr, ty = np.ascontiguousarray(arr, dtype="<f2"), GGML_F16
        else:
            raise ValueError(f"unsupported dtype {dtype}")
        if arr.ndim < 1 or arr.ndim > 4:
            raise ValueError(f"{name}: rank {arr.ndim} outside 1..4")
        self.tensors.append((name, arr, ty))

    # ---- write ----------------------------------------------------------
    def write(self):
        head = struct.pack("<IIQQ", MAGIC, VERSION, len(self.tensors), len(self.kv))
        for key, val in self.kv:
            head += _str(key) + val

        # Two passes: tensor infos carry offsets into the data section, so the
        # offsets must be known before the infos are serialized.
        infos, offset = b"", 0
        for name, arr, ty in self.tensors:
            dims = list(arr.shape)[::-1]          # ggml: fastest-varying first
            infos += _str(name) + struct.pack("<I", len(dims))
            infos += b"".join(struct.pack("<Q", d) for d in dims)
            infos += struct.pack("<IQ", ty, offset)
            offset += arr.nbytes
            offset = (offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1)

        pre = head + infos
        pad = (-len(pre)) % ALIGNMENT

        with open(self.path, "wb") as f:
            f.write(pre)
            f.write(b"\0" * pad)
            for _, arr, _ in self.tensors:
                f.write(arr.tobytes())
                f.write(b"\0" * ((-arr.nbytes) % ALIGNMENT))

        total = len(pre) + pad + offset
        return total
