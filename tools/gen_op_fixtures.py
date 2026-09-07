#!/usr/bin/env python3
"""Golden vectors for the reference kernels, computed in numpy.

Needs numpy only — no torch — so the op-level gate works before any model is
downloaded. References are computed in float64 and stored as float32, so the
tolerance the C tests assert covers legitimate fp32 rounding and nothing else.

    .venv/bin/python tools/gen_op_fixtures.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_writer import GGUFWriter  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ops.gguf"
rng = np.random.default_rng(0)


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def d(a):
    return np.asarray(a, dtype=np.float64)


def main():
    w = GGUFWriter(str(OUT))
    w.add_string("general.architecture", "tc-op-fixtures")
    w.add_string("general.name", "golden vectors for src/ops.c")
    w.add_u32("fixtures.seed", 0)

    # -- gemm, deliberately odd sizes to catch indexing bugs ----------------
    for tag, M, N, K in (("small", 7, 5, 11), ("wide", 3, 129, 1024)):
        A = f32(rng.standard_normal((M, K)))
        B = f32(rng.standard_normal((N, K)))          # [N,K] -> transposed use
        w.add_tensor(f"gemm.{tag}.A", A)
        w.add_tensor(f"gemm.{tag}.B", B)
        w.add_tensor(f"gemm.{tag}.C", f32(d(A) @ d(B).T))
        w.add_u32(f"gemm.{tag}.M", M)
        w.add_u32(f"gemm.{tag}.N", N)
        w.add_u32(f"gemm.{tag}.K", K)

    # -- linear with bias, PyTorch weight layout [out, in] ------------------
    M, N, K = 6, 17, 64
    x = f32(rng.standard_normal((M, K)))
    W = f32(rng.standard_normal((N, K)))
    b = f32(rng.standard_normal(N))
    w.add_tensor("linear.x", x); w.add_tensor("linear.W", W); w.add_tensor("linear.b", b)
    w.add_tensor("linear.y", f32(d(x) @ d(W).T + d(b)))
    w.add_u32("linear.M", M); w.add_u32("linear.N", N); w.add_u32("linear.K", K)

    # -- layernorm. One row is given a large offset on purpose: it is the case
    #    a one-pass variance loses, and post-LayerNorm ViT activations live
    #    exactly there.
    rows, cols, eps = 5, 64, 1e-5
    ln = rng.standard_normal((rows, cols))
    ln[2] += 1e4
    ln = f32(ln)
    g = f32(rng.standard_normal(cols)); be = f32(rng.standard_normal(cols))
    mu = d(ln).mean(-1, keepdims=True)
    va = d(ln).var(-1, keepdims=True)                 # biased, as PyTorch
    w.add_tensor("ln.x", ln); w.add_tensor("ln.gamma", g); w.add_tensor("ln.beta", be)
    w.add_tensor("ln.out", f32((d(ln) - mu) / np.sqrt(va + eps) * d(g) + d(be)))
    w.add_u32("ln.rows", rows); w.add_u32("ln.cols", cols); w.add_f32("ln.eps", eps)

    # -- gelu, both flavours, over a range that includes the saturating tails
    gx = f32(np.concatenate([np.linspace(-8, 8, 257), rng.standard_normal(64) * 3]))
    from math import erf
    erfv = np.vectorize(erf)
    w.add_tensor("gelu.x", gx)
    w.add_tensor("gelu.exact", f32(0.5 * d(gx) * (1.0 + erfv(d(gx) / np.sqrt(2.0)))))
    w.add_tensor("gelu.quick", f32(d(gx) / (1.0 + np.exp(-1.702 * d(gx)))))

    # -- softmax, including a row with a large constant offset (stability)
    rows, cols = 4, 33
    sx = rng.standard_normal((rows, cols))
    sx[1] += 800.0                                    # naive exp() overflows here
    sx = f32(sx)
    e = np.exp(d(sx) - d(sx).max(-1, keepdims=True))
    w.add_tensor("softmax.x", sx)
    w.add_tensor("softmax.out", f32(e / e.sum(-1, keepdims=True)))
    w.add_u32("softmax.rows", rows); w.add_u32("softmax.cols", cols)

    # -- bidirectional multi-head attention, ViT shape in miniature ---------
    T, H, HD = 17, 4, 16
    D = H * HD
    q = f32(rng.standard_normal((T, D)))
    k = f32(rng.standard_normal((T, D)))
    v = f32(rng.standard_normal((T, D)))
    out = np.zeros((T, D), dtype=np.float64)
    for h in range(H):
        sl = slice(h * HD, (h + 1) * HD)
        s = d(q[:, sl]) @ d(k[:, sl]).T / np.sqrt(HD)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        out[:, sl] = s @ d(v[:, sl])
    w.add_tensor("mha.q", q); w.add_tensor("mha.k", k); w.add_tensor("mha.v", v)
    w.add_tensor("mha.out", f32(out))
    w.add_u32("mha.T", T); w.add_u32("mha.n_head", H); w.add_u32("mha.head_dim", HD)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = w.write()
    print(f"wrote {OUT}  ({n/1024:.1f} KiB)")


main()
