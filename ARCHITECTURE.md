# Architecture

**roofline.c** is a transformer inference engine in C. It is a research project
first: the goal is correct, measurable inference, and speed only after
correctness is provable. Nothing here concerns product tiers or deployment
targets — the engine is capacity-agnostic and the same core serves a 100 MB
encoder and an 8 B language model.

---

## 1. Principles

These are not aspirations; each one already shows up in the code.

**Correctness before speed, and correctness is a number.** Every stage ends with
a numeric gate against a reference implementation, not a demo. A kernel that is
fast and subtly wrong is worse than no kernel, because it poisons everything
measured after it.

**Fail loudly on the unknown.** The GGUF reader reports `UNSUPPORTED` with
`nbytes == 0` for tensor types it cannot size, rather than guessing a block
size. A guess turns an unreadable file into silently wrong tensor offsets — the
worst failure mode a loader has.

**Zero-copy where the data is large.** Weights are `mmap`'d once and never
copied. A 4.58 GiB model loads in under 10 ms because nothing is read until it
is touched.

**All allocation flows through arenas.** There is no `free()` in the forward
pass. A forward pass allocates the same shapes in the same order every call, so
a bump pointer makes that free and removes fragmentation and leaks from the hot
loop entirely.

**Deterministic reduction order.** Floating-point addition is not associative,
so a reduction that splits differently across 4 and 8 threads produces different
bits. Every reduction here uses a fixed, thread-count-independent order. This
costs a little performance and buys the ability to regression-test accuracy at
all — without it, "did this kernel change the output?" is unanswerable.

**Measure, do not claim.** Every performance number in this repository is
reproducible from a command in it, or it is labelled a target.

## 2. Layers

Each layer depends only on those below it.

```
  models/     llama (causal LM)          vit (image encoder)
  ─────────────────────────────────────────────────────────
  graph/      op sequencing · KV cache (LM) · batching (ViT)
  ─────────────────────────────────────────────────────────
  ops/        matmul softmax attention layernorm rmsnorm
              rope gelu swiglu
  ─────────────────────────────────────────────────────────
  backend/    cpu-scalar │ cpu-simd (NEON, AVX2/VNNI) │ metal
  ─────────────────────────────────────────────────────────
  quant/      block formats · quantize · imatrix calibration
  ─────────────────────────────────────────────────────────
  io/         gguf loader · tokenizer (LM) · image preprocess (ViT)
  ─────────────────────────────────────────────────────────
  core/       arena · tensor descriptor · errors · threading
```

`backend/` is a vtable, not a compile-time switch. The same graph runs on any
backend and the outputs are compared against each other — a Metal kernel that
disagrees with the scalar CPU kernel is a bug with a known location.

## 3. Two model families, one substrate

The engine runs both a causal language model and a ViT image encoder. That is
not scope creep: it is the argument that the backend abstraction is real rather
than asserted, and the two exercise opposite halves of the performance problem.

| | Causal LM (Llama family) | Image encoder (CLIP / BioCLIP family) |
|---|---|---|
| Input | token ids | one image, or a batch |
| Normalization | RMSNorm | LayerNorm |
| Position | RoPE | learned embeddings |
| FFN | SwiGLU | GELU |
| Attention | causal mask, GQA | bidirectional, no mask |
| Entry | embedding lookup | patch-embed convolution |
| Exit | logits, sampling | CLS pooling, projection |
| State | KV cache across tokens | none |
| Bottleneck | **memory-bound GEMV** | **compute-bound GEMM** |

**Shared:** `core/`, `io/gguf`, `quant/`, `backend/`, and the matmul, softmax
and attention ops.

The last row is why both are worth having. Autoregressive decode reads every
weight once per token, so it is bandwidth-bound and the ceiling is arithmetic —
that is what the project is named for. Encoding images is compute-bound at high
arithmetic intensity. One engine, two rooflines, and a kernel tuned for one is
measurably wrong for the other.

## 4. Numerics

**Accumulate in fp32, always** — including for quantized inputs, where the
products are integer but the block scales are not. fp16 accumulation is a
frequent source of quiet drift in attention.

**Dequantize inside the inner loop.** Never materialize dequantized weights to
memory. On a bandwidth-bound decode the compression ratio *is* the speedup;
writing fp16 weights out to compare against destroys the entire benefit.

**One reference per op.** Every op has a scalar fp32 implementation that is
never optimized. It is the definition of the op. Optimized kernels are checked
against it, not against each other.

## 5. How accuracy is proven

The part that makes this a research project rather than a demo.

**Three levels of gate:**

1. **Per-op, against golden vectors.** Inputs and outputs captured from PyTorch,
   committed as fixtures. Tolerance is stated per op, not global.
2. **End-to-end, against the reference model.** For the LM, logit parity: max
   |Δ| < 1e-4 in fp32. For the encoder, embedding cosine similarity ≥ 0.9999
   against the PyTorch forward on the same image.
3. **Task metric, after quantization.** Numerical closeness is not the goal —
   task accuracy is. For an LM that is perplexity on held-out text. For a
   classifier it is top-1 and top-3 over the real label space.

**The regression harness is the deliverable, not a side effect.** Every kernel
change reruns levels 1 and 2, and any change beyond tolerance fails. Most
from-scratch engines optimize and hope; the interesting research question here
is not "can this be made fast" — it can — but **what accuracy each optimization
actually costs**, measured rather than assumed.

That question is what the quantization work in stage 02 exists to answer, and it
is only answerable because of the deterministic reduction order in §1.

## 6. Memory model

| Region | Lifetime | Backing |
|---|---|---|
| Weights | process | `mmap`, `PROT_READ`, never copied |
| Activations | one forward pass | arena, reset per call |
| Scratch | one op | arena `mark`/`release` |
| KV cache (LM) | one sequence | its own arena, persistent |
| Metadata | process | arena, allocated once at load |

The reservation is deliberately far larger than the resident set. Pages cost
nothing until written, so arenas are sized for the worst case and the high-water
mark is reported rather than guessed.

## 7. Status

| Component | State |
|---|---|
| `core/arena` | **built**, 54 checks, clean under ASAN+UBSan |
| `io/gguf` | **built**, 71 checks, verified against a 4.58 GiB model |
| `tc-inspect` | **built** |
| Unicode tables | generated, not yet wired up |
| Everything else | planned — see `README.md` |

The GGUF reader's correctness argument is worth recording: across 292 tensors of
a real Llama-3.1-8B Q4_K_M file, tensor payloads tile the data section exactly —
4,912,898,304 bytes of 4,912,898,304, none unaccounted. A single wrong block
size in the type table would leave a gap or an overlap. That is evidence, not a
claim.
