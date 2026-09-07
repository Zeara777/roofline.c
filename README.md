# roofline.c

A transformer inference engine written from scratch in C.

Named for the constraint the whole project turns on: autoregressive decode reads
every weight once per token, so it is bandwidth-bound. That gives a hard ceiling
before a single kernel is written — on this M1, ~68 GB/s divided by the model's
size in bytes. It is both the optimization target and the signal to stop.

Building a transformer inference engine from scratch in C, one layer at a time.
Design and the accuracy discipline: [`ARCHITECTURE.md`](ARCHITECTURE.md).
Research findings for the image-encoder path: [`docs/VIT_RESEARCH.md`](docs/VIT_RESEARCH.md).
The staged build plan: [`docs/bare-metal-transformers.html`](docs/bare-metal-transformers.html).

```
01  fp32 forward pass — GGUF loader, BPE tokenizer, RMSNorm/RoPE/GQA/SwiGLU   ← in progress
02  quantization + hand-written SIMD (NEON on M1, AVX2/VNNI on x86)
03  Metal backend
04  serving layer — paged KV cache, continuous batching, streaming HTTP
05  training in C — every backward pass derived by hand   (companion binary)
```

## Build

```sh
make            # build tools and run tests (disk-checked first)
make tools      # just the binaries
make test       # just the tests
make STRICT=1   # add -Wconversion -Wsign-conversion -Wdouble-promotion
make ASAN=1     # address + UB sanitizers (use for tests, never benchmarks)
make disk       # just the disk guard
```

### Session summaries

Every session writes a summary to `~/.claude/session-summaries/CPT/YYYY-MM-DD-<slug>.md`
— deliberately outside the repo, so one can never end up in a commit or a PR. A `Stop`
hook checks the rule and reminds when a summary is owed; it stays quiet when today's file
already exists, and quiet when the session produced no commits and no working-tree
changes, since a read-only session has nothing to summarize.

```sh
make summary SLUG=vit-forward     # scaffold today's file
scripts/session-summary.sh list   # what exists
scripts/session-summary.sh path   # where they go
```

### Disk

This project pulls model checkpoints measured in gigabytes and writes GGUF
exports beside them. Running out of disk mid-export leaves a truncated file that
still parses far enough to look plausible, so `scripts/check-disk.sh` runs
before every `make test` and, via a `SessionStart` hook in
`.claude/settings.json`, at the start of every session. Thresholds are
`ROOFLINE_DISK_FAIL_GIB` (default 5) and `ROOFLINE_DISK_WARN_GIB` (default 15).

C11, no dependencies beyond libc. Clean under `-Wall -Wextra -Wpedantic`, under
`STRICT=1`, and under ASAN+UBSan.

Binaries are prefixed `tc-` — transformer-in-C — (`tc-inspect`, later
`tc-quant`, `tc-serve`). Deliberately never a bare `tc`: that is iproute2's
traffic-control binary on Linux, and stage 02 targets an x86 Linux box.

## What works now

**`arena.c`** — bump allocator over a lazily-committed `mmap` reservation.
Every allocation in the engine goes through it. No `free()`: memory returns via
`arena_reset()` or scoped `arena_mark()`/`arena_release()`. A forward pass
allocates the same shapes in the same order every token, so a bump pointer makes
that free and removes any chance of a fragmentation or leak bug in the hot loop.

**`gguf.c`** — zero-copy GGUF v2/v3 reader. The file is mapped once, `PROT_READ`;
metadata is parsed into arena-allocated descriptors whose strings and tensor
payloads are pointers *into* the mapping. Nothing is copied. Every read is bounds
checked against the mapping, so a truncated or corrupt file produces a message
rather than a segfault.

**`tc-inspect`** — dumps a model's structure. Stage 01's first checkpoint.

```sh
./build/tc-inspect model.gguf              # header + metadata + type summary
./build/tc-inspect model.gguf --tensors    # also list every tensor
```

Against Llama-3.1-8B-Instruct Q4_K_M (4.58 GiB):

```
  gguf v3   alignment 32   file 4.58 GiB   tensor data 4.58 GiB
  architecture llama   layers 32   d_model 4096   heads 32   kv-heads 8   ctx 131072

  type      tensors       parameters          bytes    bits/wt
  F32            66           266304       1.02 MiB      32.00
  Q4_K          193       6498025472       3.40 GiB       4.50
  Q6_K           33       1531969536       1.17 GiB       6.56
  total         292       8030261312       4.58 GiB       4.89

  tensor data: 4912898304 bytes used of 4912898304 in the section  (exact tiling)
  decode ceiling at 68 GB/s: 13.8 tok/s
```

Two things worth noting there. **Exact tiling** — 292 tensors' payloads cover the
data section with zero bytes unaccounted — is strong evidence every block size in
the type table is correct; a single wrong size would leave a gap or an overlap.
And the whole 4.58 GiB file loads and parses in under 10 ms, because nothing is
read until it is touched.

## Notes on the type table

Block sizes are the on-disk ggml layouts. Q4_K is a 256-element superblock:
2 (`d`) + 2 (`dmin`) + 12 (6-bit scales and mins) + 128 (nibbles) = 144 bytes,
i.e. 4.50 bits per weight. The `IQ*` and `TQ*` families are deliberately absent —
this engine will never read them, and guessing a block size would turn an
unsupported file into silently wrong tensor offsets. Unknown types report
`UNSUPPORTED` and `nbytes == 0`.

The reader assumes a little-endian host (both targets are). A big-endian GGUF is
detected by its reversed magic and rejected by name rather than misparsed.

GGUF array elements sit at whatever file offset they land on and are *not*
naturally aligned, so values are copied out with `memcpy` rather than cast.
On arm64 a cast happens to work; it is still undefined behavior, and UBSan
flags it.

## The image-encoder path

`tools/export_vit_gguf.py` exports an open_clip ViT to GGUF plus reference
activations. `--config-only` needs no torch and answers the architecture
question straight from the hub:

```
$ .venv/bin/python tools/export_vit_gguf.py --config-only
  layers 24 · width 1024 · heads 16 · ffn 4096 · patch 14 / image 224
  tokens 16x16 + 1 CLS = 257 · projection dim 768 · parameters 303,966,208
    fp16 607.9 MB   Q8_0 323.0 MB   Q4_K 171.0 MB
```

Fixtures are written as GGUF too, so the C tests read them with the loader that
already exists rather than a second file format. `tools/gguf_writer.py` is a
pure-Python mirror of `src/gguf.c` — writing with one and reading with the other
round-trips the format through two independent implementations.

## Next

- [ ] Tensor descriptor and op dispatch
- [ ] Scalar fp32 reference kernels: matmul, softmax, layernorm, gelu
- [ ] Golden-vector harness reading `*-fixtures.gguf`
- [ ] ViT forward: patch embed → pos → 24 blocks → CLS pool → projection
- [ ] **Gate:** embedding cosine ≥ 0.9999 vs the PyTorch reference
- [ ] Prototype classifier + roster filter
- [ ] **Result:** top-1/top-3 as a function of encoder weight precision
- [ ] *Deferred:* BPE tokenizer and the causal-LM path
