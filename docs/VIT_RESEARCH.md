# ViT path — research findings

**2026-09-07.** What the literature and the existing divegomobile record say about
running a CLIP-family image encoder in this engine, and what it changes.

Context: the application is species identification from diver photos in
divegomobile. That work already exists as `speciesEval/` on the `seaLifeAI`
branch. This document records only what bears on the engine.

---

## 1. The model is ViT-L/14 — confirmed

BioCLIP 2 uses a **ViT-L/14 image encoder** plus a masked self-attention text
encoder, trained on TreeOfLife-200M (214M images across seven Linnaean ranks),
initialized from LAION-2B ViT-L/14. Loaded through `open_clip`.

| | |
|---|---|
| Input | 224 x 224 |
| Patch | 14 -> 16 x 16 = 256 patches + 1 CLS = **257 tokens** |
| Image encoder | ~304 M parameters |
| Checkpoint (both towers, fp32) | ~1.6 GB, consistent with ViT-L/14 + text tower |

⚠️ **Layer count, width, head count and projection dim are NOT in the model
card.** The standard open_clip `ViT-L-14` is 24 layers / width 1024 / 16 heads /
projection 768, but that must be read from the checkpoint config, not assumed.
The export script's first job is to print it.

A newer `bioclip-2.5-vith14` (ViT-H/14) also exists and is larger.

**Image encoder alone, at weight precision:**

| Precision | Size | On an 8 GB M1 |
|---|---|---|
| fp32 | ~1.2 GB | fine |
| fp16 | ~608 MB | fine |
| Q8_0 | ~323 MB | fine |
| Q4_K | ~171 MB | fine |

The encoder is comfortable on this machine at any precision. Size is not the
constraint here — accuracy is.

## 2. The quantization literature, and the distinction that matters

Most published ViT post-training-quantization results are alarming:

- Naive PTQ on ViTs loses **>1% even at 8-bit**; PTQ4ViT recovers to **<0.5%**.
- At 6-bit, naive PTQ loses **9.8%** on average; PTQ4ViT **2.1%**.
- CLIP vision encoders specifically have **massive-scale activation outliers**,
  concentrated in a few channels in the middle-to-final blocks.
- Post-LayerNorm activations are irregularly distributed; post-Softmax values
  near 1 carry importance that log-scale quantizers miss; **patch embedding is
  especially sensitive** — disabling its quantizer recovers most of the loss.

⭐ **Almost all of that concerns quantizing *activations*.** GGUF-style
quantization is **weight-only**: activations stay fp32. That sidesteps the
outlier problem entirely, which is the single hardest part of ViT PTQ.

So the numbers above are an upper bound on our risk, not a forecast. Two further
findings point the same way:

- **Larger ViTs are less sensitive to quantization.** ViT-L is a favourable case.
- The known-sensitive tensors — patch embedding, final projection — are exactly
  the ones a mixed scheme keeps at higher precision. This is what `Q4_K_M`
  already does for language models: mostly Q4_K, with Q6_K on sensitive tensors.
  Confirmed on a real file in `tc-inspect`: 193 Q4_K tensors at 4.50 bpw plus 33
  Q6_K at 6.56.

## 3. The gap this lands in

⭐ **llama.cpp does not quantize vision encoders.** Multimodal projector
(`mmproj`) files are conventionally shipped at F16 or F32. The conversion script
accepts `{f32, f16, bf16, q8_0, tq1_0, tq2_0}`, but lower formats are "not
commonly recommended or distributed due to quality concerns" — the stated
reasoning being that the encoder's output feeds the language model, so its
quality bounds everything downstream.

There is an **open feature request** for quantized mmproj support (#18881) and a
discussion of Q4_0 for mmproj (#15453). The motivation given is exactly the size
problem: an mmproj at 1.33 GB in f16 is a bottleneck.

So the question *"what does weight-only 4-bit quantization actually cost a ViT
encoder on a real downstream task"* is **open in the ecosystem, with an issue
asking for it, and unanswered because nobody has measured it on a task with a
real label space.**

That is the research result stage 02 is positioned to produce, and it is also a
plausible route into the stage 06 contribution.

## 4. What the divegomobile record already establishes

Read from the project's own memory and `speciesEval/`. These constrain the plan
and correct some natural assumptions.

**Gemini is the primary path; BioCLIP is a validated shadow.** The 2026-08-13
decision was explicit: *"Gemini stays P0 primary; shadow validated; Stage 2
justified."* Adjudicated top-1/top-3, BioCLIP-2 with roster **82/92** against
Gemini with roster **95/95**. BioCLIP currently loses.

⚠️ **The main loss source is name matching, not vision.** BioCLIP speaks formal
taxonomy; the app's vocabulary is distinct `photos.label` common names. The
mismatch, not the encoder, dominates the error.

⭐ **Prototypes eliminate that problem**, which is why they gain so much: the
prediction space *is* the label vocabulary, so no synonym adjudication happens
at all. Over 97 labels and 756 photos: zero-shot **73/90**, prototype
**88.9/97.2**, blend **89.2/97.6**. The >=10-shot bucket reaches 92.6/97.7.

⚠️ **Caveats on those numbers, from the same record.** Ground truth is
unconfirmed user labels. Leave-one-out over same-dive photos of the same
individual likely inflates the prototypes, because near-duplicate frames leak
between fold and holdout. Treat 89/97 as optimistic.

**The data is thin.** 732 labels over 1,770 labeled photos; only **one** species
has >=20 photos, 24 have >=10, 97 have >=5. Growth is 30–90 photos/month, so any
plan that needs more labeled data organically is years out.

**On-device was already their Stage 3** — "on-device distill (product-gated)".
This engine lands on an existing plan, not a new idea.

**Reported eval latency was ~70 s/photo** for the open TreeOfLife condition, but
that scans ~950k labels and is explicitly *not* the serving shape. A 97-label
prototype comparison is a trivially small matmul by comparison.

## 5. What this changes for the engine

1. **Target ViT-L/14, 257 tokens, fp16 reference.** Size is not a constraint on
   this machine; correctness and accuracy are the whole problem.
2. **Weight-only quantization only.** Activation quantization is out of scope —
   it is where the ViT literature's difficulty lives, and GGUF does not do it.
3. **Expect a mixed scheme to be necessary.** Keep patch embedding and the final
   projection above 4-bit. Measure with them quantized *and* not, since the
   literature says patch embed is where the loss concentrates.
4. **The measurable result** is top-1/top-3 on the 97-label prototype task as a
   function of encoder weight precision — fp16, Q8_0, Q6_K, Q4_K — reusing the
   existing manifest and scoring so the comparison is apples-to-apples.
5. **State the caveats with the result.** The 89/97 baseline is optimistic for
   the reasons in §4. A quantization *delta* is far more robust than an absolute
   number, because both sides inherit the same bias.

## Sources

- BioCLIP 2 — [arXiv:2505.23883](https://arxiv.org/abs/2505.23883),
  [Imageomics/bioclip-2](https://github.com/Imageomics/bioclip-2),
  [model card](https://huggingface.co/imageomics/bioclip-2)
- [PTQ4ViT (ECCV 2022)](https://arxiv.org/pdf/2111.12293) — twin uniform quantization
- [Post-training quantization of vision encoders needs prefixing registers](https://arxiv.org/abs/2510.04547v1) — CLIP activation outliers
- [RepQ-ViT](https://arxiv.org/pdf/2212.08254), [ADFQ-ViT](https://arxiv.org/pdf/2407.02763) — post-LayerNorm / post-Softmax distributions
- llama.cpp [multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md),
  [issue #18881](https://github.com/ggml-org/llama.cpp/issues/18881),
  [discussion #15453](https://github.com/ggml-org/llama.cpp/discussions/15453)
