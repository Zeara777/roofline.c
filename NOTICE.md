# Third-party notices

`roofline.c` itself is MIT — see [LICENSE](LICENSE). Nothing third-party is
vendored into this repository. What follows covers the artifacts the tooling
*produces*, because a `.gguf` this repo writes carries someone else's weights
and travels without the repo attached to it.

## BioCLIP 2 — `imageomics/bioclip-2`

Licensed **MIT**. Commercial use and redistribution of derivative works —
including quantized `.gguf` conversions — are permitted, provided the copyright
notice and license text travel with them.

Which is the whole reason `tools/export_vit_gguf.py` writes `general.license`
and `general.license.link` into every file it produces: a model file gets copied
around on its own, so the attribution has to live *inside* it rather than in a
README the file will be separated from.

BioCLIP 2 is fine-tuned from CLIP pre-trained on LAION-2B. The weights are MIT;
the upstream training corpus has its own, murkier situation, which is worth
knowing but does not bear on use of the weights themselves.

If you publish a converted model, cite the original work:

```bibtex
@software{bioclip2,
  title  = {BioCLIP 2},
  author = {Imageomics},
  url    = {https://huggingface.co/imageomics/bioclip-2},
  license = {MIT}
}
```

## OpenCLIP

`tools/export_vit_gguf.py` imports `open_clip_torch` (MIT) at export time only.
It is a build-time dependency and is not linked into any binary here.

## GGUF

The format and the ggml type/block-size table are from `ggml` / `llama.cpp`
(MIT). `src/gguf.c` is an independent implementation written against the format
specification — no ggml source is copied into this repository.
