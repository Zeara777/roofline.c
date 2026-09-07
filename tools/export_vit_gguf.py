#!/usr/bin/env python3
"""Export an open_clip ViT image encoder to GGUF, with reference fixtures.

Stage 01 of the ViT path. Two modes:

  --config-only   Print the architecture and preprocessing. Needs NO torch —
                  it reads open_clip_config.json from the hub. Run this first.

  (default)       Load the model, write <out>.gguf (weights) and
                  <out>-fixtures.gguf (reference activations for the C tests).

Fixtures are written as GGUF so the C test harness reads them with the loader
it already has, rather than inventing a second file format.

Setup for the full mode:
    python3 -m venv .venv && .venv/bin/pip install numpy torch open_clip_torch
    .venv/bin/python tools/export_vit_gguf.py --model hf-hub:imageomics/bioclip-2

torch + open_clip cost ~600 MB on macOS arm64 (measured); the BioCLIP-2
checkpoint is ~1.6 GB.
"""
import argparse
import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_writer import GGUFWriter  # noqa: E402

ARCH = "clip-vit"

# open_clip visual.* -> our GGUF names. Every parameter must match exactly one
# rule; an unmatched key aborts the export rather than being silently dropped,
# because a missing tensor is a wrong model that still runs.
NAME_MAP = [
    (r"^conv1\.weight$",                      "v.patch_embd.weight"),
    (r"^class_embedding$",                    "v.cls_token"),
    (r"^positional_embedding$",               "v.pos_embd"),
    (r"^ln_pre\.(weight|bias)$",              r"v.ln_pre.\1"),
    (r"^ln_post\.(weight|bias)$",             r"v.ln_post.\1"),
    (r"^proj$",                               "v.proj"),
    (r"^transformer\.resblocks\.(\d+)\.ln_1\.(weight|bias)$",           r"v.blk.\1.ln1.\2"),
    (r"^transformer\.resblocks\.(\d+)\.ln_2\.(weight|bias)$",           r"v.blk.\1.ln2.\2"),
    (r"^transformer\.resblocks\.(\d+)\.attn\.in_proj_(weight|bias)$",   r"v.blk.\1.attn_qkv.\2"),
    (r"^transformer\.resblocks\.(\d+)\.attn\.out_proj\.(weight|bias)$", r"v.blk.\1.attn_out.\2"),
    (r"^transformer\.resblocks\.(\d+)\.mlp\.c_fc\.(weight|bias)$",      r"v.blk.\1.ffn_up.\2"),
    (r"^transformer\.resblocks\.(\d+)\.mlp\.c_proj\.(weight|bias)$",    r"v.blk.\1.ffn_down.\2"),
]


def map_name(k):
    for pat, rep in NAME_MAP:
        if re.match(pat, k):
            return re.sub(pat, rep, k)
    return None


def detect_activation(mlp):
    """Which GELU does this checkpoint want? Probe it, do not assume.

    open_clip_config.json does not say, and the two differ by ~2e-2 at x=-2 —
    small enough to look like a kernel bug and large enough to change an
    embedding. So run the checkpoint's own activation module on fixed inputs
    and compare against both definitions. The answer goes in the GGUF, so the
    C side reads it rather than inheriting this guess.
    """
    import math
    import torch

    act = None
    for _, mod in mlp.named_children():
        if "Linear" not in type(mod).__name__:
            act = mod
    if act is None:
        sys.exit("ABORT: no activation module found in the MLP")

    x = torch.tensor([-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0])
    with torch.no_grad():
        got = act(x)
    exact = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
    quick = x * torch.sigmoid(1.702 * x)
    if torch.allclose(got, exact, atol=1e-6):
        return "gelu"
    if torch.allclose(got, quick, atol=1e-6):
        return "gelu_quick"
    sys.exit(f"ABORT: activation {type(act).__name__} matches neither GELU "
             f"nor QuickGELU: {got.tolist()}")


def detect_pooling(visual):
    """CLS token, or mean over patches? Getting this wrong still produces a
    plausible embedding, so it is recorded rather than assumed.

    open_clip spells this `pool_type` ('tok'/'avg'/'none'); older forks used a
    `global_average_pool` bool. Both are read, because a missing attribute
    silently defaulting to 'cls' would be a guess wearing a check's clothes."""
    if getattr(visual, "attn_pool", None) is not None:
        return "attn"
    pool_type = getattr(visual, "pool_type", None)
    if pool_type is not None:
        return {"tok": "cls", "avg": "mean", "none": "none"}[pool_type]
    if getattr(visual, "global_average_pool", False):
        return "mean"
    sys.exit("ABORT: cannot determine pooling — no pool_type, no global_average_pool")

def _ssl_ctx():
    """The python.org framework Python ships without root CAs wired into ssl.
    Same trap speciesEval/shadow/run_bioclip.py hits; same fix."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_config(repo, cached=None):
    """Read open_clip_config.json from a local path if given, else the hub."""
    if cached:
        with open(cached) as fh:
            return json.load(fh)
    url = f"https://huggingface.co/{repo}/raw/main/open_clip_config.json"
    with urllib.request.urlopen(url, timeout=30, context=_ssl_ctx()) as r:
        return json.load(r)


def describe(cfg, source):
    m = cfg["model_cfg"]
    v = m["vision_cfg"]
    width, layers = v["width"], v["layers"]
    patch, image = v["patch_size"], v["image_size"]
    # open_clip derives vision heads as width // head_width, head_width default 64.
    heads = v.get("heads") or width // v.get("head_width", 64)
    mlp_ratio = v.get("mlp_ratio", 4.0)
    ffn = int(width * mlp_ratio)
    grid = image // patch
    tokens = grid * grid + 1

    print(f"--- vision tower  ({source}) ---")
    print(f"  layers            {layers}")
    print(f"  width             {width}")
    print(f"  heads             {heads}   (head_dim {width // heads})")
    print(f"  ffn               {ffn}   (mlp_ratio {mlp_ratio})")
    print(f"  patch / image     {patch} / {image}")
    print(f"  tokens            {grid}x{grid} + 1 CLS = {tokens}")
    print(f"  projection dim    {m['embed_dim']}")

    # Parameter count, derived rather than looked up.
    n  = 3 * patch * patch * width                 # patch embed conv (no bias)
    n += width                                     # class embedding
    n += tokens * width                            # position embedding
    n += 2 * width                                 # ln_pre
    per = (2 * width                               # ln_1
           + width * 3 * width + 3 * width         # qkv
           + width * width + width                 # out proj
           + 2 * width                             # ln_2
           + width * ffn + ffn                     # fc
           + ffn * width + width)                  # proj
    n += layers * per
    n += 2 * width                                 # ln_post
    n += width * m["embed_dim"]                    # visual projection
    print(f"  parameters        {n:,}")
    for label, bpw in (("fp32", 32), ("fp16", 16), ("Q8_0", 8.5), ("Q4_K", 4.5)):
        print(f"    {label:<5} {n * bpw / 8 / 1e6:>10.1f} MB")

    p = cfg.get("preprocess_cfg", {})
    print("--- preprocessing (the C side must match this exactly) ---")
    print(f"  mean   {p.get('mean')}")
    print(f"  std    {p.get('std')}")
    print(f"  interp {p.get('interpolation')}   resize {p.get('resize_mode')}")
    return dict(layers=layers, width=width, heads=heads, ffn=ffn, patch=patch,
                image=image, tokens=tokens, embed_dim=m["embed_dim"], pre=p)


def export(args, spec):
    import numpy as np
    import torch
    import open_clip

    print(f"\nloading {args.model} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    model.eval()
    visual = model.visual

    # ---- weights ----
    w = GGUFWriter(args.out)
    w.add_string("general.architecture", ARCH)
    w.add_string("general.name", args.model)
    w.add_u32(f"{ARCH}.block_count", spec["layers"])
    w.add_u32(f"{ARCH}.embedding_length", spec["width"])
    w.add_u32(f"{ARCH}.attention.head_count", spec["heads"])
    w.add_u32(f"{ARCH}.feed_forward_length", spec["ffn"])
    w.add_u32(f"{ARCH}.patch_size", spec["patch"])
    w.add_u32(f"{ARCH}.image_size", spec["image"])
    w.add_u32(f"{ARCH}.context_length", spec["tokens"])
    w.add_u32(f"{ARCH}.projection_dim", spec["embed_dim"])
    w.add_f32(f"{ARCH}.layer_norm_epsilon", 1e-5)
    # Detected off the checkpoint, never assumed — see detect_activation.
    w.add_string(f"{ARCH}.activation", detect_activation(visual.transformer.resblocks[0].mlp))
    w.add_string(f"{ARCH}.pooling", detect_pooling(visual))
    w.add_f32_array(f"{ARCH}.preprocess.mean", spec["pre"]["mean"])
    w.add_f32_array(f"{ARCH}.preprocess.std", spec["pre"]["std"])
    w.add_string(f"{ARCH}.preprocess.interpolation", spec["pre"]["interpolation"])
    w.add_string(f"{ARCH}.preprocess.resize_mode", spec["pre"]["resize_mode"])

    unmapped, n_par = [], 0
    for k, t in visual.state_dict().items():
        name = map_name(k)
        if name is None:
            unmapped.append(k)
            continue
        w.add_tensor(name, t.detach().cpu().float().numpy(), dtype=args.dtype)
        n_par += t.numel()
    if unmapped:
        sys.exit(f"ABORT: {len(unmapped)} unmapped parameter(s): {unmapped[:8]}")

    total = w.write()
    print(f"wrote {args.out}  ({n_par:,} params, {total/1e6:.1f} MB, {args.dtype})")

    # ---- reference fixtures ----
    # A seeded synthetic image, so the fixture is reproducible without shipping
    # a photo. Real-image parity is a separate check once the C path runs.
    g = torch.Generator().manual_seed(0)
    px = torch.randn(1, 3, spec["image"], spec["image"], generator=g)

    taps = {}
    hooks = [
        visual.ln_pre.register_forward_hook(lambda m, i, o: taps.update(ln_pre=o)),
        visual.transformer.resblocks[0].register_forward_hook(lambda m, i, o: taps.update(blk0=o)),
        visual.transformer.resblocks[-1].register_forward_hook(lambda m, i, o: taps.update(blk_last=o)),
        visual.ln_post.register_forward_hook(lambda m, i, o: taps.update(ln_post=o)),
    ]
    with torch.no_grad():
        emb = visual(px)
    for h in hooks:
        h.remove()

    f = GGUFWriter(args.out.replace(".gguf", "-fixtures.gguf"))
    f.add_string("general.architecture", ARCH + "-fixtures")
    f.add_string("general.name", f"reference activations for {args.model}")
    f.add_u32("fixtures.seed", 0)
    f.add_tensor("input.pixels", px.numpy())
    for k, v in taps.items():
        a = v[0] if isinstance(v, tuple) else v
        f.add_tensor(f"ref.{k}", a.detach().cpu().float().numpy())
    f.add_tensor("ref.embedding", emb.detach().cpu().float().numpy())
    n = emb / emb.norm(dim=-1, keepdim=True)
    f.add_tensor("ref.embedding_normalized", n.detach().cpu().float().numpy())
    tot = f.write()
    print(f"wrote {f.path}  ({tot/1e3:.1f} KB)")
    print(f"  embedding norm {float(emb.norm()):.6f}  first 4 {emb[0,:4].tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf-hub:imageomics/bioclip-2")
    ap.add_argument("--out", default="bioclip2-vit-l14-f16.gguf")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16"])
    ap.add_argument("--config-only", action="store_true",
                    help="print the architecture without loading torch")
    ap.add_argument("--config-file",
                    help="read open_clip_config.json locally instead of the hub")
    args = ap.parse_args()

    repo = args.model.split("hf-hub:", 1)[-1]
    cfg = fetch_config(repo, args.config_file)
    spec = describe(cfg, args.config_file or f"hub: {repo}")

    if args.config_only:
        print("\n(config only — pass no --config-only to export)")
        return
    export(args, spec)


main()
