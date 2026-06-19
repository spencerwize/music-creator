#!/usr/bin/env python3
"""Phase 0 spike: introspect the ACE-Step transformer's control conditioning.

READ-ONLY. Loads the pipeline, runs a couple of forward passes, and prints the
shapes we need to implement the remix operator's control encoder
(DESIGN_remix_operator.md):

  - transformer block count and hidden/inner dim
  - the DCAE latent shape for a fixed-length audio clip
  - the per-block hidden_states sequence length (captured via a hook)
  - whether `block_controlnet_hidden_states` is accepted, and whether feeding it
    actually changes the output (i.e. the control path is live)

Run on the GPU box:
    python spike_control_shapes.py 2>&1 | tee spike_output.txt

Paste spike_output.txt back and the control encoder can be implemented directly.
"""

from __future__ import annotations

import inspect
import logging

import torch

from finetune import _build_conditioning
from generate import ACE_STEP_CHECKPOINT
from utils import configure_logging, pick_device

logger = logging.getLogger("music-gen")

CLIP_SECONDS = 10.0
SAMPLE_RATE = 48_000


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"== {title}")
    print("=" * 70)


def main() -> int:
    configure_logging(verbose=False)
    device = pick_device()
    print(f"device: {device}")

    from acestep.pipeline_ace_step import ACEStepPipeline

    banner("Loading pipeline")
    pipeline = ACEStepPipeline(
        checkpoint_dir=ACE_STEP_CHECKPOINT,
        dtype="bfloat16" if device == "cuda" else "float32",
        torch_compile=False,
    )
    if not getattr(pipeline, "loaded", False):
        pipeline.load_checkpoint(pipeline.checkpoint_dir)

    transformer = pipeline.ace_step_transformer
    music_dcae = pipeline.music_dcae
    model_dtype = next(music_dcae.parameters()).dtype
    print(f"transformer class: {type(transformer).__name__}")
    print(f"model dtype: {model_dtype}")

    # --- Transformer config / blocks ------------------------------------- #
    banner("Transformer config")
    cfg = getattr(transformer, "config", None)
    if cfg is not None:
        try:
            print(dict(cfg))
        except Exception:
            print(cfg)

    # Find the ModuleList of transformer blocks (name varies across impls).
    block_list = None
    block_attr = None
    for name, module in transformer.named_children():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            print(f"ModuleList child: {name} (len {len(module)})")
            if "block" in name.lower() or block_list is None:
                block_list, block_attr = module, name
    if block_list is not None:
        print(f"--> using block list '{block_attr}', num_blocks = {len(block_list)}")

    # --- forward() signature --------------------------------------------- #
    banner("transformer.forward signature")
    print(inspect.signature(transformer.forward))

    # --- DCAE latent shape ----------------------------------------------- #
    banner("DCAE latent shape")
    audio = torch.zeros(1, 2, int(CLIP_SECONDS * SAMPLE_RATE), device=device, dtype=model_dtype)
    with torch.no_grad():
        encoded = music_dcae.encode(audio)
    latents = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
    print(f"latents shape for {CLIP_SECONDS}s: {tuple(latents.shape)}  dtype={latents.dtype}")

    # --- Hook the first block to capture per-block hidden_states shape ---- #
    banner("Per-block hidden_states shape (hook)")
    captured = {}

    def hook(_module, args, kwargs):
        if args:
            captured["shape"] = tuple(args[0].shape)
        elif "hidden_states" in kwargs:
            captured["shape"] = tuple(kwargs["hidden_states"].shape)

    handle = None
    if block_list is not None:
        handle = block_list[0].register_forward_pre_hook(hook, with_kwargs=True)

    base_cond = _build_conditioning(pipeline, "instrumental, techno", device, model_dtype)
    bsz = latents.shape[0]
    attn = torch.ones(bsz, latents.shape[-1], device=device, dtype=model_dtype)
    t = torch.rand(bsz, device=device, dtype=model_dtype)

    banner("Forward WITHOUT control")
    try:
        with torch.no_grad():
            out = transformer(
                hidden_states=latents,
                attention_mask=attn,
                timestep=t,
                **base_cond,
            )
        sample = getattr(out, "sample", out)
        print(f"output shape: {tuple(sample.shape)}")
        baseline = sample.float().mean().item()
        print(f"output mean: {baseline:.6f}")
    except Exception as exc:
        print(f"forward (no control) FAILED: {type(exc).__name__}: {exc}")
        baseline = None

    if "shape" in captured:
        print(f"per-block hidden_states shape: {captured['shape']}")
        print("  -> control tensors likely need this shape per block "
              "(batch, seq_len, inner_dim)")
    else:
        print("could not capture per-block shape; inspect block list naming above")
    if handle is not None:
        handle.remove()

    # --- Try feeding block_controlnet_hidden_states ---------------------- #
    banner("Forward WITH control (probe)")
    if "shape" in captured and block_list is not None:
        seq_shape = captured["shape"]
        n_blocks = len(block_list)
        # Use a clearly non-zero control so any effect is visible.
        ctrl = [torch.ones(seq_shape, device=device, dtype=model_dtype) for _ in range(n_blocks)]
        for variant in (ctrl, ctrl[0]):  # try list-per-block, then single tensor
            try:
                with torch.no_grad():
                    out = transformer(
                        hidden_states=latents,
                        attention_mask=attn,
                        timestep=t,
                        block_controlnet_hidden_states=variant,
                        controlnet_scale=1.0,
                        **base_cond,
                    )
                sample = getattr(out, "sample", out)
                kind = "list[per-block]" if isinstance(variant, list) else "single tensor"
                mean = sample.float().mean().item()
                changed = baseline is None or abs(mean - baseline) > 1e-6
                print(f"ACCEPTED control as {kind}: output mean {mean:.6f} "
                      f"({'changed' if changed else 'no change'} vs baseline)")
                break
            except Exception as exc:
                kind = "list[per-block]" if isinstance(variant, list) else "single tensor"
                print(f"control as {kind} rejected: {type(exc).__name__}: {exc}")
    else:
        print("skipped (no captured per-block shape)")

    banner("DONE")
    print("Copy everything above back to continue implementing the control encoder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
