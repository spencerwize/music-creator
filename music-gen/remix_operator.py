"""Remix operator — conditioning module + paired training and inference.

Learns the transformation original -> remix from paired data (pairs.py) and
applies it to a new original. The base ACE-Step model is frozen; we train a
small ControlEncoder plus an attention LoRA.

Grounded in the Phase 0 spike (DESIGN_remix_operator.md):
  - ACEStepTransformer2DModel: 24 blocks, in_channels=8, inner_dim=2560,
    max_height=16, patch_size=[16,1].
  - DCAE latent for audio is [B, 8, 16, W] where W is the time axis.
  - After proj_in the per-block hidden states are [B, W, 2560].
  - Control is injected ONCE as a single tensor of that shape:
        control_condi = cross_norm(hidden_states, block_controlnet_hidden_states)
        hidden_states = hidden_states + control_condi * controlnet_scale
    so the ControlEncoder must map [B, 8, 16, W] -> [B, W, 2560].

STATUS: implemented but not yet run on GPU; expect to iterate on ACE-Step
internals (DCAE encode/decode return types, conditioning shapes).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finetune import _attach_lora, _build_conditioning, _expand_conditioning
from generate import ACE_STEP_CHECKPOINT
from pairs import PairedRemixDataset
from style import extract_style
from utils import (
    LORAS_OP_DIR,
    TARGET_SAMPLE_RATE,
    load_audio,
    resolve_paths,
    save_audio,
    unique_path,
)

logger = logging.getLogger("music-gen")


@dataclass
class OperatorConfig:
    name: str
    style: str
    epochs: int = 100
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32  # consumed by finetune._attach_lora
    segment_seconds: float = 20.0
    control_scale: float = 1.0
    match_key: bool = True
    save_every: int = 25


# --------------------------------------------------------------------------- #
# Control encoder
# --------------------------------------------------------------------------- #
class ControlEncoder(nn.Module):
    """Map original-song latents [B, C, H, W] -> control [B, W, inner_dim].

    Mirrors the transformer's patch embedding (a stride-(H,1) conv collapses the
    height into the feature dim and keeps the time axis), followed by a few
    depthwise temporal conv blocks for context. The output projection is
    zero-initialised (ControlNet-style) so training starts as a no-op and the
    control influence grows smoothly.
    """

    def __init__(
        self,
        in_channels: int = 8,
        height: int = 16,
        inner_dim: int = 2560,
        depth: int = 3,
    ):
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels, inner_dim, kernel_size=(height, 1), stride=(height, 1)
        )
        groups = max(1, inner_dim // 64)
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(inner_dim, inner_dim, kernel_size=3, padding=1, groups=groups)
                for _ in range(depth)
            ]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(8, inner_dim) for _ in range(depth)])
        self.act = nn.SiLU()
        self.out = nn.Conv1d(inner_dim, inner_dim, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B, C, H, W]
        x = self.patch(latents)          # [B, inner_dim, 1, W]
        x = x.squeeze(2)                 # [B, inner_dim, W]
        for blk, norm in zip(self.blocks, self.norms):
            x = x + self.act(blk(norm(x)))
        x = self.out(x)                  # [B, inner_dim, W]
        return x.transpose(1, 2)         # [B, W, inner_dim]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _encode_latents(music_dcae, audio: torch.Tensor) -> torch.Tensor:
    """Encode [B, 2, N] audio to latents [B, 8, 16, W], handling return shapes."""
    encoded = music_dcae.encode(audio)
    return encoded[0] if isinstance(encoded, (tuple, list)) else encoded


def _decode_audio(music_dcae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents [B, 8, 16, W] back to audio [B, 2, N]."""
    decoded = music_dcae.decode(latents)
    return decoded[0] if isinstance(decoded, (tuple, list)) else decoded


def _load_pipeline(device: str):
    from acestep.pipeline_ace_step import ACEStepPipeline

    pipeline = ACEStepPipeline(
        checkpoint_dir=ACE_STEP_CHECKPOINT,
        dtype="bfloat16" if device == "cuda" else "float32",
        torch_compile=False,
    )
    if not getattr(pipeline, "loaded", False):
        pipeline.load_checkpoint(pipeline.checkpoint_dir)
    return pipeline


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_train_operator(
    style: str,
    name: str,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    segment_seconds: float = 20.0,
    match_key: bool = True,
    device: str = "cuda",
) -> Path:
    """Train a remix operator on references/<style>/{originals,remixes}."""
    cfg = OperatorConfig(
        name=name,
        style=style,
        epochs=epochs,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        segment_seconds=segment_seconds,
        match_key=match_key,
    )
    out_dir = LORAS_OP_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ACE-Step pipeline for operator training on %s", device)
    pipeline = _load_pipeline(device)
    transformer = pipeline.ace_step_transformer
    music_dcae = pipeline.music_dcae
    music_dcae.requires_grad_(False)
    music_dcae.eval()
    model_dtype = next(music_dcae.parameters()).dtype

    transformer.requires_grad_(False)
    peft_model = _attach_lora(transformer, cfg)
    peft_model.train()

    tconf = transformer.config
    encoder = ControlEncoder(
        in_channels=tconf.in_channels,
        height=tconf.max_height,
        inner_dim=tconf.inner_dim,
    ).to(device)  # float32 for training stability
    encoder.train()

    dataset = PairedRemixDataset(
        style,
        segment_seconds=segment_seconds,
        sample_rate=TARGET_SAMPLE_RATE,
        match_key=match_key,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Style conditioning derived from the remixes (the target sound).
    remix_paths = [p.remix for p in dataset.pairs]
    style_prompt = extract_style(remix_paths, device=device).as_prompt_fragment()
    base_cond = _build_conditioning(pipeline, style_prompt, device, model_dtype)

    trainable = list(encoder.parameters()) + [
        p for p in peft_model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    logger.info(
        "Training remix operator '%s' on %d pairs, %d segments/epoch",
        name, len(dataset.pairs), len(dataset),
    )
    avg = float("nan")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            original = batch["original"].to(device=device, dtype=model_dtype)
            remix = batch["remix"].to(device=device, dtype=model_dtype)

            with torch.no_grad():
                z_orig = _encode_latents(music_dcae, original)
                z_remix = _encode_latents(music_dcae, remix)

            control = encoder(z_orig.float()).to(model_dtype)  # [B, W, inner]

            noise = torch.randn_like(z_remix)
            t = torch.rand(z_remix.shape[0], device=device, dtype=z_remix.dtype)
            t_exp = t.view(-1, *([1] * (z_remix.dim() - 1)))
            noisy = (1 - t_exp) * z_remix + t_exp * noise
            target = noise - z_remix

            bsz = z_remix.shape[0]
            cond = _expand_conditioning(base_cond, bsz)
            attn = torch.ones(bsz, z_remix.shape[-1], device=device, dtype=model_dtype)

            out = peft_model(
                hidden_states=noisy,
                attention_mask=attn,
                timestep=t,
                block_controlnet_hidden_states=control,
                controlnet_scale=cfg.control_scale,
                **cond,
            )
            pred = getattr(out, "sample", None)
            if pred is None:
                pred = out[0] if isinstance(out, (tuple, list)) else out

            loss = F.mse_loss(pred.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        avg = epoch_loss / max(1, len(loader))
        if epoch % 5 == 0 or epoch == 1:
            logger.info("epoch %3d/%d | loss %.5f", epoch, epochs, avg)
        if epoch % cfg.save_every == 0 and epoch != epochs:
            _save_operator(peft_model, encoder, out_dir, cfg, style_prompt, tconf, epoch, avg)

    _save_operator(peft_model, encoder, out_dir, cfg, style_prompt, tconf, epochs, avg)
    logger.info("Remix operator saved to %s", out_dir)
    return out_dir


def _save_operator(peft_model, encoder, out_dir, cfg, style_prompt, tconf, epoch, loss):
    peft_model.save_pretrained(str(out_dir))
    torch.save(encoder.state_dict(), out_dir / "control_encoder.pt")
    meta = {
        "name": cfg.name,
        "style": cfg.style,
        "base_model": ACE_STEP_CHECKPOINT,
        "style_prompt": style_prompt,
        "epochs_completed": epoch,
        "last_loss": loss,
        "lora_rank": cfg.lora_rank,
        "control_scale": cfg.control_scale,
        # ControlEncoder construction args, so apply-remix can rebuild it.
        "in_channels": tconf.in_channels,
        "height": tconf.max_height,
        "inner_dim": tconf.inner_dim,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "operator_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Operator checkpoint saved (epoch %d) -> %s", epoch, out_dir)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run_apply_remix(
    operator: str,
    original: str,
    output: str = "outputs/remixed.wav",
    steps: int = 60,
    control_scale: Optional[float] = None,
    device: str = "cuda",
) -> Path:
    """Apply a trained remix operator to a new original song."""
    op_dir = Path(operator)
    meta_path = op_dir / "operator_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Not a remix operator (no operator_meta.json): {op_dir}")
    meta = json.loads(meta_path.read_text())

    logger.info("Loading ACE-Step pipeline for remix on %s", device)
    pipeline = _load_pipeline(device)
    music_dcae = pipeline.music_dcae
    music_dcae.eval()
    model_dtype = next(music_dcae.parameters()).dtype

    # Attach the LoRA (PEFT format, symmetric with training save).
    from peft import PeftModel

    transformer = PeftModel.from_pretrained(
        pipeline.ace_step_transformer, str(op_dir), adapter_name="style", is_trainable=False
    )
    transformer.eval()

    # Rebuild + load the control encoder.
    encoder = ControlEncoder(
        in_channels=meta["in_channels"],
        height=meta["height"],
        inner_dim=meta["inner_dim"],
    ).to(device)
    encoder.load_state_dict(torch.load(op_dir / "control_encoder.pt", map_location=device))
    encoder.eval()

    if control_scale is None:
        control_scale = meta.get("control_scale", 1.0)

    # Encode the new original.
    orig_path = resolve_paths([original])[0]
    wav, _ = load_audio(orig_path, sample_rate=TARGET_SAMPLE_RATE, mono=False)
    wav = wav.repeat(2, 1) if wav.shape[0] == 1 else wav[:2]
    with torch.no_grad():
        z_orig = _encode_latents(
            music_dcae, wav.unsqueeze(0).to(device=device, dtype=model_dtype)
        )
        control = encoder(z_orig.float()).to(model_dtype)

    style_prompt = meta.get("style_prompt", "")
    base_cond = _build_conditioning(pipeline, style_prompt, device, model_dtype)
    attn = torch.ones(1, z_orig.shape[-1], device=device, dtype=model_dtype)

    # Flow-matching Euler sampler: integrate from noise (t=1) to data (t=0).
    # Convention matches training: noisy = (1-t)*z + t*noise, velocity = noise - z.
    logger.info("Sampling remix (%d steps, control_scale %.2f)", steps, control_scale)
    x = torch.randn_like(z_orig)
    schedule = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = schedule[i].repeat(1).to(model_dtype)
        dt = (schedule[i] - schedule[i + 1]).to(model_dtype)
        with torch.no_grad():
            out = transformer(
                hidden_states=x,
                attention_mask=attn,
                timestep=t,
                block_controlnet_hidden_states=control,
                controlnet_scale=control_scale,
                **base_cond,
            )
            v = getattr(out, "sample", out)
        x = x - dt * v

    with torch.no_grad():
        audio = _decode_audio(music_dcae, x.to(model_dtype))

    audio = audio[0] if audio.dim() == 3 else audio
    out_path = save_audio(unique_path(Path(output)), audio.float().cpu(), TARGET_SAMPLE_RATE)
    logger.info("Remix complete: %s", out_path)
    return out_path
