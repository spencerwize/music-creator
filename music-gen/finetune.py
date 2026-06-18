"""LoRA fine-tuning of ACE-Step on a small set of reference songs.

This trains a low-rank adapter that captures the timbre/production of the
reference set. The adapter is saved under loras/<name>/ and can later be
loaded by the `generate` command for high-fidelity style transfer.

The training loop is a standard rectified-flow / diffusion denoising
objective: encode each reference to the ACE-Step latent space, add noise at a
random timestep, and train the (LoRA-adapted) transformer to predict the
flow target. Only the LoRA parameters receive gradients.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from generate import ACE_STEP_CHECKPOINT, extract_style_summary
from utils import LORAS_DIR, TARGET_SAMPLE_RATE, load_audio, resolve_paths

logger = logging.getLogger("music-gen")


@dataclass
class FinetuneConfig:
    name: str
    epochs: int = 100
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    batch_size: int = 1
    segment_seconds: float = 20.0
    grad_accum: int = 1
    save_every: int = 25


class ReferenceSegmentDataset(Dataset):
    """Yields fixed-length audio segments cropped from the reference songs."""

    def __init__(self, references: list[Path], segment_seconds: float, sample_rate: int):
        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate)
        self.waveforms: list[torch.Tensor] = []
        for ref in references:
            wav, _ = load_audio(ref, sample_rate=sample_rate, mono=False)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            self.waveforms.append(wav)
        # One sample == one random crop; oversample short sets so an "epoch"
        # sees a reasonable number of segments.
        self.length = max(len(self.waveforms) * 8, 16)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav = self.waveforms[idx % len(self.waveforms)]
        total = wav.shape[1]
        if total <= self.segment_len:
            pad = self.segment_len - total
            return F.pad(wav, (0, pad))
        start = int(torch.randint(0, total - self.segment_len, (1,)).item())
        return wav[:, start : start + self.segment_len]


def _attach_lora(transformer, cfg: FinetuneConfig):
    """Wrap the transformer's attention projections with LoRA layers."""
    from peft import LoraConfig, get_peft_model

    # Target the attention projection matrices that exist in the ACE-Step DiT.
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(transformer, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def run_finetune(
    references: list[str],
    name: str,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    segment_seconds: float = 20.0,
    device: str = "cuda",
) -> Path:
    """Fine-tune a LoRA adapter and save it to loras/<name>/."""
    from acestep.pipeline_ace_step import ACEStepPipeline

    ref_paths = resolve_paths(references)
    cfg = FinetuneConfig(
        name=name,
        epochs=epochs,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        segment_seconds=segment_seconds,
    )

    out_dir = LORAS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ACE-Step pipeline for fine-tuning on %s", device)
    pipeline = ACEStepPipeline(
        checkpoint_dir=ACE_STEP_CHECKPOINT,
        dtype="bfloat16" if device == "cuda" else "float32",
        torch_compile=False,
    )

    transformer = pipeline.ace_step_transformer
    transformer.requires_grad_(False)
    peft_model = _attach_lora(transformer, cfg)
    peft_model.train()

    # The DCAE music autoencoder maps audio <-> latent for the diffusion model.
    music_dcae = pipeline.music_dcae
    music_dcae.requires_grad_(False)
    music_dcae.eval()

    dataset = ReferenceSegmentDataset(ref_paths, cfg.segment_seconds, TARGET_SAMPLE_RATE)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)

    # A style summary of the references, stored alongside the adapter so the
    # generate command can default to a matching prompt.
    style_prompt = extract_style_summary(ref_paths, device=device).as_prompt_fragment()

    logger.info("Starting LoRA training: %d epochs, %d segments/epoch", epochs, len(dataset))
    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for i, batch in enumerate(loader):
            batch = batch.to(device)

            with torch.no_grad():
                # Encode audio to latents. Returns (latents, ...) depending on
                # version; we take the latent tensor.
                encoded = music_dcae.encode(batch)
                latents = encoded[0] if isinstance(encoded, (tuple, list)) else encoded

            noise = torch.randn_like(latents)
            # Rectified-flow timesteps in [0, 1].
            t = torch.rand(latents.shape[0], device=device)
            t_exp = t.view(-1, *([1] * (latents.dim() - 1)))
            noisy = (1 - t_exp) * latents + t_exp * noise
            target = noise - latents  # flow-matching velocity target

            pred = peft_model(
                hidden_states=noisy,
                timestep=t,
            )
            pred = pred[0] if isinstance(pred, (tuple, list)) else pred

            loss = F.mse_loss(pred.float(), target.float())
            (loss / cfg.grad_accum).backward()

            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()

        avg = epoch_loss / max(1, len(loader))
        if epoch % 5 == 0 or epoch == 1:
            logger.info("epoch %3d/%d | loss %.5f", epoch, epochs, avg)

        if epoch % cfg.save_every == 0 and epoch != epochs:
            _save_adapter(peft_model, out_dir, cfg, style_prompt, epoch, avg)

    final_loss = avg if "avg" in dir() else math.nan
    _save_adapter(peft_model, out_dir, cfg, style_prompt, epochs, final_loss)
    logger.info("LoRA saved to %s", out_dir)
    return out_dir


def _save_adapter(
    peft_model,
    out_dir: Path,
    cfg: FinetuneConfig,
    style_prompt: str,
    epoch: int,
    loss: float,
) -> None:
    peft_model.save_pretrained(str(out_dir))
    meta = {
        "name": cfg.name,
        "base_model": ACE_STEP_CHECKPOINT,
        "epochs_completed": epoch,
        "epochs_requested": cfg.epochs,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "last_loss": loss,
        "style_prompt": style_prompt,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Checkpoint saved (epoch %d) -> %s", epoch, out_dir)
