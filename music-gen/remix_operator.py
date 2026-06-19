"""Remix operator — conditioning module + paired training. (Phase 2 — WIP)

This trains a conditional model that maps an *original* song to a *remix* in a
learned style, supervised by paired data (see pairs.py). The base ACE-Step model
stays frozen; we train a small control encoder that injects the original into the
transformer plus an attention LoRA.

STATUS: scaffold only. The conditioning mechanism is pending the Phase 0 spike
(see DESIGN_remix_operator.md) — we must confirm the exact shape/semantics the
transformer's `block_controlnet_hidden_states` expects before implementing the
control encoder. The training loop below mirrors finetune.py's flow-matching
objective; the marked TODOs are the only unknowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("music-gen")


@dataclass
class OperatorConfig:
    name: str
    style: str
    epochs: int = 100
    learning_rate: float = 1e-4
    lora_rank: int = 16
    segment_seconds: float = 20.0
    control_scale: float = 1.0
    match_key: bool = True


class ControlEncoder(torch.nn.Module):
    """Maps original-song latents -> per-block control hidden states.

    TODO(Phase 0): determine the number of transformer blocks, the hidden dim,
    and the latent/sequence length so the produced control tensors match what
    `block_controlnet_hidden_states` expects. Until then this is a placeholder.
    """

    def __init__(self, latent_channels: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.num_blocks = num_blocks
        # Placeholder projection — real architecture decided after the spike.
        self.proj = torch.nn.Conv1d(latent_channels, hidden_dim, kernel_size=1)

    def forward(self, original_latents: torch.Tensor):  # pragma: no cover - WIP
        raise NotImplementedError(
            "ControlEncoder is pending the Phase 0 control-shape spike."
        )


def run_train_operator(
    style: str,
    name: str,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    segment_seconds: float = 20.0,
    match_key: bool = True,
    device: str = "cuda",
) -> Path:  # pragma: no cover - WIP
    """Train a remix operator on references/<style>/{originals,remixes}.

    Outline (see DESIGN_remix_operator.md):
      1. PairedRemixDataset(style) -> aligned (original, remix) crops.
      2. Frozen DCAE encodes both -> z_orig, z_remix.
      3. ControlEncoder(z_orig) -> block_controlnet_hidden_states.
      4. Flow-matching denoise of z_remix conditioned on style prompt + control.
      5. Train control encoder + attention LoRA; save to loras_op/<name>/.
    """
    raise NotImplementedError(
        "Remix operator training is scaffolded but not yet implemented. "
        "Next step: the Phase 0 spike to pin down control-tensor shapes."
    )


def run_apply_remix(
    operator: str,
    original: str,
    output: str = "outputs/remixed.wav",
    device: str = "cuda",
) -> Path:  # pragma: no cover - WIP
    """Apply a trained remix operator to a new original song."""
    raise NotImplementedError("apply-remix is pending the operator training (Phase 2/3).")
