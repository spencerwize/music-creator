"""ACE-Step inference: turn anchor stems + style references into a full mix.

Two style-conditioning paths are supported, and they compose:

1. Inference-time conditioning (no training): we extract a learned CLAP style
   embedding from the reference songs (see style.py), turn it into prompt-ready
   descriptors, and fold them into the text prompt. This carries real style
   information without competing with the anchor stems for the audio2audio slot.

2. LoRA conditioning: a previously fine-tuned adapter (see finetune.py) is
   loaded into the ACE-Step transformer for much higher style fidelity.

Because (1) only touches the prompt and (2) only touches the weights, both can
be active at once: a LoRA drives the core sound while references nudge each run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from style import extract_style
from utils import (
    OUTPUTS_DIR,
    TARGET_SAMPLE_RATE,
    load_audio,
    resolve_paths,
    save_audio,
    stack_anchor_stems,
    unique_path,
)

logger = logging.getLogger("music-gen")

ACE_STEP_CHECKPOINT = "ACE-Step/ACE-Step-v1-3.5B"


# --------------------------------------------------------------------------- #
# ACE-Step generation
# --------------------------------------------------------------------------- #
@dataclass
class GenerationConfig:
    prompt: str = ""
    lyrics: str = "[inst]"
    duration: float = 60.0
    infer_steps: int = 60
    guidance_scale: float = 15.0
    ref_audio_strength: float = 0.5
    seed: Optional[int] = None


class MusicGenerator:
    """Wraps the ACE-Step pipeline for stem-conditioned generation."""

    def __init__(self, checkpoint: str = ACE_STEP_CHECKPOINT, device: str = "cuda"):
        from acestep.pipeline_ace_step import ACEStepPipeline

        self.device = device
        logger.info("Loading ACE-Step pipeline (%s) on %s", checkpoint, device)
        # bf16 is the recommended dtype on A100-class hardware.
        self.pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint,
            dtype="bfloat16" if device == "cuda" else "float32",
            torch_compile=False,
        )
        self._loaded_lora: Optional[str] = None

    # -- LoRA management ---------------------------------------------------- #
    def load_lora(self, lora_dir: str | Path) -> None:
        """Attach a saved LoRA adapter to the diffusion transformer."""
        lora_dir = Path(lora_dir)
        if not lora_dir.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_dir}")
        logger.info("Loading LoRA adapter: %s", lora_dir)
        transformer = self.pipeline.ace_step_transformer
        transformer.load_lora_adapter(str(lora_dir), adapter_name="style")
        transformer.set_adapters(["style"])
        self._loaded_lora = str(lora_dir)

        # Pull any prompt hints the LoRA was trained with so generation can
        # default to them when the user doesn't pass a prompt.
        meta = lora_dir / "training_meta.json"
        self.lora_prompt_hint = ""
        if meta.exists():
            try:
                self.lora_prompt_hint = json.loads(meta.read_text()).get("style_prompt", "")
            except json.JSONDecodeError:
                pass

    # -- Conditioning ------------------------------------------------------- #
    def _prepare_anchor(self, stem_paths: list[Path]) -> Path:
        """Mix the anchor stems to a temp WAV used as audio2audio reference."""
        mixed, sr = stack_anchor_stems(stem_paths, sample_rate=TARGET_SAMPLE_RATE)
        anchor_path = OUTPUTS_DIR / "_anchor_mix.wav"
        save_audio(anchor_path, mixed, sr)
        return anchor_path

    def _anchor_duration(self, stem_paths: list[Path]) -> float:
        mixed, sr = stack_anchor_stems(stem_paths, sample_rate=TARGET_SAMPLE_RATE)
        return mixed.shape[1] / sr

    # -- Generation --------------------------------------------------------- #
    def generate(
        self,
        stem_paths: list[Path],
        config: GenerationConfig,
        output_path: Path,
    ) -> Path:
        """Run ACE-Step conditioned on the anchor stems and prompt.

        If no anchor stems are given, audio2audio is disabled and ACE-Step does
        a pure text+style generation (driven by the prompt, CLAP tags, and any
        LoRA). Duration then comes from --duration, falling back to 60s.
        """
        use_anchor = bool(stem_paths)
        if use_anchor:
            anchor_path = self._prepare_anchor(stem_paths)
            # Match the generated length to the anchor so the stems line up.
            duration = config.duration or self._anchor_duration(stem_paths)
        else:
            anchor_path = None
            duration = config.duration or 60.0

        prompt = config.prompt
        if not prompt and getattr(self, "lora_prompt_hint", ""):
            prompt = self.lora_prompt_hint

        logger.info(
            "Generating %.1fs | anchor=%s | prompt=%r | steps=%d | guidance=%.1f | ref_strength=%.2f",
            duration,
            "yes" if use_anchor else "no (pure text+style)",
            prompt,
            config.infer_steps,
            config.guidance_scale,
            config.ref_audio_strength,
        )

        output_path = unique_path(output_path)
        # ACE-Step writes to the path(s) it is handed and returns them.
        self.pipeline(
            prompt=prompt,
            lyrics=config.lyrics,
            audio_duration=duration,
            infer_step=config.infer_steps,
            guidance_scale=config.guidance_scale,
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10.0,
            manual_seeds=str(config.seed) if config.seed is not None else None,
            # audio2audio: the anchor stems steer the structure/rhythm.
            audio2audio_enable=use_anchor,
            ref_audio_strength=config.ref_audio_strength,
            ref_audio_input=str(anchor_path) if anchor_path else None,
            save_path=str(output_path),
            format="wav",
        )
        logger.info("Generation complete: %s", output_path)
        return output_path


def run_generation(
    stems: list[str],
    references: Optional[list[str]] = None,
    lora: Optional[str] = None,
    prompt: str = "",
    output: str = "outputs/result.wav",
    duration: float = 0.0,
    infer_steps: int = 60,
    guidance_scale: float = 15.0,
    ref_audio_strength: float = 0.5,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> Path:
    """High-level entry point used by the CLI `generate` command."""
    # Stems are optional: with none, generation is pure text+style+LoRA.
    stem_paths = resolve_paths(stems) if stems else []

    # The learned CLAP style embedding lives in the prompt path, so references
    # contribute even when a LoRA is loaded (the two compose).
    style_fragment = ""
    if references:
        ref_paths = resolve_paths(references)
        style = extract_style(ref_paths, device=device)
        style_fragment = style.as_prompt_fragment()

    full_prompt = ", ".join(p for p in (prompt, style_fragment) if p).strip(", ")

    generator = MusicGenerator(device=device)
    if lora:
        generator.load_lora(lora)

    config = GenerationConfig(
        prompt=full_prompt,
        duration=duration,
        infer_steps=infer_steps,
        guidance_scale=guidance_scale,
        ref_audio_strength=ref_audio_strength,
        seed=seed,
    )
    return generator.generate(stem_paths, config, Path(output))
