"""ACE-Step inference: turn anchor stems + style references into a full mix.

Two style-conditioning paths are supported:

1. Inference-time conditioning (no training): we separate the reference songs,
   derive a compact "style summary" from them, fold it into the text prompt,
   and feed the references to ACE-Step as audio2audio guidance alongside the
   user's anchor stems.

2. LoRA conditioning: a previously fine-tuned adapter (see finetune.py) is
   loaded into the ACE-Step transformer for much higher style fidelity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch
import torchaudio

from separate import separate_references
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
# Style extraction
# --------------------------------------------------------------------------- #
@dataclass
class StyleSummary:
    """A lightweight, human-readable description of the reference material.

    These are cheap signal-processing descriptors (tempo, brightness, energy
    balance across stems) rather than a learned embedding. They are folded
    into the text prompt so the base model leans toward the reference sound
    even without a LoRA.
    """

    tempo_bpm: float = 0.0
    brightness: float = 0.0  # spectral centroid, normalised 0..1
    stem_energy: dict[str, float] = field(default_factory=dict)
    descriptors: list[str] = field(default_factory=list)

    def as_prompt_fragment(self) -> str:
        parts: list[str] = []
        if self.tempo_bpm:
            parts.append(f"{round(self.tempo_bpm)} BPM")
        parts.extend(self.descriptors)
        return ", ".join(parts)


def _estimate_tempo(waveform: torch.Tensor, sample_rate: int) -> float:
    """Rough onset-autocorrelation tempo estimate (no external deps)."""
    mono = waveform.mean(0)
    # Onset envelope via the positive first difference of a coarse envelope.
    win = max(1, sample_rate // 100)
    env = mono.abs().unfold(0, win, win).mean(1)
    env = torch.clamp(env[1:] - env[:-1], min=0.0)
    if env.numel() < 4:
        return 0.0
    env = env - env.mean()
    ac = torch.nn.functional.conv1d(
        env.view(1, 1, -1), env.flip(0).view(1, 1, -1), padding=env.numel() - 1
    ).view(-1)
    ac = ac[ac.numel() // 2 :]
    # Frames per second of the onset envelope.
    fps = sample_rate / win
    min_lag = int(fps * 60 / 200)  # 200 BPM ceiling
    max_lag = int(fps * 60 / 60)   # 60 BPM floor
    if max_lag <= min_lag or max_lag >= ac.numel():
        return 0.0
    lag = int(torch.argmax(ac[min_lag:max_lag]).item()) + min_lag
    return float(60.0 * fps / lag) if lag else 0.0


def _spectral_brightness(waveform: torch.Tensor, sample_rate: int) -> float:
    mono = waveform.mean(0)
    spec = torch.stft(
        mono, n_fft=2048, hop_length=512, return_complex=True, window=torch.hann_window(2048)
    ).abs()
    freqs = torch.linspace(0, sample_rate / 2, spec.shape[0]).unsqueeze(1)
    centroid = (spec * freqs).sum() / (spec.sum() + 1e-8)
    return float((centroid / (sample_rate / 2)).clamp(0, 1).item())


def extract_style_summary(
    references: Iterable[str | Path],
    device: str = "cuda",
    skip_separation: bool = False,
) -> StyleSummary:
    """Separate references and summarise their style as prompt-ready descriptors."""
    references = list(references)
    stem_energy: dict[str, float] = {}

    if not skip_separation:
        try:
            separated = separate_references(references, device=device)
            for stems in separated.values():
                for name, path in stems.items():
                    wav, sr = load_audio(path, mono=False)
                    stem_energy[name] = stem_energy.get(name, 0.0) + float(
                        wav.pow(2).mean().sqrt().item()
                    )
        except Exception as exc:  # pragma: no cover - depends on demucs runtime
            logger.warning("Stem separation failed (%s); using mix-only style.", exc)

    tempos: list[float] = []
    brightnesses: list[float] = []
    for ref in references:
        wav, sr = load_audio(ref, mono=False)
        t = _estimate_tempo(wav, sr)
        if t:
            tempos.append(t)
        brightnesses.append(_spectral_brightness(wav, sr))

    tempo = sum(tempos) / len(tempos) if tempos else 0.0
    brightness = sum(brightnesses) / len(brightnesses) if brightnesses else 0.0

    descriptors: list[str] = []
    if brightness:
        descriptors.append("bright and airy" if brightness > 0.45 else "warm and dark")
    if stem_energy:
        loudest = max(stem_energy, key=stem_energy.get)
        descriptors.append(f"{loudest}-forward")

    summary = StyleSummary(
        tempo_bpm=tempo,
        brightness=brightness,
        stem_energy=stem_energy,
        descriptors=descriptors,
    )
    logger.info("Style summary: %s", summary.as_prompt_fragment() or "(none)")
    return summary


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
        """Run ACE-Step conditioned on the anchor stems and prompt."""
        anchor_path = self._prepare_anchor(stem_paths)
        # Match the generated length to the anchor so the stems line up.
        duration = config.duration or self._anchor_duration(stem_paths)

        prompt = config.prompt
        if not prompt and getattr(self, "lora_prompt_hint", ""):
            prompt = self.lora_prompt_hint

        logger.info(
            "Generating %.1fs | prompt=%r | steps=%d | guidance=%.1f | ref_strength=%.2f",
            duration,
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
            audio2audio_enable=True,
            ref_audio_strength=config.ref_audio_strength,
            ref_audio_input=str(anchor_path),
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
    stem_paths = resolve_paths(stems)

    style_fragment = ""
    if references and not lora:
        ref_paths = resolve_paths(references)
        summary = extract_style_summary(ref_paths, device=device)
        style_fragment = summary.as_prompt_fragment()

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
