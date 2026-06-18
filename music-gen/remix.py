"""Remix workflow: re-tempo an acapella and build an in-style instrumental.

This is the vocal-preserving path. Unlike `generate`, the vocal is NOT fed
through ACE-Step's audio2audio (which would resynthesise and smear it). Instead:

1. The real vocal is kept, but pitch-preservingly **time-stretched** from its
   source BPM to the target BPM (so a 100 BPM pop acapella locks to a 128 BPM
   EDM grid while staying in its original key), with an optional key shift.
2. An **instrumental** is generated in the reference style at the target BPM,
   matched to the stretched vocal's length.
3. The two are **mixed**, vocal on top — your performance preserved exactly.

Tight bar-by-bar arrangement (drops landing on the hook, etc.) still needs a
DAW; this produces a tempo-locked, in-style rough remix to build from.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from generate import GenerationConfig, MusicGenerator
from style import detect_key, estimate_tempo, extract_style, semitone_shift
from utils import (
    OUTPUTS_DIR,
    TARGET_SAMPLE_RATE,
    load_audio,
    mix_overlay,
    pitch_shift,
    resolve_audio_paths,
    resolve_paths,
    save_audio,
    time_stretch,
    unique_path,
)

logger = logging.getLogger("music-gen")


def run_remix(
    vocal: str,
    references: Optional[list[str]] = None,
    lora: Optional[str] = None,
    prompt: str = "",
    output: str = "outputs/remix.wav",
    source_bpm: float = 0.0,
    target_bpm: float = 0.0,
    pitch_shift_semitones: float = 0.0,
    match_key: bool = False,
    vocal_gain_db: float = 0.0,
    infer_steps: int = 60,
    guidance_scale: float = 15.0,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> Path:
    """Build a remix: re-tempo the vocal and lay it over an in-style instrumental."""
    vocal_path = resolve_paths([vocal])[0]
    vocal_wav, sr = load_audio(vocal_path, sample_rate=TARGET_SAMPLE_RATE, mono=False)

    # 1. Analyse the references first — this gives both the style tags and a
    #    reliable target tempo (full songs have far clearer beats than an
    #    acapella), which we use as the default --target-bpm.
    style_fragment = ""
    reference_bpm = 0.0
    if references:
        style = extract_style(resolve_audio_paths(references), device=device)
        style_fragment = style.as_prompt_fragment()
        reference_bpm = style.tempo_bpm

    # 2. Determine source (vocal) tempo.
    if not source_bpm:
        source_bpm = estimate_tempo(vocal_wav, sr)
        logger.info("Auto-detected source vocal tempo: %.1f BPM", source_bpm)
    if not source_bpm:
        raise ValueError(
            "Could not determine the vocal's source BPM automatically; "
            "pass --source-bpm explicitly."
        )

    # 3. Determine target tempo: explicit > reference tempo > keep vocal tempo.
    if not target_bpm:
        if reference_bpm:
            target_bpm = reference_bpm
            logger.info("Auto-target tempo from references: %.1f BPM", target_bpm)
        else:
            target_bpm = source_bpm
            logger.info(
                "No --target-bpm and no references to detect one; keeping %.1f BPM",
                target_bpm,
            )

    # 4. Re-tempo the vocal (pitch-preserving). Key shift happens after the
    #    instrumental exists, so --match-key can align to the actual generation.
    rate = target_bpm / source_bpm
    if abs(rate - 1.0) > 1e-3:
        logger.info(
            "Time-stretching vocal %.1f -> %.1f BPM (rate %.3f)",
            source_bpm, target_bpm, rate,
        )
        if rate > 1.6 or rate < 0.625:
            logger.warning(
                "Large tempo change (%.2fx) may introduce stretch artifacts; "
                "consider a half/double-time target instead.", rate,
            )
        vocal_wav = time_stretch(vocal_wav, rate)

    vocal_seconds = vocal_wav.shape[1] / sr
    logger.info("Re-tempo'd vocal length: %.1fs", vocal_seconds)

    # 5. Build the instrumental prompt (instrumental + style tags + target BPM).
    bpm_tag = f"{round(target_bpm)} BPM"
    full_prompt = ", ".join(
        p for p in ("instrumental", prompt, style_fragment, bpm_tag) if p
    )

    # 6. Generate the instrumental, matched to the stretched vocal's length.
    generator = MusicGenerator(device=device)
    if lora:
        generator.load_lora(lora)
    config = GenerationConfig(
        prompt=full_prompt,
        lyrics="[inst]",  # keep the bed instrumental so it won't fight the vocal
        duration=vocal_seconds,
        infer_steps=infer_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    inst_path = generator.generate(
        stem_paths=[],  # no anchor: pure in-style instrumental
        config=config,
        output_path=OUTPUTS_DIR / "_remix_instrumental.wav",
    )

    inst_wav, _ = load_audio(inst_path, sample_rate=TARGET_SAMPLE_RATE, mono=False)

    # 7. Key alignment. Explicit --pitch-shift wins; otherwise --match-key
    #    detects the generated instrumental's key and the vocal's key and
    #    shifts the vocal by the minimal number of semitones to align them.
    shift = pitch_shift_semitones
    if abs(shift) < 1e-3 and match_key:
        voc_pc, voc_mode, voc_name = detect_key(vocal_wav, sr)
        inst_pc, inst_mode, inst_name = detect_key(inst_wav, sr)
        shift = semitone_shift(voc_pc, inst_pc)
        logger.info(
            "Key match: vocal %s -> instrumental %s = %+d semitones",
            voc_name, inst_name, shift,
        )
        if voc_mode != inst_mode:
            logger.warning(
                "Vocal is %s but instrumental is %s; pitch shift aligns the "
                "tonic but cannot convert mode — expect some tension.",
                voc_mode, inst_mode,
            )
    elif abs(shift) >= 1e-3 and match_key:
        logger.info("Explicit --pitch-shift %.2f given; ignoring --match-key.", shift)

    if abs(shift) > 1e-3:
        logger.info("Pitch-shifting vocal %+.2f semitones", shift)
        vocal_wav = pitch_shift(vocal_wav, sr, shift)

    # 8. Mix the preserved vocal on top of the instrumental.
    mixed = mix_overlay(inst_wav, vocal_wav, overlay_gain_db=vocal_gain_db)

    out_path = save_audio(unique_path(Path(output)), mixed, sr)
    logger.info("Remix complete: %s", out_path)
    return out_path
