"""Paired (original, remix) dataset + alignment for the remix operator.

Layout (matched by filename stem):

    references/<style>/originals/  songA.wav  songB.mp3 ...
    references/<style>/remixes/    songA.wav  songB.wav ...

A pair is formed when a stem appears in both folders. Before training, each
pair is globally aligned (tempo, optionally key) reusing the DSP from style.py /
utils.py, then sliced into time-aligned crops. Fine structural differences
(reordered sections, added drops) are intentionally left for the diffusion model
to absorb.

Phase 1 of DESIGN_remix_operator.md. This module is CPU-testable on its own; it
returns aligned audio crops and the trainer (remix_operator.py) handles latent
encoding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from style import detect_key, estimate_tempo, semitone_shift
from utils import (
    AUDIO_EXTENSIONS,
    REFERENCES_DIR,
    TARGET_SAMPLE_RATE,
    load_audio,
    pitch_shift,
    time_stretch,
)

logger = logging.getLogger("music-gen")


@dataclass
class Pair:
    """One (original, remix) example located on disk."""

    stem: str
    original: Path
    remix: Path


# Trailing role suffixes stripped before matching, so e.g. songA_original pairs
# with songA_remix. Separator can be _, -, space, or a dot.
_ROLE_SUFFIX = re.compile(
    r"[\s_\-.]+[\(\[]?"
    r"(originals?|orig|remix(?:es)?|rmx|rework|reedit|edit|flip|bootleg|vip)"
    r"[\)\]]?$",
    re.IGNORECASE,
)


def _normalize_stem(stem: str) -> str:
    """Lowercase and strip trailing role suffixes (repeatedly) for matching."""
    prev = None
    s = stem
    while s != prev:
        prev = s
        s = _ROLE_SUFFIX.sub("", s).strip()
    return s.lower()


def discover_pairs(style: str, base_dir: Path = REFERENCES_DIR) -> list[Pair]:
    """Find (original, remix) pairs under references/<style>/{originals,remixes}.

    Pairs are matched by a normalised filename stem: common role suffixes like
    ``_original`` / ``_remix`` (and ``-orig``, `` rmx``, etc.) are stripped
    first, so ``songA_original.wav`` pairs with ``songA_remix.wav``. Exact same
    names (``songA.wav`` in both folders) also work. The extension may differ.
    Unmatched files on either side are skipped with a warning.
    """
    style_dir = base_dir / style
    orig_dir = style_dir / "originals"
    remix_dir = style_dir / "remixes"
    for d in (orig_dir, remix_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Expected folder not found: {d}\n"
                f"Lay out pairs as {style_dir}/originals/ and {style_dir}/remixes/."
            )

    def index(d: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                out.setdefault(_normalize_stem(p.stem), p)
        return out

    originals = index(orig_dir)
    remixes = index(remix_dir)

    matched = sorted(set(originals) & set(remixes))
    for stem in sorted(set(originals) ^ set(remixes)):
        side = "remix" if stem in originals else "original"
        path = (originals if stem in originals else remixes)[stem]
        logger.warning("Unpaired file (no matching %s): %s", side, path.name)

    pairs = [Pair(s, originals[s], remixes[s]) for s in matched]
    if not pairs:
        raise FileNotFoundError(
            f"No matching original/remix filename stems found under {style_dir}."
        )
    logger.info("Discovered %d (original, remix) pairs for style '%s'", len(pairs), style)
    return pairs


def align_pair(
    original: torch.Tensor,
    remix: torch.Tensor,
    sample_rate: int,
    match_key: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Globally align the original to the remix (tempo, optional key).

    The remix defines the target grid; the original is time-stretched to the
    remix's tempo and (optionally) pitch-shifted to its key, then both are
    trimmed to a common length. Returns (aligned_original, remix).
    """
    orig_bpm = estimate_tempo(original, sample_rate)
    remix_bpm = estimate_tempo(remix, sample_rate)
    if orig_bpm and remix_bpm:
        rate = remix_bpm / orig_bpm
        if abs(rate - 1.0) > 1e-3:
            original = time_stretch(original, rate)
            logger.debug("Aligned tempo %.1f -> %.1f BPM", orig_bpm, remix_bpm)

    if match_key:
        o_pc, _, _ = detect_key(original, sample_rate)
        r_pc, _, _ = detect_key(remix, sample_rate)
        shift = semitone_shift(o_pc, r_pc)
        if shift:
            original = pitch_shift(original, sample_rate, shift)
            logger.debug("Aligned key by %+d semitones", shift)

    length = min(original.shape[1], remix.shape[1])
    return original[:, :length], remix[:, :length]


class PairedRemixDataset(Dataset):
    """Yields time-aligned (original_crop, remix_crop) tensors for training.

    Each pair is loaded, globally aligned, and randomly cropped to
    `segment_seconds`. Both crops share the same time window so the conditioning
    (original) and target (remix) correspond.
    """

    def __init__(
        self,
        style: str,
        segment_seconds: float = 20.0,
        sample_rate: int = TARGET_SAMPLE_RATE,
        match_key: bool = True,
        base_dir: Path = REFERENCES_DIR,
        oversample: int = 8,
    ):
        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate)
        self.pairs = discover_pairs(style, base_dir=base_dir)

        # Pre-load and align once; remixing sets are small enough to hold in RAM.
        self.aligned: list[tuple[torch.Tensor, torch.Tensor]] = []
        for pair in self.pairs:
            orig, _ = load_audio(pair.original, sample_rate=sample_rate, mono=False)
            remix, _ = load_audio(pair.remix, sample_rate=sample_rate, mono=False)
            orig = _force_stereo(orig)
            remix = _force_stereo(remix)
            self.aligned.append(align_pair(orig, remix, sample_rate, match_key))

        self.length = max(len(self.aligned) * oversample, 16)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        orig, remix = self.aligned[idx % len(self.aligned)]
        total = min(orig.shape[1], remix.shape[1])
        if total <= self.segment_len:
            orig = F.pad(orig, (0, self.segment_len - orig.shape[1]))
            remix = F.pad(remix, (0, self.segment_len - remix.shape[1]))
            start = 0
        else:
            start = int(torch.randint(0, total - self.segment_len, (1,)).item())
        sl = slice(start, start + self.segment_len)
        return {"original": orig[:, sl], "remix": remix[:, sl]}


def _force_stereo(wav: torch.Tensor) -> torch.Tensor:
    return wav.repeat(2, 1) if wav.shape[0] == 1 else wav[:2]
