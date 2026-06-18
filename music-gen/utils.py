"""Audio I/O, resampling, and file-management helpers shared across the tool."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import torch
import torchaudio

logger = logging.getLogger("music-gen")

# Project root is the directory that contains this file (music-gen/).
PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LORAS_DIR = PROJECT_ROOT / "loras"
REFERENCES_DIR = PROJECT_ROOT / "references"
STEMS_DIR = PROJECT_ROOT / "stems"

# ACE-Step operates at 48 kHz stereo internally.
TARGET_SAMPLE_RATE = 48_000

# Audio file types we discover when collecting a group folder.
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}


def configure_logging(verbose: bool = False) -> None:
    """Set up a single, readable log format for the whole CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dirs() -> None:
    """Create the working directories if they do not exist yet."""
    for d in (OUTPUTS_DIR, LORAS_DIR, REFERENCES_DIR, STEMS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def pick_device() -> str:
    """Return the best available compute device.

    The project targets a CUDA A100 box, but we fall back gracefully so the
    CLI is still runnable (e.g. for `--help` or dry runs) on a laptop.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def resolve_paths(paths: Iterable[str | os.PathLike]) -> list[Path]:
    """Expand and validate a list of input file paths.

    Raises FileNotFoundError listing every missing path at once so the user
    can fix them in a single pass rather than one failure at a time.
    """
    resolved: list[Path] = []
    missing: list[str] = []
    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            missing.append(str(path))
        resolved.append(path)
    if missing:
        raise FileNotFoundError(
            "The following input files were not found:\n  - "
            + "\n  - ".join(missing)
        )
    return resolved


def collect_group(base_dir: Path, group: str) -> list[Path]:
    """Return all audio files inside `base_dir/<group>/`, sorted by name.

    Used so a user can organise material into named project folders, e.g.
    references/jdilla_vibes/*.mp3 and stems/jdilla_vibes/*.wav, and reference
    the whole set with a single `--group jdilla_vibes` flag.

    The internal references/_separated cache folder is skipped.
    """
    group_dir = base_dir / group
    if not group_dir.is_dir():
        raise FileNotFoundError(
            f"Group folder not found: {group_dir}\n"
            f"Create it and add audio files, e.g. {group_dir / 'song1.mp3'}"
        )
    files = sorted(
        p
        for p in group_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
        and "_separated" not in p.parts
    )
    if not files:
        raise FileNotFoundError(
            f"No audio files found in {group_dir} "
            f"(looked for: {', '.join(sorted(AUDIO_EXTENSIONS))})"
        )
    return files


def load_audio(
    path: str | os.PathLike,
    sample_rate: int = TARGET_SAMPLE_RATE,
    mono: bool = False,
) -> tuple[torch.Tensor, int]:
    """Load an audio file and resample it to `sample_rate`.

    Returns a (waveform, sample_rate) tuple where waveform has shape
    (channels, samples). When `mono` is True the channels are averaged down
    to a single channel.
    """
    path = Path(path)
    waveform, sr = torchaudio.load(str(path))

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate

    return waveform, sr


def save_audio(
    path: str | os.PathLike,
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Path:
    """Write a waveform (channels, samples) to disk, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # torchaudio expects CPU float tensors for encoding.
    waveform = waveform.detach().to("cpu", dtype=torch.float32)
    waveform = waveform.clamp(-1.0, 1.0)

    torchaudio.save(str(path), waveform, sample_rate)
    logger.info("Wrote audio: %s", path)
    return path


def stack_anchor_stems(
    stem_paths: Iterable[str | os.PathLike],
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """Load and sum a set of anchor stems into a single conditioning mix.

    Each stem is loaded at the target sample rate, padded to the longest
    length, and summed. The result is peak-normalised to avoid clipping when
    multiple loud stems overlap.
    """
    waveforms: list[torch.Tensor] = []
    for sp in stem_paths:
        wav, _ = load_audio(sp, sample_rate=sample_rate, mono=False)
        # Force stereo for a consistent channel count.
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        waveforms.append(wav)

    if not waveforms:
        raise ValueError("No anchor stems were provided to stack.")

    max_len = max(w.shape[1] for w in waveforms)
    mixed = torch.zeros(2, max_len)
    for w in waveforms:
        mixed[:, : w.shape[1]] += w[:2]

    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak

    return mixed, sample_rate


def unique_path(path: str | os.PathLike) -> Path:
    """Return `path` or, if it exists, a non-clobbering variant with a suffix."""
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1
