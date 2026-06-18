"""Demucs-based stem separation for the style-reference songs.

Separating the references into stems lets the style extractor look at the
musical layers (drums, bass, vocals, other) independently, which produces a
cleaner style embedding than a single mixed track.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import torch

from utils import PROJECT_ROOT, load_audio, save_audio

logger = logging.getLogger("music-gen")

# htdemucs is the current default 4-stem hybrid-transformer model.
DEFAULT_MODEL = "htdemucs"

# Where separated stems are cached so repeated runs don't re-separate.
SEPARATED_DIR = PROJECT_ROOT / "references" / "_separated"


class StemSeparator:
    """Thin wrapper around Demucs' pretrained models."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        # Imported lazily so `--help` and unrelated commands don't pay the
        # cost of importing the (heavy) demucs stack.
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        self.device = device
        self.model_name = model_name
        logger.info("Loading Demucs model '%s' on %s", model_name, device)
        self.model = get_model(model_name)
        self.model.to(device)
        self.model.eval()
        self._apply_model = apply_model
        # The source names this model produces, e.g. ["drums","bass","other","vocals"]
        self.sources: list[str] = list(self.model.sources)
        self.sample_rate: int = self.model.samplerate

    def separate_file(self, path: str | Path, overwrite: bool = False) -> dict[str, Path]:
        """Separate a single song into per-stem WAV files.

        Returns a mapping of stem name -> file path. Results are cached under
        references/_separated/<song-stem>/ and reused unless `overwrite`.
        """
        path = Path(path)
        out_dir = SEPARATED_DIR / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = {s: out_dir / f"{s}.wav" for s in self.sources}
        if not overwrite and all(p.exists() for p in existing.values()):
            logger.info("Using cached stems for %s", path.name)
            return existing

        logger.info("Separating %s into %s stems", path.name, len(self.sources))
        waveform, _ = load_audio(path, sample_rate=self.sample_rate, mono=False)

        # Demucs expects shape (batch, channels, samples) with stereo channels.
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            sources = self._apply_model(
                self.model,
                waveform[None].to(self.device),
                device=self.device,
                split=True,
                overlap=0.25,
                progress=True,
            )[0]
        sources = sources * ref.std() + ref.mean()
        sources = sources.cpu()

        result: dict[str, Path] = {}
        for name, source in zip(self.sources, sources):
            out_path = out_dir / f"{name}.wav"
            save_audio(out_path, source, self.sample_rate)
            result[name] = out_path
        return result


def separate_references(
    references: Iterable[str | Path],
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    overwrite: bool = False,
) -> dict[str, dict[str, Path]]:
    """Separate every reference song and return {song_path: {stem: path}}."""
    separator = StemSeparator(model_name=model_name, device=device)
    out: dict[str, dict[str, Path]] = {}
    for ref in references:
        ref = Path(ref)
        out[str(ref)] = separator.separate_file(ref, overwrite=overwrite)
    return out
