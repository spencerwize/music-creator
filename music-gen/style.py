"""Learned audio style embeddings for the style-reference songs.

We use CLAP (Contrastive Language-Audio Pretraining, trained on music) to turn
the reference songs into a 512-d *style embedding* that captures real timbral,
genre and production character — far more information than hand-rolled
tempo/brightness descriptors.

The embedding is used two ways:

1. Zero-shot tagging: CLAP shares an audio<->text space, so we score the
   embedding against a curated vocabulary of musical descriptors and fold the
   closest tags into the text prompt. This carries genuine style information
   into generation *without* competing with the anchor stems for the
   audio2audio slot, so it composes cleanly with a LoRA.

2. Persistence: the raw vector is cached and saved alongside LoRAs, so a future
   trained projector (IP-Adapter style) can consume it directly.

A cheap onset-autocorrelation tempo estimate is kept as a complementary numeric
signal (CLAP does not produce a precise BPM).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch

from utils import PROJECT_ROOT, load_audio

logger = logging.getLogger("music-gen")

# Music-focused CLAP checkpoint (also handles speech). ~1.5 GB.
CLAP_CHECKPOINT = "laion/larger_clap_music_and_speech"

# CLAP operates at 48 kHz and ingests up to 10 s windows.
CLAP_SAMPLE_RATE = 48_000
CLAP_WINDOW_SAMPLES = CLAP_SAMPLE_RATE * 10

STYLE_CACHE_DIR = PROJECT_ROOT / "references" / "_style_cache"

# Curated vocabulary, grouped by axis. Labels are short (they go into the
# prompt); the CLAP query is templated per-axis for better audio<->text match.
VOCABULARY: dict[str, list[str]] = {
    "genre": [
        "boom bap", "lo-fi hip hop", "trap", "soul", "jazz", "funk", "ambient",
        "house", "techno", "drum and bass", "rock", "r&b", "gospel", "neo-soul",
        "trip hop", "downtempo", "disco", "afrobeat",
    ],
    "mood": [
        "dark", "melancholic", "uplifting", "aggressive", "dreamy", "nostalgic",
        "warm", "cold", "ethereal", "gritty", "romantic", "tense", "playful",
        "hypnotic", "triumphant", "mellow",
    ],
    "texture": [
        "dusty", "vinyl crackle", "tape saturated", "lo-fi", "clean hi-fi",
        "distorted", "reverb-drenched", "dry", "muffled", "crisp", "analog",
        "spacious",
    ],
    "instrumentation": [
        "piano", "electric guitar", "acoustic guitar", "saxophone", "strings",
        "synth pads", "rhodes piano", "organ", "brass", "flute", "vocal chops",
        "upright bass", "808 bass", "live drums",
    ],
}

# How to phrase each axis when querying CLAP's text encoder.
_AXIS_TEMPLATE: dict[str, str] = {
    "genre": "{label} music",
    "mood": "a {label} sounding song",
    "texture": "music with a {label} sound",
    "instrumentation": "music featuring {label}",
}


@dataclass
class StyleEmbedding:
    """A learned style descriptor for a set of reference songs."""

    vector: list[float] = field(default_factory=list)  # L2-normalised CLAP embedding
    tags: list[str] = field(default_factory=list)      # zero-shot descriptors
    tempo_bpm: float = 0.0
    references: list[str] = field(default_factory=list)

    def as_prompt_fragment(self) -> str:
        parts: list[str] = []
        if self.tempo_bpm:
            parts.append(f"{round(self.tempo_bpm)} BPM")
        parts.extend(self.tags)
        return ", ".join(parts)

    def to_json(self) -> str:
        return json.dumps(
            {
                "tags": self.tags,
                "tempo_bpm": self.tempo_bpm,
                "references": self.references,
                "vector": self.vector,
            }
        )

    def save_vector(self, path: str | Path) -> None:
        """Persist just the raw vector (for a future trained projector)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor(self.vector), str(path))


def estimate_tempo(waveform: torch.Tensor, sample_rate: int) -> float:
    """Rough onset-autocorrelation tempo estimate (no external deps)."""
    mono = waveform.mean(0)
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
    fps = sample_rate / win
    min_lag = int(fps * 60 / 200)  # 200 BPM ceiling
    max_lag = int(fps * 60 / 60)   # 60 BPM floor
    if max_lag <= min_lag or max_lag >= ac.numel():
        return 0.0
    lag = int(torch.argmax(ac[min_lag:max_lag]).item()) + min_lag
    return float(60.0 * fps / lag) if lag else 0.0


# Krumhansl-Kessler key profiles for major/minor key detection.
_KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key(waveform: torch.Tensor, sample_rate: int) -> tuple[int, str, str]:
    """Estimate musical key via Krumhansl-Schmuckler profile correlation.

    Returns (tonic_pitch_class 0-11 where 0=C, mode "major"/"minor", name).
    Falls back to (0, "major", "C major") if librosa is unavailable.
    """
    try:
        import librosa
        import numpy as np
    except Exception:  # pragma: no cover
        logger.warning("librosa unavailable; cannot detect key.")
        return 0, "major", "C major"

    mono = waveform.mean(0).detach().cpu().numpy()
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sample_rate)
    chroma_mean = chroma.mean(axis=1)

    best = None
    for mode, profile in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
        prof = np.asarray(profile)
        for tonic in range(12):
            rotated = np.roll(prof, tonic)
            corr = float(np.corrcoef(chroma_mean, rotated)[0, 1])
            if best is None or corr > best[0]:
                best = (corr, tonic, mode)

    _, tonic, mode = best
    return tonic, mode, f"{_PITCH_NAMES[tonic]} {mode}"


def semitone_shift(from_pc: int, to_pc: int) -> int:
    """Minimal signed semitone shift to move pitch class `from_pc` to `to_pc`."""
    diff = (to_pc - from_pc) % 12
    if diff > 6:
        diff -= 12
    return diff


class StyleEncoder:
    """Wraps CLAP to embed audio and score it against the tag vocabulary."""

    def __init__(self, checkpoint: str = CLAP_CHECKPOINT, device: str = "cuda"):
        from transformers import ClapModel, ClapProcessor

        self.device = device
        logger.info("Loading CLAP style encoder (%s) on %s", checkpoint, device)
        self.model = ClapModel.from_pretrained(checkpoint).to(device).eval()
        self.processor = ClapProcessor.from_pretrained(checkpoint)
        self._text_cache: dict[str, torch.Tensor] = {}

    # -- audio -------------------------------------------------------------- #
    def embed_audio(self, references: list[Path]) -> torch.Tensor:
        """Return one L2-normalised style vector averaged over all references.

        Each song is chunked into 10 s windows; every chunk is embedded and the
        results are mean-pooled, giving a stable whole-track style vector.
        """
        chunks: list = []
        for ref in references:
            wav, _ = load_audio(ref, sample_rate=CLAP_SAMPLE_RATE, mono=True)
            samples = wav.squeeze(0)
            if samples.numel() < CLAP_SAMPLE_RATE:  # pad clips under 1 s
                samples = torch.nn.functional.pad(
                    samples, (0, CLAP_SAMPLE_RATE - samples.numel())
                )
            for start in range(0, samples.numel(), CLAP_WINDOW_SAMPLES):
                window = samples[start : start + CLAP_WINDOW_SAMPLES]
                if window.numel() < CLAP_SAMPLE_RATE:  # skip tiny tail
                    continue
                chunks.append(window.cpu().numpy())

        if not chunks:
            raise ValueError("No audio long enough to embed for style.")

        inputs = self.processor(
            audios=chunks, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.get_audio_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        pooled = torch.nn.functional.normalize(feats.mean(0), dim=-1)
        return pooled

    # -- text --------------------------------------------------------------- #
    def _axis_text_features(self, axis: str) -> torch.Tensor:
        if axis in self._text_cache:
            return self._text_cache[axis]
        template = _AXIS_TEMPLATE[axis]
        phrases = [template.format(label=lbl) for lbl in VOCABULARY[axis]]
        inputs = self.processor(text=phrases, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        self._text_cache[axis] = feats
        return feats

    def retrieve_tags(
        self, audio_vector: torch.Tensor, min_similarity: float = 0.0
    ) -> list[str]:
        """Pick the closest label per axis (above a similarity floor)."""
        tags: list[str] = []
        for axis, labels in VOCABULARY.items():
            text_feats = self._axis_text_features(axis)
            sims = (audio_vector.unsqueeze(0) @ text_feats.T).squeeze(0)
            best = int(torch.argmax(sims).item())
            if float(sims[best].item()) >= min_similarity:
                tags.append(labels[best])
        return tags


def _cache_key(references: list[Path]) -> str:
    h = hashlib.sha256()
    for ref in sorted(references):
        stat = ref.stat()
        h.update(str(ref).encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()[:16]


def extract_style(
    references: Iterable[str | Path],
    device: str = "cuda",
    use_cache: bool = True,
) -> StyleEmbedding:
    """Compute (or load from cache) the learned style embedding for references.

    Falls back to a tempo-only embedding if CLAP cannot be loaded (e.g. offline
    or the dependency is missing), so generation still proceeds.
    """
    references = [Path(r) for r in references]

    cache_path = STYLE_CACHE_DIR / f"{_cache_key(references)}.json"
    if use_cache and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            logger.info("Loaded cached style embedding (%s)", cache_path.name)
            return StyleEmbedding(
                vector=data.get("vector", []),
                tags=data.get("tags", []),
                tempo_bpm=data.get("tempo_bpm", 0.0),
                references=data.get("references", [str(r) for r in references]),
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Style cache unreadable; recomputing.")

    # Tempo is cheap and complementary to CLAP, so always compute it.
    tempos: list[float] = []
    for ref in references:
        wav, sr = load_audio(ref, mono=False)
        t = estimate_tempo(wav, sr)
        if t:
            tempos.append(t)
    tempo = sum(tempos) / len(tempos) if tempos else 0.0

    vector: list[float] = []
    tags: list[str] = []
    try:
        encoder = StyleEncoder(device=device)
        vec = encoder.embed_audio(references)
        tags = encoder.retrieve_tags(vec)
        vector = vec.cpu().tolist()
    except Exception as exc:  # pragma: no cover - depends on CLAP runtime
        logger.warning(
            "CLAP style embedding unavailable (%s); using tempo-only style.", exc
        )

    embedding = StyleEmbedding(
        vector=vector,
        tags=tags,
        tempo_bpm=tempo,
        references=[str(r) for r in references],
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(embedding.to_json())

    logger.info("Style: %s", embedding.as_prompt_fragment() or "(none)")
    return embedding


def load_style_vector(path: str | Path) -> Optional[torch.Tensor]:
    """Load a persisted style vector if present."""
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(str(path), map_location="cpu")
