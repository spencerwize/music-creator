# music-gen

A local Python CLI for **stem-conditioned, style-referenced music generation**.

Feed it:
- one or more **style-reference songs** (full tracks that define the target sound),
- one or more **anchor stems** (your own drums, vocals, etc.) that the output is built around,
- an optional **text prompt** (e.g. `"dark and atmospheric, 90 BPM"`),

and it produces a full mix that *sounds like the references* but is *structured around your stems*.

Under the hood:
- A **learned CLAP style embedding** (`laion/larger_clap_music_and_speech`) is extracted from
  the references and turned into descriptive prompt tags (genre, mood, texture, instrumentation)
  via zero-shot audio→text retrieval — real style information, not just tempo/brightness.
- **Demucs** (`htdemucs`) separates references when you want per-stem analysis.
- **ACE-Step** (`ACE-Step/ACE-Step-v1-3.5B`) generates the mix, using the anchor stems as
  audio2audio conditioning.
- Optionally, a **LoRA** fine-tuned on the references gives much higher style fidelity.

The style embedding lives in the **prompt path**, so it composes cleanly with both the anchor
stems (which own the audio2audio slot) and a LoRA (which owns the weights). That means you can
run a LoRA *and* references together: the LoRA drives the core sound, the references nudge each
generation. A fine-tuned LoRA also stores its references' embedding, so generating from a LoRA
alone still benefits from the learned style even without passing `--references`.

## Project layout

```
music-gen/
  cli.py          # entry point (generate / finetune)
  separate.py     # Demucs stem separation
  style.py        # learned CLAP style embedding + zero-shot tagging
  generate.py     # style conditioning + ACE-Step inference
  finetune.py     # LoRA fine-tuning on reference songs
  utils.py        # audio I/O, resampling, file management
  outputs/        # generated tracks
  loras/          # saved LoRA adapters, named by you
  references/     # drop reference songs here
  stems/          # drop your anchor stems here
```

## Setup

Designed for a fresh **RunPod PyTorch** box with a CUDA GPU (A100 recommended):

```bash
bash setup.sh
python cli.py --help
```

`setup.sh` installs the pinned dependencies and pre-fetches the ACE-Step and Demucs weights.

## Organising material into project groups

Rather than listing every file, you can group material into named project
folders and reference the whole set with `--group <name>`:

```
references/jdilla_vibes/song1.mp3
references/jdilla_vibes/song2.mp3
stems/jdilla_vibes/my_drums.wav
stems/jdilla_vibes/my_vocals.wav
```

```bash
# Collects all of references/jdilla_vibes/* and stems/jdilla_vibes/*
python cli.py generate --group jdilla_vibes \
  --prompt "dusty boom bap" --output outputs/result.wav

# Collects references/jdilla_vibes/* and names the LoRA "jdilla_vibes"
python cli.py finetune --group jdilla_vibes --epochs 100
```

Explicit `--references` / `--stems` / `--name` always override the
auto-collected group. Any audio extension is picked up
(`.wav .mp3 .flac .m4a .ogg .aiff`).

## Usage

### Generate (inference-time style conditioning)

```bash
python cli.py generate \
  --references references/song1.mp3 references/song2.mp3 \
  --stems stems/my_drums.wav \
  --prompt "dark and atmospheric 90 BPM" \
  --output outputs/result.wav
```

### Fine-tune a LoRA on reference songs

```bash
python cli.py finetune \
  --references references/song1.mp3 references/song2.mp3 references/song3.mp3 \
  --name "jdilla_vibes" \
  --epochs 100
```

Saved to `loras/jdilla_vibes/`.

### Generate with a saved LoRA (higher fidelity)

```bash
python cli.py generate \
  --lora loras/jdilla_vibes \
  --stems stems/my_drums.wav stems/my_vocals.wav \
  --prompt "dusty boom bap" \
  --output outputs/result.wav
```

## Useful flags

| Flag | Applies to | Meaning |
|------|------------|---------|
| `--duration` | generate | Output length in seconds (`0` = match the anchor stems). |
| `--ref-strength` | generate | How strongly the anchor stems steer generation (0–1). |
| `--infer-steps` | generate | Diffusion steps (quality vs. speed). |
| `--guidance-scale` | generate | Classifier-free guidance strength. |
| `--seed` | generate | Reproducible output. |
| `--lora-rank` | finetune | LoRA capacity. |
| `--segment-seconds` | finetune | Audio crop length per training step. |
| `--device` | both | Override auto-detected device (`cuda`/`mps`/`cpu`). |

## Notes

- The anchor stems define the length and rhythmic backbone; `--ref-strength`
  trades off fidelity to your stems vs. creative freedom for the model.
- Separated reference stems are cached under `references/_separated/` and reused.
- LoRA adapters and audio are git-ignored by default (see `.gitignore`).
