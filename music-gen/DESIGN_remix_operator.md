# Remix Operator — Design

Status: **design / scaffolding**. Branch: `claude/remix-operator`.

## Goal

Learn the *transformation* of remixing from **paired data** — `(original, remix)`
examples — then apply it to a brand-new original to produce a remix.

This is different from the style LoRA already in `main`:

| | Style LoRA (existing) | Remix Operator (this branch) |
|---|---|---|
| Trains on | remix audio only | (original → remix) pairs |
| Learns | what remixes *sound like* | what remixing *does* |
| Conditioned on | text/style prompt | the input song + style |
| Inference | build new audio in that style | feed a new original, get its remix |

It is the audio analog of paired image-to-image translation.

## Data principle (decided)

- **Destinations consistent, sources varied.** All *remixes* should share one
  target style S; the *originals* should be diverse. The model generalizes the
  arrow `original → remix-in-style-S`.
- One trained operator == one target remix style. Different styles → different
  operators (like different LoRAs).
- Target: **20–50 pairs** minimum for a coherent operator. Fewer overfits;
  stylistically scattered remixes won't converge.

### Directory layout

Originals and remixes live in sibling folders under a named style, **matched by
filename** (same basename in each folder == one pair):

```
references/<style>/originals/   songA.wav  songB.mp3  songC.wav
references/<style>/remixes/     songA.wav  songB.wav  songC.wav
```

e.g. `references/lo-fi/originals/` + `references/lo-fi/remixes/`. A pair is
formed when a basename appears in both folders; unmatched files are skipped with
a warning. (Extension may differ between the two sides; the stem is what's
matched.)

## The crux: how to condition on the original

The transformer's `forward` already accepts conditioning we can exploit:
`hidden_states, attention_mask, encoder_text_hidden_states, text_attention_mask,
speaker_embeds, lyric_token_idx, lyric_mask, timestep,
block_controlnet_hidden_states, controlnet_scale`.

Three candidate mechanisms, in order of fidelity vs. effort:

### Option A — Control encoder via `block_controlnet_hidden_states` (recommended)
Build a small trainable encoder: `original audio → DCAE latents → control
hidden states` (one tensor per transformer block, matching hidden dim and
sequence length), injected through the existing `block_controlnet_hidden_states`
hook and scaled by `controlnet_scale`. Train this encoder + an attention LoRA on
the paired denoising objective.
- ✅ Uses the model's intended control path; time-aligned conditioning.
- ⚠️ Must reverse-engineer the exact expected shapes/semantics (ACE-Step ships
  the hook but not the ControlNet) — this is the Phase 0 spike.

### Option B — In-context concatenation
Concatenate original latents to the noisy remix latents (channel or sequence
dim) and adapt the patch-embed + LoRA to consume them.
- ✅ Conceptually simple, time-aligned.
- ⚠️ Changes transformer input dims → patch-embed surgery; more fragile.

### Option C — Global embedding (IP-Adapter-lite, fallback)
Encode the original with the **CLAP encoder we already have**, project it into
the text-encoder hidden space, and append as extra conditioning tokens. Train
the projection + LoRA.
- ✅ Lightest; reuses existing CLAP path.
- ⚠️ Global (not time-aligned) — captures the original's character but not its
  structure. Acceptable only if combined with audio2audio for temporal anchor.

**Plan:** spike Option A first; fall back to C (or A+C) if the control-shape
work proves too costly.

## Training objective

Same flow-matching denoising as the existing LoRA, but conditioned on the
original:
1. Encode both songs to latents (frozen DCAE): `z_orig`, `z_remix`.
2. Noise the **remix**: `noisy = (1-t)·z_remix + t·noise`.
3. Predict velocity `noise - z_remix`, conditioned on **style prompt + the
   original** (via the chosen mechanism above).
4. MSE loss; update only the control encoder + LoRA. Base model frozen.

The diffusion objective (vs. a direct regression) is important: it models the
*distribution* of valid remixes, so the one-original-to-many-remixes ambiguity
doesn't collapse to mush.

## Alignment (data pipeline)

Originals and remixes differ in tempo/key/length/structure. Before training we
align each pair as much as is cheap, reusing the DSP already in `utils.py` /
`style.py`:
- **Tempo**: detect both BPMs, time-stretch the original to the remix's grid.
- **Key** (optional): detect both, pitch-shift the original to match.
- **Length**: trim/pad to a common length; segment into aligned crops.

Perfect structural alignment (reordered sections, added drops) is *not* solved
here — the diffusion model is expected to absorb that residual variation. We
align the easy, global axes (tempo/key) and let training handle the rest.

## Inference

New CLI surface (working name):
```
python cli.py train-operator --style lo-fi --name lofi_remixer --epochs N
#   (reads references/lo-fi/originals + references/lo-fi/remixes)
python cli.py apply-remix    --operator loras_op/lofi_remixer --original song.wav \
                             --output outputs/song_remixed.wav
```
`apply-remix` encodes the new original, runs conditioned generation, and writes
the remix. May reuse the existing `remix` mix/key/tempo utilities for polish.

## Phases & milestones

- **Phase 0 — Spike (derisk the crux).** In a throwaway script on the GPU box,
  introspect `ACEStepTransformer2DModel`: number of blocks, hidden dim, latent
  sequence length, and the exact shape/semantics `block_controlnet_hidden_states`
  expects. Decide A vs B vs C. *Exit criterion: a forward pass that runs with a
  hand-constructed control tensor and visibly changes the output.*
- **Phase 1 — Data pipeline.** `pairs.py`: discover pairs, align (tempo/key),
  segment into aligned `(orig_crop, remix_crop)` tensors. Unit-testable on CPU.
- **Phase 2 — Conditioning + training.** `operator.py`: the control encoder and
  a training loop extending `finetune.py`. Saves operator + style metadata.
- **Phase 3 — Inference.** `apply-remix` command end-to-end on one new song.
- **Phase 4 — Eval & iterate.** Listen tests; tune control scale, LoRA rank,
  alignment strength, epochs; watch overfitting.

## File plan

```
music-gen/
  pairs.py            # Phase 1: paired dataset + alignment (scaffolded here)
  remix_operator.py   # Phase 2: control encoder + training (stub here)
                      #   (named remix_operator, not operator, to avoid
                      #    shadowing Python's stdlib `operator` module)
  references/<style>/originals/   # user data: input songs
  references/<style>/remixes/     # user data: matching remixes (by filename)
  loras_op/           # trained remix operators (separate from style loras/)
  DESIGN_remix_operator.md
```

## Risks / open questions

1. **Control shapes (Phase 0)** — biggest unknown; everything hinges on it.
2. **Data volume/consistency** — needs a real audit of available pairs.
3. **Structural misalignment** — pairs that reorder sections may train fuzzy;
   may need section-level alignment later.
4. **Compute** — paired training encodes two songs/step; ~2× the I/O of the
   style LoRA. Still single-GPU-feasible.
5. **Evaluation** — no automatic "good remix" metric; relies on listening.

## Reuse from `main`

- DCAE encode, conditioning builder, flow-matching loop → `finetune.py`.
- Tempo/key detection, time-stretch, pitch-shift, mixing → `style.py`/`utils.py`.
- CLAP encoder (for Option C and/or operator style metadata) → `style.py`.
