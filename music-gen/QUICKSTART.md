# Quickstart — running music-gen on a RunPod GPU

A step-by-step guide to go from a fresh RunPod pod to a generated track.

## 0. Pick a pod

- **GPU:** RTX 6000 Ada (48 GB) recommended — fast and roomy. An A6000 (48 GB,
  cheaper) or A5000 (24 GB, budget) also work.
- **GPU count:** **1** (the tool is single-GPU).
- **Template:** **RunPod PyTorch** (ships with CUDA + PyTorch preinstalled).
- **Disk/volume:** ~50–60 GB so the model weights fit (ACE-Step ~7 GB,
  CLAP ~1.5 GB, Demucs, plus your audio and saved LoRAs).

## 1. Open a terminal

Easiest path — **no SSH keys required**: open your pod's **Jupyter Lab** link,
then **File → New → Terminal**. (The "Enable web terminal" button works too.)
Jupyter also gives you drag-and-drop file upload, which you'll want in step 4.

## 2. Get the code

```bash
cd /workspace
git clone https://github.com/spencerwize/music-creator.git
cd music-creator
git checkout claude/friendly-cerf-izh2j0
cd music-gen
```

Private repo? When prompted, use your GitHub username and a **Personal Access
Token** as the password (GitHub → Settings → Developer settings → Tokens).

## 3. Install everything (~10–15 min the first time)

```bash
bash setup.sh
```

This installs the pinned dependencies and pre-downloads the ACE-Step, Demucs,
and CLAP weights. Sanity check:

```bash
python cli.py --help
```

## 4. Add your audio

Organise material into a named **group** folder so you can reference it all with
a single `--group` flag. In the Jupyter file browser, inside `music-gen/`,
create and fill:

```
references/<group>/   ← your style-reference songs (e.g. references/mysound/*.mp3)
stems/<group>/        ← your anchor stems         (e.g. stems/mysound/*.wav)
```

## 5. Build the model (fine-tune a LoRA on the references)

```bash
python cli.py finetune --group mysound --epochs 100
```

Saves to `loras/mysound/` (named from the group), including the references'
learned style embedding. Watch the `loss` printouts; it checkpoints every 25
epochs.

## 6. Generate

```bash
python cli.py generate \
  --group mysound \
  --lora loras/mysound \
  --prompt "dusty boom bap, 90 BPM" \
  --output outputs/track01.wav
```

With `--group mysound` this stacks all three style levers: the **LoRA** (weights)
+ the **references'** CLAP style tags (prompt) + your **stems** as the audio2audio
backbone.

## 7. Get your track

In the Jupyter file browser, open `music-gen/outputs/`, right-click the `.wav`
→ **Download**.

---

## Common variations

```bash
# No fine-tuning — fast path using style tags + stems only
python cli.py generate --group mysound --prompt "dusty boom bap"

# No stems — pure text+style generation (set a length, default 60s)
python cli.py generate --group mysound --lora loras/mysound \
  --prompt "dark ambient drone" --duration 90

# Stick tighter to your stems (higher) or give the model more freedom (lower)
python cli.py generate --group mysound --lora loras/mysound --ref-strength 0.7
python cli.py generate --group mysound --lora loras/mysound --ref-strength 0.35

# Reproducible take, then change the seed for alternates
python cli.py generate --group mysound --lora loras/mysound --seed 42
```

## What needs what

| You provide | Result |
|-------------|--------|
| stems + references + `--lora` | Full combo: structure + style tags + trained style |
| stems + references | Built around stems, styled by reference tags (no training) |
| stems + `--lora` | Built around stems, styled by the trained LoRA |
| references only (no stems) | Pure text+style generation, no anchor |
| `--lora` only (no stems) | Pure generation in the trained style |
| nothing (no references, no `--lora`) | **Error** — you must define a target style |

## Don't lose your work

Storage can be wiped when a pod stops. Before stopping the pod, either
**download** your `loras/` and `outputs/`, or commit them back to GitHub. Cloning
into `/workspace` (as above) keeps the code on the pod's persistent volume, but
treat anything you care about as worth backing up.
