#!/usr/bin/env bash
#
# setup.sh — provision a fresh RunPod PyTorch box for music-gen.
#
# Installs Python deps (Demucs, ACE-Step, peft, ...) and pre-fetches the model
# weights so the first generation run doesn't stall on a multi-GB download.
#
# Usage:
#   bash setup.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "==> music-gen setup starting in $HERE"

# --------------------------------------------------------------------------- #
# 1. Python environment
# --------------------------------------------------------------------------- #
PYTHON="${PYTHON:-python3}"
echo "==> Using interpreter: $($PYTHON --version 2>&1)"

$PYTHON -m pip install --upgrade pip setuptools wheel

# On the stock RunPod PyTorch image, torch/torchaudio ship with the matching
# CUDA build. Reinstalling from PyPI can pull a CPU-only or mismatched wheel,
# so only install them if missing.
if $PYTHON -c "import torch, torchaudio" 2>/dev/null; then
  echo "==> torch/torchaudio already present:"
  $PYTHON -c "import torch; print('    torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
  # Install everything except the torch lines (they're already satisfied).
  grep -vE '^(torch|torchaudio)==' requirements.txt > /tmp/requirements.notorch.txt
  $PYTHON -m pip install -r /tmp/requirements.notorch.txt
else
  echo "==> Installing full requirements (including torch)"
  $PYTHON -m pip install -r requirements.txt
fi

# --------------------------------------------------------------------------- #
# 2. Pre-fetch model weights
# --------------------------------------------------------------------------- #
# Cache lives under the HF cache dir; reused across runs on a persistent volume.
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "==> Pre-fetching ACE-Step weights (ACE-Step/ACE-Step-v1-3.5B, ~7GB)"
$PYTHON - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("ACE-Step/ACE-Step-v1-3.5B")
print("    ACE-Step weights cached.")
PY

echo "==> Pre-fetching Demucs weights (htdemucs)"
$PYTHON - <<'PY'
from demucs.pretrained import get_model
get_model("htdemucs")
print("    Demucs htdemucs weights cached.")
PY

echo "==> Pre-fetching CLAP style-encoder weights (laion/larger_clap_music_and_speech, ~1.5GB)"
$PYTHON - <<'PY'
from transformers import ClapModel, ClapProcessor
name = "laion/larger_clap_music_and_speech"
ClapModel.from_pretrained(name)
ClapProcessor.from_pretrained(name)
print("    CLAP weights cached.")
PY

# --------------------------------------------------------------------------- #
# 3. Working directories
# --------------------------------------------------------------------------- #
mkdir -p outputs loras references stems
echo "==> Created working directories: outputs/ loras/ references/ stems/"

echo ""
echo "==> Setup complete. Quick smoke test:"
echo "    python cli.py --help"
echo ""
echo "    Drop reference songs into references/ and anchor stems into stems/, then:"
echo "    python cli.py generate --references references/song1.mp3 \\"
echo "      --stems stems/my_drums.wav --prompt 'dark and atmospheric 90 BPM' \\"
echo "      --output outputs/result.wav"
