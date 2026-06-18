#!/usr/bin/env python3
"""music-gen — stem-conditioned, style-referenced music generation.

Commands
--------
  generate   Generate a full mix around your anchor stems, styled after a set
             of reference songs and/or a saved LoRA.
  finetune   Train a LoRA style adapter on a set of reference songs.

Run `python cli.py <command> --help` for per-command options.
"""

from __future__ import annotations

import argparse
import sys

from utils import configure_logging, ensure_dirs, pick_device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-gen",
        description="Stem-conditioned, style-referenced music generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (cuda/mps/cpu). Auto-detected when omitted.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- generate ----------------------------------------------------------- #
    g = sub.add_parser(
        "generate",
        help="Generate a track around anchor stems, styled by references or a LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g.add_argument(
        "--stems",
        nargs="+",
        required=True,
        help="Anchor stem audio file(s) the output is built around.",
    )
    g.add_argument(
        "--references",
        nargs="+",
        default=None,
        help="Style-reference song(s) for inference-time conditioning.",
    )
    g.add_argument(
        "--lora",
        default=None,
        help="Path to a saved LoRA adapter dir (loras/<name>) for high style fidelity.",
    )
    g.add_argument("--prompt", default="", help="Optional text prompt.")
    g.add_argument("--output", default="outputs/result.wav", help="Output audio path.")
    g.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Output length in seconds. 0 = match the anchor stems.",
    )
    g.add_argument("--infer-steps", type=int, default=60, help="Diffusion steps.")
    g.add_argument("--guidance-scale", type=float, default=15.0, help="CFG guidance scale.")
    g.add_argument(
        "--ref-strength",
        type=float,
        default=0.5,
        help="How strongly the anchor stems steer generation (0..1).",
    )
    g.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    # -- finetune ----------------------------------------------------------- #
    f = sub.add_parser(
        "finetune",
        help="Train and save a LoRA style adapter on reference songs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    f.add_argument("--references", nargs="+", required=True, help="Reference song(s) to learn.")
    f.add_argument("--name", required=True, help="Name for the saved LoRA (loras/<name>).")
    f.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    f.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    f.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (capacity).")
    f.add_argument(
        "--segment-seconds",
        type=float,
        default=20.0,
        help="Length of audio segments cropped per training step.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    ensure_dirs()

    device = args.device or pick_device()

    if args.command == "generate":
        if not args.references and not args.lora:
            print(
                "error: provide --references and/or --lora to define the target style.",
                file=sys.stderr,
            )
            return 2
        # Imported lazily so heavy ML deps load only when actually generating.
        from generate import run_generation

        out = run_generation(
            stems=args.stems,
            references=args.references,
            lora=args.lora,
            prompt=args.prompt,
            output=args.output,
            duration=args.duration,
            infer_steps=args.infer_steps,
            guidance_scale=args.guidance_scale,
            ref_audio_strength=args.ref_strength,
            seed=args.seed,
            device=device,
        )
        print(f"\n✓ Generated: {out}")
        return 0

    if args.command == "finetune":
        from finetune import run_finetune

        out = run_finetune(
            references=args.references,
            name=args.name,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            segment_seconds=args.segment_seconds,
            device=device,
        )
        print(f"\n✓ LoRA saved: {out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
