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

from utils import (
    REFERENCES_DIR,
    STEMS_DIR,
    collect_group,
    configure_logging,
    ensure_dirs,
    pick_device,
)


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
        "--group",
        default=None,
        help=(
            "Project name. Auto-collects anchor stems from stems/<group>/ and "
            "reference songs from references/<group>/. Explicit --stems / "
            "--references override the auto-collected sets."
        ),
    )
    g.add_argument(
        "--stems",
        nargs="+",
        default=None,
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
    f.add_argument(
        "--group",
        default=None,
        help=(
            "Project name. Auto-collects reference songs from references/<group>/ "
            "and defaults the LoRA --name to the group name."
        ),
    )
    f.add_argument(
        "--references",
        nargs="+",
        default=None,
        help="Reference song(s) to learn (overrides --group auto-collection).",
    )
    f.add_argument(
        "--name",
        default=None,
        help="Name for the saved LoRA (loras/<name>). Defaults to --group.",
    )
    f.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    f.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    f.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (capacity).")
    f.add_argument(
        "--segment-seconds",
        type=float,
        default=20.0,
        help="Length of audio segments cropped per training step.",
    )

    # -- remix -------------------------------------------------------------- #
    r = sub.add_parser(
        "remix",
        help="Re-tempo an acapella and lay it over a freshly generated, in-style instrumental.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    r.add_argument("--vocal", required=True, help="Acapella / vocal stem to preserve.")
    r.add_argument(
        "--group",
        default=None,
        help="Project name. Auto-collects style references from references/<group>/.",
    )
    r.add_argument(
        "--references",
        nargs="+",
        default=None,
        help="Style-reference remix(es) defining the target sound.",
    )
    r.add_argument("--lora", default=None, help="Saved LoRA adapter for high style fidelity.")
    r.add_argument("--prompt", default="", help="Optional text prompt for the instrumental.")
    r.add_argument("--output", default="outputs/remix.wav", help="Output audio path.")
    r.add_argument(
        "--source-bpm",
        type=float,
        default=0.0,
        help="Original tempo of the vocal. Auto-detected from the vocal if omitted.",
    )
    r.add_argument(
        "--target-bpm",
        type=float,
        default=0.0,
        help="Target tempo for the remix (vocal is stretched to match). If "
        "omitted, auto-detected from the references; otherwise keeps the vocal tempo.",
    )
    r.add_argument(
        "--pitch-shift",
        type=float,
        default=0.0,
        help="Shift the vocal by N semitones (for key matching). 0 = no change.",
    )
    r.add_argument(
        "--match-key",
        action="store_true",
        help="Auto-detect the generated instrumental's key and shift the vocal "
        "to match it (overridden by an explicit --pitch-shift).",
    )
    r.add_argument(
        "--vocal-gain",
        type=float,
        default=0.0,
        help="Vocal level relative to the instrumental, in dB.",
    )
    r.add_argument("--infer-steps", type=int, default=60, help="Diffusion steps.")
    r.add_argument("--guidance-scale", type=float, default=15.0, help="CFG guidance scale.")
    r.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    ensure_dirs()

    device = args.device or pick_device()

    if args.command == "generate":
        # Resolve a --group into stem/reference file lists; explicit flags win.
        stems = args.stems
        references = args.references
        if args.group:
            # Stems are optional, so a missing stems/<group>/ folder is fine —
            # generation just falls back to pure text+style.
            if stems is None:
                try:
                    stems = [str(p) for p in collect_group(STEMS_DIR, args.group)]
                except FileNotFoundError:
                    stems = []
            if references is None:
                # References compose with a LoRA, so collect them either way.
                # In LoRA mode a reference folder is optional, so don't error
                # if it's absent — the LoRA already carries the style.
                try:
                    references = [str(p) for p in collect_group(REFERENCES_DIR, args.group)]
                except FileNotFoundError:
                    if not args.lora:
                        raise

        # Stems are optional; style source is not. Without stems, ACE-Step does
        # a pure text+style generation rather than building around an anchor.
        if not references and not args.lora:
            print(
                "error: provide --references, --group, and/or --lora to define the target style.",
                file=sys.stderr,
            )
            return 2
        # Imported lazily so heavy ML deps load only when actually generating.
        from generate import run_generation

        out = run_generation(
            stems=stems,
            references=references,
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
        references = args.references
        name = args.name or args.group
        if args.group and references is None:
            references = [str(p) for p in collect_group(REFERENCES_DIR, args.group)]

        if not references:
            print(
                "error: provide --references or --group to supply reference songs.",
                file=sys.stderr,
            )
            return 2
        if not name:
            print(
                "error: provide --name (or --group) to name the saved LoRA.",
                file=sys.stderr,
            )
            return 2

        from finetune import run_finetune

        out = run_finetune(
            references=references,
            name=name,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            segment_seconds=args.segment_seconds,
            device=device,
        )
        print(f"\n✓ LoRA saved: {out}")
        return 0

    if args.command == "remix":
        references = args.references
        if args.group and references is None:
            try:
                references = [str(p) for p in collect_group(REFERENCES_DIR, args.group)]
            except FileNotFoundError:
                if not args.lora:
                    raise
        if not references and not args.lora:
            print(
                "error: provide --references, --group, and/or --lora to define the remix style.",
                file=sys.stderr,
            )
            return 2

        from remix import run_remix

        out = run_remix(
            vocal=args.vocal,
            references=references,
            lora=args.lora,
            prompt=args.prompt,
            output=args.output,
            source_bpm=args.source_bpm,
            target_bpm=args.target_bpm,
            pitch_shift_semitones=args.pitch_shift,
            match_key=args.match_key,
            vocal_gain_db=args.vocal_gain,
            infer_steps=args.infer_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            device=device,
        )
        print(f"\n✓ Remix saved: {out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
