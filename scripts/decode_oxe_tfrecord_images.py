#!/usr/bin/env python3
"""
Decode image observations from OpenX-Embodiment TFRecord into PNG files.

Example:
  python scripts/decode_oxe_tfrecord_images.py \
    --tfrecord /share/OpenXEmbodiment-Full/aloha_mobile/0.0.1/aloha_mobile-train.tfrecord-00000-of-00160 \
    --output-dir ./decoded_aloha_mobile \
    --max-episodes 1
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image
from tfrecord.reader import tfrecord_loader


DEFAULT_IMAGE_KEYS = [
    "steps/observation/cam_high",
    "steps/observation/cam_left_wrist",
    "steps/observation/cam_right_wrist",
]


def _safe_decode_image(image_bytes: bytes):
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def _to_text(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).decode("utf-8", errors="replace")
    return str(x)


def decode_tfrecord_images(
    tfrecord_path: Path,
    output_dir: Path,
    image_keys: list[str],
    max_episodes: int | None,
    max_steps: int | None,
    write_metadata: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_count = 0
    saved_count = 0
    skipped_count = 0

    for episode_idx, example in enumerate(tfrecord_loader(str(tfrecord_path), None, None)):
        if max_episodes is not None and episode_idx >= max_episodes:
            break

        episode_dir = output_dir / f"episode_{episode_idx:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Optional: write episode-level metadata + per-frame language instruction.
        if write_metadata:
            episode_metadata = {}
            for k, v in example.items():
                if k.startswith("episode_metadata/"):
                    episode_metadata[k] = _to_text(v)

            language_per_frame = []
            if "steps/language_instruction" in example:
                seq = example["steps/language_instruction"]
                steps_for_lang = len(seq) if max_steps is None else min(len(seq), max_steps)
                language_per_frame = [_to_text(seq[t]) for t in range(steps_for_lang)]

            meta = {
                "episode_index": episode_idx,
                "episode_metadata": episode_metadata,
                "num_language_frames": len(language_per_frame),
                "language_instruction_per_frame": language_per_frame,
            }
            with open(episode_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        for key in image_keys:
            if key not in example:
                continue

            cam_name = key.split("/")[-1]
            cam_dir = episode_dir / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)

            sequence = example[key]
            steps = len(sequence)
            if max_steps is not None:
                steps = min(steps, max_steps)

            for t in range(steps):
                raw = sequence[t]
                # np.bytes_ is bytes-like; make sure we pass plain bytes to PIL
                image = _safe_decode_image(bytes(raw))
                if image is None:
                    skipped_count += 1
                    continue
                out_path = cam_dir / f"{t:06d}.png"
                image.save(out_path)
                saved_count += 1

        episode_count += 1

    print(f"done. episodes={episode_count}, saved_png={saved_count}, skipped={skipped_count}")
    print(f"output: {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="Decode OXE TFRecord image observations to PNG.")
    ap.add_argument("--tfrecord", required=True, type=Path, help="Path to one .tfrecord shard")
    ap.add_argument("--output-dir", required=True, type=Path, help="Directory to save decoded PNG files")
    ap.add_argument(
        "--image-keys",
        type=str,
        default=",".join(DEFAULT_IMAGE_KEYS),
        help=(
            "Comma-separated TFRecord image keys under steps/observation. "
            f"Default: {','.join(DEFAULT_IMAGE_KEYS)}"
        ),
    )
    ap.add_argument("--max-episodes", type=int, default=None, help="Decode at most this many episodes")
    ap.add_argument("--max-steps", type=int, default=None, help="Decode at most this many steps per episode")
    ap.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write metadata.json (episode_metadata + language per frame).",
    )
    args = ap.parse_args()

    keys = [k.strip() for k in args.image_keys.split(",") if k.strip()]
    if not keys:
        raise ValueError("No image keys provided.")

    decode_tfrecord_images(
        tfrecord_path=args.tfrecord,
        output_dir=args.output_dir,
        image_keys=keys,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        write_metadata=not args.no_metadata,
    )


if __name__ == "__main__":
    main()

