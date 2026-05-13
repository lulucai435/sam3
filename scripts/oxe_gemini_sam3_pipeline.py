#!/usr/bin/env python3
"""
OpenXEmbodiment TFRecord -> Gemini object identification (one call per episode)
                          -> SAM3 segmentation per frame.

For each episode:
  1. Decode camera observations from the TFRecord.
  2. Pick a primary camera and sample N frames at equal intervals
     (default N=10) from its decodable frames.
  3. Send those N frames + the episode's `steps/language_instruction`
     to Gemini in ONE request and parse a unified object list. Doing
     it per-episode (not per-frame) keeps object names consistent.
  4. Run SAM3 on every frame of every requested camera using that
     object list, saving merged-per-label masks + scores to npz.

Output layout:
  output_dir/
    episode_000000/
      metadata.json
      cam_high/
        000000.npz        # masks (K,H,W) bool, scores (K,) float32
        ...
        vis_000040.png    # optional, only Gemini-sampled frames
      cam_left_wrist/
        ...

Example:
  export GOOGLE_API_KEY=...
  python scripts/oxe_gemini_sam3_pipeline.py \\
      --tfrecord /share/OpenXEmbodiment-Full/aloha_mobile/0.0.1/aloha_mobile-train.tfrecord-00000-of-00160 \\
      --output-dir ./oxe_seg_out \\
      --max-episodes 1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tfrecord.reader import tfrecord_loader

# Reuse SAM3 helpers from the single-image pipeline in this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_sam3_pipeline import (  # noqa: E402
    load_sam3_model,
    run_sam3_per_object,
    _merge_masks_per_label,
)

# --- Gemini SDK detection ----------------------------------------------------
_GEMINI_BACKEND = None
try:
    from google import genai
    from google.genai import types
    _GEMINI_BACKEND = "google-genai"
except ImportError:
    try:
        import google.generativeai as genai  # type: ignore
        types = None  # type: ignore
        _GEMINI_BACKEND = "google-generativeai"
    except ImportError:
        genai = None  # type: ignore
        types = None  # type: ignore


DEFAULT_CAM_KEYS = [
    "steps/observation/cam_high",
    "steps/observation/cam_left_wrist",
    "steps/observation/cam_right_wrist",
]


# --- TFRecord helpers --------------------------------------------------------
def _to_text(x) -> str:
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
        else:
            try:
                x = x.tobytes()
            except Exception:
                x = str(x)
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).decode("utf-8", errors="replace")
    return str(x)


def _safe_decode_image(image_bytes: bytes) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def _decode_camera_frames(seq) -> list[Image.Image | None]:
    return [_safe_decode_image(bytes(raw)) for raw in seq]


def _episode_instruction(example: dict) -> str:
    seq = example.get("steps/language_instruction")
    if seq is None:
        return ""
    for v in seq:
        s = _to_text(v).strip()
        if s:
            return s
    return ""


def _sample_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    k = min(k, n)
    if k == 1:
        return [0]
    return [int(round(x)) for x in np.linspace(0, n - 1, k)]


def _resize_for_gemini(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)


def _parse_object_list(text: str) -> list[str]:
    text = (text or "").strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r"^\s*\[\s*", "", text)
        text = re.sub(r"\s*\]\s*$", "", text)
        obj = [p.strip().strip('"').strip("'") for p in re.split(r"[,\n]+", text) if p.strip()]
    if not isinstance(obj, list):
        obj = [str(obj)]
    seen, out = set(), []
    for x in obj:
        s = str(x).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# --- Gemini: one multi-image call per episode --------------------------------
def gemini_objects_for_episode(
    images: list[Image.Image],
    instruction: str,
    api_key: str | None,
    model_name: str = "gemini-2.0-flash",
    image_max_side: int = 512,
) -> list[str]:
    """Send ALL sampled frames + instruction to Gemini in a single request."""
    if _GEMINI_BACKEND is None:
        raise ImportError("Install google-genai or google-generativeai")
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or pass --gemini-api-key")

    prompt = (
        "Identify unique objects in these frames.\n"
        f'Task: "{instruction}"\n'
        "Requirements:\n"
        '- Include objects from Task, "robotic arm", and environment.\n'
        '- 1-3 words per name (e.g., "silver fork", "red block").\n'
        "- Output ONLY a JSON array of strings.\n"
        'Example: ["robotic arm", "green apple", "metal tray"]'
    )

    if _GEMINI_BACKEND == "google-genai":
        contents = []
        for img in images:
            buf = io.BytesIO()
            _resize_for_gemini(img, image_max_side).save(buf, format="JPEG", quality=90)
            contents.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
        contents.append(prompt)
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=1024),
        )
        text = (resp.text or "").strip()
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resized = [_resize_for_gemini(im, image_max_side) for im in images]
        resp = model.generate_content(
            list(resized) + [prompt],
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=1024),
        )
        text = (resp.text or "").strip()

    return _parse_object_list(text)


# --- SAM3 per-frame ----------------------------------------------------------
def _build_frame_arrays(merged_results, object_names, h, w):
    """Stack merged-per-label results into (K,H,W) bool + (K,) float arrays.
    Row i corresponds to object_names[i]; missing detections are zero."""
    K = len(object_names)
    masks = np.zeros((K, h, w), dtype=np.bool_)
    scores = np.zeros((K,), dtype=np.float32)
    label_to_idx = {n: i for i, n in enumerate(object_names)}
    for r in merged_results:
        i = label_to_idx.get(r["label"])
        if i is None or r["masks"] is None:
            continue
        m = np.asarray(r["masks"]).astype(np.bool_)
        while m.ndim > 2:
            m = m.any(axis=0)
        masks[i] = m
        if r["scores"] is not None and len(r["scores"]) > 0:
            scores[i] = float(np.asarray(r["scores"], dtype=np.float32).max())
    return masks, scores


def _save_vis(image: Image.Image, merged_results, object_names, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    arr = np.array(image.convert("RGB")) / 255.0
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(arr)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(object_names), 1)))
    for r in merged_results:
        if r["masks"] is None:
            continue
        try:
            i = object_names.index(r["label"])
        except ValueError:
            continue
        color = colors[i % len(colors)][:3]
        m = np.asarray(r["masks"]).astype(np.float32)
        while m.ndim > 2:
            m = m.max(axis=0)
        overlay = np.zeros((*m.shape, 4))
        overlay[..., :3] = color
        overlay[..., 3] = 0.4 * m
        ax.imshow(overlay)
        ax.text(0.02, 0.98 - i * 0.025, r["label"],
                transform=ax.transAxes, color=color, fontsize=9,
                verticalalignment="top")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()


# --- Episode processing ------------------------------------------------------
def process_episode(
    episode_idx: int,
    example: dict,
    processor,
    args,
    api_key: str | None,
    output_dir: Path,
):
    instruction = _episode_instruction(example)

    cam_frames: dict[str, list[Image.Image | None]] = {}
    for key in args.cam_keys:
        if key not in example:
            continue
        cam_frames[key.split("/")[-1]] = _decode_camera_frames(example[key])

    if not cam_frames:
        print(f"  episode {episode_idx}: no camera frames in {args.cam_keys}, skip")
        return

    primary_cam = args.cam_keys[0].split("/")[-1]
    if primary_cam not in cam_frames:
        primary_cam = next(iter(cam_frames))

    primary_seq = cam_frames[primary_cam]
    valid_primary = [(i, f) for i, f in enumerate(primary_seq) if f is not None]
    if not valid_primary:
        print(f"  episode {episode_idx}: primary cam '{primary_cam}' has no decodable frames, skip")
        return

    sample_pos = _sample_indices(len(valid_primary), args.num_gemini_frames)
    sampled = [valid_primary[p] for p in sample_pos]
    sampled_indices = [i for i, _ in sampled]
    sampled_imgs = [im for _, im in sampled]

    episode_dir = output_dir / f"episode_{episode_idx:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    meta_path = episode_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        objects = list(meta.get("objects") or [])
        print(f"  episode {episode_idx}: resume -> metadata.json exists, objects={objects}")
    else:
        if args.skip_gemini:
            objects = [s.strip() for s in (args.objects or "").split(",") if s.strip()]
            print(f"  episode {episode_idx}: skip_gemini -> objects={objects}")
        else:
            try:
                objects = gemini_objects_for_episode(
                    sampled_imgs,
                    instruction or "(no language instruction)",
                    api_key=api_key,
                    model_name=args.gemini_model,
                    image_max_side=args.gemini_image_size,
                )
            except Exception as e:
                # Skip metadata.json so the next run retries this episode
                # instead of treating it as already processed.
                print(f"  episode {episode_idx}: Gemini call failed: {e} "
                      f"-> no metadata.json written, will retry on next run")
                return
            print(f"  episode {episode_idx}: instruction='{instruction}'")
            print(f"  episode {episode_idx}: objects={objects}")

        meta = {
            "episode_index": episode_idx,
            "language_instruction": instruction,
            "objects": objects,
            "primary_camera": primary_cam,
            "num_gemini_frames": len(sampled_imgs),
            "gemini_frame_indices": sampled_indices,
            "gemini_model": None if args.skip_gemini else args.gemini_model,
            "cameras": {cam: len(seq) for cam, seq in cam_frames.items()},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    if not objects:
        print(f"  episode {episode_idx}: no objects, skipping SAM3")
        return

    for cam, seq in cam_frames.items():
        cam_dir = episode_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)

        if args.segment_mode == "sampled":
            if cam == primary_cam:
                indices = sampled_indices
            else:
                valid = [i for i, f in enumerate(seq) if f is not None]
                pos = _sample_indices(len(valid), args.num_gemini_frames)
                indices = [valid[p] for p in pos]
        elif args.segment_mode == "every_n":
            stride = max(1, int(getattr(args, "segment_every_n", 1)))
            indices = list(range(0, len(seq), stride))
            if args.max_segment_frames is not None:
                indices = indices[: args.max_segment_frames]
        else:
            indices = list(range(len(seq)))
            if args.max_segment_frames is not None:
                indices = indices[: args.max_segment_frames]

        vis_set = set(sampled_indices) if (args.save_vis and cam == primary_cam) else set()

        for t in indices:
            img = seq[t] if 0 <= t < len(seq) else None
            if img is None:
                continue
            npz_path = cam_dir / f"{t:06d}.npz"
            vis_path = cam_dir / f"vis_{t:06d}.png"
            need_sam3 = not npz_path.exists()
            need_vis = t in vis_set and not vis_path.exists()
            if not need_sam3 and not need_vis:
                continue
            results = run_sam3_per_object(processor, img, objects)
            merged = _merge_masks_per_label(results)
            if need_sam3:
                w, h = img.size
                masks, scores = _build_frame_arrays(merged, objects, h, w)
                np.savez_compressed(npz_path, masks=masks, scores=scores)
            if need_vis:
                _save_vis(img, merged, objects, vis_path)

    total_frames = sum(len(seq) for seq in cam_frames.values())
    print(f"  episode {episode_idx}: done ({len(objects)} objects, "
          f"{total_frames} total frames across {len(cam_frames)} cameras)")


# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="OXE TFRecord -> per-episode Gemini object list -> per-frame SAM3 segmentation."
    )
    ap.add_argument("--tfrecord", required=True, type=Path, help="Path to one .tfrecord shard")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument(
        "--cam-keys",
        type=str,
        default=",".join(DEFAULT_CAM_KEYS),
        help=f"Comma-separated camera observation keys. Default: {','.join(DEFAULT_CAM_KEYS)}",
    )
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--num-gemini-frames", type=int, default=5,
                    help="How many frames to sample per episode for the single Gemini call")
    ap.add_argument("--gemini-image-size", type=int, default=512,
                    help="Max side length (px) for frames sent to Gemini. 0 = no resize")
    ap.add_argument("--gemini-model", default="gemini-3-flash-preview")
    ap.add_argument("--gemini-api-key", default=None)

    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--confidence", type=float, default=0.5, help="SAM3 confidence threshold")
    ap.add_argument("--checkpoint-path", type=str,
                    default="/data/lulucai/code/sam3/weights/sam3.pt",
                    help="Local SAM3 checkpoint .pt; otherwise download from HuggingFace")
    ap.add_argument("--no-download-hf", action="store_true",
                    help="Disallow HuggingFace download (requires --checkpoint-path)")

    ap.add_argument("--segment-mode", choices=("all", "sampled", "every_n"), default="all",
                    help="Segment every frame ('all', default), only the Gemini-sampled frames "
                         "('sampled'), or every Nth frame ('every_n', stride from --segment-every-n)")
    ap.add_argument("--segment-every-n", type=int, default=10,
                    help="Frame stride for --segment-mode every_n (default: 10)")
    ap.add_argument("--max-segment-frames", type=int, default=None,
                    help="Cap segmented frames per camera (used with --segment-mode all or every_n)")
    ap.add_argument("--save-vis", action="store_true",
                    help="Also save overlay PNGs for the Gemini-sampled frames on the primary camera")

    ap.add_argument("--skip-gemini", action="store_true",
                    help="Skip Gemini and use --objects directly (debug)")
    ap.add_argument("--objects", type=str, default=None,
                    help="Comma-separated object list when --skip-gemini")

    args = ap.parse_args()
    args.cam_keys = [k.strip() for k in args.cam_keys.split(",") if k.strip()]
    if not args.cam_keys:
        raise ValueError("Need at least one --cam-keys entry")

    api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
    if not args.skip_gemini and not api_key:
        raise ValueError("Set GOOGLE_API_KEY or pass --gemini-api-key (or use --skip-gemini)")

    load_from_hf = not args.no_download_hf
    if args.no_download_hf and not args.checkpoint_path:
        raise ValueError("--no-download-hf requires --checkpoint-path")

    print("Loading SAM3...")
    _, processor = load_sam3_model(
        device=args.device,
        confidence_threshold=args.confidence,
        checkpoint_path=args.checkpoint_path,
        load_from_hf=load_from_hf,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading TFRecord: {args.tfrecord}")

    for ep_idx, example in enumerate(tfrecord_loader(str(args.tfrecord), None, None)):
        if args.max_episodes is not None and ep_idx >= args.max_episodes:
            break
        process_episode(ep_idx, example, processor, args, api_key, args.output_dir)

    print(f"All done. Output -> {args.output_dir}")


if __name__ == "__main__":
    main()
