#!/usr/bin/env python3
"""最高效的 OXE -> Gemini -> SAM3 pipeline。

与 batch_oxe_full_dataset.py 同等输出（episode_XXXXXX/metadata.json + 每 cam 的
npz/vis），但更快。三个关键优化：

  1. Gemini async prefetch（单 worker 内）—— 用 ThreadPoolExecutor 维护一个
     lookahead 窗口，提前发起后续 episode 的 Gemini call。主线程跑 SAM3 时
     下个 Gemini 已经在路上，**消除 GPU 等网络的 idle 窗口**（之前 ~25%
     util 的根因）。
  2. SAM3 forward 用 bf16 autocast —— ViT backbone 在 bf16 下数值安全且
     1.5-2x 提速，激活显存减半。配合 (2) 可以多塞 worker。
  3. 默认 2 workers/GPU —— bf16 省下的显存能放两份 model + activations，
     两 worker 进一步错峰 Gemini I/O 与 SAM3 GPU。

Gemini 调用内置 3 次指数 backoff（2/5/10s）应对 clash 抖动；不再像旧版
一次 SSL EOF 就丢 episode 等下次 sbatch 补。

不修改 batch_oxe_full_dataset.py / oxe_gemini_sam3_pipeline.py /
gemini_sam3_pipeline.py 任何一行 —— 只 import 它们的 helper。

用法：
  sbatch run_sam3_oxe_fast.slurm

或前台：
  python scripts/batch_oxe_fast.py \\
      --dataset-root /public/home/lisunhku/tensorflow_datasets \\
      --output-root  /public/home/lulucai/data/oxe_seg \\
      --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 2 --gemini-lookahead 2 \\
      --batch-size 8 --max-objects 10 \\
      --segment-mode every_n --segment-every-n 10 \\
      --num-gemini-frames 5 --gemini-model gemini-2.5-flash-lite \\
      --confidence 0.5 --save-vis \\
      --datasets fractal20220817_data,droid,...
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import queue
import signal
import socket
import sys
import threading
import time
import traceback
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# 复用原脚本 helpers —— 不修改原脚本。
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from batch_oxe_full_dataset import (  # noqa: E402
    ShardTask,
    detect_camera_keys,
    discover_shards,
    notify_serverchan,
)
from oxe_gemini_sam3_pipeline import (  # noqa: E402
    _build_frame_arrays,
    _decode_camera_frames,
    _episode_instruction,
    _parse_object_list,
    _resize_for_gemini,
    _sample_indices,
    _save_vis,
)


# ---------------- Gemini SDK detection ----------------
_GEMINI_BACKEND: str | None = None
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


_GEMINI_PROMPT = (
    "Enumerate EVERY distinct physical object visible across these frames "
    "(scene inventory, not just task-relevant ones).\n"
    'Task context: "{instruction}"\n'
    "Requirements:\n"
    "- List ALL distinct visible objects: distractor items, containers, tools.\n"
    '- Include the robot as a SINGLE entry "robotic arm". Do NOT add separate '
    '  entries for its parts ("robot gripper", "robot base", "robot wrist") '
    "  — the arm entry covers the whole manipulator.\n"
    '- Also include the main support surface (e.g. "table", "tray") '
    '  and any large fixed elements visible (e.g. "wall", "shelf").\n'
    "- Each name 1-3 words, descriptive but generic enough that SAM3 "
    '  can ground it (e.g. "red can", "green bottle", "lavender container").\n'
    "- Disambiguate by color/shape when multiple similar objects are present.\n"
    "- One entry per unique object instance class. No duplicates, no parts "
    '  ("can lid"), no actions, no abstract nouns.\n'
    "- Only list objects you can ACTUALLY see in the frames. Do not include "
    '  generic items ("green bottle", "white table") unless they are '
    "  truly present and match the stated color.\n"
    "- Aim for ~5-15 entries. Output ONLY a JSON array of strings, "
    "  no commentary.\n"
    'Example: ["robotic arm", "red soda can", "green bottle", '
    '"orange snack bag", "purple container", "white table"]'
)


def _gemini_call_with_retry(
    images: list[Image.Image],
    instruction: str,
    api_key: str,
    model_name: str,
    image_max_side: int,
    backoffs: tuple[int, ...] = (2, 5, 10),
) -> list[str]:
    """Send sampled frames + task context to Gemini, return parsed object list.

    Bounded retry：默认 4 次尝试，最坏 17s wall clock。所有异常都重试（不区分
    类型，因为 Google SDK 异常层次复杂；最坏多浪费 17s 也比丢 episode 划算）。
    被 ThreadPoolExecutor 调用，所以异常会被封装在 Future 里。"""
    if _GEMINI_BACKEND is None:
        raise ImportError("Install google-genai or google-generativeai")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or pass --gemini-api-key")
    prompt = _GEMINI_PROMPT.format(
        instruction=instruction or "(no language instruction)"
    )

    last_err: Exception | None = None
    for attempt in range(len(backoffs) + 1):
        try:
            if _GEMINI_BACKEND == "google-genai":
                contents = []
                for img in images:
                    buf = io.BytesIO()
                    _resize_for_gemini(img, image_max_side).save(
                        buf, format="JPEG", quality=90
                    )
                    contents.append(
                        types.Part.from_bytes(
                            data=buf.getvalue(), mime_type="image/jpeg"
                        )
                    )
                contents.append(prompt)
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.1, max_output_tokens=1024
                    ),
                )
                text = (resp.text or "").strip()
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                resized = [_resize_for_gemini(im, image_max_side) for im in images]
                resp = model.generate_content(
                    list(resized) + [prompt],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1, max_output_tokens=1024
                    ),
                )
                text = (resp.text or "").strip()
            return _parse_object_list(text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= len(backoffs):
                break
            time.sleep(backoffs[attempt])
    assert last_err is not None
    raise last_err


# ---------------- SAM3 batched forward (bf16 autocast) ----------------
def _run_sam3_batched_bf16(
    model,
    transform,
    postprocessor,
    images: list[Image.Image],
    object_names: list[str],
    batch_size: int,
    log_fn=print,
) -> list[list[dict]]:
    """与 gemini_sam3_pipeline.run_sam3_batched 等价输出格式，但：

    - forward 包 `torch.autocast("cuda", dtype=torch.bfloat16)` —— SAM3 的
      ViT backbone 在 bf16 下安全（layernorm/softmax 由 autocast 自动 fp32
      fallback），1.5-2x 提速，激活显存减半。模型权重仍是 fp32 不动。
    - 保留 OOM 减半重试 + ramp back（同原版）。
    """
    import torch
    from gemini_sam3_pipeline import _make_batched_datapoint
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api as collate

    if not images:
        return []
    if not object_names:
        return [[] for _ in images]

    out: list[list[dict]] = []
    chunk_start = 0
    cur_b = batch_size
    while chunk_start < len(images):
        chunk = images[chunk_start : chunk_start + cur_b]
        dps = []
        chunk_qids: list[list[int]] = []
        for img in chunk:
            dp, qids = _make_batched_datapoint(img, object_names, transform)
            dps.append(dp)
            chunk_qids.append(qids)
        batch = collate(dps, dict_key="d")["d"]
        batch = copy_data_to_device(
            batch, torch.device("cuda"), non_blocking=True
        )
        try:
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                output = model(batch)
        except torch.cuda.OutOfMemoryError:
            del batch, dps
            torch.cuda.empty_cache()
            if cur_b <= 1:
                raise
            cur_b = max(1, cur_b // 2)
            log_fn(
                f"[sam3-bf16] OOM at B={cur_b * 2} K={len(object_names)}; "
                f"retry with B={cur_b}"
            )
            continue
        processed = postprocessor.process_results(output, batch.find_metadatas)
        for qids in chunk_qids:
            per_obj: list[dict] = []
            for name, qid in zip(object_names, qids):
                r = processed.get(qid)
                if r is None:
                    per_obj.append(
                        {"label": name, "masks": None, "boxes": None, "scores": None}
                    )
                    continue
                masks = r.get("masks")
                scores = r.get("scores")
                boxes = r.get("boxes")
                if masks is None or len(masks) == 0:
                    per_obj.append(
                        {"label": name, "masks": None, "boxes": None, "scores": None}
                    )
                    continue
                # bf16 autocast 可能让 mask/score/box 仍是 bf16 tensor；numpy
                # 不支持 bf16，需要先 .float() 提到 fp32 再转。
                def _t2np(t):
                    if t.is_floating_point():
                        t = t.float()
                    return t.detach().cpu().numpy()

                m_np = _t2np(masks).astype(bool)
                s_np = (
                    _t2np(scores)
                    if scores is not None and len(scores) > 0
                    else None
                )
                b_np = (
                    _t2np(boxes)
                    if boxes is not None and len(boxes) > 0
                    else None
                )
                per_obj.append(
                    {"label": name, "masks": m_np, "boxes": b_np, "scores": s_np}
                )
            out.append(per_obj)
        chunk_start += cur_b
        if cur_b < batch_size:
            cur_b = min(batch_size, cur_b * 2)
    return out


# ---------------- Decoded-and-sampled episode payload ----------------
@dataclass
class _PreparedEpisode:
    ep_idx: int
    instruction: str
    cam_frames: dict[str, list]  # cam short name -> [PIL.Image|None, ...]
    primary_cam: str
    sampled_indices: list[int]
    sampled_imgs: list[Image.Image]
    # Filled before SAM3 stage:
    objects: list[str] | None = None
    resume_from_meta: bool = False


def _prepare_episode(
    ep_idx: int, example: dict, cam_keys: list[str], num_gemini_frames: int
) -> _PreparedEpisode | None:
    """Decode every camera in cam_keys, pick primary cam, sample N frames for
    Gemini. Return None on structural skip (no camera / no decodable frame)."""
    instruction = _episode_instruction(example)
    cam_frames: dict[str, list] = {}
    for key in cam_keys:
        if key not in example:
            continue
        cam_frames[key.split("/")[-1]] = _decode_camera_frames(example[key])
    if not cam_frames:
        return None
    primary_cam = cam_keys[0].split("/")[-1]
    if primary_cam not in cam_frames:
        primary_cam = next(iter(cam_frames))
    primary_seq = cam_frames[primary_cam]
    valid_primary = [(i, f) for i, f in enumerate(primary_seq) if f is not None]
    if not valid_primary:
        return None
    sample_pos = _sample_indices(len(valid_primary), num_gemini_frames)
    sampled = [valid_primary[p] for p in sample_pos]
    return _PreparedEpisode(
        ep_idx=ep_idx,
        instruction=instruction,
        cam_frames=cam_frames,
        primary_cam=primary_cam,
        sampled_indices=[i for i, _ in sampled],
        sampled_imgs=[im for _, im in sampled],
    )


# ---------------- Per-episode SAM3 + I/O ----------------
def _sam3_and_save(
    prepared: _PreparedEpisode,
    objects: list[str],
    sam3_runtime,
    cli_args: Namespace,
    output_dir: Path,
    log_fn,
) -> tuple[int, int]:
    """Run SAM3 on every cam's required frames and write npz/vis. Returns
    (total_frames_in_all_cams, n_cams)."""
    from gemini_sam3_pipeline import _merge_masks_per_label

    model, transform, postproc = sam3_runtime
    episode_dir = output_dir / f"episode_{prepared.ep_idx:06d}"

    for cam, seq in prepared.cam_frames.items():
        cam_dir = episode_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        if cli_args.segment_mode == "sampled":
            if cam == prepared.primary_cam:
                indices = prepared.sampled_indices
            else:
                valid = [i for i, f in enumerate(seq) if f is not None]
                pos = _sample_indices(len(valid), cli_args.num_gemini_frames)
                indices = [valid[p] for p in pos]
        elif cli_args.segment_mode == "every_n":
            stride = max(1, int(cli_args.segment_every_n))
            indices = list(range(0, len(seq), stride))
            if cli_args.max_segment_frames is not None:
                indices = indices[: cli_args.max_segment_frames]
        else:
            indices = list(range(len(seq)))
            if cli_args.max_segment_frames is not None:
                indices = indices[: cli_args.max_segment_frames]

        vis_set = (
            set(prepared.sampled_indices)
            if (cli_args.save_vis and cam == prepared.primary_cam)
            else set()
        )

        to_do: list[tuple[int, Image.Image, bool, bool]] = []
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
            to_do.append((t, img, need_sam3, need_vis))
        if not to_do:
            continue
        imgs_to_run = [item[1] for item in to_do]
        per_image_results = _run_sam3_batched_bf16(
            model,
            transform,
            postproc,
            imgs_to_run,
            objects,
            batch_size=cli_args.batch_size,
            log_fn=log_fn,
        )
        for (t, img, need_sam3, need_vis), results in zip(to_do, per_image_results):
            merged = _merge_masks_per_label(results)
            if need_sam3:
                w, h = img.size
                npz_path = cam_dir / f"{t:06d}.npz"
                masks, scores = _build_frame_arrays(merged, objects, h, w)
                np.savez_compressed(npz_path, masks=masks, scores=scores)
            if need_vis:
                vis_path = cam_dir / f"vis_{t:06d}.png"
                _save_vis(
                    img,
                    merged,
                    objects,
                    vis_path,
                    gemini_image_size=cli_args.gemini_image_size,
                )

    total_frames = sum(len(seq) for seq in prepared.cam_frames.values())
    return total_frames, len(prepared.cam_frames)


# ---------------- Worker main ----------------
def _worker_main(
    worker_id: int,
    gpu_id: str,
    task_q: mp.Queue,
    log_q: mp.Queue,
    cli_args: Namespace,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def log(msg: str):
        log_q.put(f"[w{worker_id} gpu={gpu_id}] {msg}")

    # Lazy imports after CUDA_VISIBLE_DEVICES pin.
    try:
        from gemini_sam3_pipeline import build_batched_runtime, load_sam3_model
        from tfrecord.reader import tfrecord_loader
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: import failed: {e}\n{traceback.format_exc()}")
        return

    if getattr(cli_args, "gemini_only", False):
        sam3_runtime = None
        log("gemini-only mode: skipping SAM3 load")
    else:
        try:
            log("loading SAM3 ...")
            model, _ = load_sam3_model(
                device="cuda",
                confidence_threshold=cli_args.confidence,
                checkpoint_path=cli_args.checkpoint_path,
                load_from_hf=not cli_args.no_download_hf,
            )
            transform, postprocessor = build_batched_runtime(
                detection_threshold=cli_args.confidence
            )
            sam3_runtime = (model, transform, postprocessor)
            log(
                f"SAM3 ready (bf16 autocast, B={cli_args.batch_size}, "
                f"gemini_lookahead={cli_args.gemini_lookahead})"
            )
        except Exception as e:  # noqa: BLE001
            log(f"FATAL: SAM3 load failed: {e}\n{traceback.format_exc()}")
            return

    api_key = cli_args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
    max_streak = int(getattr(cli_args, "gemini_max_consecutive_failures", 0) or 0)
    gemini_fail_streak = 0

    if getattr(cli_args, "sam3_only", False):
        gemini_pool = None
    else:
        gemini_pool = ThreadPoolExecutor(
            max_workers=cli_args.gemini_lookahead,
            thread_name_prefix=f"w{worker_id}-gemini",
        )

    def gemini_submit(prepared: _PreparedEpisode) -> Future:
        return gemini_pool.submit(
            _gemini_call_with_retry,
            prepared.sampled_imgs,
            prepared.instruction,
            api_key,
            cli_args.gemini_model,
            cli_args.gemini_image_size,
        )

    def trip_breaker(reason: str, shard_path: str = ""):
        log(f"FATAL: {reason}; signaling parent to drain & stop all workers")
        try:
            os.kill(os.getppid(), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass
        host = os.environ.get("HOSTNAME", "")
        jobid = os.environ.get("SLURM_JOB_ID", "")
        notify_serverchan(
            title=f"⚠️ SAM3 fast 熔断 job={jobid or '?'}",
            body=(
                f"**SAM3 OXE fast pipeline tripped breaker**\n\n"
                f"- Worker: w{worker_id} on gpu={gpu_id}\n"
                f"- Host: {host}\n- SLURM job: {jobid}\n"
                f"- Reason: {reason}\n- Last shard: `{shard_path}`\n\n"
                f"Action: check Gemini key / quota / clash proxy, then `sbatch` again."
            ),
        )

    while True:
        try:
            shard: ShardTask | None = task_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if shard is None:
            break

        output_dir = Path(cli_args.output_root) / shard.rel_out
        output_dir.mkdir(parents=True, exist_ok=True)
        done_marker = output_dir / "_DONE"
        gemini_done_marker = output_dir / "_GEMINI_DONE"
        if done_marker.exists() and not cli_args.force:
            log(f"[skip] {shard.path} (done)")
            continue
        if getattr(cli_args, "gemini_only", False):
            if gemini_done_marker.exists() and not cli_args.force:
                log(f"[skip] {shard.path} (gemini done)")
                continue
        elif getattr(cli_args, "sam3_only", False):
            if not gemini_done_marker.exists():
                log(f"[skip] {shard.path} (gemini not done yet)")
                continue

        # Per-shard claim lock (parity with batch_oxe_full_dataset.py).
        claim_path = output_dir / "_CLAIMED"
        claim_stale = int(getattr(cli_args, "claim_stale_secs", 7200) or 7200)
        if claim_path.exists():
            age = time.time() - claim_path.stat().st_mtime
            if age < claim_stale:
                log(f"[skip] {shard.path} (claimed elsewhere, age={age:.0f}s)")
                continue
            log(f"[steal] {shard.path} stale claim (age={age:.0f}s)")
            claim_path.unlink(missing_ok=True)
        try:
            with open(claim_path, "x") as f:
                f.write(
                    f"job={os.environ.get('SLURM_JOB_ID', '?')} "
                    f"host={socket.gethostname()} pid={os.getpid()} "
                    f"t={time.time():.0f}\n"
                )
        except FileExistsError:
            log(f"[skip] {shard.path} (race lost)")
            continue

        log(f"[start] {shard.path}")
        start_t = time.time()
        n_ok = 0
        n_fail = 0
        n_skip = 0
        try:
            cam_keys_for_shard: list[str] | None = None
            if cli_args.cam_keys != "auto":
                cam_keys_for_shard = [
                    k.strip() for k in cli_args.cam_keys.split(",") if k.strip()
                ]

            ep_iter = enumerate(tfrecord_loader(shard.path, None, None))
            # buffered holds (prepared, future_or_None) in episode order.
            # future is None for structural skips, resume, or skip_gemini cases.
            buffered: list[tuple[_PreparedEpisode, Future | None]] = []

            def submit_next() -> bool:
                """Pull one more episode, decode, kick off Gemini if needed.
                Return False when iterator exhausted (or shard cap hit)."""
                nonlocal cam_keys_for_shard
                try:
                    ep_idx, example = next(ep_iter)
                except StopIteration:
                    return False
                if (
                    cli_args.max_episodes_per_shard is not None
                    and ep_idx >= cli_args.max_episodes_per_shard
                ):
                    return False
                if cam_keys_for_shard is None:
                    cam_keys_for_shard = detect_camera_keys(example)
                    log(
                        f"  [auto-cams] {shard.dataset}/{shard.version}: "
                        f"{cam_keys_for_shard}"
                    )
                    if not cam_keys_for_shard:
                        log(
                            f"  [warn] {shard.path}: no decodable cams; "
                            f"shard cannot proceed"
                        )
                        return False
                prepared = _prepare_episode(
                    ep_idx, example, cam_keys_for_shard, cli_args.num_gemini_frames
                )
                if prepared is None:
                    placeholder = _PreparedEpisode(
                        ep_idx=ep_idx,
                        instruction="",
                        cam_frames={},
                        primary_cam="",
                        sampled_indices=[],
                        sampled_imgs=[],
                    )
                    buffered.append((placeholder, None))
                    return True
                # Resume: metadata.json already on disk -> skip Gemini.
                episode_dir = output_dir / f"episode_{ep_idx:06d}"
                meta_path = episode_dir / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        prepared.objects = list(meta.get("objects") or [])
                        prepared.resume_from_meta = True
                        buffered.append((prepared, None))
                        return True
                    except Exception:
                        pass  # fall through and call Gemini
                if getattr(cli_args, "sam3_only", False):
                    # No metadata.json in SAM3-only mode → skip episode.
                    buffered.append((_PreparedEpisode(
                        ep_idx=ep_idx, instruction="", cam_frames={},
                        primary_cam="", sampled_indices=[], sampled_imgs=[],
                    ), None))
                    return True
                if cli_args.skip_gemini:
                    prepared.objects = [
                        s.strip()
                        for s in (cli_args.objects or "").split(",")
                        if s.strip()
                    ]
                    buffered.append((prepared, None))
                    return True
                buffered.append((prepared, gemini_submit(prepared)))
                return True

            # Warm up: fill the lookahead window with in-flight Gemini calls.
            for _ in range(cli_args.gemini_lookahead):
                if not submit_next():
                    break

            # Consume in order. For each pop, top up first so a Gemini call is
            # always in flight while SAM3 forward runs on the popped episode.
            while buffered:
                # Keep claim fresh — long droid shards (~50min) would otherwise
                # be stolen by another worker.
                try:
                    claim_path.touch()
                except OSError:
                    pass

                prepared, fut = buffered.pop(0)
                submit_next()  # top up before we start SAM3

                if not prepared.cam_frames:
                    log(f"  episode {prepared.ep_idx}: no cameras, skip")
                    n_skip += 1
                    continue

                if prepared.objects is None and fut is not None:
                    t_g = time.time()
                    try:
                        objects = fut.result()
                    except Exception as e:  # noqa: BLE001
                        log(
                            f"  episode {prepared.ep_idx}: Gemini call failed "
                            f"after retries: {e} -> no metadata.json, retry "
                            f"on next sbatch"
                        )
                        n_fail += 1
                        gemini_fail_streak += 1
                        if max_streak > 0 and gemini_fail_streak >= max_streak:
                            trip_breaker(
                                f"{gemini_fail_streak} consecutive Gemini "
                                f"failures (threshold={max_streak})",
                                shard_path=shard.path,
                            )
                            break
                        continue
                    gemini_ms = (time.time() - t_g) * 1000
                    prepared.objects = objects
                    log(
                        f"  episode {prepared.ep_idx}: "
                        f"instruction='{prepared.instruction}' "
                        f"objects={objects} [gemini {gemini_ms:.0f}ms]"
                    )
                else:
                    objects = prepared.objects or []
                    if prepared.resume_from_meta:
                        log(
                            f"  episode {prepared.ep_idx}: resume -> "
                            f"objects={objects}"
                        )

                # K cap (parity with batch_oxe_full_dataset.py).
                max_obj = int(getattr(cli_args, "max_objects", 0) or 0)
                if max_obj > 0 and len(objects) > max_obj:
                    dropped = objects[max_obj:]
                    objects = objects[:max_obj]
                    log(
                        f"  episode {prepared.ep_idx}: K cap {max_obj} -> "
                        f"kept={objects} dropped={dropped}"
                    )

                # Write metadata.json before SAM3 so a mid-shard crash still
                # leaves a resumable artifact.
                episode_dir = output_dir / f"episode_{prepared.ep_idx:06d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                meta_path = episode_dir / "metadata.json"
                if not meta_path.exists():
                    meta = {
                        "episode_index": prepared.ep_idx,
                        "language_instruction": prepared.instruction,
                        "objects": objects,
                        "primary_camera": prepared.primary_cam,
                        "num_gemini_frames": len(prepared.sampled_imgs),
                        "gemini_frame_indices": prepared.sampled_indices,
                        "gemini_model": (
                            None if cli_args.skip_gemini else cli_args.gemini_model
                        ),
                        "cameras": {
                            cam: len(seq)
                            for cam, seq in prepared.cam_frames.items()
                        },
                    }
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)

                if getattr(cli_args, "gemini_only", False):
                    n_ok += 1
                    gemini_fail_streak = 0
                    continue

                if not objects:
                    log(f"  episode {prepared.ep_idx}: no objects, skip SAM3")
                    n_ok += 1
                    gemini_fail_streak = 0
                    continue

                t_s = time.time()
                try:
                    total_frames, n_cams = _sam3_and_save(
                        prepared, objects, sam3_runtime, cli_args,
                        output_dir, log,
                    )
                except Exception as e:  # noqa: BLE001
                    log(
                        f"  episode {prepared.ep_idx}: SAM3 forward failed: "
                        f"{e}\n{traceback.format_exc()}"
                    )
                    n_fail += 1
                    continue
                sam3_ms = (time.time() - t_s) * 1000
                log(
                    f"  episode {prepared.ep_idx}: done ({len(objects)} "
                    f"objects, {total_frames} total frames across {n_cams} "
                    f"cameras) [sam3 {sam3_ms:.0f}ms]"
                )
                n_ok += 1
                gemini_fail_streak = 0

            elapsed = time.time() - start_t
            if n_fail == 0:
                if getattr(cli_args, "gemini_only", False):
                    gemini_done_marker.write_text(
                        f"episodes={n_ok}\nskipped={n_skip}\nt={time.time():.0f}\n"
                    )
                    log(
                        f"[gemini-done] {shard.path} ok={n_ok} skip={n_skip} "
                        f"elapsed={elapsed:.1f}s"
                    )
                else:
                    done_marker.write_text(
                        f"episodes={n_ok}\nskipped={n_skip}\nt={time.time():.0f}\n"
                    )
                    log(
                        f"[done] {shard.path} ok={n_ok} skip={n_skip} "
                        f"elapsed={elapsed:.1f}s"
                    )
            else:
                log(
                    f"[partial] {shard.path} ok={n_ok} fail={n_fail} "
                    f"skip={n_skip} -> _DONE NOT written, retry next run "
                    f"(elapsed={elapsed:.1f}s)"
                )
        except Exception as e:  # noqa: BLE001
            log(f"[err] shard {shard.path}: {e}\n{traceback.format_exc()}")
        finally:
            claim_path.unlink(missing_ok=True)

    if gemini_pool is not None:
        gemini_pool.shutdown(wait=True, cancel_futures=True)
    log("exit")


# ---------------- Log pump ----------------
def _log_pump(log_q: mp.Queue, stop_evt: threading.Event):
    while not stop_evt.is_set() or not log_q.empty():
        try:
            msg = log_q.get(timeout=0.5)
        except queue.Empty:
            continue
        print(msg, flush=True)


# ---------------- Scheduler main ----------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "最高效 OXE pipeline：worker 内 Gemini async prefetch 消除 GPU "
            "等网络的 idle + SAM3 bf16 autocast 提速 + 多 worker per GPU。"
            "复用 batch_oxe_full_dataset.py 的 shard discovery / claim lock。"
        )
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/public/home/lulucai/tensorflow_datasets"),
    )
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument(
        "--version-policy", choices=("latest", "all"), default="latest"
    )
    ap.add_argument(
        "--shard-group-policy", choices=("largest", "all"), default="largest"
    )
    ap.add_argument("--shard-limit", type=int, default=None)
    ap.add_argument("--max-episodes-per-shard", type=int, default=None)

    # Topology.
    ap.add_argument("--gpus", type=str, default="0")
    ap.add_argument(
        "--workers-per-gpu",
        type=int,
        default=2,
        help=(
            "每 GPU 进程数。bf16 后单 worker 显存 ~35-45%%，2 worker 安全；"
            "若 model.to(bf16) 也启用，可上调 3。"
        ),
    )
    ap.add_argument(
        "--gemini-lookahead",
        type=int,
        default=2,
        help=(
            "单 worker 内同时在飞 Gemini call 数。默认 2：1 个 in flight + 1 "
            "个正在被 SAM3 消费。Gemini 比 SAM3 慢得多时可调到 3-4，但每加 "
            "一档 buffer 多占 ~800MB RAM（droid episode 全 cam decode 后）。"
        ),
    )

    ap.add_argument("--cam-keys", type=str, default="auto")
    ap.add_argument("--num-gemini-frames", type=int, default=5)
    ap.add_argument("--gemini-image-size", type=int, default=512)
    ap.add_argument("--gemini-model", default="gemini-2.5-flash-lite")
    ap.add_argument("--gemini-api-key", default=None)

    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--checkpoint-path", type=str, default=None)
    ap.add_argument("--no-download-hf", action="store_true")

    ap.add_argument(
        "--segment-mode",
        choices=("all", "sampled", "every_n"),
        default="every_n",
    )
    ap.add_argument("--segment-every-n", type=int, default=10)
    ap.add_argument("--max-segment-frames", type=int, default=None)
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "SAM3 batched-forward image batch size per worker. Default 8 — "
            "bf16 + 2 workers/GPU 下安全。"
        ),
    )
    ap.add_argument("--max-objects", type=int, default=10)
    ap.add_argument(
        "--gemini-max-consecutive-failures",
        type=int,
        default=10,
        help="单 worker 连续 Gemini 失败到此数即熔断。0 关闭。",
    )

    ap.add_argument("--skip-gemini", action="store_true")
    ap.add_argument("--objects", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--claim-stale-secs", type=int, default=7200)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument(
        "--gemini-only", action="store_true",
        help="只调 Gemini 写 metadata.json，跳过 SAM3。完成后写 _GEMINI_DONE。",
    )
    ap.add_argument(
        "--sam3-only", action="store_true",
        help="只跑 SAM3，读已有 metadata.json，不调 Gemini。需先跑 --gemini-only。",
    )
    ap.add_argument(
        "--poll-interval", type=int, default=0,
        help=(
            "sam3-only 模式下队列耗尽后轮询新 _GEMINI_DONE shard 的间隔（秒）。"
            "0=不轮询（跑完就退出）。>0=持续等待，直到 "
            "output_root/_ALL_GEMINI_DONE 出现且队列空才退出。"
            "用于与 gemini-only job 并行运行。"
        ),
    )

    args = ap.parse_args()
    if args.gemini_only and args.sam3_only:
        raise ValueError("--gemini-only 和 --sam3-only 不能同时使用")

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must list at least one GPU id")
    ds_filter = (
        [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    )

    api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
    if not args.skip_gemini and not api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY or pass --gemini-api-key (or --skip-gemini)"
        )
    args.gemini_api_key = api_key
    if args.no_download_hf and not args.checkpoint_path:
        raise ValueError("--no-download-hf requires --checkpoint-path")

    shards = discover_shards(
        root=args.dataset_root,
        datasets=ds_filter,
        version_policy=args.version_policy,
        shard_group_policy=args.shard_group_policy,
    )
    if args.shard_limit is not None:
        shards = shards[: args.shard_limit]

    by_ds: dict[str, int] = {}
    for s in shards:
        by_ds[s.dataset] = by_ds.get(s.dataset, 0) + 1
    print(f"Discovered {len(shards)} shards across {len(by_ds)} datasets.")
    for ds, n in sorted(by_ds.items()):
        print(f"  {ds}: {n}")
    print(f"Output root: {args.output_root}")
    print(
        f"GPUs: {gpu_ids} x {args.workers_per_gpu} workers/gpu = "
        f"{len(gpu_ids) * args.workers_per_gpu} workers total "
        f"(Gemini lookahead per worker = {args.gemini_lookahead}, "
        f"SAM3 bf16, B={args.batch_size})"
    )

    if args.list_only or not shards:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)

    # sam3-only + poll-interval: 持续轮询新 _GEMINI_DONE shard，与 gemini-only
    # job 并行运行。初始 shards 里过滤掉还没有 _GEMINI_DONE 的（等下轮再捡）。
    poll_interval = int(getattr(args, "poll_interval", 0) or 0)
    sam3_only_live = args.sam3_only and poll_interval > 0
    if sam3_only_live:
        shards = [
            s for s in shards
            if (args.output_root / s.rel_out / "_GEMINI_DONE").exists()
            and not (args.output_root / s.rel_out / "_DONE").exists()
        ]
        print(f"[live] initial eligible shards: {len(shards)}")

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    log_q = ctx.Queue()
    queued_paths: set[str] = set()
    for s in shards:
        task_q.put(s)
        queued_paths.add(s.path)
    num_workers = len(gpu_ids) * args.workers_per_gpu
    if not sam3_only_live:
        # 普通模式：提前放 None sentinel，workers 跑完就退出
        for _ in range(num_workers):
            task_q.put(None)

    stop_evt = threading.Event()
    log_t = threading.Thread(
        target=_log_pump, args=(log_q, stop_evt), daemon=True
    )
    log_t.start()

    procs: list[mp.Process] = []
    wid = 0
    for gpu in gpu_ids:
        for _ in range(args.workers_per_gpu):
            p = ctx.Process(
                target=_worker_main, args=(wid, gpu, task_q, log_q, args)
            )
            p.start()
            procs.append(p)
            wid += 1

    def _send_stop():
        for _ in range(num_workers):
            try:
                task_q.put_nowait(None)
            except Exception:  # noqa: BLE001
                pass

    def _graceful(signum, frame):  # noqa: ARG001
        print(
            "\n[parent] Ctrl-C: draining remaining shards; in-flight will finish.",
            flush=True,
        )
        try:
            while True:
                task_q.get_nowait()
        except queue.Empty:
            pass
        _send_stop()

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    if sam3_only_live:
        # 轮询主循环：每 poll_interval 秒检查新 _GEMINI_DONE shard
        sentinel = args.output_root / "_ALL_GEMINI_DONE"
        all_shards_src = discover_shards(
            root=args.dataset_root,
            datasets=ds_filter,
            version_policy=args.version_policy,
            shard_group_policy=args.shard_group_policy,
        )
        while True:
            time.sleep(poll_interval)
            new = [
                s for s in all_shards_src
                if s.path not in queued_paths
                and (args.output_root / s.rel_out / "_GEMINI_DONE").exists()
                and not (args.output_root / s.rel_out / "_DONE").exists()
            ]
            for s in new:
                task_q.put(s)
                queued_paths.add(s.path)
            if new:
                print(f"[live] queued {len(new)} new shards (total queued: {len(queued_paths)})", flush=True)
            # 退出条件：sentinel 存在且没有新 shard
            if sentinel.exists() and not new:
                print("[live] _ALL_GEMINI_DONE seen + no new shards → stopping workers", flush=True)
                _send_stop()
                break

    for p in procs:
        p.join()

    stop_evt.set()
    log_t.join(timeout=5)

    # gemini-only 完成后写 sentinel，通知 sam3-only-live workers 退出
    if getattr(args, "gemini_only", False):
        sentinel = args.output_root / "_ALL_GEMINI_DONE"
        sentinel.write_text(f"t={time.time():.0f}\n")
        print(f"[gemini-only] sentinel written: {sentinel}")

    print(f"All done. Output -> {args.output_root}")


if __name__ == "__main__":
    main()
