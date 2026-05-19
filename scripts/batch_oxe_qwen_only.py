#!/usr/bin/env python3
"""Qwen-VL gemini-only 替代脚本：无需 clash 代理，直连 DashScope API。

与 batch_oxe_fast.py --gemini-only 输出完全兼容：
  - 写 episode_XXXXXX/metadata.json
  - shard 全部成功后写 _GEMINI_DONE
  - 写完所有 shard 后写 output_root/_ALL_GEMINI_DONE

不修改任何已有文件，可与 run_sam3_oxe_sam3_only.slurm 配合使用。

用法：
  sbatch run_sam3_oxe_qwen_only.slurm
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from batch_oxe_full_dataset import (  # noqa: E402
    ShardTask,
    detect_camera_keys,
    discover_shards,
    notify_serverchan,
)
from oxe_gemini_sam3_pipeline import (  # noqa: E402
    _decode_camera_frames,
    _episode_instruction,
    _parse_object_list,
    _resize_for_gemini,
    _sample_indices,
)
# Qwen 专用 prompt：相比 _GEMINI_PROMPT 强化三点 ——
#   1. 多部件强制拆开（pot+lid 而非 "pot with lid"）—— SAM3 需要分别 grounding
#   2. 名称必须 1-3 词，禁止 "X with Y" 描述性子句
#   3. 显式提醒不要遗漏小物体（egg、block 等）
_QWEN_PROMPT = (
    "Enumerate EVERY distinct physical object visible across these frames "
    "(scene inventory, not just task-relevant ones).\n"
    'Task context: "{instruction}"\n'
    "CRITICAL rules:\n"
    "- If an object has separable parts that the task interacts with "
    "  (e.g. pot+lid, jar+cap, box+cover), list them as SEPARATE entries: "
    '  e.g. ["pot", "lid"], NOT "pot with lid".\n'
    "- Names must be SHORT (1-3 words): \"red can\", \"green bottle\", \"toy banana\". "
    '  Avoid descriptive sub-clauses like "blue brush with white bristles" — '
    '  just say "blue brush".\n'
    "- DO NOT skip small objects (eggs, blocks, screws, food items). "
    "  Even tiny visible things should be listed.\n"
    "- Include the robot as a SINGLE entry \"robotic arm\". No separate "
    "  entries for parts (gripper, wrist, base).\n"
    "- Also include the main support surface and any large fixed elements.\n"
    "- Disambiguate similar objects by color/shape.\n"
    "- One entry per unique object instance class. No duplicates, no "
    "  abstract nouns, no actions.\n"
    "- Only list objects you can ACTUALLY see.\n"
    "- Aim for 5-15 entries. Output ONLY a JSON array of strings, no commentary.\n"
    'Example: ["robotic arm", "pot", "lid", "white plate", "spatula", '
    '"egg", "stove top", "table"]'
)


# ------------------------------------------------------------------ Qwen call

def _qwen_call(
    images: list[Image.Image],
    instruction: str,
    api_key: str,
    model_name: str,
    image_max_side: int,
    backoffs: tuple[int, ...] = (2, 5, 10),
) -> list[str]:
    prompt = _QWEN_PROMPT.format(instruction=instruction or "(no language instruction)")
    content: list[dict] = []
    for img in images:
        buf = io.BytesIO()
        _resize_for_gemini(img, image_max_side).save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    last_err: Exception | None = None
    for attempt in range(len(backoffs) + 1):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=60.0, proxy=None)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return _parse_object_list(text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= len(backoffs):
                break
            time.sleep(backoffs[attempt])
    assert last_err is not None
    raise last_err


# ------------------------------------------------------------------ worker

def _worker_main(worker_id: int, task_q: mp.Queue, log_q: mp.Queue, cli_args) -> None:
    def log(msg: str):
        log_q.put(f"[w{worker_id}] {msg}")

    try:
        from tfrecord.reader import tfrecord_loader
    except Exception as e:
        log(f"FATAL: import failed: {e}")
        return

    api_key = cli_args.api_key
    max_streak = int(getattr(cli_args, "max_consecutive_failures", 0) or 0)
    fail_streak = 0

    while True:
        try:
            shard: ShardTask | None = task_q.get(timeout=1.0)
        except Exception:  # noqa: BLE001
            continue
        if shard is None:
            break

        output_dir = Path(cli_args.output_root) / shard.rel_out
        output_dir.mkdir(parents=True, exist_ok=True)
        done_marker = output_dir / "_DONE"
        gemini_done_marker = output_dir / "_GEMINI_DONE"

        if done_marker.exists() or gemini_done_marker.exists():
            log(f"[skip] {shard.path} (already done)")
            continue

        claim_path = output_dir / "_CLAIMED"
        claim_stale = 7200
        if claim_path.exists():
            age = time.time() - claim_path.stat().st_mtime
            if age < claim_stale:
                log(f"[skip] {shard.path} (claimed elsewhere, age={age:.0f}s)")
                continue
        try:
            claim_path.write_text(f"w{worker_id}\t{time.time():.0f}\n")
        except OSError:
            log(f"[skip] {shard.path} (race lost)")
            continue
        if claim_path.read_text().split("\t")[0] != f"w{worker_id}":
            log(f"[skip] {shard.path} (race lost)")
            continue

        log(f"[start] {shard.path}")
        start_t = time.time()
        n_ok = 0
        n_fail = 0
        n_skip = 0

        try:
            cam_keys: list[str] | None = None
            for ep_idx, example in enumerate(tfrecord_loader(shard.path, None, None)):
                claim_path.touch()

                if cam_keys is None:
                    cam_keys = detect_camera_keys(example)
                    if not cam_keys:
                        log(f"  no cameras in shard, abort")
                        break

                instruction = _episode_instruction(example)

                # resume if metadata.json already exists
                episode_dir = output_dir / f"episode_{ep_idx:06d}"
                meta_path = episode_dir / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        log(f"  episode {ep_idx}: resume -> objects={meta.get('objects')}")
                        n_ok += 1
                        continue
                    except Exception:
                        pass  # fall through, re-call Qwen

                # decode frames for primary cam
                primary_key = cam_keys[0]
                frames = _decode_camera_frames(example.get(primary_key, []))
                valid = [f for f in frames if f is not None]
                if not valid:
                    log(f"  episode {ep_idx}: no frames, skip")
                    n_skip += 1
                    continue

                indices = _sample_indices(len(valid), cli_args.num_frames)
                sampled = [valid[i] for i in indices]

                t0 = time.time()
                try:
                    objects = _qwen_call(
                        sampled, instruction, api_key,
                        cli_args.model, cli_args.image_max_side,
                    )
                except Exception as e:  # noqa: BLE001
                    log(f"  episode {ep_idx}: Qwen call failed: {e} -> retry next run")
                    n_fail += 1
                    fail_streak += 1
                    if max_streak > 0 and fail_streak >= max_streak:
                        jobid = os.environ.get("SLURM_JOB_ID", "")
                        notify_serverchan(
                            title=f"⚠️ Qwen-only 熔断 job={jobid}",
                            body=f"Worker w{worker_id} hit {fail_streak} consecutive failures. Last shard: {shard.path}",
                        )
                        log(f"FATAL: {fail_streak} consecutive failures, signaling stop")
                        task_q.put(None)
                        return
                    continue

                qwen_ms = (time.time() - t0) * 1000
                log(f"  episode {ep_idx}: instruction='{instruction}' objects={objects} [qwen {qwen_ms:.0f}ms]")
                fail_streak = 0

                # write metadata.json
                episode_dir.mkdir(parents=True, exist_ok=True)
                if not meta_path.exists():
                    meta = {
                        "episode_index": ep_idx,
                        "language_instruction": instruction,
                        "objects": objects,
                        "primary_camera": primary_key.split("/")[-1],
                        "num_gemini_frames": len(sampled),
                        "gemini_frame_indices": indices,
                        "gemini_model": cli_args.model,
                        "cameras": {k.split("/")[-1]: len(_decode_camera_frames(example.get(k, []))) for k in cam_keys},
                    }
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                n_ok += 1

        except Exception as e:  # noqa: BLE001
            log(f"  shard crashed: {e}\n{traceback.format_exc()}")
            n_fail += 1

        elapsed = time.time() - start_t
        if n_fail == 0:
            gemini_done_marker.write_text(f"episodes={n_ok}\nskipped={n_skip}\nt={time.time():.0f}\n")
            log(f"[gemini-done] {shard.path} ok={n_ok} skip={n_skip} elapsed={elapsed:.1f}s")
        else:
            log(f"[partial] {shard.path} ok={n_ok} fail={n_fail} skip={n_skip} -> _GEMINI_DONE NOT written (elapsed={elapsed:.1f}s)")

        try:
            claim_path.unlink(missing_ok=True)
        except OSError:
            pass


# ------------------------------------------------------------------ log pump

def _log_pump(log_q: mp.Queue, stop_evt) -> None:
    while not stop_evt.is_set() or not log_q.empty():
        try:
            msg = log_q.get(timeout=0.2)
            print(msg, flush=True)
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------ main

def main() -> None:
    import threading

    parser = argparse.ArgumentParser(description="Qwen-VL gemini-only OXE pipeline")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--image-max-side", type=int, default=512)
    parser.add_argument("--max-consecutive-failures", type=int, default=50)
    parser.add_argument("--version-policy", default="latest")
    parser.add_argument("--shard-group-policy", default="largest")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY")
    args.api_key = api_key

    ds_filter = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    shards = discover_shards(
        root=args.dataset_root,
        datasets=ds_filter,
        version_policy=args.version_policy,
        shard_group_policy=args.shard_group_policy,
    )
    print(f"Discovered {len(shards)} shards, {args.num_workers} workers, model={args.model}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    log_q = ctx.Queue()
    for s in shards:
        task_q.put(s)
    for _ in range(args.num_workers):
        task_q.put(None)

    import threading as _threading
    stop_evt = _threading.Event()
    log_t = _threading.Thread(target=_log_pump, args=(log_q, stop_evt), daemon=True)
    log_t.start()

    procs = []
    for wid in range(args.num_workers):
        p = ctx.Process(target=_worker_main, args=(wid, task_q, log_q, args))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    stop_evt.set()
    log_t.join(timeout=5)

    sentinel = args.output_root / "_ALL_GEMINI_DONE"
    sentinel.write_text(f"t={time.time():.0f}\n")
    print(f"[qwen-only] sentinel written: {sentinel}")
    print(f"All done. Output -> {args.output_root}")


if __name__ == "__main__":
    main()
