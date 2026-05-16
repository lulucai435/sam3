#!/usr/bin/env python3
"""
Multi-GPU, multi-process driver that runs `oxe_gemini_sam3_pipeline.process_episode`
over every TFRecord shard in the full OpenX-Embodiment dataset.

This is a thin scheduler — all episode-level logic (Gemini call per episode,
per-frame SAM3, metadata.json, resume) lives in `oxe_gemini_sam3_pipeline.py`
and is reused verbatim.

Parallelism:
  * `--gpus 0,1,2,3`: one worker process per physical GPU, pinned via
    CUDA_VISIBLE_DEVICES before CUDA/torch imports.
  * `--workers-per-gpu N`: extra concurrent workers per GPU (N>1 lets one
    worker's Gemini network I/O overlap with another's GPU-bound SAM3 pass,
    which is how this script is "multi-threaded" — at the OS-process level).
  * Shards are pulled from a shared queue so the slowest-shard tail is balanced.

Output layout (same as single-shard script, one level deeper):
  <output_root>/<dataset>/<version>/<shard_basename>/episode_XXXXXX/...

Example:
  export GOOGLE_API_KEY=...
  python scripts/batch_oxe_full_dataset.py \\
      --dataset-root ~/data/OpenXEmbodiment-Full \\
      --output-root  ~/data/oxe_seg \\
      --gpus 0,1,2,3 --workers-per-gpu 2 \\
      --segment-mode sampled --num-gemini-frames 5
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

# NB: torch / sam3 / google-genai are imported lazily inside workers so that
# CUDA_VISIBLE_DEVICES pinning takes effect before CUDA is initialized.

from PIL import Image


_SHARD_RE = re.compile(r"^(?P<prefix>.+)\.tfrecord-(?P<idx>\d{5})-of-(?P<total>\d{5})$")


def notify_serverchan(title: str, body: str, timeout: float = 10.0) -> bool:
    """Push a message to WeChat via Server酱. Reads SCT_KEY from env.
    Returns True if delivered, False otherwise. Never raises — notification
    failure must not poison the caller."""
    key = os.environ.get("SCT_KEY", "").strip()
    if not key:
        return False
    try:
        import urllib.parse
        import urllib.request
        url = f"https://sctapi.ftqq.com/{urllib.parse.quote(key)}.send"
        data = urllib.parse.urlencode({"title": title, "desp": body}).encode("utf-8")
        # The script runs with HTTPS_PROXY=clash already exported in SLURM, so
        # urllib follows that env automatically.
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
        return ok
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class ShardTask:
    dataset: str   # "aloha_mobile"
    version: str   # "0.0.1"
    prefix: str    # "aloha_mobile-train"
    total: str     # "00160"
    path: str      # absolute path

    @property
    def rel_out(self) -> str:
        return f"{self.dataset}/{self.version}/{self.prefix}__of_{self.total}/{Path(self.path).name}"


# -------------------------- shard discovery --------------------------

def discover_shards(
    root: Path,
    datasets: list[str] | None,
    version_policy: str,
    shard_group_policy: str,
) -> list[ShardTask]:
    """Walk <root>/<dataset>/<version>/*.tfrecord-NNNNN-of-MMMMM.

    * version_policy:
        - "latest": keep only the lexicographically largest version dir per dataset.
        - "all":    keep every version dir.
    * shard_group_policy (handles datasets where multiple shard groups coexist,
      e.g. aloha_mobile has both `-of-00160` and `-of-00166` side-by-side):
        - "largest": per (dataset, version, prefix), keep the group with the largest MMMMM.
        - "all":     keep every group.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    wanted = set(datasets) if datasets else None
    # dataset -> version -> list[ShardTask]
    by_ds: dict[str, dict[str, list[ShardTask]]] = {}

    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        if wanted is not None and ds_name not in wanted:
            continue
        for ver_dir in sorted(ds_dir.iterdir()):
            if not ver_dir.is_dir():
                continue
            try:
                entries = os.listdir(ver_dir)
            except OSError:
                continue
            shards_here: list[ShardTask] = []
            for name in entries:
                m = _SHARD_RE.match(name)
                if not m:
                    continue
                p = ver_dir / name
                if not p.is_file():
                    continue
                shards_here.append(ShardTask(
                    dataset=ds_name,
                    version=ver_dir.name,
                    prefix=m.group("prefix"),
                    total=m.group("total"),
                    path=str(p),
                ))
            if shards_here:
                by_ds.setdefault(ds_name, {}).setdefault(ver_dir.name, []).extend(shards_here)

    if version_policy == "latest":
        for ds_name, ver_map in list(by_ds.items()):
            keep = max(ver_map.keys())
            by_ds[ds_name] = {keep: ver_map[keep]}
    elif version_policy != "all":
        raise ValueError(f"bad version_policy: {version_policy}")

    selected: list[ShardTask] = []
    for ver_map in by_ds.values():
        for shards in ver_map.values():
            groups: dict[tuple[str, str], list[ShardTask]] = {}
            for s in shards:
                groups.setdefault((s.prefix, s.total), []).append(s)
            if shard_group_policy == "largest":
                by_prefix: dict[str, list[ShardTask]] = {}
                for (prefix, total), group in groups.items():
                    cur = by_prefix.get(prefix)
                    if cur is None or int(total) > int(cur[0].total):
                        by_prefix[prefix] = group
                for g in by_prefix.values():
                    selected.extend(g)
            elif shard_group_policy == "all":
                for g in groups.values():
                    selected.extend(g)
            else:
                raise ValueError(f"bad shard_group_policy: {shard_group_policy}")

    selected.sort(key=lambda s: (s.dataset, s.version, s.prefix, s.total, s.path))
    return selected


# -------------------------- camera key auto-detection --------------------------

def _safe_decode_image(image_bytes: bytes) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def detect_camera_keys(example: dict) -> list[str]:
    """PIL-probe first element of each `steps/observation/*` field."""
    cams: list[str] = []
    for key, val in example.items():
        if not key.startswith("steps/observation/") or val is None or len(val) == 0:
            continue
        try:
            raw = bytes(val[0])
        except Exception:
            continue
        if _safe_decode_image(raw) is not None:
            cams.append(key)
    cams.sort()
    return cams


# -------------------------- worker --------------------------

def _build_episode_args(cli_args, cam_keys: list[str]) -> Namespace:
    """Build an argparse.Namespace in the shape `process_episode` expects."""
    return Namespace(
        cam_keys=cam_keys,
        num_gemini_frames=cli_args.num_gemini_frames,
        gemini_image_size=cli_args.gemini_image_size,
        gemini_model=cli_args.gemini_model,
        skip_gemini=cli_args.skip_gemini,
        objects=cli_args.objects,
        segment_mode=cli_args.segment_mode,
        segment_every_n=cli_args.segment_every_n,
        max_segment_frames=cli_args.max_segment_frames,
        save_vis=cli_args.save_vis,
        batch_size=cli_args.batch_size,
        max_objects=cli_args.max_objects,
    )


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

    # Lazy imports after GPU pinning.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tfrecord.reader import tfrecord_loader
        from gemini_sam3_pipeline import load_sam3_model, build_batched_runtime
        import oxe_gemini_sam3_pipeline as oxe_pipe
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: import failed: {e}\n{traceback.format_exc()}")
        return

    # Load SAM3 once per worker.
    try:
        log("loading SAM3 ...")
        model, _ = load_sam3_model(
            device="cuda",
            confidence_threshold=cli_args.confidence,
            checkpoint_path=cli_args.checkpoint_path,
            load_from_hf=not cli_args.no_download_hf,
        )
        transform, postprocessor = build_batched_runtime(
            detection_threshold=cli_args.confidence,
        )
        sam3_runtime = (model, transform, postprocessor)
        log(f"SAM3 ready (batched path, batch_size={cli_args.batch_size})")
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: SAM3 load failed: {e}\n{traceback.format_exc()}")
        return

    api_key = cli_args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
    max_streak = int(getattr(cli_args, "gemini_max_consecutive_failures", 0) or 0)
    gemini_fail_streak = 0  # consecutive Gemini-failed episodes seen by this worker

    def trip_breaker(reason: str, shard_path: str = ""):
        log(f"FATAL: {reason}; signaling parent to drain & stop all workers")
        try:
            os.kill(os.getppid(), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass
        # WeChat push via Server酱 (SCT_KEY env). Best-effort, silent on failure.
        host = os.environ.get("HOSTNAME", "")
        jobid = os.environ.get("SLURM_JOB_ID", "")
        body = (
            f"**SAM3 OXE pipeline tripped circuit breaker**\n\n"
            f"- Worker: w{worker_id} on gpu={gpu_id}\n"
            f"- Host: {host}\n"
            f"- SLURM job: {jobid}\n"
            f"- Reason: {reason}\n"
            f"- Last shard: `{shard_path}`\n\n"
            f"Action: check Gemini key / quota / clash proxy, then `sbatch` again."
        )
        notify_serverchan(title=f"⚠️ SAM3 熔断 job={jobid or '?'}", body=body)

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
        if done_marker.exists() and not cli_args.force:
            log(f"[skip] {shard.path} (done)")
            continue

        # Cross-job claim lock: prevents two concurrent jobs (e.g. an 8-GPU
        # sbatch + a 1-GPU dev-container worker writing to the same output
        # root) from picking the same shard. Stale claims (no mtime bump for
        # claim_stale_secs) get stolen on the assumption the holder crashed.
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
                    f"host={socket.gethostname()} "
                    f"pid={os.getpid()} "
                    f"t={time.time():.0f}\n"
                )
        except FileExistsError:
            log(f"[skip] {shard.path} (race lost to another worker)")
            continue

        log(f"[start] {shard.path}")
        start_t = time.time()
        n_ok = 0
        n_fail = 0  # episodes that returned False (Gemini failure) or raised
        try:
            cam_keys_for_shard: list[str] | None = None
            if cli_args.cam_keys != "auto":
                cam_keys_for_shard = [
                    k.strip() for k in cli_args.cam_keys.split(",") if k.strip()
                ]

            ep_args: Namespace | None = None
            for ep_idx, example in enumerate(tfrecord_loader(shard.path, None, None)):
                if cli_args.max_episodes_per_shard is not None and ep_idx >= cli_args.max_episodes_per_shard:
                    break
                if cam_keys_for_shard is None:
                    cam_keys_for_shard = detect_camera_keys(example)
                    log(f"  [auto-cams] {shard.dataset}/{shard.version}: {cam_keys_for_shard}")
                    if not cam_keys_for_shard:
                        log(f"  [warn] {shard.path}: no decodable cameras, skipping shard")
                        break
                if ep_args is None:
                    ep_args = _build_episode_args(cli_args, cam_keys_for_shard)

                # Keep claim fresh so long-running shards don't get stolen.
                try:
                    claim_path.touch()
                except OSError:
                    pass

                try:
                    ok = oxe_pipe.process_episode(
                        ep_idx, example, sam3_runtime, ep_args, api_key, output_dir,
                    )
                    if ok:
                        n_ok += 1
                        gemini_fail_streak = 0
                    else:
                        n_fail += 1
                        gemini_fail_streak += 1
                except Exception as e:  # noqa: BLE001
                    n_fail += 1
                    gemini_fail_streak += 1
                    log(f"[err] ep {ep_idx} in {shard.path}: {e}\n{traceback.format_exc()}")

                if max_streak > 0 and gemini_fail_streak >= max_streak:
                    trip_breaker(
                        f"{gemini_fail_streak} consecutive Gemini failures "
                        f"(threshold={max_streak}); current shard finishes its "
                        f"loop then this worker exits",
                        shard_path=shard.path,
                    )
                    break  # bail out of episode loop in this shard

            if n_fail == 0:
                done_marker.write_text(f"episodes={n_ok}\nt={time.time():.0f}\n")
                log(f"[done] {shard.path} episodes={n_ok} elapsed={time.time() - start_t:.1f}s")
            else:
                log(f"[partial] {shard.path} ok={n_ok} fail={n_fail} "
                    f"-> _DONE NOT written, retry next run "
                    f"(elapsed={time.time() - start_t:.1f}s)")
        except Exception as e:  # noqa: BLE001
            log(f"[err] shard {shard.path}: {e}\n{traceback.format_exc()}")
        finally:
            # Always release the claim, otherwise the next sbatch would have to
            # wait claim_stale_secs to steal it.
            claim_path.unlink(missing_ok=True)

    log("exit")


def _log_pump(log_q: mp.Queue, stop_evt: threading.Event):
    while not stop_evt.is_set() or not log_q.empty():
        try:
            msg = log_q.get(timeout=0.5)
        except queue.Empty:
            continue
        print(msg, flush=True)


# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Multi-GPU driver over the full OpenX-Embodiment dataset. "
                    "Reuses process_episode() from oxe_gemini_sam3_pipeline.py.",
    )
    ap.add_argument("--dataset-root", type=Path,
                    default=Path("/public/home/lulucai/tensorflow_datasets"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--datasets", type=str, default=None,
                    help="Comma-separated dataset directory names to include (default: all)")
    ap.add_argument("--version-policy", choices=("latest", "all"), default="latest")
    ap.add_argument("--shard-group-policy", choices=("largest", "all"), default="largest",
                    help="For a dataset with multiple shard groups (e.g. ...-of-00160 and "
                         "...-of-00166 side-by-side), which to keep")
    ap.add_argument("--shard-limit", type=int, default=None, help="Cap total shards (debug)")
    ap.add_argument("--max-episodes-per-shard", type=int, default=None, help="Cap episodes per shard (debug)")

    # Topology.
    ap.add_argument("--gpus", type=str, default="0",
                    help="Comma-separated physical GPU IDs (nvidia-smi). Default: 0")
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="Concurrent worker processes per GPU. >1 overlaps Gemini I/O with SAM3 GPU work.")

    # Forwarded to process_episode (matches oxe_gemini_sam3_pipeline.main).
    ap.add_argument("--cam-keys", type=str, default="auto",
                    help="'auto' to detect from each shard's first example, or comma-separated "
                         "TFRecord keys like 'steps/observation/image'")
    ap.add_argument("--num-gemini-frames", type=int, default=5)
    ap.add_argument("--gemini-image-size", type=int, default=512)
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--gemini-api-key", default=None)

    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--checkpoint-path", type=str, default=None,
                    help="Local SAM3 checkpoint .pt; otherwise resolved via the "
                         "HuggingFace cache (models--facebook--sam3) unless "
                         "--no-download-hf is also passed")
    ap.add_argument("--no-download-hf", action="store_true")

    ap.add_argument("--segment-mode", choices=("all", "sampled", "every_n"), default="all")
    ap.add_argument("--segment-every-n", type=int, default=10,
                    help="Frame stride for --segment-mode every_n (default: 10)")
    ap.add_argument("--max-segment-frames", type=int, default=None)
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="SAM3 batched-forward image batch size per worker. "
                         "Roughly +4.5 GiB VRAM per image at 1008x1008 resolution. "
                         "B=16 ≈ 70 GiB (half of an H200).")
    ap.add_argument("--max-objects", type=int, default=10,
                    help="Cap K (objects per episode). K>~10 forces SAM3 batched path to "
                         "downgrade to B=4 (OOM), which is ~4x slower. 0 disables the cap.")
    ap.add_argument("--gemini-max-consecutive-failures", type=int, default=5,
                    help="Abort the job after this many consecutive Gemini-failed "
                         "episodes in a single worker. Usually means key revoked, "
                         "quota hit, or proxy down — better to stop early than burn "
                         "through thousands of shards. 0 disables the circuit breaker.")

    ap.add_argument("--skip-gemini", action="store_true")
    ap.add_argument("--objects", type=str, default=None)

    ap.add_argument("--force", action="store_true", help="Re-process shards with an existing _DONE marker")
    ap.add_argument("--claim-stale-secs", type=int, default=7200,
                    help="A shard's _CLAIMED file is considered stale (and can be stolen) "
                         "after this many seconds with no mtime bump. Default 7200 (2h).")
    ap.add_argument("--list-only", action="store_true", help="Discover shards and exit")

    args = ap.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must list at least one GPU id")
    ds_filter = [d.strip() for d in args.datasets.split(",")] if args.datasets else None

    api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
    if not args.skip_gemini and not api_key:
        raise ValueError("Set GOOGLE_API_KEY or pass --gemini-api-key (or use --skip-gemini)")
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
    print(f"GPUs: {gpu_ids} x {args.workers_per_gpu} workers/gpu "
          f"= {len(gpu_ids) * args.workers_per_gpu} workers total")

    if args.list_only or not shards:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    log_q = ctx.Queue()
    for s in shards:
        task_q.put(s)
    num_workers = len(gpu_ids) * args.workers_per_gpu
    for _ in range(num_workers):
        task_q.put(None)

    stop_evt = threading.Event()
    log_t = threading.Thread(target=_log_pump, args=(log_q, stop_evt), daemon=True)
    log_t.start()

    procs: list[mp.Process] = []
    wid = 0
    for gpu in gpu_ids:
        for _ in range(args.workers_per_gpu):
            p = ctx.Process(target=_worker_main, args=(wid, gpu, task_q, log_q, args))
            p.start()
            procs.append(p)
            wid += 1

    def _graceful(signum, frame):  # noqa: ARG001
        print("\n[parent] Ctrl-C: draining remaining shards; in-flight will finish.", flush=True)
        try:
            while True:
                task_q.get_nowait()
        except queue.Empty:
            pass
        for _ in range(num_workers):
            try:
                task_q.put_nowait(None)
            except Exception:
                pass

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    for p in procs:
        p.join()

    stop_evt.set()
    log_t.join(timeout=5)
    print(f"All done. Output -> {args.output_root}")


if __name__ == "__main__":
    main()
