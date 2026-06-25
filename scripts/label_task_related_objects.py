#!/usr/bin/env python3
"""Label which SAM3 objects are task-related to each episode's instruction.

For every OXE episode `metadata.json` this asks Qwen (text-only) to select,
*from the existing `objects` list only*, which object instances are relevant to
completing `language_instruction`, and assigns each a role. The result is written
to `task_related.json` next to the episode so the hypolicy-target Stage-1 loader
can build a task-focused union mask instead of unioning every detected object.

Pipeline position
-----------------
  build_oxe_robot_label_registry.py  -> robot vs non-robot (per *label*)
  THIS script                        -> task-related vs distractor (per *episode*)

The two are orthogonal: the robot registry suppresses the robot arm regardless of
task; this stage decides which of the *remaining* objects actually matter for the
instruction. Both can be consumed together by the loader.

Object indexing
---------------
`metadata.json["objects"]` is an ordered list of label strings. Its position is
the SAM3 instance index (the loader maps `objects[instance_idx]`). So
`object_index` in the output is exactly that mask instance index — usable directly
to select masks from the per-step `.npz`.

Output schema (task_related.json)
----------------------------------
  {
    "instruction": str,
    "task_related_instances": [{object_index, label, role, confidence}],  # union-eligible
    "excluded_instances":     [{object_index, label, role, confidence}],  # distractor/robot/...
    "missing_or_uncertain":   [{object_index, label, role, confidence}],  # uncertain or not returned
    "label_source": str,   # e.g. "qwen_text:qwen3-vl-flash"
    "label_version": str,  # prompt schema version
  }

Roles
-----
ALLOWED_ROLES below. Union-eligible (-> task_related_instances):
  manipulated_object, target_object, target_container,
  source_object, source_support_object
  (+ support_object only when --include-support-object is passed)
Excluded (-> excluded_instances): distractor, robot, support_object (by default)
Uncertain / not-returned (-> missing_or_uncertain): uncertain

Resumability & IO discipline
----------------------------
- One small file per episode on shared storage: written atomically (tmp + os.replace)
  and skipped if it already exists (the file *is* the checkpoint). Use --force to
  overwrite. Keep --workers modest; the call is network-bound on Qwen, not CPU.
- DASHSCOPE_API_KEY must be set in env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

import httpx

# --------------------------------------------------------------------- config

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen3-vl-flash"
LABEL_VERSION = "task-related-v1"
OUTPUT_NAME = "task_related.json"

ALLOWED_ROLES = (
    "manipulated_object",
    "target_object",
    "target_container",
    "source_object",
    "source_support_object",
    "support_object",
    "distractor",
    "robot",
    "uncertain",
)
# Roles that go into the training union mask.
UNION_ROLES_BASE = frozenset({
    "manipulated_object",
    "target_object",
    "target_container",
    "source_object",
    "source_support_object",
})
UNCERTAIN_ROLES = frozenset({"uncertain"})


SYSTEM_PROMPT = (
    "You are labeling task-related objects for a robot manipulation dataset.\n"
    "You are given a natural-language instruction and a fixed, indexed list of "
    "objects detected in the scene.\n"
    "Select which objects are related to completing the instruction.\n"
    "Rules:\n"
    "- Only select objects from the provided list. Do not invent new objects.\n"
    "- Refer to each object by its given integer index; echo its label verbatim.\n"
    "- If the instruction is empty or unrelated to any listed object, return an "
    "empty list.\n"
    "- Assign each selected object exactly one role from: "
    + ", ".join(ALLOWED_ROLES)
    + ".\n"
    "Role guidance: manipulated_object = the thing the gripper grasps/moves; "
    "target_object/target_container = where it goes; source_object/"
    "source_support_object = where it comes from; support_object = a passive "
    "surface (table/shelf) that is mentioned but not acted on; distractor = "
    "present but irrelevant to the instruction; robot = the robot/arm/gripper "
    "itself; uncertain = relevant but you cannot confidently assign a role.\n"
    'Return strict JSON only, no prose, of the form: '
    '{"task_related_instances": [{"object_index": <int>, "label": "<verbatim>", '
    '"role": "<role>", "confidence": <0..1>}]}'
)


# ----------------------------------------------------------------- enumerate

def iter_metadata_files(root: Path) -> Iterator[Path]:
    """Yield every metadata.json under root using os.scandir (faster than rglob)."""
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and entry.name == "metadata.json":
                        yield Path(entry.path)
        except (PermissionError, OSError):
            continue


def iter_from_index(index_path: Path) -> Iterator[Path]:
    """Yield metadata.json paths recorded in a prior scan's metadata_index.jsonl."""
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield Path(json.loads(line)["path"])
            except Exception:
                continue


def dataset_of(metadata_path: Path, root: Path) -> str:
    try:
        return metadata_path.relative_to(root).parts[0]
    except ValueError:
        return metadata_path.parts[-6] if len(metadata_path.parts) >= 6 else "_unknown"


# ----------------------------------------------------------------- metadata io

def read_instruction_and_objects(meta: dict) -> tuple[str, list[str]]:
    """Extract (instruction, objects-as-ordered-label-list) from one metadata dict.

    `objects` may be a list (position == instance index) or, in some export
    variants, a dict keyed by stringified index. Always returns a dense list whose
    position is the instance index; gaps in a dict layout become "".
    """
    instruction = meta.get("language_instruction") or meta.get("instruction") or ""
    instruction = str(instruction).strip()

    raw = meta.get("objects")
    if raw is None:
        raw = meta.get("instances")

    objects: list[str] = []
    if isinstance(raw, list):
        for obj in raw:
            if isinstance(obj, dict):
                objects.append(str(obj.get("label") or obj.get("name")
                                   or obj.get("category") or ""))
            else:
                objects.append(str(obj))
    elif isinstance(raw, dict):
        keyed: dict[int, str] = {}
        for k, v in raw.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict):
                keyed[idx] = str(v.get("label") or v.get("name")
                                 or v.get("category") or "")
            else:
                keyed[idx] = str(v)
        if keyed:
            objects = [keyed.get(i, "") for i in range(max(keyed) + 1)]
    return instruction, objects


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


# ----------------------------------------------------------------- qwen

def call_qwen(instruction: str, objects: list[str], model: str, api_key: str,
              timeout: float, max_retries: int, proxy: str | None = None,
              backoff: tuple[float, ...] = (2.0, 5.0, 10.0)) -> dict:
    """Ask Qwen to select task-related objects. Returns parsed JSON dict.

    `proxy` is an explicit HTTP(S) proxy URL. We never trust_env: the cluster sets
    ALL_PROXY=socks5://... which httpx can't use without socksio, so the caller
    passes the HTTP proxy explicitly (see --proxy / its env default).
    """
    indexed = "\n".join(f"{i}: {lbl}" for i, lbl in enumerate(objects))
    user_msg = (
        f"Instruction:\n{instruction}\n\n"
        f"Objects (index: label):\n{indexed}\n\n"
        "Select which objects are related to completing the instruction and "
        "assign each a role. Return strict JSON only."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout, trust_env=False, proxy=proxy) as client:
                resp = client.post(DASHSCOPE_URL, json=body, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["_usage"] = data.get("usage", {})
            return parsed
        except Exception as e:  # noqa: BLE001 - retry on any transport/parse error
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
    raise RuntimeError(f"Qwen call failed after {max_retries} attempts: {last_err}")


def partition_result(parsed: dict, objects: list[str],
                     union_roles: frozenset[str]) -> dict:
    """Map a Qwen response onto the three output buckets, validated against objects.

    - object_index must be a valid index into `objects`; out-of-range entries are
      dropped (object stays in missing_or_uncertain).
    - label is overwritten with objects[object_index] (the canonical, on-disk
      label) so it always matches the mask the loader will select.
    - role not in ALLOWED_ROLES is coerced to "uncertain".
    - Objects never returned by Qwen fall into missing_or_uncertain.
    """
    rows = parsed.get("task_related_instances")
    if not isinstance(rows, list):
        rows = []

    by_index: dict[int, dict] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("object_index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(objects):
            continue
        if idx in by_index:  # keep first mention, ignore duplicates
            continue
        role = r.get("role")
        if role not in ALLOWED_ROLES:
            role = "uncertain"
        try:
            conf = float(r.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(1.0, max(0.0, conf))
        by_index[idx] = {
            "object_index": idx,
            "label": objects[idx],  # canonical label from disk, not Qwen's echo
            "role": role,
            "confidence": round(conf, 3),
        }

    task_related, excluded, missing = [], [], []
    for idx in range(len(objects)):
        ent = by_index.get(idx)
        if ent is None:
            missing.append({
                "object_index": idx, "label": objects[idx],
                "role": "uncertain", "confidence": 0.0,
            })
            continue
        if ent["role"] in union_roles:
            task_related.append(ent)
        elif ent["role"] in UNCERTAIN_ROLES:
            missing.append(ent)
        else:
            excluded.append(ent)
    return {
        "task_related_instances": task_related,
        "excluded_instances": excluded,
        "missing_or_uncertain": missing,
    }


# ----------------------------------------------------------------- driver

def _wait_one(in_flight: set):
    """Block until >=1 future completes; return (done_set, still_pending_set)."""
    done, pending = wait(in_flight, return_when=FIRST_COMPLETED)
    return done, pending


def process_episode(meta_path: Path, *, model: str, api_key: str,
                    union_roles: frozenset[str], timeout: float,
                    max_retries: int, force: bool, proxy: str | None = None) -> str:
    """Process one episode. Returns a short status string for logging."""
    out_path = meta_path.with_name(OUTPUT_NAME)
    if out_path.exists() and not force:
        return "skip_exists"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return f"err_read:{e}"

    instruction, objects = read_instruction_and_objects(meta)
    if not instruction:
        return "skip_no_instruction"

    label_source = f"qwen_text:{model}"
    if not objects:
        atomic_write_json(out_path, {
            "instruction": instruction,
            "task_related_instances": [],
            "excluded_instances": [],
            "missing_or_uncertain": [],
            "label_source": label_source,
            "label_version": LABEL_VERSION,
        })
        return "ok_no_objects"

    try:
        parsed = call_qwen(instruction, objects, model, api_key, timeout,
                           max_retries, proxy=proxy)
    except Exception as e:  # noqa: BLE001
        return f"err_qwen:{e}"

    buckets = partition_result(parsed, objects, union_roles)
    payload = {
        "instruction": instruction,
        **buckets,
        "label_source": label_source,
        "label_version": LABEL_VERSION,
    }
    atomic_write_json(out_path, payload)
    return "ok"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--metadata-root", type=Path,
                     help="Walk this OXE seg root for metadata.json files.")
    src.add_argument("--index", type=Path,
                     help="Read episode paths from a prior metadata_index.jsonl.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent Qwen calls (network-bound; keep modest on shared FS).")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--proxy", default=None,
                   help="Explicit HTTP proxy for the Qwen API. Defaults to "
                        "HTTPS_PROXY/HTTP_PROXY env (the cluster's SOCKS ALL_PROXY "
                        "is intentionally ignored).")
    p.add_argument("--include-support-object", action="store_true",
                   help="Put support_object into the training union (default: excluded).")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Optional dataset-name filter (first path component).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N episodes (0 = no limit); for smoke tests.")
    p.add_argument("--force", action="store_true",
                   help="Re-label episodes that already have task_related.json.")
    p.add_argument("--dry-run", action="store_true",
                   help="Enumerate and report counts without calling Qwen or writing.")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("DASHSCOPE_API_KEY not set in environment")

    union_roles = UNION_ROLES_BASE
    if args.include_support_object:
        union_roles = union_roles | {"support_object"}

    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy and not args.dry_run:
        print(f"[task-related] using HTTP proxy {proxy}", flush=True)

    if args.metadata_root is not None:
        root = args.metadata_root
        paths = iter_metadata_files(root)
    else:
        root = None
        paths = iter_from_index(args.index)

    ds_filter = set(args.datasets) if args.datasets else None

    def keep(path: Path) -> bool:
        if ds_filter is None:
            return True
        base = root if root is not None else path.parents[5] if len(path.parents) >= 6 else path.parent
        return dataset_of(path, base) in ds_filter

    counts: dict[str, int] = {}
    lock = threading.Lock()
    t0 = time.time()
    submitted = 0

    def record(status: str) -> None:
        key = status.split(":", 1)[0]
        with lock:
            counts[key] = counts.get(key, 0) + 1
            done = sum(counts.values())
            if done % 200 == 0 or key.startswith("err"):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-3)
                print(f"[task-related] done={done} rate={rate:.1f}/s "
                      f"counts={counts} elapsed={elapsed:.0f}s", flush=True)

    if args.dry_run:
        for path in paths:
            if not keep(path):
                continue
            submitted += 1
            if args.limit and submitted >= args.limit:
                break
        print(f"[task-related] dry-run: {submitted} episodes would be processed "
              f"(union_roles={sorted(union_roles)})", flush=True)
        return 0

    def submit_iter() -> Iterator[Path]:
        nonlocal submitted
        for path in paths:
            if not keep(path):
                continue
            submitted += 1
            yield path
            if args.limit and submitted >= args.limit:
                break

    # Bounded-inflight streaming: keep at most `max_in_flight` futures pending so
    # enumeration of a 1.38M-episode tree never materializes all futures at once.
    src_iter = submit_iter()
    max_in_flight = max(1, args.workers * 4)

    def _submit_next(ex) -> Any:
        try:
            path = next(src_iter)
        except StopIteration:
            return None
        return ex.submit(
            process_episode, path, model=args.model, api_key=api_key,
            union_roles=union_roles, timeout=args.timeout,
            max_retries=args.max_retries, force=args.force, proxy=proxy,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        in_flight: set = set()
        for _ in range(max_in_flight):
            fut = _submit_next(ex)
            if fut is None:
                break
            in_flight.add(fut)
        while in_flight:
            done_set, in_flight = _wait_one(in_flight)
            for f in done_set:
                record(f.result())
                nxt = _submit_next(ex)
                if nxt is not None:
                    in_flight.add(nxt)

    elapsed = time.time() - t0
    print(f"[task-related] FINISHED submitted={submitted} counts={counts} "
          f"elapsed={elapsed:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
