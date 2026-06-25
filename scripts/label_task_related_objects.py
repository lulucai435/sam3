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
    "task_related_instances": [{object_index, label, role, confidence, reason}],
    "missing_or_uncertain": [{object_index, label, reason}],
    "valid_for_action_seg": bool,   # true iff >=1 task-related object selected
    "label_source": "qwen_text_only",
    "label_version": "v1",
  }

Roles (all union-eligible; only these survive in task_related_instances)
-----
  manipulated_object, target_object, target_container,
  source_object, source_support_object

Robot parts are handled separately by robot_label_registry.yaml and are NOT
selected here. Distractors / background / uncertain objects are simply omitted by
the model (or listed under missing_or_uncertain). The objects list is shown to the
model in full and NEVER reindexed, so object_index always equals the SAM3 mask
instance index.

confidence is audit/debug only for v1 — the loader/training must NOT filter on it.

Large support surfaces (table/counter/floor/wall/background) tagged as
source_support_object are demoted to missing_or_uncertain by default
(--exclude-large-support-objects, on by default) so the union mask =
task_related_instances exactly. Use --no-exclude-large-support-objects to keep them.

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
import re
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
LABEL_VERSION = "v1"
LABEL_SOURCE = "qwen_text_only"
OUTPUT_NAME = "task_related.json"

# Only the union-eligible roles exist now. Robot parts are handled separately by
# robot_label_registry.yaml; distractors / background / uncertain are simply not
# selected (the model omits them, or lists them under missing_or_uncertain).
ALLOWED_ROLES = (
    "manipulated_object",
    "target_object",
    "target_container",
    "source_object",
    "source_support_object",
)

# Large, low-information support surfaces. Even when Qwen tags one as
# source_support_object, it should not enter the training union mask by default
# (it dominates the frame and adds little task signal). Matched as a substring of
# the lowercased label, so "black table" / "wall panel" are caught. Applied only
# to the source_support_object role — a table that is a genuine target_object
# (e.g. "put X on the table") is kept.
DEFAULT_LARGE_SUPPORT_LABELS = ("table", "counter", "floor", "wall", "background")
LARGE_SUPPORT_ROLES = frozenset({"source_support_object"})

# Robot parts are handled separately by the loader via robot_label_registry.yaml.
# Qwen still occasionally selects "robotic arm" as manipulated_object (~2.6% of
# episodes in audit), which would double-union the robot mask. We cross-check every
# selected label against the same registry and demote any robot match.
DEFAULT_ROBOT_REGISTRY = "reports/oxe_robot_labels/robot_label_registry.yaml"
ROBOT_CATEGORIES = frozenset({"robotic_arm", "gripper", "end_effector"})


def normalize_robot_label(label: object) -> str:
    """Match hypolicy-target RobotLabelRegistry.normalize so lookups agree."""
    text = str(label).casefold().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    return " ".join(text.split())


def load_robot_label_set(path: Path) -> frozenset[str]:
    """Load the set of normalized robot labels from robot_label_registry.yaml."""
    import yaml  # local import: only needed when robot filtering is enabled
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: set[str] = set()

    def _add(label: object) -> None:
        n = normalize_robot_label(label)
        if len(n) > 1:
            out.add(n)

    for cat, labels in (data.get("categories") or {}).items():
        if cat in ROBOT_CATEGORIES:
            for lbl in labels or []:
                _add(lbl)
    for label, meta in (data.get("labels") or {}).items():
        if not isinstance(meta, dict):
            continue
        if meta.get("category") in ROBOT_CATEGORIES and meta.get("is_robot_related", True):
            _add(label)
            for variant in meta.get("raw_variants") or []:
                _add(variant)
    return frozenset(out)

# Prompt is a single self-contained user message. Two placeholders are filled by
# str.replace (NOT str.format — the body contains literal JSON braces).
PROMPT_TEMPLATE = """You are labeling task-related objects for robot manipulation.

Given an action instruction and an indexed object list, select only the objects that are directly needed to complete the instruction.

The object index is the position shown in the indexed object list. Only choose from the provided indices. Do not invent objects.

Instruction:
__INSTRUCTION__

Objects:
__INDEXED_OBJECTS__

Return strict JSON only:

{
  "instruction": "...",
  "task_related_instances": [
    {
      "object_index": 0,
      "label": "...",
      "role": "...",
      "confidence": 0.95,
      "reason": "..."
    }
  ],
  "missing_or_uncertain": [],
  "valid_for_action_seg": true,
  "label_source": "qwen_text_only",
  "label_version": "v1"
}

Allowed roles for task_related_instances:

- manipulated_object: object directly moved, picked, placed, pushed, inserted, opened, closed, turned on/off, etc.
- target_object: target object/surface/location for the action. For "put X on/under/beside Y", Y is target_object, even if Y is a shelf, cabinet, counter, tray, plate, or table.
- target_container: container/appliance/holder/drawer/basket/box that receives or is used with the manipulated object.
- source_object: object/location the manipulated object is removed from.
- source_support_object: object supporting the manipulated object at the start, e.g. "bowl on the ramekin" -> ramekin.

Do not select robot parts, distractors, uncertain objects, or generic background/support objects.

Important:
- A cabinet shelf in "put the pan under/on the cabinet shelf" is target_object, not background support.
- A table/counter/shelf is selected only if explicitly used as source or target.
- Do not select table/counter/floor/wall/background as source_support_object unless explicitly mentioned in the instruction and necessary to identify the task. If the instruction refers to a target serving area/tray/plate/container, select that target object instead of the table.
- If the instruction is empty or no action-related object can be selected, return an empty task_related_instances list and set valid_for_action_seg=false.
- The label must exactly match the object label from the provided indexed list.

Confidence:
- Set confidence to your estimated confidence between 0 and 1. Do not copy the placeholder value. Use higher confidence for exact matches and lower confidence for approximate or ambiguous matches."""


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
    prompt = (PROMPT_TEMPLATE
              .replace("__INSTRUCTION__", instruction)
              .replace("__INDEXED_OBJECTS__", indexed))
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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


def is_large_support_label(label: str, large_support_labels: frozenset[str]) -> bool:
    """True if `label` names a large support surface (substring match, lowercased)."""
    low = label.lower()
    return any(kw in low for kw in large_support_labels)


def partition_result(parsed: dict, objects: list[str],
                     exclude_large_support: bool = True,
                     large_support_labels: frozenset[str] = frozenset(
                         DEFAULT_LARGE_SUPPORT_LABELS),
                     robot_labels: frozenset[str] | None = None) -> dict:
    """Validate a Qwen response against the on-disk objects list.

    - object_index must be a valid index into `objects`; out-of-range or
      bad-role entries are demoted into missing_or_uncertain instead of silently
      dropped, so the object is still accounted for.
    - label is overwritten with objects[object_index] (the canonical, on-disk
      label) so it always matches the mask the loader will select by index.
    - Only the 5 allowed roles survive in task_related_instances.
    - When robot_labels is given, a selected label that resolves to a robot
      category in robot_label_registry.yaml is demoted to missing_or_uncertain:
      robot masks are unioned separately by the loader, so they must not appear
      here (Qwen sometimes mislabels "robotic arm" as manipulated_object).
    - When exclude_large_support is set, a source_support_object whose label names
      a large support surface (table/counter/floor/wall/background) is demoted to
      missing_or_uncertain so it never enters the union mask. The demotion is
      recorded in the entry's reason for audit; the union = task_related_instances
      exactly, so the loader needs no extra config.
    - A selected instance with confidence exactly 0.0 is demoted: Qwen uses the 0.0
      sentinel to signal self-rejection ("likely a distractor") while still forced
      to assign a role. This is NOT threshold filtering — only the exact 0.0 value
      is removed, never conf < some-threshold.
    - valid_for_action_seg is derived from the *validated* list (not trusted from
      the model) so it can never disagree with task_related_instances.
    """
    rows = parsed.get("task_related_instances")
    if not isinstance(rows, list):
        rows = []

    task_related: list[dict] = []
    missing: list[dict] = []
    seen: set[int] = set()

    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("object_index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(objects) or idx in seen:
            continue
        seen.add(idx)
        try:
            conf = float(r.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(1.0, max(0.0, conf))
        reason = str(r.get("reason", ""))
        role = r.get("role")
        if role not in ALLOWED_ROLES:
            missing.append({
                "object_index": idx, "label": objects[idx],
                "reason": f"unrecognized role {role!r}; {reason}".strip("; "),
            })
            continue
        if robot_labels and normalize_robot_label(objects[idx]) in robot_labels:
            missing.append({
                "object_index": idx, "label": objects[idx],
                "reason": (f"demoted from {role}: robot label handled by "
                           f"robot_label_registry, excluded from union; {reason}"
                           ).strip("; "),
            })
            continue
        if (exclude_large_support and role in LARGE_SUPPORT_ROLES
                and is_large_support_label(objects[idx], large_support_labels)):
            missing.append({
                "object_index": idx, "label": objects[idx],
                "reason": (f"demoted from {role}: large support surface excluded "
                           f"from union; {reason}").strip("; "),
            })
            continue
        # Exact 0.0 sentinel only (NOT a confidence threshold): Qwen uses 0.0 to
        # signal self-rejection / "this is likely a distractor" while still being
        # forced to assign a role. Such entries must not enter the union.
        if conf == 0.0:
            missing.append({
                "object_index": idx, "label": objects[idx],
                "reason": ("Removed because Qwen assigned confidence 0.0, which "
                           "indicates self-rejection / likely distractor."),
            })
            continue
        task_related.append({
            "object_index": idx,
            "label": objects[idx],  # canonical label from disk, not Qwen's echo
            "role": role,
            "confidence": round(conf, 3),
            "reason": reason,
        })

    # Pass through the model's own missing_or_uncertain, sanitized to valid,
    # not-already-selected indices.
    for r in parsed.get("missing_or_uncertain") or []:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("object_index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(objects) or idx in seen:
            continue
        seen.add(idx)
        missing.append({
            "object_index": idx,
            "label": objects[idx],
            "reason": str(r.get("reason", "")),
        })

    return {
        "task_related_instances": task_related,
        "missing_or_uncertain": missing,
        "valid_for_action_seg": len(task_related) > 0,
    }


# ----------------------------------------------------------------- driver

def _wait_one(in_flight: set):
    """Block until >=1 future completes; return (done_set, still_pending_set)."""
    done, pending = wait(in_flight, return_when=FIRST_COMPLETED)
    return done, pending


def process_episode(meta_path: Path, *, model: str, api_key: str,
                    timeout: float, max_retries: int, force: bool,
                    proxy: str | None = None,
                    exclude_large_support: bool = True,
                    large_support_labels: frozenset[str] = frozenset(
                        DEFAULT_LARGE_SUPPORT_LABELS),
                    robot_labels: frozenset[str] | None = None) -> str:
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

    if not objects:
        atomic_write_json(out_path, {
            "instruction": instruction,
            "task_related_instances": [],
            "missing_or_uncertain": [],
            "valid_for_action_seg": False,
            "label_source": LABEL_SOURCE,
            "label_version": LABEL_VERSION,
        })
        return "ok_no_objects"

    try:
        parsed = call_qwen(instruction, objects, model, api_key, timeout,
                           max_retries, proxy=proxy)
    except Exception as e:  # noqa: BLE001
        return f"err_qwen:{e}"

    buckets = partition_result(parsed, objects,
                               exclude_large_support=exclude_large_support,
                               large_support_labels=large_support_labels,
                               robot_labels=robot_labels)
    payload = {
        "instruction": instruction,
        **buckets,
        "label_source": LABEL_SOURCE,
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
    p.add_argument("--exclude-large-support-objects", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Demote source_support_object instances whose label names a "
                        "large support surface (see --large-support-labels) into "
                        "missing_or_uncertain so they never enter the union mask. "
                        "On by default; use --no-exclude-large-support-objects to keep them.")
    p.add_argument("--large-support-labels", nargs="*", default=list(DEFAULT_LARGE_SUPPORT_LABELS),
                   help="Substring keywords (lowercased) treated as large support surfaces.")
    p.add_argument("--robot-label-registry", default=DEFAULT_ROBOT_REGISTRY,
                   help="robot_label_registry.yaml used to drop robot labels that Qwen "
                        "wrongly selects (handled separately by the loader). Set to '' to "
                        "disable. Missing file -> robot filtering off with a warning.")
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

    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy and not args.dry_run:
        print(f"[task-related] using HTTP proxy {proxy}", flush=True)

    large_support = frozenset(s.lower() for s in args.large_support_labels)
    if args.exclude_large_support_objects and not args.dry_run:
        print(f"[task-related] excluding large support surfaces from union: "
              f"{sorted(large_support)}", flush=True)

    robot_labels: frozenset[str] | None = None
    if args.robot_label_registry:
        reg_path = Path(args.robot_label_registry)
        if reg_path.is_file():
            robot_labels = load_robot_label_set(reg_path)
            if not args.dry_run:
                print(f"[task-related] robot-label filter ON: {len(robot_labels)} "
                      f"normalized robot labels from {reg_path}", flush=True)
        elif not args.dry_run:
            print(f"[task-related] WARNING: robot registry {reg_path} not found; "
                  f"robot-label filtering OFF", flush=True)

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
              f"(roles={list(ALLOWED_ROLES)})", flush=True)
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
            timeout=args.timeout, max_retries=args.max_retries,
            force=args.force, proxy=proxy,
            exclude_large_support=args.exclude_large_support_objects,
            large_support_labels=large_support,
            robot_labels=robot_labels,
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
