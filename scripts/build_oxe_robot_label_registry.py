#!/usr/bin/env python3
"""Build a robot-related label registry from OXE SAM3 metadata.

Output consumed by /public/home/lulucai/code/hypolicy-target Stage 1 loader to
union robot masks with non-robot object masks.

Pipeline (see /public/home/lulucai/.claude/plans/curious-launching-matsumoto.md):
  scan       Stage 0: walk all metadata.json, build per-file index + signature
  classify   Stage 1: Qwen-text classify each unique signature (resumable)
  bucket     Stage 2: partition metadata by # of robot-flagged labels
  verify-vl  Stage 3: Qwen-VL vision check for bucket_multi + non-canonical single
  finalize   Stage 4: emit vocabulary / registry / uncertain CSV / summary
  all        run 0..4 in order

Notes:
- Single-process scan (shared-storage IO discipline; see CLAUDE.md).
- All resumable via per-stage checkpoints under --output-dir.
- DASHSCOPE_API_KEY must be set in env for stages 1/3.
"""
from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx
import yaml
from PIL import Image

# ----------------------------------------------------------------- constants

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen3-vl-flash"

ROBOT_CATEGORIES = ("robotic_arm", "gripper", "end_effector")
NON_ROBOT_CATEGORIES = ("non_robot", "uncertain")
ALL_CATEGORIES = ROBOT_CATEGORIES + NON_ROBOT_CATEGORIES

CANONICAL_ARM_LABEL = "robotic arm"  # post-normalization

PROMPT_VERSION = "robot-label-v1"


# ----------------------------------------------------------------- normalize

_PUNCT_RE = re.compile(r"[^\w\s]|_")
_WS_RE = re.compile(r"\s+")


def normalize_label(s: Any) -> str:
    """NFKC + lower + strip + punct→space + collapse whitespace. Empty if not str."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def make_signature(dataset: str, instruction: str, objects_norm: list[str],
                   mode: str = "strict") -> str:
    """Stable hash of the Qwen-text input. Same signature ⇒ same Qwen output.

    mode="strict": include language_instruction (max safety, low dedup).
    mode="loose":  ignore instruction (high dedup, sacrifices Qwen task context).
    """
    if mode == "loose":
        key = [dataset, sorted(set(objects_norm))]
    else:
        key = [dataset, instruction or "", sorted(set(objects_norm))]
    payload = json.dumps(key, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------- iter walk


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


def dataset_of(metadata_path: Path, root: Path) -> str:
    """First path component under metadata_root is the dataset name."""
    try:
        return metadata_path.relative_to(root).parts[0]
    except ValueError:
        return "_unknown"


# ----------------------------------------------------------------- Stage 0


@dataclass
class IndexRow:
    path: str
    dataset: str
    model: str
    instruction: str
    objects_raw: list[str]
    objects_norm: list[str]
    signature: str
    primary_camera: str
    gemini_frame_indices: list[int]

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def stage_scan(metadata_root: Path, out_dir: Path, checkpoint_every: int = 50_000,
               log_every: int = 5_000) -> dict:
    """Walk all metadata.json. Append to index + resume from _scan_processed.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "metadata_index.jsonl"
    processed_path = out_dir / "_scan_processed.txt"
    errors_path = out_dir / "scan_errors.log"
    summary_path = out_dir / "scan_summary.json"

    processed: set[str] = set()
    if processed_path.exists():
        processed = set(p for p in processed_path.read_text().splitlines() if p)
        print(f"[scan] resuming, skipping {len(processed)} already-indexed files", flush=True)

    stats = Counter()
    stats["resumed_skipped"] = len(processed)
    label_set: set[str] = set()
    sig_set: set[str] = set()
    datasets_seen: set[str] = set()
    t0 = time.time()

    err_f = open(errors_path, "a", encoding="utf-8")
    with open(index_path, "a", encoding="utf-8") as out_f, \
         open(processed_path, "a", encoding="utf-8") as proc_f:
        for fp in iter_metadata_files(metadata_root):
            fp_str = str(fp)
            stats["seen"] += 1

            if fp_str in processed:
                continue

            try:
                m = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as e:
                stats["read_error"] += 1
                err_f.write(f"READ {fp_str} :: {e}\n")
                err_f.flush()
                # still mark processed so we don't retry forever
                proc_f.write(fp_str + "\n")
                continue

            objs_raw = m.get("objects") or []
            if not isinstance(objs_raw, list):
                stats["bad_objects_field"] += 1
                err_f.write(f"BADOBJ {fp_str} :: objects not a list\n")
                proc_f.write(fp_str + "\n")
                continue

            objs_raw_clean: list[str] = []
            objs_norm: list[str] = []
            for o in objs_raw:
                if not isinstance(o, str):
                    continue
                n = normalize_label(o)
                if not n:
                    continue
                objs_raw_clean.append(o)
                objs_norm.append(n)

            if not objs_norm:
                stats["empty_objects"] += 1
                # still index it (downstream wants to know about bucket_zero)

            ds = dataset_of(fp, metadata_root)
            instruction = m.get("language_instruction") or ""
            sig = make_signature(ds, instruction, objs_norm)

            row = IndexRow(
                path=fp_str,
                dataset=ds,
                model=str(m.get("gemini_model") or ""),
                instruction=instruction,
                objects_raw=objs_raw_clean,
                objects_norm=objs_norm,
                signature=sig,
                primary_camera=str(m.get("primary_camera") or ""),
                gemini_frame_indices=list(m.get("gemini_frame_indices") or []),
            )

            out_f.write(row.to_json_line() + "\n")
            proc_f.write(fp_str + "\n")

            label_set.update(objs_norm)
            sig_set.add(sig)
            datasets_seen.add(ds)
            stats["indexed"] += 1

            if stats["indexed"] % checkpoint_every == 0:
                out_f.flush()
                proc_f.flush()
            if stats["seen"] % log_every == 0:
                elapsed = time.time() - t0
                rate = stats["indexed"] / max(elapsed, 1e-3)
                print(
                    f"[scan] seen={stats['seen']} indexed={stats['indexed']} "
                    f"err={stats['read_error']} sigs={len(sig_set)} labels={len(label_set)} "
                    f"rate={rate:.0f}/s elapsed={elapsed:.0f}s",
                    flush=True,
                )

    err_f.close()
    summary = {
        "metadata_files_seen": stats["seen"],
        "metadata_files_indexed": stats["indexed"],
        "metadata_files_resumed_skipped": stats["resumed_skipped"],
        "read_errors": stats["read_error"],
        "bad_objects_field": stats["bad_objects_field"],
        "empty_objects": stats["empty_objects"],
        "unique_signatures": len(sig_set),
        "unique_normalized_labels": len(label_set),
        "datasets_seen": sorted(datasets_seen),
        "wall_seconds": round(time.time() - t0, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[scan] done. summary -> {summary_path}", flush=True)
    return summary


# ----------------------------------------------------------------- Stage 1


_SYSTEM_PROMPT_LABEL_ONLY = """You classify single English object-label strings extracted from robot-manipulation episodes.

INPUT
A JSON array of {"id": <int>, "label": "<normalized label string>"}.
Each label is ONE word/phrase only — there is NO surrounding context, NO scene image, NO instruction.

TASK
For every input id return exactly one category. Choose from:
- robotic_arm: the operating robot arm/manipulator OR its main body / structural
  parts (robot base, joint, link, torso). Multi-robot platform body counts here
  (Spot quadruped body, SARA platforms, mobile manipulator bases).
- gripper: gripper, robotic claw, parallel/suction gripper, jaws of a robot.
- end_effector: robot hand, robot wrist, tool flange (the terminal mechanism
  that holds tools).
- non_robot: any non-robot scene object — furniture, food, tools, people,
  toys/figurines, stickers, posters, decorations.
- uncertain: cannot decide from the string alone. Examples that MUST go here:
    "yellow robot", "blue robot", "silver robot", "red robot"
       (color + "robot" without platform context — could be Spot, could be a toy)
    "manipulator" alone (could be human or robot)
    "arm" alone (anatomical or robot?)
    "robot" alone

HARD EXCLUSIONS — these are non_robot, NEVER robot:
- arm chair, armrest, recliner, sofa arm
- wrist watch, wristband, wristlet
- claw machine, claw hammer
- hand soap, hand towel, hand sanitizer
- human hand, human arm, person, hand, arm  (anatomical, no robot modifier)
- toy robot, robot sticker, robot poster, robot figurine, robot toy, robot magnet
- robot dog/cat/animal  (decorative unless clearly an operating platform)

RULES
1. Do NOT call a label robot-related just because it contains the substring
   "robot" / "arm" / "hand" / "claw" / "wrist". Read the WHOLE label.
2. Common color/material modifiers on the robot itself ("black robotic arm",
   "white robotic arm", "silver robotic arm", "metal robotic arm") → robotic_arm.
3. "robotic arm" → robotic_arm. "robot arm" → robotic_arm. "robot manipulator" → robotic_arm.
4. When in doubt, prefer "uncertain" over guessing.

OUTPUT — STRICT JSON ONLY, no markdown:
{
  "results": [
    {"id": <int>, "category": "<one of 5>", "confidence": <float 0..1>, "reason": "<<=12 words>"},
    ...
  ]
}
One result entry per input id. ALL ids must be present.
"""


_SYSTEM_PROMPT_TEXT = """You classify object labels that came from robot-manipulation video episodes.

CONTEXT
- Each input represents one episode's objects list (auto-detected by a vision model).
- The detection prompt enforced: (a) name the operating robot exactly "robotic arm"
  as a SINGLE entry, no separate gripper/wrist/base entries; (b) one entry per
  unique object instance class — no duplicates.
- Therefore: if two labels both look "robot-related" inside one objects list, they
  MUST refer to different physical objects in that scene.

TASK
Return ONLY the labels that are robot-related. Most metadata will yield just
[{"label": "robotic arm", ...}]. Skip non-robot labels entirely — do not echo
tables, fixtures, food, tools, monitors, etc.

CATEGORIES (use exactly one)
- robotic_arm: the operating robot arm/manipulator and its main body parts
  (base, joint, link, torso). Includes color/material variants like
  "black robotic arm". For multi-robot platforms (e.g., Boston Dynamics Spot
  quadruped with both a body and an arm, or SARA platforms), the body/base
  is also robotic_arm.
- gripper: gripper, robotic claw, parallel/suction gripper.
- end_effector: robot hand / robot wrist / tool flange (terminal mechanism).
- uncertain: cannot decide robot vs scene from the label + co-occurring labels.

HARD RULES
1. Do NOT call a label robot-related just because it contains
   "robot"/"arm"/"hand"/"claw"/"wrist". Counterexamples that are NON-ROBOT and
   must be EXCLUDED from output: arm chair, armrest, wrist watch, claw machine,
   hand soap, human hand, human arm, toy robot, robot sticker, robot poster,
   robot figurine, robot toy.
2. If a metadata's objects include BOTH "robotic arm" AND another robot-ish
   label (e.g., "yellow robot"), they CANNOT be the same object. The second is
   either (a) another robot part / multi-robot platform body → keep with the
   appropriate robot category, or (b) a scene object/toy → exclude from output.
   If unsure → include with category "uncertain".
3. Color-modified robot labels with no platform context ("yellow robot",
   "blue robot", "red robot") → category "uncertain" unless dataset/instruction
   clearly disambiguates.
4. Use the language_instruction and co-occurring labels as context.

OUTPUT — STRICT JSON ONLY, no markdown, no commentary:
{
  "results": [
    {"id": <int matching input>,
     "robot_labels": [
        {"label": "<exact raw label string from input>",
         "category": "<robotic_arm|gripper|end_effector|uncertain>",
         "confidence": <float 0..1>,
         "reason": "<short>"
        }, ...
     ]
    }, ...
  ]
}
- One result entry per input id (even if robot_labels is empty).
- "label" must be an EXACT string from that input's objects list.
"""


def _http_post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


def call_qwen_text(items: list[dict], model: str, api_key: str,
                   timeout: float = 120.0,
                   max_retries: int = 3,
                   backoff: tuple[float, ...] = (2.0, 5.0, 10.0)) -> dict:
    """Call Qwen text model. items = [{id, dataset, instruction, objects(raw strings)}].

    Returns the parsed Qwen response dict {results: [...]}. Raises on final failure.
    """
    user_msg = "Classify these metadata:\n" + json.dumps(items, ensure_ascii=False)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT_TEXT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _http_post_json(DASHSCOPE_URL, headers, body, timeout)
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            # Embed usage for caller stats
            parsed["_usage"] = resp.get("usage", {})
            return parsed
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
    raise RuntimeError(f"Qwen text call failed after {max_retries} attempts: {last_err}")


def validate_qwen_batch(parsed: dict, items: list[dict]) -> list[dict]:
    """Validate Qwen output. Returns per-item robot_labels lists.

    Raises ValueError if response is malformed (missing ids, bad categories, etc.).
    """
    if not isinstance(parsed, dict) or "results" not in parsed:
        raise ValueError("response missing 'results' key")
    results = parsed["results"]
    if not isinstance(results, list):
        raise ValueError("'results' is not a list")
    id_to_input = {it["id"]: it for it in items}
    out: dict[int, list[dict]] = {}
    seen_ids: set[int] = set()
    for r in results:
        if not isinstance(r, dict) or "id" not in r:
            raise ValueError(f"result row missing id: {r}")
        rid = r["id"]
        if rid in seen_ids:
            raise ValueError(f"duplicate id {rid}")
        seen_ids.add(rid)
        if rid not in id_to_input:
            raise ValueError(f"unexpected id {rid}")
        labels = r.get("robot_labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"id {rid}: robot_labels not a list")
        allowed_labels = set(id_to_input[rid]["objects"])
        normalized_out: list[dict] = []
        for ent in labels:
            if not isinstance(ent, dict):
                raise ValueError(f"id {rid}: non-dict robot_label entry")
            lbl = ent.get("label")
            cat = ent.get("category")
            if lbl not in allowed_labels:
                raise ValueError(
                    f"id {rid}: label {lbl!r} not in input objects"
                )
            if cat not in ALL_CATEGORIES:
                raise ValueError(f"id {rid}: bad category {cat!r}")
            normalized_out.append({
                "label": lbl,
                "category": cat,
                "confidence": float(ent.get("confidence", 0.0)),
                "reason": str(ent.get("reason", "")),
            })
        out[rid] = normalized_out
    missing = [it["id"] for it in items if it["id"] not in out]
    if missing:
        raise ValueError(f"missing ids {missing}")
    return [out[it["id"]] for it in items]


def load_sig_input_map(index_path: Path,
                       signature_mode: str = "strict"
                       ) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Build sig→canonical input + sig→[paths] from the metadata index JSONL.

    If signature_mode != the mode used during scan, recompute on the fly.
    """
    sig_input: dict[str, dict] = {}
    sig_paths: dict[str, list[str]] = defaultdict(list)
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sig = make_signature(row["dataset"], row["instruction"],
                                 row["objects_norm"], mode=signature_mode)
            if sig not in sig_input:
                raw_for_norm: dict[str, str] = {}
                for raw, norm in zip(row["objects_raw"], row["objects_norm"]):
                    raw_for_norm.setdefault(norm, raw)
                ordered_raw = [raw_for_norm[n] for n in sorted(set(row["objects_norm"]))]
                # For loose mode the instruction is not part of signature; pick one
                # representative instruction (first seen) for context in the prompt.
                sig_input[sig] = {
                    "dataset": row["dataset"],
                    "instruction": row["instruction"],
                    "objects": ordered_raw,
                    "objects_norm": sorted(set(row["objects_norm"])),
                }
            sig_paths[sig].append(row["path"])
    return sig_input, dict(sig_paths)


def stage_classify(out_dir: Path, model: str, batch_size: int,
                   timeout: float, max_retries: int,
                   force: bool = False,
                   signature_mode: str = "strict") -> dict:
    """Stage 1: per-signature Qwen text classification with resume."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY not set in environment")

    index_path = out_dir / "metadata_index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"metadata index missing: {index_path}; run `scan` first")

    sig_input, sig_paths = load_sig_input_map(index_path, signature_mode=signature_mode)
    print(f"[classify] {len(sig_input)} unique signatures from index "
          f"(signature_mode={signature_mode})", flush=True)

    ckp_path = out_dir / "qwen_classification_checkpoint.jsonl"
    failed_path = out_dir / "failed_signatures.jsonl"
    done_sigs: set[str] = set()
    if not force and ckp_path.exists():
        with open(ckp_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_sigs.add(json.loads(line)["signature"])
                except Exception:
                    continue
        print(f"[classify] resuming, {len(done_sigs)} sigs already done", flush=True)
    elif force and ckp_path.exists():
        ckp_path.unlink()

    todo = [s for s in sig_input if s not in done_sigs]
    print(f"[classify] {len(todo)} signatures to classify", flush=True)

    total_usage = Counter()
    t0 = time.time()
    n_failed = 0

    with open(ckp_path, "a", encoding="utf-8") as ckp_f, \
         open(failed_path, "a", encoding="utf-8") as fail_f:
        for batch_start in range(0, len(todo), batch_size):
            batch_sigs = todo[batch_start:batch_start + batch_size]
            items = [
                {"id": i,
                 "dataset": sig_input[s]["dataset"],
                 "instruction": sig_input[s]["instruction"],
                 "objects": sig_input[s]["objects"]}
                for i, s in enumerate(batch_sigs)
            ]
            try:
                parsed = call_qwen_text(items, model, api_key, timeout, max_retries)
                per_item = validate_qwen_batch(parsed, items)
                for sig, robot_labels in zip(batch_sigs, per_item):
                    ckp_f.write(json.dumps({
                        "signature": sig,
                        "robot_labels": robot_labels,
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                        "ts": time.time(),
                    }, ensure_ascii=False) + "\n")
                usage = parsed.get("_usage") or {}
                for k, v in usage.items():
                    if isinstance(v, int):
                        total_usage[k] += v
                if (batch_start // batch_size) % 5 == 0:
                    ckp_f.flush()
            except Exception as e:
                n_failed += len(batch_sigs)
                fail_f.write(json.dumps({
                    "signatures": batch_sigs,
                    "error": str(e),
                    "ts": time.time(),
                }, ensure_ascii=False) + "\n")
                fail_f.flush()

            elapsed = time.time() - t0
            done = batch_start + len(batch_sigs)
            print(
                f"[classify] {done}/{len(todo)} sigs ({100*done/max(len(todo),1):.1f}%) "
                f"failed={n_failed} usage={dict(total_usage)} elapsed={elapsed:.0f}s",
                flush=True,
            )

    summary = {
        "total_signatures": len(sig_input),
        "already_done_on_start": len(done_sigs),
        "classified_this_run": len(todo) - n_failed,
        "failed_this_run": n_failed,
        "usage": dict(total_usage),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "classify_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[classify] done. summary -> {out_dir / 'classify_summary.json'}", flush=True)
    return summary


# ----------------------------------------------------------------- Stage 1b (per-label)


def call_qwen_label(items: list[dict], model: str, api_key: str,
                    timeout: float = 120.0,
                    max_retries: int = 3,
                    backoff: tuple[float, ...] = (2.0, 5.0, 10.0)) -> dict:
    """Per-label Qwen text call. items = [{id, label}]. Returns parsed dict."""
    user_msg = "Classify each label string:\n" + json.dumps(items, ensure_ascii=False)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT_LABEL_ONLY},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _http_post_json(DASHSCOPE_URL, headers, body, timeout)
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["_usage"] = resp.get("usage", {})
            return parsed
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
    raise RuntimeError(f"Qwen label call failed: {last_err}")


def validate_qwen_label_batch(parsed: dict, items: list[dict]) -> list[dict]:
    """Validate per-label Qwen output. Returns list aligned with items order."""
    if not isinstance(parsed, dict) or "results" not in parsed:
        raise ValueError("response missing 'results'")
    results = parsed["results"]
    if not isinstance(results, list):
        raise ValueError("'results' not a list")
    id_to_item = {it["id"]: it for it in items}
    by_id: dict[int, dict] = {}
    for r in results:
        if not isinstance(r, dict) or "id" not in r:
            raise ValueError(f"bad row: {r}")
        rid = r["id"]
        if rid not in id_to_item:
            raise ValueError(f"unexpected id {rid}")
        cat = r.get("category")
        if cat not in ALL_CATEGORIES:
            raise ValueError(f"id {rid}: bad category {cat!r}")
        # Last-wins on duplicate id (Qwen occasionally repeats at larger batches).
        by_id[rid] = {
            "label": id_to_item[rid]["label"],
            "category": cat,
            "confidence": float(r.get("confidence", 0.0)),
            "reason": str(r.get("reason", ""))[:200],
        }
    missing = [it["id"] for it in items if it["id"] not in by_id]
    if missing:
        raise ValueError(f"missing ids {missing}")
    return [by_id[it["id"]] for it in items]


def load_unique_normalized_labels(index_path: Path) -> list[str]:
    """Scan metadata_index.jsonl, return sorted unique normalized labels."""
    labels: set[str] = set()
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            for n in row.get("objects_norm", []):
                if n:
                    labels.add(n)
    return sorted(labels)


def stage_classify_per_label(out_dir: Path, model: str, batch_size: int,
                             timeout: float, max_retries: int,
                             force: bool = False, workers: int = 1) -> dict:
    """Stage 1 (per-label mode): classify each unique normalized label in isolation.

    Output: qwen_label_classification.jsonl, one row per label.
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY not set in environment")

    index_path = out_dir / "metadata_index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"metadata index missing: {index_path}; run `scan` first")

    labels = load_unique_normalized_labels(index_path)
    print(f"[classify-label] {len(labels)} unique normalized labels", flush=True)

    ckp_path = out_dir / "qwen_label_classification.jsonl"
    failed_path = out_dir / "failed_label_batches.jsonl"
    done_labels: set[str] = set()
    if not force and ckp_path.exists():
        with open(ckp_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_labels.add(json.loads(line)["label"])
                except Exception:
                    continue
        print(f"[classify-label] resuming, {len(done_labels)} labels already done", flush=True)
    elif force and ckp_path.exists():
        ckp_path.unlink()

    todo = [lbl for lbl in labels if lbl not in done_labels]
    print(f"[classify-label] {len(todo)} labels to classify", flush=True)

    total_usage = Counter()
    n_failed = 0
    t0 = time.time()

    def _try_batch(batch_labels: list[str]) -> tuple[list[dict], dict, list[str]]:
        """Run one batch with split-retry on validation failure.
        Returns (successful_entries, accumulated_usage, failed_labels)."""
        items = [{"id": i, "label": lbl} for i, lbl in enumerate(batch_labels)]
        try:
            parsed = call_qwen_label(items, model, api_key, timeout, max_retries)
            per_item = validate_qwen_label_batch(parsed, items)
            usage = {k: v for k, v in (parsed.get("_usage") or {}).items()
                     if isinstance(v, int)}
            return per_item, usage, []
        except Exception as e:
            if len(batch_labels) <= 1:
                print(f"[classify-label] giving up on {batch_labels!r}: {e}", flush=True)
                return [], {}, list(batch_labels)
            mid = len(batch_labels) // 2
            print(f"[classify-label] split-retry: {len(batch_labels)} → "
                  f"{mid}+{len(batch_labels)-mid} ({e})", flush=True)
            a_items, a_usage, a_fail = _try_batch(batch_labels[:mid])
            b_items, b_usage, b_fail = _try_batch(batch_labels[mid:])
            merged_usage = {k: a_usage.get(k, 0) + b_usage.get(k, 0)
                            for k in set(a_usage) | set(b_usage)}
            return a_items + b_items, merged_usage, a_fail + b_fail

    batches: list[list[str]] = [todo[i:i+batch_size]
                                 for i in range(0, len(todo), batch_size)]

    def _process_one(batch: list[str]) -> tuple[list[dict], dict, list[str]]:
        return _try_batch(batch)

    done = 0
    write_lock = None
    if workers > 1:
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        write_lock = threading.Lock()

    with open(ckp_path, "a", encoding="utf-8") as ckp_f, \
         open(failed_path, "a", encoding="utf-8") as fail_f:

        def _drain(per_item, usage, fail, batch_len):
            nonlocal n_failed, done
            for ent in per_item:
                ckp_f.write(json.dumps({
                    "label": ent["label"],
                    "category": ent["category"],
                    "confidence": ent["confidence"],
                    "reason": ent["reason"],
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "ts": time.time(),
                }, ensure_ascii=False) + "\n")
            for k, v in usage.items():
                total_usage[k] += v
            for lbl in fail:
                n_failed += 1
                fail_f.write(json.dumps({"label": lbl, "ts": time.time()},
                                        ensure_ascii=False) + "\n")
            if fail:
                fail_f.flush()
            done += batch_len
            ckp_f.flush()
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-3)
            eta = (len(todo) - done) / max(rate, 1e-3)
            print(
                f"[classify-label] {done}/{len(todo)} "
                f"({100*done/max(len(todo),1):.1f}%) failed={n_failed} "
                f"rate={rate:.1f}/s eta={eta:.0f}s "
                f"usage={dict(total_usage)} elapsed={elapsed:.0f}s",
                flush=True,
            )

        if workers <= 1:
            for batch in batches:
                per_item, usage, fail = _process_one(batch)
                _drain(per_item, usage, fail, len(batch))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_process_one, b): b for b in batches}
                for fut in as_completed(futs):
                    batch = futs[fut]
                    try:
                        per_item, usage, fail = fut.result()
                    except Exception as e:
                        per_item, usage, fail = [], {}, list(batch)
                        print(f"[classify-label] worker raised: {e}", flush=True)
                    with write_lock:
                        _drain(per_item, usage, fail, len(batch))

    summary = {
        "mode": "per_label",
        "total_labels": len(labels),
        "already_done_on_start": len(done_labels),
        "classified_this_run": len(todo) - n_failed,
        "failed_this_run": n_failed,
        "usage": dict(total_usage),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "classify_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[classify-label] done. summary -> {out_dir / 'classify_summary.json'}", flush=True)
    return summary


def load_label_classifications(out_dir: Path) -> dict[str, dict]:
    """normalized_label → {category, confidence, reason} (last-write-wins)."""
    ckp_path = out_dir / "qwen_label_classification.jsonl"
    if not ckp_path.exists():
        raise SystemExit(f"label classification missing: {ckp_path}; "
                         f"run `classify --classify-mode per-label`")
    out: dict[str, dict] = {}
    with open(ckp_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                out[row["label"]] = {
                    "category": row["category"],
                    "confidence": float(row.get("confidence", 0.0)),
                    "reason": row.get("reason", ""),
                }
            except Exception:
                continue
    return out


# ----------------------------------------------------------------- Stage 2


def load_classifications(out_dir: Path) -> dict[str, list[dict]]:
    """sig → robot_labels list (last write wins for duplicate sigs)."""
    ckp_path = out_dir / "qwen_classification_checkpoint.jsonl"
    if not ckp_path.exists():
        raise SystemExit(f"classification checkpoint missing: {ckp_path}; run `classify`")
    out: dict[str, list[dict]] = {}
    with open(ckp_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                out[row["signature"]] = row["robot_labels"]
            except Exception:
                continue
    return out


def bucket_labels(robot_label_entries: list[dict]) -> tuple[str, tuple[str, ...]]:
    """Return (bucket_name, sorted_label_tuple) for one metadata's robot_labels.

    bucket_name in {bucket_zero, bucket_single, bucket_multi}.
    """
    labels = sorted({e["label"] for e in robot_label_entries
                     if e.get("category") in ROBOT_CATEGORIES})
    if len(labels) == 0:
        return "bucket_zero", tuple()
    if len(labels) == 1:
        return "bucket_single", tuple(labels)
    return "bucket_multi", tuple(labels)


def stage_bucket(out_dir: Path, signature_mode: str = "strict",
                 classify_mode: str = "per_signature") -> dict:
    """Stage 2: assign every metadata to a bucket.

    classify_mode="per_signature" — use signature → robot_labels (Stage 1a).
    classify_mode="per_label"     — use global normalized_label → category (Stage 1b).
    """
    index_path = out_dir / "metadata_index.jsonl"

    label_cls: dict[str, dict] = {}
    sig_classifications: dict[str, list[dict]] = {}
    if classify_mode == "per_label":
        label_cls = load_label_classifications(out_dir)
        print(f"[bucket] per-label mode: loaded {len(label_cls)} label classifications",
              flush=True)
    else:
        sig_classifications = load_classifications(out_dir)

    bucket_assignments: list[dict] = []
    bucket_zero_paths: list[str] = []
    bucket_single: dict[str, list[str]] = defaultdict(list)
    bucket_multi: dict[tuple, list[str]] = defaultdict(list)
    missing_sig = 0

    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if classify_mode == "per_label":
                # Look up each normalized label, keep those in ROBOT_CATEGORIES
                norm_cls: list[dict] = []
                for nlbl in set(row.get("objects_norm", [])):
                    info = label_cls.get(nlbl)
                    if info is None:
                        continue
                    if info["category"] in ROBOT_CATEGORIES:
                        norm_cls.append({"label": nlbl,
                                         "category": info["category"],
                                         "confidence": info["confidence"]})
                sig = ""  # not used in per_label mode
            else:
                sig = make_signature(row["dataset"], row["instruction"],
                                     row["objects_norm"], mode=signature_mode)
                cls = sig_classifications.get(sig)
                if cls is None:
                    missing_sig += 1
                    continue
                norm_cls = []
                for e in cls:
                    nlbl = normalize_label(e["label"])
                    if nlbl:
                        norm_cls.append({**e, "label": nlbl})

            bucket, key = bucket_labels(norm_cls)
            bucket_assignments.append({"path": row["path"], "sig": sig, "bucket": bucket,
                                       "robot_labels": list(key)})
            if bucket == "bucket_zero":
                bucket_zero_paths.append(row["path"])
            elif bucket == "bucket_single":
                bucket_single[key[0]].append(row["path"])
            else:
                bucket_multi[key].append(row["path"])

    assignments_path = out_dir / "bucket_assignments.jsonl"
    with open(assignments_path, "w", encoding="utf-8") as f:
        for a in bucket_assignments:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    (out_dir / "bucket_zero.jsonl").write_text(
        "\n".join(json.dumps({"path": p}) for p in bucket_zero_paths) + ("\n" if bucket_zero_paths else "")
    )
    (out_dir / "bucket_single.json").write_text(
        json.dumps({k: v for k, v in sorted(bucket_single.items(), key=lambda x: -len(x[1]))},
                   indent=2, ensure_ascii=False)
    )
    (out_dir / "bucket_multi.json").write_text(
        json.dumps({" | ".join(k): v for k, v in
                    sorted(bucket_multi.items(), key=lambda x: -len(x[1]))},
                   indent=2, ensure_ascii=False)
    )

    summary = {
        "total_metadata": len(bucket_assignments),
        "missing_sig_classification": missing_sig,
        "bucket_zero": len(bucket_zero_paths),
        "bucket_single_total": sum(len(v) for v in bucket_single.values()),
        "bucket_single_unique_labels": len(bucket_single),
        "bucket_single_top": dict(Counter({k: len(v) for k, v in bucket_single.items()}).most_common(20)),
        "bucket_multi_total": sum(len(v) for v in bucket_multi.values()),
        "bucket_multi_unique_combos": len(bucket_multi),
        "bucket_multi_top": {" | ".join(k): len(v) for k, v in
                              sorted(bucket_multi.items(), key=lambda x: -len(x[1]))[:20]},
    }
    (out_dir / "bucket_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[bucket] done. summary -> {out_dir / 'bucket_summary.json'}", flush=True)
    return summary


# ----------------------------------------------------------------- Stage 3


_SYSTEM_PROMPT_VL = """You are verifying object labels detected from a robot-manipulation episode.

CONTEXT YOU WILL RECEIVE
- The dataset name (e.g., conq_hose_manipulation, kuka, droid).
- The episode's language_instruction.
- The COMPLETE objects list detected for this episode (so you know what else is in the scene).
- One or more visualization images (vis_*.png) from the episode.
- The specific candidate label(s) to verify.

YOUR TASK
For each candidate label, decide its category:
- robotic_arm: ANY structural part of the operating robot platform — arm,
  base, body, head, torso, foot, joint, shoulder, chassis. Multi-robot
  platforms (Boston Dynamics Spot quadruped, mobile manipulators, SARA) — the
  WHOLE body counts, not just the arm. Color/material variants are fine
  ("yellow robot" on a yellow Spot → robotic_arm).
- gripper: gripper, robotic claw, parallel/suction gripper, jaws.
- end_effector: robot hand, robot wrist, tool flange (terminal mechanism).
- non_robot: scene object, toy, sticker, decoration, person, furniture, food.
- uncertain: only when images are too unclear to decide.

CRITICAL DECISION RULES
1. If the images clearly show the label points to a physical part of the
   operating robot platform (including quadruped body for Spot), classify it
   as robotic_arm/gripper/end_effector. Do NOT hedge to non_robot just because
   the label string is ambiguous in isolation.
2. Dedup rule: another model already produced this objects list with the
   instruction "name the robot as a SINGLE entry 'robotic arm', one entry per
   unique object instance class". So if "robotic arm" AND another robot-ish
   label both appear in the objects list, they MUST refer to DIFFERENT physical
   things — typically the arm vs. the platform body (Spot body, base, head, foot).
3. Use the dataset name as a strong prior:
   - conq_hose_manipulation → Boston Dynamics Spot, body is YELLOW
   - droid → Franka Panda arm (mostly black/white, on a stand)
   - kuka → KUKA arm (often orange/blue/silver)
   - language_table → small tabletop, may have toy robots or no robot body
4. If 3 images disagree (one shows robot body, another shows a toy), set
   "inconsistent": true and choose the dominant category.

OUTPUT — STRICT JSON ONLY:
{
  "labels": [
    {"label": "<exact label from input>",
     "category": "robotic_arm|gripper|end_effector|non_robot|uncertain",
     "confidence": <float 0..1>,
     "inconsistent": <bool>,
     "reason": "<short>"
    }, ...
  ]
}
"""


def _resize_for_vl(img: Image.Image, max_side: int = 768) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img.convert("RGB") if img.mode != "RGB" else img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS).convert("RGB")


def _img_to_data_url(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def find_vis_for_path(metadata_path: Path) -> Path | None:
    """Find a vis_*.png in the primary camera dir for an episode."""
    try:
        m = json.loads(metadata_path.read_text())
    except Exception:
        return None
    cam = m.get("primary_camera")
    if not cam:
        return None
    cam_dir = metadata_path.parent / cam
    if not cam_dir.is_dir():
        return None
    indices = m.get("gemini_frame_indices") or []
    for idx in indices:
        candidate = cam_dir / f"vis_{idx:06d}.png"
        if candidate.exists():
            return candidate
    # fallback: any vis_*.png
    try:
        for p in sorted(cam_dir.iterdir()):
            if p.name.startswith("vis_") and p.suffix == ".png":
                return p
    except OSError:
        pass
    return None


def call_qwen_vl(candidate_labels: list[str],
                 context: dict,
                 vis_paths: list[Path],
                 model: str,
                 api_key: str,
                 timeout: float = 180.0,
                 max_retries: int = 3,
                 image_max_side: int = 768) -> dict:
    """Call Qwen-VL with text + images. Returns parsed dict {labels: [...]}.

    context: {dataset, instruction, all_objects: list[str]}
    """
    content: list[dict] = []
    for p in vis_paths:
        try:
            with Image.open(p) as im:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _img_to_data_url(_resize_for_vl(im, image_max_side))},
                })
        except Exception:
            continue
    user_text = (
        "Dataset: " + (context.get("dataset") or "(unknown)") + "\n"
        "language_instruction: " + (context.get("instruction") or "(none)") + "\n"
        "Complete objects list for this episode: "
        + json.dumps(context.get("all_objects") or [], ensure_ascii=False) + "\n"
        "Candidate labels to verify: "
        + json.dumps(candidate_labels, ensure_ascii=False) + "\n"
        "Look at the image(s), use the dataset/instruction as priors, and classify each candidate."
    )
    content.append({"type": "text", "text": user_text})
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT_VL},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _http_post_json(DASHSCOPE_URL, headers, body, timeout)
            text = resp["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            parsed["_usage"] = resp.get("usage", {})
            return parsed
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Qwen-VL call failed: {last_err}")


def stage_verify_vl(out_dir: Path, model: str, vis_per_unit: int = 3,
                    timeout: float = 180.0, max_retries: int = 3,
                    seed: int = 42,
                    include_multi: bool = True,
                    include_single_nc: bool = True,
                    include_uncertain: bool = True,
                    workers: int = 1,
                    reverify_labels_matching: list[str] | None = None,
                    force_reverify: bool = False) -> dict:
    """Stage 3: vision verify ambiguous units.

    Three kinds of units (any combination via include_* flags):
    - multi: bucket_multi unique combos (different physical robot parts in same scene)
    - single_non_canonical: bucket_single with label != 'robotic arm'
    - uncertain: labels Qwen text-classified as 'uncertain' (color+robot, lone arm, etc.)
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY not set in environment")

    rng = random.Random(seed)
    units: list[dict] = []

    if include_multi:
        bucket_multi = json.loads((out_dir / "bucket_multi.json").read_text())
        for combo_str, paths in bucket_multi.items():
            labels = combo_str.split(" | ")
            units.append({"kind": "multi", "labels": labels, "paths": paths})
    if include_single_nc:
        bucket_single = json.loads((out_dir / "bucket_single.json").read_text())
        for lbl, paths in bucket_single.items():
            if lbl == CANONICAL_ARM_LABEL:
                continue
            units.append({"kind": "single_non_canonical", "labels": [lbl], "paths": paths})
    if include_uncertain:
        all_cls_path = out_dir / "all_label_classifications.json"
        vocab_path = out_dir / "object_label_vocabulary.json"
        if not all_cls_path.exists() or not vocab_path.exists():
            raise SystemExit("uncertain mode needs all_label_classifications.json and "
                             "object_label_vocabulary.json; run `finalize` first")
        all_cls = json.loads(all_cls_path.read_text())
        vocab = json.loads(vocab_path.read_text()).get("labels", {})
        for lbl, info in all_cls.items():
            if info.get("category") != "uncertain":
                continue
            paths = vocab.get(lbl, {}).get("example_paths", [])
            if paths:
                units.append({"kind": "uncertain", "labels": [lbl], "paths": paths})

    # Optional substring filter on labels (used for targeted re-verification)
    if reverify_labels_matching:
        substrs = [s.strip().lower() for s in reverify_labels_matching if s.strip()]
        def _hit(u):
            return any(any(sub in lbl.lower() for sub in substrs) for lbl in u["labels"])
        units = [u for u in units if _hit(u)]
        print(f"[verify-vl] label-substr filter {substrs}: kept {len(units)} units",
              flush=True)

    kinds_count = Counter(u["kind"] for u in units)
    print(f"[verify-vl] {len(units)} units to verify ({dict(kinds_count)})", flush=True)

    out_path = out_dir / "vision_verifications.jsonl"
    done_keys: set[str] = set()
    if out_path.exists() and not force_reverify:
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_keys.add(json.loads(line)["unit_key"])
                except Exception:
                    continue
    elif out_path.exists() and force_reverify:
        # Remove only entries for units we're about to re-verify (keep others intact)
        targeted_keys = set(u["kind"] + "::" + " | ".join(u["labels"]) for u in units)
        kept_lines: list[str] = []
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("unit_key") in targeted_keys:
                        continue
                except Exception:
                    pass
                kept_lines.append(line)
        out_path.write_text("".join(kept_lines))
        print(f"[verify-vl] force-reverify: stripped {len(targeted_keys)} target unit "
              f"records from existing JSONL", flush=True)

    total_usage = Counter()
    n_fail = 0
    n_done_total = 0
    t0 = time.time()

    def _process_unit(unit: dict) -> dict:
        """Run one unit. Returns a dict ready to write to JSONL."""
        key = unit["kind"] + "::" + " | ".join(unit["labels"])
        paths = unit["paths"]
        sampled = rng.sample(paths, min(vis_per_unit, len(paths)))
        vis_paths: list[Path] = []
        for p_str in sampled:
            vp = find_vis_for_path(Path(p_str))
            if vp is not None:
                vis_paths.append(vp)
        if not vis_paths:
            return {"unit_key": key, "kind": unit["kind"], "labels": unit["labels"],
                    "error": "no vis images found", "sampled_paths": sampled}
        ctx = {"dataset": "", "instruction": "", "all_objects": []}
        try:
            m = json.loads(Path(sampled[0]).read_text())
            ctx["instruction"] = m.get("language_instruction", "") or ""
            ctx["all_objects"] = list(m.get("objects") or [])
            # dataset = first component under metadata_root (best-effort by path split)
            parts = Path(sampled[0]).parts
            if "oxe_seg" in parts:
                idx = parts.index("oxe_seg")
                if idx + 1 < len(parts):
                    ctx["dataset"] = parts[idx + 1]
        except Exception:
            pass
        try:
            parsed = call_qwen_vl(unit["labels"], ctx, vis_paths,
                                  model, api_key, timeout, max_retries)
            return {"unit_key": key, "kind": unit["kind"], "labels": unit["labels"],
                    "dataset": ctx["dataset"], "instruction": ctx["instruction"],
                    "vis_paths": [str(p) for p in vis_paths],
                    "result": parsed.get("labels", []),
                    "_usage": parsed.get("_usage") or {}}
        except Exception as e:
            return {"unit_key": key, "kind": unit["kind"], "labels": unit["labels"],
                    "error": str(e)}

    todo_units = [u for u in units
                  if (u["kind"] + "::" + " | ".join(u["labels"])) not in done_keys]
    print(f"[verify-vl] resuming, {len(done_keys)} already done, {len(todo_units)} to do",
          flush=True)

    with open(out_path, "a", encoding="utf-8") as out_f:
        if workers <= 1:
            for ui, unit in enumerate(todo_units):
                res = _process_unit(unit)
                if "error" in res:
                    n_fail += 1
                else:
                    usage = res.pop("_usage", {})
                    for k, v in usage.items():
                        if isinstance(v, int):
                            total_usage[k] += v
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done_total += 1
                elapsed = time.time() - t0
                rate = n_done_total / max(elapsed, 1e-3)
                eta = (len(todo_units) - n_done_total) / max(rate, 1e-3)
                print(f"[verify-vl] {n_done_total}/{len(todo_units)} fails={n_fail} "
                      f"rate={rate:.2f}/s eta={eta:.0f}s usage={dict(total_usage)}",
                      flush=True)
        else:
            import threading
            from concurrent.futures import ThreadPoolExecutor, as_completed
            write_lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_process_unit, u): u for u in todo_units}
                for fut in as_completed(futs):
                    try:
                        res = fut.result()
                    except Exception as e:
                        u = futs[fut]
                        res = {"unit_key": u["kind"] + "::" + " | ".join(u["labels"]),
                               "kind": u["kind"], "labels": u["labels"],
                               "error": f"worker raised: {e}"}
                    with write_lock:
                        if "error" in res:
                            n_fail += 1
                        else:
                            usage = res.pop("_usage", {})
                            for k, v in usage.items():
                                if isinstance(v, int):
                                    total_usage[k] += v
                        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        out_f.flush()
                        n_done_total += 1
                        elapsed = time.time() - t0
                        rate = n_done_total / max(elapsed, 1e-3)
                        eta = (len(todo_units) - n_done_total) / max(rate, 1e-3)
                        print(f"[verify-vl] {n_done_total}/{len(todo_units)} fails={n_fail} "
                              f"rate={rate:.2f}/s eta={eta:.0f}s usage={dict(total_usage)}",
                              flush=True)

    summary = {
        "total_units": len(units),
        "previously_done": len(done_keys),
        "failures": n_fail,
        "usage": dict(total_usage),
    }
    (out_dir / "verify_vl_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[verify-vl] done. summary -> {out_dir / 'verify_vl_summary.json'}", flush=True)
    return summary


# ----------------------------------------------------------------- Stage 4


def load_manual_overrides(path: Path | None) -> dict[str, str]:
    """Load manual overrides yaml. Returns {normalized_label: category}."""
    if not path or not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if v not in ALL_CATEGORIES:
            raise ValueError(f"manual_overrides: {k!r}={v!r} not in {ALL_CATEGORIES}")
        out[normalize_label(k)] = v
    return out


def build_vocabulary(index_path: Path, sample_n: int = 5) -> dict:
    """Aggregate per-label vocabulary from the index. Stream, no full retain."""
    counts: Counter[str] = Counter()
    raw_variants: dict[str, set[str]] = defaultdict(set)
    datasets_per: dict[str, set[str]] = defaultdict(set)
    paths_sample: dict[str, list[str]] = defaultdict(list)
    instr_sample: dict[str, list[str]] = defaultdict(list)
    n_metadata = 0
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_metadata += 1
            ds = row["dataset"]
            instr = row.get("instruction") or ""
            for raw, norm in zip(row["objects_raw"], row["objects_norm"]):
                counts[norm] += 1
                raw_variants[norm].add(raw)
                datasets_per[norm].add(ds)
                if len(paths_sample[norm]) < sample_n:
                    paths_sample[norm].append(row["path"])
                if instr and len(instr_sample[norm]) < sample_n and instr not in instr_sample[norm]:
                    instr_sample[norm].append(instr)
    vocab = {}
    for lbl in sorted(counts):
        vocab[lbl] = {
            "count": counts[lbl],
            "raw_variants": sorted(raw_variants[lbl]),
            "datasets": sorted(datasets_per[lbl]),
            "example_paths": paths_sample[lbl],
            "sample_instructions": instr_sample[lbl],
        }
    return {"n_metadata": n_metadata, "n_unique_labels": len(vocab), "labels": vocab}


def stage_finalize(out_dir: Path, manual_overrides_path: Path | None,
                   low_conf_threshold: float, model: str,
                   strict_robot_conf: float | None = None) -> dict:
    """Stage 4: build the 5 output files."""
    index_path = out_dir / "metadata_index.jsonl"

    # Vocabulary
    vocab = build_vocabulary(index_path)
    (out_dir / "object_label_vocabulary.json").write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False))

    # Load Stage 1 results (sig → robot_labels) — per-signature mode
    sig_classifications = load_classifications(out_dir) if (out_dir / "qwen_classification_checkpoint.jsonl").exists() else {}

    # Aggregate per-label classification across all sigs.
    per_label_votes: dict[str, list[dict]] = defaultdict(list)
    for sig, robot_labels in sig_classifications.items():
        for ent in robot_labels:
            nlbl = normalize_label(ent["label"])
            if not nlbl:
                continue
            per_label_votes[nlbl].append({
                "category": ent["category"],
                "confidence": float(ent.get("confidence", 0.0)),
                "source": "qwen_text",
            })

    # Load Stage 1b results (per-label classification, if present)
    label_cls_path = out_dir / "qwen_label_classification.jsonl"
    if label_cls_path.exists():
        for lbl, info in load_label_classifications(out_dir).items():
            per_label_votes[lbl].append({
                "category": info["category"],
                "confidence": info["confidence"],
                "source": "qwen_text_label",
            })

    # Load vision verifications and overwrite per-label decisions if present
    vl_path = out_dir / "vision_verifications.jsonl"
    vl_results: dict[str, list[dict]] = defaultdict(list)
    if vl_path.exists():
        with open(vl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                for ent in row.get("result", []):
                    nlbl = normalize_label(ent.get("label", ""))
                    if not nlbl:
                        continue
                    cat = ent.get("category")
                    if cat not in ALL_CATEGORIES:
                        continue
                    vl_results[nlbl].append({
                        "category": cat,
                        "confidence": float(ent.get("confidence", 0.0)),
                        "inconsistent": bool(ent.get("inconsistent", False)),
                        "source": "qwen_vl",
                    })

    overrides = load_manual_overrides(manual_overrides_path)

    def decide(votes: list[dict], vl_votes: list[dict]) -> dict:
        """Decision: VL trumps text. Inconsistent VL → uncertain. Majority by count + max-avg-conf tiebreak.

        If strict_robot_conf is set: a ROBOT-category VL decision with avg confidence
        below the threshold is demoted to 'uncertain' (strict registry mode).
        """
        primary = vl_votes if vl_votes else votes
        source = "qwen_vl" if vl_votes else "qwen_text"
        if any(v.get("inconsistent") for v in primary):
            return {"category": "uncertain", "confidence": 0.0, "source": source,
                    "inconsistent": True}
        cat_counts: Counter[str] = Counter()
        cat_conf: dict[str, list[float]] = defaultdict(list)
        for v in primary:
            cat_counts[v["category"]] += 1
            cat_conf[v["category"]].append(v["confidence"])
        best_cat = max(cat_counts, key=lambda c: (cat_counts[c],
                       sum(cat_conf[c]) / len(cat_conf[c])))
        avg_conf = sum(cat_conf[best_cat]) / len(cat_conf[best_cat])
        if (strict_robot_conf is not None
                and source == "qwen_vl"
                and best_cat in ROBOT_CATEGORIES
                and avg_conf < strict_robot_conf):
            return {"category": "uncertain", "confidence": round(avg_conf, 3),
                    "source": source, "inconsistent": False,
                    "demoted_from": best_cat}
        return {"category": best_cat, "confidence": round(avg_conf, 3),
                "source": source, "inconsistent": False}

    all_classifications: dict[str, dict] = {}
    all_labels = set(per_label_votes) | set(vl_results) | set(vocab["labels"])
    for lbl in sorted(all_labels):
        if lbl in overrides:
            all_classifications[lbl] = {
                "category": overrides[lbl], "confidence": 1.0,
                "source": "manual_override", "inconsistent": False,
            }
        elif lbl in per_label_votes or lbl in vl_results:
            all_classifications[lbl] = decide(per_label_votes.get(lbl, []),
                                              vl_results.get(lbl, []))
        else:
            all_classifications[lbl] = {
                "category": "non_robot", "confidence": 1.0,
                "source": "default", "inconsistent": False,
            }
        # attach vocab stats
        if lbl in vocab["labels"]:
            all_classifications[lbl]["count"] = vocab["labels"][lbl]["count"]
            all_classifications[lbl]["raw_variants"] = vocab["labels"][lbl]["raw_variants"]

    (out_dir / "all_label_classifications.json").write_text(
        json.dumps(all_classifications, indent=2, ensure_ascii=False, sort_keys=True))

    # Build registry yaml
    registry_labels: dict[str, dict] = {}
    categories_grouped: dict[str, list[str]] = {c: [] for c in ROBOT_CATEGORIES}
    for lbl in sorted(all_classifications):
        info = all_classifications[lbl]
        cat = info["category"]
        if cat not in ROBOT_CATEGORIES:
            continue
        categories_grouped[cat].append(lbl)
        registry_labels[lbl] = {
            "category": cat,
            "is_robot_related": True,
            "confidence": info["confidence"],
            "count": info.get("count", 0),
            "raw_variants": info.get("raw_variants", []),
            "source": info["source"],
        }
    for c in categories_grouped:
        categories_grouped[c] = sorted(set(categories_grouped[c]))

    registry = {
        "version": 1,
        "source_root": str(out_dir.resolve().parent.parent),
        "qwen_model": model,
        "prompt_version": PROMPT_VERSION,
        "categories": categories_grouped,
        "labels": registry_labels,
    }
    (out_dir / "robot_label_registry.yaml").write_text(
        yaml.safe_dump(registry, sort_keys=True, allow_unicode=True, default_flow_style=False))

    # uncertain CSV
    uncertain_rows = []
    for lbl, info in all_classifications.items():
        is_uncertain = info["category"] == "uncertain"
        is_low_conf = (info["category"] in ROBOT_CATEGORIES + ("non_robot",)
                       and info["confidence"] < low_conf_threshold)
        if is_uncertain or is_low_conf:
            v = vocab["labels"].get(lbl, {})
            uncertain_rows.append({
                "normalized_label": lbl,
                "raw_variants": "; ".join(v.get("raw_variants", [])),
                "count": v.get("count", 0),
                "qwen_category": info["category"],
                "qwen_confidence": info["confidence"],
                "qwen_source": info["source"],
                "inconsistent": info.get("inconsistent", False),
                "datasets": "; ".join(v.get("datasets", [])),
                "example_paths": "; ".join(v.get("example_paths", [])),
                "human_decision": "",
                "human_canonical_category": "",
            })
    csv_path = out_dir / "uncertain_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        fields = ["normalized_label", "raw_variants", "count", "qwen_category",
                  "qwen_confidence", "qwen_source", "inconsistent", "datasets",
                  "example_paths", "human_decision", "human_canonical_category"]
        w = csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()
        for row in sorted(uncertain_rows, key=lambda r: (-r["count"], r["normalized_label"])):
            w.writerow(row)

    # summary md
    counts_by_cat = Counter()
    for info in all_classifications.values():
        counts_by_cat[info["category"]] += 1
    top_robot = sorted(
        [(lbl, info["count"]) for lbl, info in registry_labels.items()],
        key=lambda x: -x[1],
    )[:30]
    md = [
        "# OXE Robot Label Registry — Summary",
        "",
        f"- metadata files scanned: {vocab['n_metadata']}",
        f"- unique normalized labels: {vocab['n_unique_labels']}",
        f"- qwen model: `{model}`, prompt version: `{PROMPT_VERSION}`",
        "",
        "## Per-category label counts",
        "",
        "| category | label_count |",
        "| --- | --- |",
    ]
    for cat in ALL_CATEGORIES + ("default",):
        md.append(f"| {cat} | {counts_by_cat.get(cat, 0)} |")
    md += [
        "",
        f"## Robot-related labels (total {len(registry_labels)})",
        "",
        "Top 30 by occurrence:",
        "",
        "| label | category | count |",
        "| --- | --- | --- |",
    ]
    for lbl, c in top_robot:
        md.append(f"| {lbl} | {registry_labels[lbl]['category']} | {c} |")
    md += [
        "",
        f"## Uncertain / low-confidence: {len(uncertain_rows)} (see uncertain_labels.csv)",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))

    print(f"[finalize] wrote: vocabulary, all_classifications, registry, uncertain CSV, summary",
          flush=True)
    return {
        "n_metadata": vocab["n_metadata"],
        "n_unique_labels": vocab["n_unique_labels"],
        "n_robot_labels": len(registry_labels),
        "n_uncertain": len(uncertain_rows),
        "counts_by_cat": dict(counts_by_cat),
    }


# ----------------------------------------------------------------- CLI


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-root", type=Path,
                    default=Path("/public/home/lulucai/data/oxe_seg"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("reports/oxe_robot_labels"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--low-confidence-threshold", type=float, default=0.85)
    ap.add_argument("--manual-overrides", type=Path, default=None)
    ap.add_argument("--vis-per-unit", type=int, default=3)
    ap.add_argument("--force-reclassify", action="store_true")
    ap.add_argument("--scan-only", action="store_true",
                    help="Run scan + finalize only (no Qwen calls)")
    ap.add_argument("--signature-mode", choices=["strict", "loose"], default="strict",
                    help="strict=(dataset,instruction,objects); loose=(dataset,objects). "
                         "Use loose to maximize Qwen dedup at the cost of task context.")
    ap.add_argument("--classify-mode", choices=["per-signature", "per-label"],
                    default="per-label",
                    help="per-signature: one Qwen call per unique (dataset,instr,objects) "
                         "signature with full context. per-label: one Qwen call per unique "
                         "normalized label, isolated (much cheaper, recommended).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel HTTP workers for per-label classification "
                         "and vision verification (safe values: 1-8).")
    ap.add_argument("--verify-only", default="all",
                    choices=["all", "uncertain", "multi", "single-nc",
                             "uncertain+multi"],
                    help="Restrict Stage 3 vision verify to specific unit kinds.")
    ap.add_argument("--reverify-labels-matching", default="",
                    help="Comma-separated substrings; only verify units whose label "
                         "contains any of them (used for targeted strict re-pass).")
    ap.add_argument("--force-reverify", action="store_true",
                    help="Re-run VL even for units already in vision_verifications.jsonl. "
                         "Old entries for the targeted units are stripped first.")
    ap.add_argument("--strict-robot-conf", type=float, default=None,
                    help="Stage 4: any VL robot decision with avg confidence below "
                         "this threshold is demoted to 'uncertain' (kept out of registry).")
    ap.add_argument("subcommand",
                    choices=["scan", "classify", "bucket", "verify-vl", "finalize", "all"])
    return ap


def _run_classify(args, out_dir: Path) -> dict:
    if args.classify_mode == "per-label":
        return stage_classify_per_label(out_dir, args.model, args.batch_size,
                                        args.timeout, args.max_retries,
                                        force=args.force_reclassify,
                                        workers=args.workers)
    return stage_classify(out_dir, args.model, args.batch_size,
                          args.timeout, args.max_retries,
                          force=args.force_reclassify,
                          signature_mode=args.signature_mode)


def _resolve_verify_filters(verify_only: str) -> tuple[bool, bool, bool]:
    """Returns (include_multi, include_single_nc, include_uncertain)."""
    if verify_only == "all":
        return True, True, True
    if verify_only == "uncertain":
        return False, False, True
    if verify_only == "multi":
        return True, False, False
    if verify_only == "single-nc":
        return False, True, False
    if verify_only == "uncertain+multi":
        return True, False, True
    return True, True, True


def main() -> int:
    args = build_argparser().parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_mode = "per_label" if args.classify_mode == "per-label" else "per_signature"

    if args.scan_only:
        stage_scan(args.metadata_root, out_dir)
        stage_finalize(out_dir, args.manual_overrides, args.low_confidence_threshold,
                       args.model)
        return 0

    if args.subcommand == "scan":
        stage_scan(args.metadata_root, out_dir)
    elif args.subcommand == "classify":
        s = _run_classify(args, out_dir)
        if s["failed_this_run"] > 0:
            print(f"[classify] WARNING: {s['failed_this_run']} items failed; "
                  f"see failed_*.jsonl", file=sys.stderr)
            return 2
    elif args.subcommand == "bucket":
        stage_bucket(out_dir, signature_mode=args.signature_mode,
                     classify_mode=bucket_mode)
    elif args.subcommand == "verify-vl":
        inc_m, inc_s, inc_u = _resolve_verify_filters(args.verify_only)
        rev_match = [s for s in args.reverify_labels_matching.split(",") if s.strip()]
        stage_verify_vl(out_dir, args.model, args.vis_per_unit,
                        args.timeout, args.max_retries,
                        include_multi=inc_m, include_single_nc=inc_s,
                        include_uncertain=inc_u, workers=args.workers,
                        reverify_labels_matching=rev_match or None,
                        force_reverify=args.force_reverify)
    elif args.subcommand == "finalize":
        stage_finalize(out_dir, args.manual_overrides, args.low_confidence_threshold,
                       args.model, strict_robot_conf=args.strict_robot_conf)
    elif args.subcommand == "all":
        stage_scan(args.metadata_root, out_dir)
        s = _run_classify(args, out_dir)
        stage_bucket(out_dir, signature_mode=args.signature_mode,
                     classify_mode=bucket_mode)
        stage_finalize(out_dir, args.manual_overrides, args.low_confidence_threshold,
                       args.model)
        inc_m, inc_s, inc_u = _resolve_verify_filters(args.verify_only)
        stage_verify_vl(out_dir, args.model, args.vis_per_unit,
                        args.timeout, args.max_retries,
                        include_multi=inc_m, include_single_nc=inc_s,
                        include_uncertain=inc_u, workers=args.workers)
        stage_finalize(out_dir, args.manual_overrides, args.low_confidence_threshold,
                       args.model)
        if s.get("failed_this_run", 0) > 0:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
