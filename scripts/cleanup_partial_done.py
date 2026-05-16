#!/usr/bin/env python3
"""
扫描 batch_oxe_full_dataset.py 的输出根目录，识别"假 _DONE"（旧 code 在
Gemini 失败时仍然写了 _DONE）。

判定一个 shard 目录是 partial 的条件：
  - 目录里没有任何 episode_XXXXXX 子目录（旧 code 整批 Gemini 失败）
  - OR  存在某个 episode_XXXXXX 子目录但里面缺 metadata.json
        （单个 episode Gemini 失败留下的孤儿目录）

partial 的 _DONE 会被删除，下次 sbatch 重新跑该 shard 时：
  - 已写 metadata.json 的 episode 直接读取 objects 跳过 Gemini
  - 已存在的 .npz 不会重新 SAM3
所以重做代价 ≈ tfrecord 读取 + 失败 episode 重调 Gemini，几乎不浪费 GPU。

默认 dry-run，加 --apply 才真删。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def shard_is_partial(shard_dir: Path) -> tuple[bool, str]:
    ep_dirs = sorted(shard_dir.glob("episode_*"))
    if not ep_dirs:
        return True, "no episode_* subdir (every Gemini call likely failed)"
    missing_meta = [d for d in ep_dirs if not (d / "metadata.json").is_file()]
    if missing_meta:
        sample = ", ".join(d.name for d in missing_meta[:3])
        more = "..." if len(missing_meta) > 3 else ""
        return True, f"{len(missing_meta)}/{len(ep_dirs)} episodes missing metadata.json [{sample}{more}]"
    return False, f"{len(ep_dirs)} episodes, all have metadata.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path,
                    default=Path("/public/home/lulucai/data/oxe_seg"))
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete _DONE markers (default: dry-run)")
    args = ap.parse_args()

    if not args.output_root.is_dir():
        raise FileNotFoundError(args.output_root)

    done_files = list(args.output_root.glob("*/*/*/*/_DONE"))
    # output layout: <root>/<dataset>/<version>/<shard__of_NNNNN>/<shard_basename>/_DONE
    print(f"Scanning {args.output_root}\nFound {len(done_files)} _DONE markers.\n")

    partial: list[tuple[Path, str]] = []
    clean: list[Path] = []
    for done in done_files:
        shard_dir = done.parent
        bad, reason = shard_is_partial(shard_dir)
        if bad:
            partial.append((done, reason))
        else:
            clean.append(done)

    print(f"Clean shards : {len(clean)}")
    print(f"Partial shards: {len(partial)}\n")
    for done, reason in partial:
        rel = done.relative_to(args.output_root)
        print(f"  [partial] {rel.parent}  -- {reason}")

    if not partial:
        print("\nNothing to clean.")
        return

    if args.apply:
        for done, _ in partial:
            done.unlink()
        print(f"\nDeleted {len(partial)} _DONE markers. "
              "Next sbatch will re-process those shards "
              "(already-good episodes inside them are skipped via metadata.json/.npz checks).")
    else:
        print(f"\n(Dry-run) Rerun with --apply to delete these {len(partial)} _DONE files.")


if __name__ == "__main__":
    main()
