#!/usr/bin/env python3
"""
递归列出目录下所有 .hdf5 文件（每行一个绝对路径，已排序）。

默认根目录:
  /share/250010208/hypernet/dataset/libero_90_no_noops

示例:
  python scripts/list_libero_hdf5.py
  python scripts/list_libero_hdf5.py /path/to/other_root
  python scripts/list_libero_hdf5.py --count
  python scripts/list_libero_hdf5.py | head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def iter_hdf5(root: Path, recursive: bool) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"不是目录或不存在: {root}")
    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.iterdir()
    out = []
    for p in candidates:
        if p.is_file() and p.suffix.lower() == ".hdf5":
            out.append(p.resolve())
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="列出目录下所有 HDF5 文件")
    ap.add_argument(
        "root",
        nargs="?",
        default="/share/250010208/hypernet/dataset/libero_90_no_noops",
        type=Path,
        help="搜索根目录",
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="只列出 root 下的一层 *.hdf5，不递归子目录",
    )
    ap.add_argument(
        "--count",
        action="store_true",
        help="只打印数量，不列出路径",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    paths = iter_hdf5(root, recursive=not args.no_recursive)

    if args.count:
        print(len(paths))
        return

    for p in paths:
        print(p)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
