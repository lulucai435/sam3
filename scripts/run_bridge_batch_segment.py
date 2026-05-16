#!/usr/bin/env python3
"""
对 Bridge Dataset 批量跑 SAM3 分割：读取 bridge/bridgedatav2_image_paths.txt 中的目录列表，
对每个目录下所有 jpg 做 segment，结果保存到指定 output-dir。

用法:
  python scripts/run_bridge_batch_segment.py \\
    --list-file bridge/bridgedatav2_image_paths.txt \\
    --base-dir /path/to/data \\
    --output-dir out_bridge_segments \\
    [--skip-existing] [--max-dirs N] [--max-images-per-dir M]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
# 保证能 import 到 pipeline 模块（项目根）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PIL import Image

from scripts.gemini_sam3_pipeline import (
    DEFAULT_EXAMPLE_OBJECTS,
    get_gemini_objects,
    load_sam3_model,
    run_sam3_per_object,
    save_segment_results,
    visualize_results,
)


def collect_image_dirs(list_file: str, base_dir: str) -> list[tuple[int, str]]:
    """返回 [(dir_index, absolute_dir_path), ...]，只保留存在的目录。路径为 base_dir/bridge/<rel>。"""
    base = Path(base_dir).resolve()
    out = []
    with open(list_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rel = line.strip()
            if not rel:
                continue
            full = base / rel
            if full.is_dir():
                out.append((i, str(full)))
    return out


def list_jpgs(dir_path: str) -> list[str]:
    """返回目录下所有 .jpg/.jpeg 的完整路径，按文件名排序。"""
    p = Path(dir_path)
    if not p.is_dir():
        return []
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        files.extend(p.glob(ext))
    return sorted([str(f) for f in files])


def main():
    parser = argparse.ArgumentParser(description="Bridge 数据集批量 SAM3 分割")
    parser.add_argument(
        "--list-file",
        default="bridge/bridgedatav2_image_paths.txt",
        help="目录列表文件，每行一个目录（相对 base-dir）",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="目录列表中的路径所基于的根目录（如 /path/to/data，则列表中 raw/... 会变为 base-dir/raw/...）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="仅用于保存 manifest.json 的目录；分割结果（npz/json/vis）直接保存在每张图所在目录",
    )
    parser.add_argument("--objects", type=str, default=None, help="物体列表，逗号分隔；不传则用 pipeline 默认")
    parser.add_argument("--confidence", type=float, default=0.5, help="SAM3 置信度阈值")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--no-download-hf", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="若已存在 segments npz 则跳过该图")
    parser.add_argument("--max-dirs", type=int, default=None, help="最多处理多少个目录（用于试跑）")
    parser.add_argument("--max-images-per-dir", type=int, default=None, help="每个目录最多处理多少张图")
    parser.add_argument("--no-merge-masks", action="store_true", help="不合并同一 label 的多个 mask")
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="对每张图用 Gemini 识别物体再分割；不传则用固定物体列表（省 API、无需外网）",
    )
    parser.add_argument("--gemini-api-key", type=str, default=None, help="Gemini API Key（或设 GOOGLE_API_KEY）")
    parser.add_argument("--no-vis", action="store_true", help="不保存每张图的可视化 PNG（默认保存）")
    args = parser.parse_args()

    list_file = args.list_file
    base_dir = args.base_dir
    output_root = Path(args.output_dir)
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"列表文件不存在: {list_file}")

    if args.use_gemini and not args.gemini_api_key and not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("使用 --use-gemini 时请设置 GOOGLE_API_KEY 或传入 --gemini-api-key")

    # 非 Gemini 模式下的固定物体列表
    default_object_names = (
        [s.strip() for s in args.objects.split(",") if s.strip()]
        if args.objects
        else DEFAULT_EXAMPLE_OBJECTS.copy()
    )
    if not args.use_gemini:
        print("物体列表（固定）:", default_object_names[:5], "..." if len(default_object_names) > 5 else "")
    else:
        print("使用 Gemini 对每张图识别物体后再分割")

    dirs = collect_image_dirs(list_file, base_dir)
    if args.max_dirs is not None:
        dirs = dirs[: args.max_dirs]
    print(f"共 {len(dirs)} 个目录待处理")

    if args.no_download_hf and not args.checkpoint_path:
        raise ValueError("使用 --no-download-hf 时必须提供 --checkpoint-path")

    print("加载 SAM3 模型...")
    load_from_hf = not args.no_download_hf
    model, processor = load_sam3_model(
        device=args.device,
        confidence_threshold=args.confidence,
        checkpoint_path=args.checkpoint_path,
        load_from_hf=load_from_hf,
    )

    manifest = []
    total_done = 0
    total_skip = 0

    for dir_index, dir_path in dirs:
        jpgs = list_jpgs(dir_path)
        if args.max_images_per_dir is not None:
            jpgs = jpgs[: args.max_images_per_dir]
        if not jpgs:
            continue

        # 结果直接保存在图片所在目录 dir_path
        out_sub = Path(dir_path)
        dir_manifest = {"dir_index": dir_index, "dir_path": dir_path, "images": []}

        for img_path in jpgs:
            stem = Path(img_path).stem
            npz_name = f"{stem}_gemini_sam3_segments.npz"
            npz_path = out_sub / npz_name
            if args.skip_existing and npz_path.is_file():
                total_skip += 1
                dir_manifest["images"].append({"path": img_path, "stem": stem, "skipped": True})
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  跳过 {img_path}: {e}")
                continue

            if args.use_gemini:
                try:
                    object_names = get_gemini_objects(img_path, api_key=args.gemini_api_key)
                except Exception as e:
                    print(f"  Gemini 失败 {img_path}: {e}，使用固定列表")
                    object_names = default_object_names
                if not object_names:
                    object_names = default_object_names
            else:
                object_names = default_object_names

            results = run_sam3_per_object(processor, image, object_names)
            merge_masks = not args.no_merge_masks
            segments_path, instances = save_segment_results(
                results,
                str(out_sub),
                stem,
                merge_masks_per_label=merge_masks,
            )
            if not args.no_vis:
                vis_path = out_sub / f"{stem}_gemini_sam3_vis.png"
                visualize_results(image, results, out_path=str(vis_path), merge_masks_per_label=merge_masks)
            summary = [
                {"label": r["label"], "num_instances": r["masks"].shape[0] if r["masks"] is not None else 0}
                for r in results
            ]
            summary_path = out_sub / f"{stem}_gemini_sam3_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"objects": object_names, "per_object": summary, "instances": instances},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            dir_manifest["images"].append({"path": img_path, "stem": stem, "npz": npz_name})
            total_done += 1

        manifest.append(dir_manifest)
        if (dir_index + 1) % 100 == 0 or dir_index == dirs[-1][0]:
            print(f"  目录 {dir_index + 1}/{len(dirs)} 完成，累计处理 {total_done} 张，跳过 {total_skip} 张")

    manifest_path = output_root / "manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("manifest 已写:", manifest_path)
    print("总处理:", total_done, "张，跳过:", total_skip, "张")


if __name__ == "__main__":
    main()
