#!/usr/bin/env python3
"""
对 Bridge Dataset 批量跑 SAM3 分割（不调用 Gemini），
物体列表从每个场景根目录下的 scene_objects.json 读取。

场景定义：列表文件中的路径，例如
  raw/bridge_data_v2/.../sweep_granular/01/2023-.../raw/traj.../images0
取前 5 级：
  raw/bridge_data_v2/.../sweep_granular/01
作为场景根，预处理脚本已经在这里写了 scene_objects.json。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 保证能 import 到 pipeline 模块（项目根）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PIL import Image

from scripts.gemini_sam3_pipeline import (
    DEFAULT_EXAMPLE_OBJECTS,
    load_sam3_model,
    run_sam3_per_object,
    save_segment_results,
    visualize_results,
)


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
    ap = argparse.ArgumentParser(description="Bridge 数据集：读取 scene_objects.json，仅用 SAM3 分割（不调 Gemini）")
    ap.add_argument(
        "--list-file",
        default="bridge/bridgedatav2_image_paths.txt",
        help="images0 目录列表文件，每行一个目录（相对 base-dir）",
    )
    ap.add_argument(
        "--base-dir",
        default=".",
        help="列表中路径的根目录，比如 `bridge`，则实际目录为 base-dir/那一行",
    )
    ap.add_argument(
        "--scene-json-name",
        default="scene_objects.json",
        help="每个场景根目录下的物体 JSON 文件名",
    )
    ap.add_argument("--confidence", type=float, default=0.5, help="SAM3 置信度阈值")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--checkpoint-path", type=str, default=None)
    ap.add_argument("--no-download-hf", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="若已存在 segments npz 则跳过该图")
    ap.add_argument("--max-dirs", type=int, default=None, help="最多处理多少个目录（用于试跑）")
    ap.add_argument("--max-images-per-dir", type=int, default=None, help="每个目录最多处理多少张图")
    ap.add_argument("--no-merge-masks", action="store_true", help="不合并同一 label 的多个 mask")
    ap.add_argument("--no-vis", action="store_true", help="不保存每张图的可视化 PNG（默认保存）")
    ap.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="仅用于保存 manifest.json 的目录；分割结果（npz/json/vis）直接保存在每张图所在目录",
    )
    args = ap.parse_args()

    list_file = args.list_file
    base_dir = Path(args.base_dir).resolve()
    output_root = Path(args.output_dir)

    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"列表文件不存在: {list_file}")

    if args.no_download_hf and not args.checkpoint_path:
        raise ValueError("使用 --no-download-hf 时必须提供 --checkpoint-path")

    # 读取 images0 目录列表，并为每个目录配上对应的场景根
    dirs: list[tuple[int, Path, Path]] = []  # (行号, images0_dir_abs, scene_root_abs)
    with open(list_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rel = line.strip()
            if not rel:
                continue
            parts = rel.split("/")
            if len(parts) < 5:
                continue
            scene_rel = "/".join(parts[:5])  # 第 5 级作为场景根
            scene_root = base_dir / scene_rel
            img_dir = base_dir / rel
            if img_dir.is_dir():
                dirs.append((idx, img_dir, scene_root))

    if args.max_dirs is not None:
        dirs = dirs[: args.max_dirs]

    print(f"共 {len(dirs)} 个 images0 目录待处理")

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

    for dir_index, img_dir, scene_root in dirs:
        jpgs = list_jpgs(str(img_dir))
        if args.max_images_per_dir is not None:
            jpgs = jpgs[: args.max_images_per_dir]
        if not jpgs:
            continue

        # 读取该场景的物体列表
        scene_json = scene_root / args.scene_json_name
        if scene_json.is_file():
            with open(scene_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            object_names = data.get("objects", []) or DEFAULT_EXAMPLE_OBJECTS.copy()
        else:
            print(f"  场景 {scene_root} 没有 {args.scene_json_name}，使用默认物体列表")
            object_names = DEFAULT_EXAMPLE_OBJECTS.copy()

        print(f"[dir {dir_index}] {img_dir}，物体数: {len(object_names)}")

        dir_manifest = {
            "dir_index": dir_index,
            "dir_path": str(img_dir),
            "scene_root": str(scene_root),
            "scene_objects_json": str(scene_json) if scene_json.is_file() else None,
            "images": [],
        }

        out_sub = img_dir  # 结果直接写在 images0 目录

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

    # 写 manifest
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("manifest 已写:", manifest_path)
    print("总处理:", total_done, "张，跳过:", total_skip, "张")


if __name__ == "__main__":
    main()