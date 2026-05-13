#!/usr/bin/env python3
"""
对 Bridge Dataset 批量跑 SAM3 分割的多进程版本。
支持多 GPU 并行处理。
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
import torch.multiprocessing as mp
from queue import Empty

# 保证能 import 到 pipeline 模块（项目根）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PIL import Image

# 注意：这里只导入必要的非模型函数，模型加载函数将在 worker 内导入或调用
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


def process_single_directory(worker_rank, gpu_id, model, processor, dir_info, args):
    """
    处理单个目录的核心逻辑
    """
    dir_index, img_dir, scene_root = dir_info
    
    # 结果清单结构
    dir_manifest = {
        "dir_index": dir_index,
        "dir_path": str(img_dir),
        "scene_root": str(scene_root),
        "scene_objects_json": None,
        "images": [],
    }

    jpgs = list_jpgs(str(img_dir))
    if args.max_images_per_dir is not None:
        jpgs = jpgs[: args.max_images_per_dir]
    
    if not jpgs:
        return None

    # 读取该场景的物体列表
    scene_json = scene_root / args.scene_json_name
    object_names = DEFAULT_EXAMPLE_OBJECTS.copy()
    
    if scene_json.is_file():
        dir_manifest["scene_objects_json"] = str(scene_json)
        try:
            with open(scene_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            object_names = data.get("objects", []) or object_names
        except Exception as e:
            print(f"[Worker {worker_rank}] 读取 JSON 失败: {scene_json}, {e}")
    else:
        # print(f"[Worker {worker_rank}] 场景 {scene_root} 没有 {args.scene_json_name}，使用默认物体列表")
        pass

    print(f"[Worker {worker_rank}-GPU{gpu_id}] 处理目录 {dir_index}: {img_dir.name} (图数: {len(jpgs)})")

    out_sub = img_dir  # 结果直接写在 images0 目录
    processed_count = 0

    for img_path in jpgs:
        stem = Path(img_path).stem
        npz_name = f"{stem}_gemini_sam3_segments.npz"
        npz_path = out_sub / npz_name
        
        # 跳过已存在
        if args.skip_existing and npz_path.is_file():
            dir_manifest["images"].append({"path": img_path, "stem": stem, "skipped": True})
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Worker {worker_rank}] 跳过坏图 {img_path}: {e}")
            continue

        # --- 模型推理 ---
        try:
            results = run_sam3_per_object(processor, image, object_names)
        except Exception as e:
            print(f"[Worker {worker_rank}] 推理失败 {img_path}: {e}")
            traceback.print_exc()
            continue
        
        # --- 保存结果 ---
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
        processed_count += 1

    return dir_manifest


def worker_main(rank, args, task_queue, result_list):
    """
    工作进程主函数
    """
    import torch
    
    # 1. 确定设备
    if args.device == "cuda" and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_id = rank % gpu_count  # 循环分配 GPU
        # 关键：设置当前进程的默认 CUDA device，避免某些内部张量用 "cuda" 落到 cuda:0
        torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}"
    else:
        gpu_id = "cpu"
        device_str = "cpu"

    print(f"--- Worker {rank} 启动，使用设备: {device_str} ---")

    # 2. 加载模型 (每个进程独立加载)
    try:
        load_from_hf = not args.no_download_hf
        # 调用 pipeline 中的加载函数，传入具体的 device_str
        model, processor = load_sam3_model(
            device=device_str,
            confidence_threshold=args.confidence,
            checkpoint_path=args.checkpoint_path,
            load_from_hf=load_from_hf,
        )
    except Exception as e:
        print(f"[Fatal] Worker {rank} 模型加载失败: {e}")
        return

    # 3. 循环处理任务
    while True:
        try:
            # 获取任务，超时 5 秒防止死锁
            task = task_queue.get(timeout=5)
        except Empty:
            break
        
        if task is None:  # 结束信号
            break
            
        try:
            manifest = process_single_directory(rank, gpu_id, model, processor, task, args)
            if manifest:
                result_list.append(manifest)
        except Exception as e:
            print(f"[Error] Worker {rank} 处理目录失败: {e}")
            traceback.print_exc()

    print(f"--- Worker {rank} 结束 ---")


def main():
    ap = argparse.ArgumentParser(description="Bridge 数据集 SAM3 多进程分割")
    ap.add_argument("--list-file", default="bridge/bridgedatav2_image_paths.txt")
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--scene-json-name", default="scene_objects.json")
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--checkpoint-path", type=str, default=None)
    ap.add_argument("--no-download-hf", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--max-dirs", type=int, default=None)
    ap.add_argument("--max-images-per-dir", type=int, default=None)
    ap.add_argument("--no-merge-masks", action="store_true")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--output-dir", "-o", default=".")
    
    # 新增：多进程参数
    ap.add_argument("--num-workers", type=int, default=1, help="并行进程数（建议 <= GPU数量）")

    args = ap.parse_args()

    # 设置启动方法为 spawn (CUDA 必须)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    list_file = args.list_file
    base_dir = Path(args.base_dir).resolve()
    output_root = Path(args.output_dir)

    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"列表文件不存在: {list_file}")

    if args.no_download_hf and not args.checkpoint_path:
        raise ValueError("使用 --no-download-hf 时必须提供 --checkpoint-path")

    # --- 1. 准备任务列表 ---
    dirs_to_process = []
    with open(list_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rel = line.strip()
            if not rel: continue
            parts = rel.split("/")
            if len(parts) < 4: continue
            
            scene_rel = "/".join(parts[:4])
            scene_root = base_dir / scene_rel
            img_dir = base_dir / rel
            
            if img_dir.is_dir():
                dirs_to_process.append((idx, img_dir, scene_root))

    if args.max_dirs is not None:
        dirs_to_process = dirs_to_process[: args.max_dirs]

    print(f"共 {len(dirs_to_process)} 个目录待处理")
    print(f"启动 {args.num_workers} 个 Worker 进程...")

    # --- 2. 初始化队列和共享列表 ---
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_list = manager.list()

    # 填充队列
    for d in dirs_to_process:
        task_queue.put(d)
    
    # 填充结束信号（每个 worker 一个）
    for _ in range(args.num_workers):
        task_queue.put(None)

    # --- 3. 启动进程 ---
    processes = []
    for i in range(args.num_workers):
        p = mp.Process(
            target=worker_main,
            args=(i, args, task_queue, result_list)
        )
        p.start()
        processes.append(p)

    # --- 4. 等待完成 ---
    for p in processes:
        p.join()

    # --- 5. 汇总并写入 Manifest ---
    print("所有进程已完成，正在生成 Manifest...")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 将 Manager List 转换为普通 List 并按原始顺序排序
    final_manifest = list(result_list)
    final_manifest.sort(key=lambda x: x['dir_index'])

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2, ensure_ascii=False)
        
    print(f"Manifest 已保存至: {manifest_path}")
    print(f"处理完成。")


if __name__ == "__main__":
    main()