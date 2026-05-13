#!/usr/bin/env python3
"""
Pipeline: 输入一张图片 -> 用 Gemini 识别环境中的物体(text) -> 用 SAM3 对每个物体做分割.

Usage:
  export GOOGLE_API_KEY=your_key   # 或通过 --gemini-api-key 传入
  python scripts/gemini_sam3_pipeline.py --image path/to/image.jpg [--output-dir out/]
"""

import argparse
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Gemini: 优先使用官方 google-genai (pip install google-genai)，其次 google-generativeai
_GEMINI_BACKEND = None
try:
    from google import genai
    from google.genai import types
    _GEMINI_BACKEND = "google-genai"
except ImportError:
    try:
        import google.generativeai as genai
        _GEMINI_BACKEND = "google-generativeai"
    except ImportError:
        genai = None
        types = None


def get_gemini_objects(image_path: str, api_key: str | None = None) -> list[str]:
    if _GEMINI_BACKEND is None:
        raise ImportError(
            "需要安装 Gemini SDK: pip install google-genai 或 pip install google-generativeai"
        )
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("请设置 GOOGLE_API_KEY 或传入 --gemini-api-key")

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/jpeg"

    prompt = (
        "Look at this image and List all the objects in the environment. "
        "Reply with ONLY a JSON array of short English object names, one per item. "
        "Examples: [\"Black Robotic Arm\",\"Blue-handled fork\",\"Small green ball\",\"Silver metal pot\",\"Black stovetop burner\",\"White speckled countertop\",\"Silver kitchen sink\",\"Stove control knob\",\"Wooden cabinet\"]. "
        "Use common nouns. No explanation, no markdown, just the JSON array."
    )

    if _GEMINI_BACKEND == "google-genai":
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        text = (response.text or "").strip()
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        # 旧版 SDK 可直接传入 PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        response = model.generate_content(
            [pil_image, prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        text = (response.text or "").strip()
    # 允许被 markdown 代码块包裹
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # 尝试按行或逗号拆分
        text = re.sub(r"^\s*\[\s*", "", text)
        text = re.sub(r"\s*\]\s*$", "", text)
        parts = [p.strip().strip('"') for p in re.split(r"[\s,]+", text) if p.strip()]
        obj = parts if parts else []
    if not isinstance(obj, list):
        obj = [str(obj)]
    return [str(x).strip() for x in obj if x]


# 单独调试 SAM3 时用的默认物体列表（--skip-gemini 且未传 --objects 时使用）
DEFAULT_EXAMPLE_OBJECTS = [
    "Black Robotic Arm",
    "Blue-handled fork",
    "Small green ball",
    "Silver metal pot",
    "Black stovetop burner",
    "White speckled countertop",
    "Silver kitchen sink",
    "Stove control knob",
    "Wooden cabinet",
]


def load_sam3_model(
    device: str = "cuda",
    confidence_threshold: float = 0.5,
    checkpoint_path: str | None = None,
    load_from_hf: bool = True,
):
    """加载 SAM3 图像模型和 Processor。"""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
    )
    processor = Sam3Processor(
        model, device=device, confidence_threshold=confidence_threshold
    )
    return model, processor


def run_sam3_per_object(processor, image: Image.Image, object_names: list[str], state=None):
    """
    对每个物体名称用 SAM3 做 text prompt 分割，返回每个物体对应的 masks/boxes/scores。
    """
    if state is None:
        state = {}
    state = processor.set_image(image, state=state)
    results = []
    for name in object_names:
        state = processor.set_text_prompt(prompt=name, state=state)
        masks = state.get("masks")  # (N, H, W) bool
        boxes = state.get("boxes")  # (N, 4) xyxy
        scores = state.get("scores")  # (N,)
        n = 0
        if masks is not None:
            n = masks.shape[0]
        if boxes is not None and n == 0:
            n = boxes.shape[0]
        if scores is not None and n == 0:
            n = scores.shape[0]
        if n == 0:
            results.append({"label": name, "masks": None, "boxes": None, "scores": None})
            continue
        # 转到 CPU numpy 便于保存/可视化
        m = masks.cpu().numpy() if masks is not None else None
        b = boxes.cpu().numpy() if boxes is not None else None
        s = scores.cpu().numpy() if scores is not None else None
        results.append({"label": name, "masks": m, "boxes": b, "scores": s})
    return results


def _merge_masks_per_label(results: list) -> list:
    """同一 label 下的多个 instance mask 合并为一个（取并集），每个 label 对应一个 mask、一个 score（取最大）。"""
    merged = []
    for r in results:
        if r["masks"] is None or r["scores"] is None or r["masks"].shape[0] == 0:
            merged.append({"label": r["label"], "masks": None, "boxes": r["boxes"], "scores": None})
            continue
        # (n, H, W) 或 (n, 1, H, W) -> (H, W)，任意一处为 1 则为 1
        ms = np.asarray(r["masks"], dtype=np.uint8)
        while ms.ndim > 3:
            ms = ms.max(axis=0)  # (n, 1, H, W) -> (1, H, W) -> 下次 (H, W)
        if ms.shape[0] > 1:
            merged_mask = (ms.max(axis=0) > 0).astype(np.uint8)
        else:
            merged_mask = (ms[0] if ms.ndim == 3 else ms).astype(np.uint8)
        if merged_mask.ndim > 2:
            merged_mask = merged_mask.squeeze()
        # 该 label 的 confidence 取所有 instance 的最大值
        merged_score = np.asarray(r["scores"]).max()
        merged.append({
            "label": r["label"],
            "masks": merged_mask[np.newaxis, ...],  # (1, H, W)
            "boxes": r["boxes"],
            "scores": np.array([merged_score], dtype=np.float32),
        })
    return merged


def save_segment_results(
    results: list,
    out_dir: str,
    base_name: str,
    merge_masks_per_label: bool = True,
) -> tuple[str | None, list]:
    """
    保存 segment mask 与 confidence 到 npz。
    若 merge_masks_per_label=True，同一 text/label 的多个 mask 合并为一个（并集），一个 label 对应一个 mask、一个 score（取最大）。
    返回 (segments_npz_path, instances_list)。
    """
    if merge_masks_per_label:
        results = _merge_masks_per_label(results)

    masks_list = []
    scores_list = []
    label_names_list = []
    instances = []

    for r in results:
        if r["masks"] is None or r["scores"] is None:
            continue
        n = r["masks"].shape[0]
        for j in range(n):
            mask = np.asarray(r["masks"][j], dtype=np.uint8)
            while mask.ndim > 2:
                mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]
            masks_list.append(mask)
            scores_list.append(float(r["scores"][j]))
            label_names_list.append(r["label"])
            idx = len(masks_list) - 1
            instances.append({"label": r["label"], "score": scores_list[-1], "index": idx})

    if not masks_list:
        return None, []

    masks_stack = np.stack(masks_list, axis=0)  # (N, H, W)，N = 物体种类数（已合并时）
    scores_arr = np.array(scores_list, dtype=np.float32)
    segments_path = os.path.join(out_dir, f"{base_name}_gemini_sam3_segments.npz")
    np.savez_compressed(
        segments_path,
        masks=masks_stack,
        scores=scores_arr,
        label_names=np.array(label_names_list, dtype=object),
    )
    return segments_path, instances


def visualize_results(
    image: Image.Image,
    results: list,
    out_path: str | None = None,
    merge_masks_per_label: bool = True,
):
    """将各物体 mask 叠加到原图并保存。若 merge_masks_per_label=True，同一 label 的多个 mask 先合并再画。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if out_path:
            print("未安装 matplotlib，跳过可视化保存")
        return
    if merge_masks_per_label:
        results = _merge_masks_per_label(results)
    img = np.array(image.convert("RGB")) / 255.0
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(results), 1)))
    for i, r in enumerate(results):
        if r["masks"] is None:
            continue
        color = colors[i % len(colors)][:3]
        for j in range(r["masks"].shape[0]):
            mask = np.asarray(r["masks"][j], dtype=np.float32)
            while mask.ndim > 2:
                mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]
            overlay = np.zeros((*mask.shape, 4))
            overlay[..., :3] = color
            overlay[..., 3] = 0.4 * mask
            ax.imshow(overlay, extent=(0, img.shape[1], img.shape[0], 0), origin="upper")
        ax.text(0.02, 0.98 - i * 0.03, f"{r['label']}", transform=ax.transAxes, color=color, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Gemini 识别物体 + SAM3 分割 pipeline")
    parser.add_argument("--image", "-i", required=True, help="输入图片路径")
    parser.add_argument("--output-dir", "-o", default=None, help="输出目录（保存可视化与结果）")
    parser.add_argument("--gemini-api-key", default=None, help="Gemini API Key（或设 GOOGLE_API_KEY）")
    parser.add_argument("--confidence", type=float, default=0.5, help="SAM3 置信度阈值")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="SAM3 运行设备")
    parser.add_argument("--no-vis", action="store_true", help="不保存可视化图")
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="不调 Gemini，用默认或 --objects 的物体列表单独调试 SAM3",
    )
    parser.add_argument(
        "--objects",
        type=str,
        default=None,
        help="跳过 Gemini 时使用的物体列表，逗号分隔，如: 'person,car,table'；不传则用脚本内 Examples 默认列表",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="本地 SAM3 checkpoint 路径（.pt 文件），不传则尝试从 HuggingFace 下载",
    )
    parser.add_argument(
        "--no-download-hf",
        action="store_true",
        help="不尝试从 HuggingFace 下载 checkpoint（需提供 --checkpoint-path）",
    )
    parser.add_argument(
        "--no-merge-masks",
        action="store_true",
        help="不合并同一 text 的多个 mask，保存为每个 instance 单独一个 mask（默认合并为一 label 一 mask）",
    )
    args = parser.parse_args()

    image_path = args.image
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    if args.skip_gemini:
        if args.objects:
            object_names = [s.strip() for s in args.objects.split(",") if s.strip()]
        else:
            object_names = DEFAULT_EXAMPLE_OBJECTS.copy()
        print("Step 1: 跳过 Gemini，使用物体列表:", object_names)
    else:
        print("Step 1: 使用 Gemini 识别图中物体...")
        object_names = get_gemini_objects(image_path, api_key=args.gemini_api_key)
        print("  识别到的物体:", object_names)
    if not object_names:
        print("  未识别到物体，退出")
        return

    print("Step 2: 加载 SAM3 模型...")
    load_from_hf = not args.no_download_hf
    if args.no_download_hf and not args.checkpoint_path:
        raise ValueError("使用 --no-download-hf 时必须提供 --checkpoint-path")
    model, processor = load_sam3_model(
        device=args.device,
        confidence_threshold=args.confidence,
        checkpoint_path=args.checkpoint_path,
        load_from_hf=load_from_hf,
    )
    image = Image.open(image_path).convert("RGB")

    print("Step 3: 对每个物体做 SAM3 分割...")
    results = run_sam3_per_object(processor, image, object_names)
    for r in results:
        n = 0
        if r["masks"] is not None:
            n = r["masks"].shape[0]
        print(f"  {r['label']}: {n} 个实例")

    out_dir = args.output_dir
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        base = Path(image_path).stem
        merge_masks = not args.no_merge_masks
        if not args.no_vis:
            vis_path = os.path.join(out_dir, f"{base}_gemini_sam3_vis.png")
            visualize_results(image, results, out_path=vis_path, merge_masks_per_label=merge_masks)
            print("  可视化已保存:", vis_path)
        # 保存 segment 结果：同一 text 的多个 mask 默认合并为一个
        segments_path, instances = save_segment_results(
            results, out_dir, base, merge_masks_per_label=merge_masks
        )
        if segments_path:
            print("  分割结果已保存:", segments_path, "(masks, scores, label_names)")
        # 结果摘要 JSON（含 per_object 与 per-instance 的 label + score）
        summary = [
            {
                "label": r["label"],
                "num_instances": r["masks"].shape[0] if r["masks"] is not None else 0,
            }
            for r in results
        ]
        summary_data = {
            "objects": object_names,
            "per_object": summary,
            "instances": instances,
            "segments_npz": os.path.basename(segments_path) if segments_path else None,
        }
        summary_path = os.path.join(out_dir, f"{base}_gemini_sam3_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print("  结果摘要已保存:", summary_path)
    else:
        if not args.no_vis:
            visualize_results(image, results)


if __name__ == "__main__":
    main()
