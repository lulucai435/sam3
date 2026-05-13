#!/usr/bin/env python3
"""
从 LIBERO 风格 HDF5 读取 RGB 帧（默认 data/demo_*/obs/agentview_rgb），
对每个 demo 先采样少量帧调用 Gemini 得到物体名并集，再对该 demo 的所有帧复用该物体列表跑 SAM3；
结果保存在 --output-dir。

典型 HDF5 布局与 OpenVLA/LIBERO 再处理数据一致：data/demo_i/obs/agentview_rgb，形状 (T, H, W, 3)。

依赖:
  pip install h5py
  pip install -e ".[gemini-sam3]"   # Gemini + matplotlib；SAM3 需按项目说明装 torch 等

示例:
  export GOOGLE_API_KEY=your_key
  python scripts/run_libero_hdf5_gemini_sam3.py \
    --hdf5 /share/250010208/hypernet/dataset/libero_90_no_noops/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5 \
    --output-dir ./libero_sam3_out \
    --use-gemini \
    --gemini-sample-frames 5 \
    --gemini-sample-mode uniform \
    --frame-stride 1

仅 SAM3、固定物体列表（省 API）:
  python scripts/run_libero_hdf5_gemini_sam3.py --hdf5 path/to/demo.hdf5 -o ./out \
    --objects "cabinet drawer, countertop, robot arm"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PIL import Image

try:
    import h5py
except ImportError as e:
    raise SystemExit("请先安装 h5py: pip install h5py") from e

from scripts.gemini_sam3_pipeline import (
    DEFAULT_EXAMPLE_OBJECTS,
    get_gemini_objects,
    load_sam3_model,
    run_sam3_per_object,
    save_segment_results,
    visualize_results,
)


def _numpy_to_rgb_pil(frame: np.ndarray) -> Image.Image:
    x = np.asarray(frame)
    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0:
            x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            x = x.astype(np.uint8)
    if x.ndim == 3 and x.shape[-1] == 3:
        return Image.fromarray(x, mode="RGB")
    raise ValueError(f"期望 (H,W,3) RGB，得到 shape={x.shape}")


def _list_demos(h5: h5py.File) -> list[str]:
    if "data" not in h5:
        raise KeyError("HDF5 中未找到顶层 group 'data'（非标准 LIBERO 布局？）")
    keys = [k for k in h5["data"].keys() if str(k).startswith("demo_")]
    return sorted(
        keys,
        key=lambda s: int(str(s).split("_")[-1]) if str(s).split("_")[-1].isdigit() else s,
    )


def normalize_object_name(x: str) -> str:
    x = str(x).strip().lower().replace("_", " ")
    x = " ".join(x.split())
    return x


def dedup_keep_order(xs: list[str]) -> list[str]:
    out = []
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def iter_demo_frames(
    h5_path: str,
    demo: str,
    obs_key: str,
    frame_stride: int,
):
    with h5py.File(h5_path, "r") as f:
        if demo not in f["data"]:
            print(f"  跳过不存在的 demo: {demo}")
            return
        g = f["data"][demo]
        if "obs" not in g:
            print(f"  跳过 {demo}: 无 obs group")
            return
        obs = g["obs"]
        if obs_key not in obs:
            available = list(obs.keys())
            raise KeyError(
                f"{h5_path} [{demo}/obs] 无数据集 '{obs_key}'。可用 keys: {available}"
            )
        rgb = obs[obs_key]
        T = rgb.shape[0]
        for t in range(0, T, frame_stride):
            frame = rgb[t]
            yield t, _numpy_to_rgb_pil(frame)


def load_single_frame(
    h5_path: str,
    demo: str,
    obs_key: str,
    t: int,
) -> Image.Image:
    with h5py.File(h5_path, "r") as f:
        if demo not in f["data"]:
            raise KeyError(f"demo 不存在: {demo}")
        g = f["data"][demo]
        if "obs" not in g or obs_key not in g["obs"]:
            raise KeyError(f"{demo}/obs/{obs_key} 不存在")
        rgb = g["obs"][obs_key]
        return _numpy_to_rgb_pil(rgb[t])


def sample_demo_frames_for_gemini(
    h5_path: str,
    demo: str,
    obs_key: str,
    num_samples: int,
    sample_mode: str = "uniform",
    seed: int = 0,
):
    with h5py.File(h5_path, "r") as f:
        if demo not in f["data"]:
            return []
        g = f["data"][demo]
        if "obs" not in g or obs_key not in g["obs"]:
            return []

        rgb = g["obs"][obs_key]
        T = rgb.shape[0]
        if T <= 0:
            return []

        k = min(num_samples, T)
        if k <= 0:
            return []

        if sample_mode == "uniform":
            idxs = np.linspace(0, T - 1, k, dtype=int).tolist()
        elif sample_mode == "random":
            rng = np.random.default_rng(seed)
            idxs = sorted(rng.choice(T, size=k, replace=False).tolist())
        else:
            raise ValueError(f"未知 sample_mode: {sample_mode}")

        idxs = dedup_keep_order([int(i) for i in idxs])
        return [(t, _numpy_to_rgb_pil(rgb[t])) for t in idxs]


def gemini_objects_from_pil(image: Image.Image, api_key: str | None) -> list[str]:
    """get_gemini_objects 只接受路径，这里写临时 JPEG 再调用。"""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        path = tmp.name
    try:
        image.convert("RGB").save(path, format="JPEG", quality=95)
        return get_gemini_objects(path, api_key=api_key)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def collect_demo_objects(
    h5_path: str,
    demo: str,
    obs_key: str,
    num_samples: int,
    api_key: str | None,
    fallback_objects: list[str],
    sample_mode: str = "uniform",
    seed: int = 0,
):
    sampled = sample_demo_frames_for_gemini(
        h5_path=h5_path,
        demo=demo,
        obs_key=obs_key,
        num_samples=num_samples,
        sample_mode=sample_mode,
        seed=seed,
    )

    raw_objects = []
    sampled_ts = []

    for t, pil_image in sampled:
        sampled_ts.append(t)
        try:
            names = gemini_objects_from_pil(pil_image, api_key=api_key)
        except Exception as e:
            print(f"  Gemini 失败 {demo} t={t}: {e}")
            traceback.print_exc()
            names = []
        raw_objects.extend([str(x).strip() for x in names if str(x).strip()])

    if not raw_objects:
        raw_objects = fallback_objects.copy()

    normalized = [normalize_object_name(x) for x in raw_objects]
    normalized = [x for x in normalized if x]
    demo_objects = dedup_keep_order(normalized)

    fallback_norm = [normalize_object_name(x) for x in fallback_objects if normalize_object_name(x)]
    for x in fallback_norm:
        if x not in demo_objects:
            demo_objects.append(x)

    return {
        "raw_objects": raw_objects,
        "demo_objects": demo_objects,
        "sampled_frames": sampled_ts,
    }


def load_demo_objects_from_existing_summary(out_root: Path, h5_stem: str, demo: str):
    """
    若输出目录中已存在该 demo 的 summary 文件，则复用其中 demo_objects，避免重复调用 Gemini。
    返回 None 表示未找到可用结果。
    """
    pattern = f"{h5_stem}__{demo}__t*_gemini_sam3_summary.json"
    for summary_path in sorted(out_root.glob(pattern)):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        demo_objects = data.get("demo_objects")
        if not demo_objects:
            # 兼容旧格式
            demo_objects = data.get("objects")
        if not demo_objects:
            continue
        normalized = [normalize_object_name(x) for x in demo_objects]
        normalized = [x for x in normalized if x]
        if normalized:
            return {
                "demo_objects": dedup_keep_order(normalized),
                "raw_objects": data.get("raw_objects", demo_objects),
                "sampled_frames": data.get("gemini_sampled_frames", []),
                "from_summary": str(summary_path),
            }
    return None


def save_vis_for_demo_frames(
    *,
    demo: str,
    items: list,
    k: int,
    obs_key: str,
    out_root: Path,
    processor,
    merge_masks: bool,
) -> None:
    """对该 demo 已处理帧列表选前 k + 后 k，重跑 SAM3；写原图 RGB PNG + mask 叠加 vis PNG。"""
    if k <= 0 or not items:
        return
    if len(items) <= 2 * k:
        vis_targets = items
    else:
        vis_targets = items[:k] + items[-k:]
    print(
        f"[{demo}] 保存 vis：本 demo 内前{k}+后{k}（共处理 {len(items)} 帧，写入 {len(vis_targets)} 张；"
        f"每张含原图 *_rgb.png + 叠加图 *_vis.png）"
    )
    for item in vis_targets:
        try:
            pil_image = load_single_frame(
                h5_path=item["hdf5"],
                demo=item["demo"],
                obs_key=obs_key,
                t=item["t"],
            )
            rgb_path = out_root / f"{item['base']}_gemini_sam3_rgb.png"
            pil_image.convert("RGB").save(rgb_path)

            results = run_sam3_per_object(processor, pil_image, item["demo_objects"])
            vis_path = out_root / f"{item['base']}_gemini_sam3_vis.png"
            visualize_results(
                pil_image,
                results,
                out_path=str(vis_path),
                merge_masks_per_label=merge_masks,
            )
        except Exception as e:
            print(f"  保存vis失败 {item['base']}: {e}")
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description="LIBERO HDF5 -> demo级 Gemini + SAM3 分割")
    ap.add_argument(
        "--hdf5",
        required=True,
        help="单个 .hdf5 路径（可多次指定：--hdf5 a.hdf5 --hdf5 b.hdf5）",
        action="append",
        dest="hdf5_paths",
    )
    ap.add_argument("--output-dir", "-o", required=True, help="输出目录（npz/json/vis）")
    ap.add_argument(
        "--obs-key",
        default="agentview_rgb",
        help="obs 下的图像数据集名；常见 agentview_rgb、eye_in_hand_rgb",
    )
    ap.add_argument(
        "--demos",
        default=None,
        help="只处理这些 demo，逗号分隔，如 demo_0,demo_1；默认处理全部 demo_*",
    )
    ap.add_argument("--frame-stride", type=int, default=1, help="每隔多少帧取一帧做 SAM3；1表示所有帧都处理")
    ap.add_argument("--max-frames", type=int, default=None, help="所有文件累计最多处理帧数（调试用）")
    ap.add_argument("--use-gemini", action="store_true", help="每个 demo 先采样若干帧做 Gemini，再复用物体列表跑 SAM3")
    ap.add_argument("--gemini-api-key", default=None)
    ap.add_argument("--gemini-sample-frames", type=int, default=5, help="每个 demo 抽多少帧给 Gemini")
    ap.add_argument(
        "--gemini-sample-mode",
        choices=("uniform", "random"),
        default="uniform",
        help="Gemini 采样方式：uniform 或 random",
    )
    ap.add_argument("--gemini-sample-seed", type=int, default=0, help="random 采样的随机种子")
    ap.add_argument(
        "--objects",
        default=None,
        help="未使用 --use-gemini 时的固定物体列表（逗号分隔）；使用 --use-gemini 时作失败回退",
    )
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--checkpoint-path", type=str, default=None)
    ap.add_argument("--no-download-hf", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-merge-masks", action="store_true")
    ap.add_argument("--no-vis", action="store_true", help="不保存任何vis；segment结果照常保存")
    ap.add_argument(
        "--vis-first-last-k",
        type=int,
        default=5,
        help="每个 demo 的 SAM3 跑完后立刻保存该 demo 内最先处理的 k 帧与最后 k 帧的 vis；"
        "若该 demo 处理帧数≤2k 则全部保存。默认 k=5",
    )
    args = ap.parse_args()

    if args.use_gemini and not args.gemini_api_key and not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("使用 --use-gemini 时请设置 GOOGLE_API_KEY 或传入 --gemini-api-key")

    fallback_objects = (
        [s.strip() for s in args.objects.split(",") if s.strip()]
        if args.objects
        else DEFAULT_EXAMPLE_OBJECTS.copy()
    )

    if args.no_download_hf and not args.checkpoint_path:
        raise ValueError("使用 --no-download-hf 时必须提供 --checkpoint-path")

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    demos_filter = None
    if args.demos:
        demos_filter = [s.strip() for s in args.demos.split(",") if s.strip()]

    print("加载 SAM3...")
    model, processor = load_sam3_model(
        device=args.device,
        confidence_threshold=args.confidence,
        checkpoint_path=args.checkpoint_path,
        load_from_hf=not args.no_download_hf,
    )

    merge_masks = not args.no_merge_masks
    done = 0
    skip = 0
    frames_seen = 0
    manifest_frames = []
    k_vis = max(0, args.vis_first_last_k)

    for h5_path in args.hdf5_paths:
        h5_path = os.path.abspath(h5_path)
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(h5_path)
        h5_stem = Path(h5_path).stem

        with h5py.File(h5_path, "r") as f:
            demo_list = demos_filter if demos_filter is not None else _list_demos(f)

        for demo in demo_list:
            if args.use_gemini:
                # 续跑时，若该 demo 已有 summary，就复用里面的 demo_objects，不再重复调 Gemini
                obj_info = None
                if args.skip_existing:
                    obj_info = load_demo_objects_from_existing_summary(out_root, h5_stem, demo)
                    if obj_info is not None:
                        print(f"[{demo}] 复用已有 demo_objects: {obj_info['from_summary']}")
                if obj_info is None:
                    obj_info = collect_demo_objects(
                        h5_path=h5_path,
                        demo=demo,
                        obs_key=args.obs_key,
                        num_samples=args.gemini_sample_frames,
                        api_key=args.gemini_api_key,
                        fallback_objects=fallback_objects,
                        sample_mode=args.gemini_sample_mode,
                        seed=args.gemini_sample_seed,
                    )
                object_names = obj_info["demo_objects"]
                raw_objects = obj_info["raw_objects"]
                sampled_frames = obj_info["sampled_frames"]
                print(f"[{demo}] Gemini采样帧: {sampled_frames}")
                print(f"[{demo}] raw_objects: {raw_objects}")
                print(f"[{demo}] demo_objects: {object_names}")
            else:
                object_names = [normalize_object_name(x) for x in fallback_objects if normalize_object_name(x)]
                raw_objects = fallback_objects.copy()
                sampled_frames = []

            demo_processed: list[dict] = []

            for t, pil_image in iter_demo_frames(
                h5_path=h5_path,
                demo=demo,
                obs_key=args.obs_key,
                frame_stride=args.frame_stride,
            ):
                if args.max_frames is not None and frames_seen >= args.max_frames:
                    break
                frames_seen += 1

                base = f"{h5_stem}__{demo}__t{t:05d}"
                npz_path = out_root / f"{base}_gemini_sam3_segments.npz"

                if args.skip_existing and npz_path.is_file():
                    skip += 1
                    manifest_frames.append(
                        {
                            "hdf5": h5_path,
                            "demo": demo,
                            "t": t,
                            "skipped": True,
                            "demo_objects": object_names,
                        }
                    )
                    continue

                try:
                    results = run_sam3_per_object(processor, pil_image, object_names)
                except Exception as e:
                    print(f"  SAM3 失败 {base}: {e}")
                    traceback.print_exc()
                    continue

                segments_path, instances = save_segment_results(
                    results,
                    str(out_root),
                    base,
                    merge_masks_per_label=merge_masks,
                )

                summary = [
                    {
                        "label": r["label"],
                        "num_instances": r["masks"].shape[0] if r["masks"] is not None else 0,
                    }
                    for r in results
                ]
                summary_path = out_root / f"{base}_gemini_sam3_summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "raw_objects": raw_objects,
                            "demo_objects": object_names,
                            "gemini_sampled_frames": sampled_frames,
                            "per_object": summary,
                            "instances": instances,
                            "segments_npz": npz_path.name if segments_path else None,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                manifest_frames.append(
                    {
                        "hdf5": h5_path,
                        "demo": demo,
                        "t": t,
                        "base": base,
                        "npz": npz_path.name if segments_path else None,
                        "demo_objects": object_names,
                        "gemini_sampled_frames": sampled_frames,
                    }
                )

                demo_processed.append(
                    {
                        "hdf5": h5_path,
                        "demo": demo,
                        "t": t,
                        "base": base,
                        "demo_objects": object_names,
                    }
                )

                done += 1

            if not args.no_vis and k_vis > 0:
                save_vis_for_demo_frames(
                    demo=demo,
                    items=demo_processed,
                    k=k_vis,
                    obs_key=args.obs_key,
                    out_root=out_root,
                    processor=processor,
                    merge_masks=merge_masks,
                )

            if args.max_frames is not None and frames_seen >= args.max_frames:
                break

        if args.max_frames is not None and frames_seen >= args.max_frames:
            break

    manifest_path = out_root / "manifest_libero_hdf5.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_frames, f, indent=2, ensure_ascii=False)

    print(f"完成: 处理 {done} 帧，跳过 {skip} 帧。manifest: {manifest_path}")


if __name__ == "__main__":
    main()