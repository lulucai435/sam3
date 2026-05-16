import argparse
import json
import random
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 假设 scripts.gemini_sam3_pipeline 里有这个函数
# 如果报错找不到，请确保路径设置正确
try:
    from scripts.gemini_sam3_pipeline import get_gemini_objects
except ImportError:
    # 占位函数，防止报错（如果你还没把pipeline代码放好的话）
    def get_gemini_objects(img_path, api_key=None):
        print(f"Mock Gemini Call: {img_path}")
        return ["mock_object_1", "mock_object_2"]

def main():
    ap = argparse.ArgumentParser(description="按场景预先用 Gemini 检测 Bridge 数据集中的物体 (scripted_raw 4级目录版)")
    ap.add_argument("--list-file", default="bridge/scripted_raw_images_list.txt")
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--scene-json-name", default="scene_objects.json")
    ap.add_argument("--gemini-api-key", default=None)
    ap.add_argument("--max-scenes", type=int, default=None)
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    # scene_root_dir -> 该场景下所有 images0 目录列表
    scenes: dict[Path, list[Path]] = {}

    # 读取 scripted_raw_images_list.txt: bridge/scripted_raw/{env_type}/{timestamp}/raw/.../images0
    # 按前 4 级目录归类，场景 = bridge/scripted_raw/{env_type}/{timestamp}
    with open(args.list_file, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            parts = rel.split("/")
            if len(parts) < 4:
                continue
            # bridge/scripted_raw/2022-12-08_pnp_rigid_objects/2022-12-08_15-22-17
            scene_rel = "/".join(parts[:4])
            scene_root = base / scene_rel
            img_dir = base / rel
            if img_dir.is_dir():
                scenes.setdefault(scene_root, []).append(img_dir)

    if args.max_scenes is not None:
        scenes = dict(list(scenes.items())[: args.max_scenes])

    print(f"按 4 级目录归类 (scripted_raw)，共找到 {len(scenes)} 个场景")

    for i, (scene_root, img_dirs) in enumerate(scenes.items(), 1):
        # json 将保存在 scene_root (第 4 级目录) 下
        json_path = scene_root / args.scene_json_name
        
        if json_path.is_file():
            print(f"[{i}/{len(scenes)}] 跳过已存在: {json_path}")
            continue

        # 找该场景下（所有子文件夹里的）所有 jpg
        jpgs = []
        for img_dir in img_dirs:
            jpgs.extend(
                p
                for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
                for p in img_dir.glob(ext)
            )
        
        if not jpgs:
            print(f"[{i}/{len(scenes)}] 场景 {scene_root} 没有图片，跳过")
            continue

        # 随机选最多 3 张代表图
        random.shuffle(jpgs)
        sample_jpgs = jpgs[:3]

        print(f"[{i}/{len(scenes)}] 处理场景: {scene_root}")
        print(f"  包含 {len(img_dirs)} 个子数据文件夹，随机采样 {len(sample_jpgs)} 张图调用 Gemini")

        all_objs: list[str] = []
        for rep_path in sample_jpgs:
            rep_img = str(rep_path)
            # print("  调用 Gemini:", rep_img) # 减少刷屏
            try:
                objs = get_gemini_objects(rep_img, api_key=args.gemini_api_key)
            except Exception as e:
                print(f"    [Error] Gemini 处理 {rep_path.name} 失败: {e}")
                continue
            
            if objs:
                all_objs.extend([o.strip() for o in objs if o and o.strip()])

        # 去重
        seen = set()
        merged_objs: list[str] = []
        for o in all_objs:
            if o not in seen:
                seen.add(o)
                merged_objs.append(o)

        if not merged_objs:
            print("  警告: Gemini 未检测到任何物体，不写入文件")
            continue

        # 确保目录存在
        scene_root.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"objects": merged_objs}, f, indent=2, ensure_ascii=False)
        print(f"  已保存: {json_path} (包含 {len(merged_objs)} 个物体)")


if __name__ == "__main__":
    main()