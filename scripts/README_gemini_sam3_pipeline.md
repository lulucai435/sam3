# Gemini + SAM3 Pipeline

输入一张图片，使用 **Gemini API** 识别图中的物体（文本列表），再用 **SAM3** 对每个物体做实例分割。

## 依赖

```bash
# 方式一：安装可选依赖（推荐）
pip install -e ".[gemini-sam3]"

# 方式二：仅安装 Gemini SDK（二选一）
pip install google-genai
# 或
pip install google-generativeai
```

需要 **PyTorch** 和 **SAM3** 已安装（见项目根目录 README）。

## 环境变量

- `GOOGLE_API_KEY`：Gemini API Key（在 [Google AI Studio](https://aistudio.google.com/apikey) 申请）

## 用法

```bash
# 基本用法（输出到终端 + 可选保存可视化）
python scripts/gemini_sam3_pipeline.py --image path/to/image.jpg

# 指定输出目录（会保存可视化图与结果摘要 JSON）
python scripts/gemini_sam3_pipeline.py --image path/to/image.jpg --output-dir out/

# 单独调试 SAM3、不调 Gemini（不消耗 API）：用脚本内 Examples 默认物体列表
python scripts/gemini_sam3_pipeline.py --image path/to/image.jpg --skip-gemini --output-dir out/

# 单独调试 SAM3 并自定义物体列表（逗号分隔）
python scripts/gemini_sam3_pipeline.py --image path/to/image.jpg --skip-gemini --objects "person,car,table,bottle"

# 指定 Gemini API Key、置信度与设备
python scripts/gemini_sam3_pipeline.py --image path/to/image.jpg \
  --gemini-api-key YOUR_KEY \
  --confidence 0.4 \
  --device cuda

# 使用本地 checkpoint（避免 HuggingFace 访问问题）
python scripts/gemini_sam3_pipeline.py --image path/to/image.jpg \
  --checkpoint-path /path/to/sam3.pt \
  --no-download-hf
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--image`, `-i` | 输入图片路径（必填） |
| `--output-dir`, `-o` | 输出目录；不指定则只打印不写文件 |
| `--gemini-api-key` | Gemini API Key（也可用环境变量 `GOOGLE_API_KEY`） |
| `--confidence` | SAM3 分割置信度阈值，默认 0.5 |
| `--device` | `cuda` 或 `cpu` |
| `--no-vis` | 不保存可视化图 |
| `--skip-gemini` | 不调用 Gemini，用默认或 `--objects` 的物体列表，仅跑 SAM3（省 API） |
| `--objects` | 与 `--skip-gemini` 同用：物体列表，逗号分隔，如 `person,car,table` |
| `--checkpoint-path` | 本地 SAM3 checkpoint 路径（.pt 文件），避免从 HuggingFace 下载 |
| `--no-download-hf` | 不尝试从 HuggingFace 下载 checkpoint（需配合 `--checkpoint-path`） |

## 流程简述

1. **Gemini**：读入图片，请求模型输出图中物体的 JSON 列表（英文名词）。
2. **SAM3**：加载 SAM3 图像模型与 Processor，对原图做一次 `set_image`，再对每个物体名称调用 `set_text_prompt`，得到该物体的 masks/boxes/scores。
3. **输出**：在终端打印每个物体的实例数；若指定 `--output-dir`，会保存叠加了所有 mask 的可视化图与摘要 JSON。

## 输出文件（当使用 `--output-dir` 时）

- `{图片名}_gemini_sam3_vis.png`：原图 + 各物体分割 mask 叠加的可视化。
- `{图片名}_gemini_sam3_summary.json`：物体列表及每个物体检测到的实例数量。

## 常见问题

### 1. HuggingFace 访问受限（403 Forbidden）

如果遇到 `GatedRepoError: 403 Client Error`，说明 `facebook/sam3` 是受限仓库，需要：

**方案 A：申请访问权限**
1. 访问 https://huggingface.co/facebook/sam3
2. 点击 "Request access" 申请访问权限
3. 等待批准后，登录 HuggingFace：
   ```bash
   huggingface-cli login
   ```

**方案 B：使用本地 checkpoint（推荐）**
如果你已有 SAM3 checkpoint 文件（`sam3.pt`），直接指定路径：
```bash
python scripts/gemini_sam3_pipeline.py --image path/to/image.jpg \
  --checkpoint-path /path/to/sam3.pt \
  --no-download-hf
```

### 2. ModuleNotFoundError: No module named 'decord'

如果遇到 `decord` 缺失错误，安装它：
```bash
pip install decord
# 或安装 notebooks 依赖组（包含 decord）
pip install -e ".[notebooks]"
```

注意：`decord` 主要用于视频处理，图像 pipeline 通常不需要，但如果某个间接导入触发了它，安装即可。

---

## Bridge 数据集批量处理

对 `bridge/bridgedatav2_image_paths.txt` 中列出的所有目录、每个目录下全部 jpg 做 SAM3 分割（不调 Gemini，用固定物体列表）：

```bash
# 在项目根目录执行
python scripts/run_bridge_batch_segment.py \
  --list-file bridge/bridgedatav2_image_paths.txt \
  --base-dir /path/to/data \
  --output-dir out_bridge_segments \
  --skip-existing
```

- **--base-dir**：列表里每行路径相对此目录，例如列表为 `raw/bridge_data_v2/.../images0`，则实际目录为 `<base-dir>/raw/bridge_data_v2/.../images0`。
- **--output-dir**：输出根目录；每个输入目录对应子目录 `output-dir/<dir_index>/`，内含每张图的 `*_gemini_sam3_segments.npz` 与 `*_gemini_sam3_summary.json`。
- **--skip-existing**：若某张图已有对应的 segments npz 则跳过。
- **--max-dirs N** / **--max-images-per-dir M**：试跑时限制目录数或每目录图片数。
- **--checkpoint-path** / **--no-download-hf**：同单图 pipeline。

结束后会生成 `output-dir/manifest.json`，记录每个目录路径及处理的图片列表。
