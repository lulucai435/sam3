#!/bin/bash
#SBATCH --job-name=sam3_oxe_lt
#SBATCH --chdir=/public/home/lulucai/code/sam3
#SBATCH --output=/public/home/lulucai/code/sam3/logs/sam3_oxe_lt_%j.out
#SBATCH --error=/public/home/lulucai/code/sam3/logs/sam3_oxe_lt_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=120G

# 单独跑 language_table —— 原 27 列表里它在 discovery 阶段被旧 _SHARD_RE
# 过滤掉（文件名缺 -of-MMMMM 后缀，见 batch_oxe_full_dataset.py 修复）。
# 改 dataset 列表为单个，复用 batch_oxe_fast.py 的全部加速与 claim_lock。
# 用法（在 mgmt 节点）：
#   sbatch run_sam3_oxe_language_table.slurm

set -euo pipefail

PROJ=/public/home/lulucai/code/sam3
cd "$PROJ"
mkdir -p logs

PY="$PROJ/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY 不存在。先在 $PROJ 跑过 uv sync / uv pip install -e \".[gemini-sam3]\"。" >&2
  exit 1
fi

if [ -z "${GOOGLE_API_KEY:-}" ]; then
  export GOOGLE_API_KEY=AIzaSyCPlpLbHXadTWIaIu_4SEWNa2yDmqR-q78
fi
if [ -z "${SCT_KEY:-}" ]; then
  export SCT_KEY=SCT350071TJOoEoItfmHndY5peyRCcCrDZ
fi

# ---- clash 代理 ----
CLASH_BIN=/public/home/lulucai/vpn/clash/clash-linux-amd64-v1.10.0
CLASH_CFG=/public/home/lulucai/vpn/clash/BitzNet_us_only.yaml
CLASH_DIR=/tmp/clash-${SLURM_JOB_ID:-$$}
mkdir -p "$CLASH_DIR"
cp "$CLASH_CFG" "$CLASH_DIR/"
[ -f /public/home/lulucai/vpn/clash/cache.db ] && cp /public/home/lulucai/vpn/clash/cache.db "$CLASH_DIR/" || true
if [ -f /public/home/lulucai/vpn/clash/Country.mmdb ]; then
  cp /public/home/lulucai/vpn/clash/Country.mmdb "$CLASH_DIR/"
elif [ -f ~/.config/clash/Country.mmdb ]; then
  cp ~/.config/clash/Country.mmdb "$CLASH_DIR/"
else
  echo "WARN: Country.mmdb 找不到，clash 可能会卡在下载阶段" >&2
fi

echo "Starting clash in $CLASH_DIR ..."
"$CLASH_BIN" -f "$CLASH_DIR/$(basename "$CLASH_CFG")" -d "$CLASH_DIR" \
  > "$CLASH_DIR/clash.log" 2>&1 &
CLASH_PID=$!
cleanup() {
  echo "Killing clash pid=$CLASH_PID"
  kill "$CLASH_PID" 2>/dev/null || true
  wait "$CLASH_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PROXY_URL=http://127.0.0.1:7890
ok=0
for i in $(seq 1 30); do
  if curl -s --max-time 3 --proxy "$PROXY_URL" -o /dev/null -w "%{http_code}" \
       https://generativelanguage.googleapis.com/ 2>/dev/null | grep -qE "^(200|301|302|401|403|404)$"; then
    ok=1; break
  fi
  sleep 2
done
if [ "$ok" -ne 1 ]; then
  echo "ERROR: clash 代理未能在 60s 内连通 Gemini endpoint。clash 日志：" >&2
  tail -50 "$CLASH_DIR/clash.log" >&2 || true
  exit 1
fi
echo "Clash proxy ready ($PROXY_URL)"

export HTTPS_PROXY="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export ALL_PROXY="$PROXY_URL"
export NO_PROXY=localhost,127.0.0.1,::1,huggingface.co
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== 环境 ==="
echo "Python:      $PY"
$PY --version
echo "CUDA avail:  $($PY -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count:   $($PY -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
echo "Hostname:    $(hostname)"
echo "============"

NUM_GPUS=$($PY -c 'import torch; print(torch.cuda.device_count())')
GPU_IDS=$($PY -c "print(','.join(str(i) for i in range($NUM_GPUS)))")
echo "Using GPU ids: ${GPU_IDS} (workers/gpu=2, lookahead=2, B=8, SAM3 bf16)"

$PY scripts/batch_oxe_fast.py \
  --dataset-root /public/home/lisunhku/tensorflow_datasets \
  --output-root  /public/home/lulucai/data/oxe_seg \
  --gpus "${GPU_IDS}" \
  --workers-per-gpu 2 \
  --gemini-lookahead 2 \
  --batch-size 8 \
  --max-objects 10 \
  --segment-mode every_n --segment-every-n 10 \
  --num-gemini-frames 5 \
  --gemini-model gemini-2.5-flash-lite \
  --confidence 0.5 \
  --save-vis \
  --datasets language_table

echo "Done -> /public/home/lulucai/data/oxe_seg/language_table"
