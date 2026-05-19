#!/bin/bash
#SBATCH --job-name=sam3_oxe_gemini
#SBATCH --chdir=/public/home/lulucai/code/sam3
#SBATCH --output=/public/home/lulucai/code/sam3/logs/sam3_oxe_gemini_%j.out
#SBATCH --error=/public/home/lulucai/code/sam3/logs/sam3_oxe_gemini_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=32G

# Stage 1：只跑 Gemini，写 metadata.json + _GEMINI_DONE，不加载 SAM3。
# CPU 节点即可，不占 GPU 资源。纯网络 IO，workers 可多开。
# 用法（mgmt 节点）：
#   sbatch run_sam3_oxe_gemini_only.slurm

set -euo pipefail

PROJ=/public/home/lulucai/code/sam3
cd "$PROJ"
mkdir -p logs

PY="$PROJ/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY 不存在。" >&2; exit 1
fi

if [ -z "${GOOGLE_API_KEY:-}" ]; then
  export GOOGLE_API_KEY=AIzaSyCPlpLbHXadTWIaIu_4SEWNa2yDmqR-q78
fi
if [ -z "${SCT_KEY:-}" ]; then
  export SCT_KEY=SCT350071TJOoEoItfmHndY5peyRCcCrDZ
fi

# ---- clash 代理 ----
CLASH_BIN=/public/home/lulucai/vpn/clash/clash-linux-amd64-v1.10.0
CLASH_CFG=/public/home/lulucai/vpn/clash/BitzNet.yaml
CLASH_DIR=/tmp/clash-${SLURM_JOB_ID:-$$}
mkdir -p "$CLASH_DIR"
cp "$CLASH_CFG" "$CLASH_DIR/"
[ -f /public/home/lulucai/vpn/clash/cache.db ] && cp /public/home/lulucai/vpn/clash/cache.db "$CLASH_DIR/" || true
if [ -f /public/home/lulucai/vpn/clash/Country.mmdb ]; then
  cp /public/home/lulucai/vpn/clash/Country.mmdb "$CLASH_DIR/"
elif [ -f ~/.config/clash/Country.mmdb ]; then
  cp ~/.config/clash/Country.mmdb "$CLASH_DIR/"
fi

echo "Starting clash in $CLASH_DIR ..."
"$CLASH_BIN" -f "$CLASH_DIR/$(basename "$CLASH_CFG")" -d "$CLASH_DIR" \
  > "$CLASH_DIR/clash.log" 2>&1 &
CLASH_PID=$!
cleanup() { kill "$CLASH_PID" 2>/dev/null || true; wait "$CLASH_PID" 2>/dev/null || true; }
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
[ "$ok" -ne 1 ] && { echo "ERROR: clash 代理未能连通" >&2; tail -50 "$CLASH_DIR/clash.log" >&2; exit 1; }
echo "Clash proxy ready ($PROXY_URL)"

export HTTPS_PROXY="$PROXY_URL" HTTP_PROXY="$PROXY_URL" ALL_PROXY="$PROXY_URL"
export NO_PROXY=localhost,127.0.0.1,::1,huggingface.co

echo "=== 环境 ==="
echo "Python: $PY ($($PY --version))"
echo "Hostname: $(hostname)"
echo "============"

# CPU 节点：用假 GPU ID "0" × N_WORKERS，workers-per-gpu=1 → N_WORKERS 个进程
# gemini-only 不加载 SAM3，CUDA_VISIBLE_DEVICES 被设置但不使用
N_WORKERS=8
GPU_IDS=$($PY -c "print(','.join(['0']*$N_WORKERS))")
echo "CPU mode: ${N_WORKERS} Gemini workers (no GPU, gemini-only)"
echo "Gemini 并发数 = ${N_WORKERS} workers × lookahead=2 = $((N_WORKERS*2)) 连接"

$PY scripts/batch_oxe_fast.py \
  --dataset-root /public/home/lisunhku/tensorflow_datasets \
  --output-root  /public/home/lulucai/data/oxe_seg \
  --gpus "${GPU_IDS}" \
  --workers-per-gpu 1 \
  --gemini-lookahead 2 \
  --num-gemini-frames 5 \
  --gemini-model gemini-2.5-flash-lite \
  --confidence 0.5 \
  --gemini-max-consecutive-failures 50 \
  --gemini-only \
  --datasets fractal20220817_data,kuka,bridge,taco_play,jaco_play,berkeley_cable_routing,roboturk,viola,berkeley_autolab_ur5,toto,language_table,stanford_hydra_dataset_converted_externally_to_rlds,austin_buds_dataset_converted_externally_to_rlds,nyu_franka_play_dataset_converted_externally_to_rlds,furniture_bench_dataset_converted_externally_to_rlds,ucsd_kitchen_dataset_converted_externally_to_rlds,austin_sailor_dataset_converted_externally_to_rlds,austin_sirius_dataset_converted_externally_to_rlds,dlr_edan_shared_control_converted_externally_to_rlds,iamlab_cmu_pickup_insert_converted_externally_to_rlds,utaustin_mutex,berkeley_fanuc_manipulation,cmu_stretch,bc_z,fmb,dobbe,droid

echo "Gemini stage done -> /public/home/lulucai/data/oxe_seg"
