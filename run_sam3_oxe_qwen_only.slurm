#!/bin/bash
#SBATCH --job-name=sam3_oxe_qwen
#SBATCH --chdir=/public/home/lulucai/code/sam3
#SBATCH --output=/public/home/lulucai/code/sam3/logs/sam3_oxe_qwen_%j.out
#SBATCH --error=/public/home/lulucai/code/sam3/logs/sam3_oxe_qwen_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=32G

# Qwen-VL 替代 Gemini，无需 clash 代理，直连 DashScope API。
# 与 run_sam3_oxe_sam3_only.slurm 配合使用（写 _GEMINI_DONE + metadata.json）。
# 用法：
#   sbatch run_sam3_oxe_qwen_only.slurm

set -euo pipefail

PROJ=/public/home/lulucai/code/sam3
cd "$PROJ"
mkdir -p logs

PY="$PROJ/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY 不存在。" >&2; exit 1
fi

source /public/home/lulucai/.sam3_secrets

# DashScope 直连，不走代理
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY

echo "=== 环境 ==="
echo "Python: $PY ($($PY --version))"
echo "Hostname: $(hostname)"
echo "============"

$PY scripts/batch_oxe_qwen_only.py \
  --dataset-root /public/home/lisunhku/tensorflow_datasets \
  --output-root  /public/home/lulucai/data/oxe_seg \
  --num-workers 32 \
  --model qwen3-vl-flash \
  --num-frames 5 \
  --max-consecutive-failures 50 \
  --datasets fractal20220817_data,kuka,bridge,taco_play,jaco_play,berkeley_cable_routing,roboturk,viola,berkeley_autolab_ur5,toto,language_table,stanford_hydra_dataset_converted_externally_to_rlds,austin_buds_dataset_converted_externally_to_rlds,nyu_franka_play_dataset_converted_externally_to_rlds,furniture_bench_dataset_converted_externally_to_rlds,ucsd_kitchen_dataset_converted_externally_to_rlds,austin_sailor_dataset_converted_externally_to_rlds,austin_sirius_dataset_converted_externally_to_rlds,dlr_edan_shared_control_converted_externally_to_rlds,iamlab_cmu_pickup_insert_converted_externally_to_rlds,utaustin_mutex,berkeley_fanuc_manipulation,cmu_stretch,bc_z,fmb,dobbe,droid

echo "Qwen stage done -> /public/home/lulucai/data/oxe_seg"
