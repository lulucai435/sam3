#!/bin/bash
#SBATCH --job-name=sam3_oxe_sam3
#SBATCH --chdir=/public/home/lulucai/code/sam3
#SBATCH --output=/public/home/lulucai/code/sam3/logs/sam3_oxe_sam3_%j.out
#SBATCH --error=/public/home/lulucai/code/sam3/logs/sam3_oxe_sam3_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=120:00:00
#SBATCH --mem=120G

# Stage 2：读 metadata.json（_GEMINI_DONE 的 shard）只跑 SAM3，不调 Gemini。
# 不需要 clash 代理，纯 GPU 计算。
# 先确保 run_sam3_oxe_gemini_only.slurm 跑完再提交本脚本。
# 用法（mgmt 节点）：
#   sbatch run_sam3_oxe_sam3_only.slurm

set -euo pipefail

PROJ=/public/home/lulucai/code/sam3
cd "$PROJ"
mkdir -p logs

PY="$PROJ/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY 不存在。" >&2; exit 1
fi

source /public/home/lulucai/.sam3_secrets

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== 环境 ==="
echo "Python: $PY ($($PY --version))"
echo "CUDA avail: $($PY -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count:  $($PY -c 'import torch; print(torch.cuda.device_count())')"
echo "Hostname:   $(hostname)"
echo "============"

NUM_GPUS=$($PY -c 'import torch; print(torch.cuda.device_count())')
GPU_IDS=$($PY -c "print(','.join(str(i) for i in range($NUM_GPUS)))")
echo "Using GPU ids: ${GPU_IDS} (2 workers/gpu=16 total, sam3-only-live, polling every 120s)"

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
  --confidence 0.5 \
  --save-vis \
  --sam3-only \
  --poll-interval 120 \
  --datasets fractal20220817_data,kuka,bridge,taco_play,jaco_play,berkeley_cable_routing,roboturk,viola,berkeley_autolab_ur5,toto,language_table,stanford_hydra_dataset_converted_externally_to_rlds,austin_buds_dataset_converted_externally_to_rlds,nyu_franka_play_dataset_converted_externally_to_rlds,furniture_bench_dataset_converted_externally_to_rlds,ucsd_kitchen_dataset_converted_externally_to_rlds,austin_sailor_dataset_converted_externally_to_rlds,austin_sirius_dataset_converted_externally_to_rlds,dlr_edan_shared_control_converted_externally_to_rlds,iamlab_cmu_pickup_insert_converted_externally_to_rlds,utaustin_mutex,berkeley_fanuc_manipulation,cmu_stretch,bc_z,fmb,dobbe,droid

echo "SAM3 stage done -> /public/home/lulucai/data/oxe_seg"
