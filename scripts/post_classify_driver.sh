#!/bin/bash
# Wait for Stage 1 (classify per-label) to finish, then run bucket + finalize.
set -u
cd /public/home/lulucai/code/sam3

CLASSIFY_LOG=logs/classify_per_label.log
DRIVER_LOG=logs/post_classify_driver.log

echo "[driver] started at $(date)" >> "$DRIVER_LOG"

# Poll for completion marker
while true; do
    if grep -q "^\[classify-label\] done\." "$CLASSIFY_LOG" 2>/dev/null; then
        break
    fi
    sleep 60
done

echo "[driver] classify done at $(date), running bucket" >> "$DRIVER_LOG"

export DASHSCOPE_API_KEY="sk-76c66bb100d944b681bb3457c749c752"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=/tmp/mpl-$USER-postdriver
export MPLBACKEND=Agg

.venv/bin/python scripts/build_oxe_robot_label_registry.py bucket \
    --classify-mode per-label \
    --output-dir reports/oxe_robot_labels >> "$DRIVER_LOG" 2>&1
bucket_rc=$?
echo "[driver] bucket rc=$bucket_rc at $(date)" >> "$DRIVER_LOG"
if [ "$bucket_rc" -ne 0 ]; then
    echo "[driver] bucket failed, aborting before finalize" >> "$DRIVER_LOG"
    exit 1
fi

.venv/bin/python scripts/build_oxe_robot_label_registry.py finalize \
    --output-dir reports/oxe_robot_labels >> "$DRIVER_LOG" 2>&1
final_rc=$?
echo "[driver] finalize rc=$final_rc at $(date)" >> "$DRIVER_LOG"
echo "[driver] all done" >> "$DRIVER_LOG"
