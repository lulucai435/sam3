#!/usr/bin/env bash
# 遍历 LIBERO 目录下所有 .hdf5，对每个文件调用 run_libero_hdf5_gemini_sam3.py，
# 输出写入 $OUT/<hdf5 文件名不含后缀>/ 。
#
# 用法:
#   export GOOGLE_API_KEY=...
#   ./scripts/batch_libero_gemini_sam3.sh
#   OUT=/other/out LIBERO_ROOT=/other/libero ./scripts/batch_libero_gemini_sam3.sh
#   ./scripts/batch_libero_gemini_sam3.sh -- --skip-existing --gemini-sample-frames 3
#
# 多卡并行（示例：4 进程，绑定 GPU 0–3，每批最多同时跑 4 个 hdf5）:
#   PARALLEL=4 GPU_IDS=0,1,2,3 ./scripts/batch_libero_gemini_sam3.sh
#
# 环境变量:
#   OUT          输出根目录，默认 /share/250010208/hypernet/dataset/libero_90_segment
#   LIBERO_ROOT  含 .hdf5 的数据根目录，默认 .../libero_90_no_noops
#   SAM3_REPO    本仓库根目录，默认根据脚本位置推断
#   PYTHON       python 可执行文件，默认 python3
#   PARALLEL     同时跑几个 hdf5 任务；默认 1（顺序执行）
#   GPU_IDS      物理 GPU 编号，逗号分隔，按任务序号轮询绑定，例如 0,1,2,3；默认 0
#                每个子进程会设置 CUDA_VISIBLE_DEVICES 为其中一个 ID（进程内即为 cuda:0）
#   SAM3_CKPT    默认 /data/lulucai/code/sam3/weights/sam3.pt；若文件存在则附加
#                --no-download-hf --checkpoint-path。可 export SAM3_CKPT=其他路径 覆盖。
#   GOOGLE_API_KEY  必设其一：export GOOGLE_API_KEY=...（勿写进本脚本或提交 Git），
#                或 export GOOGLE_API_KEY_FILE=/path/to/key.txt（文件一行密钥，chmod 600）

set -u

OUT="${OUT:-/share/250010208/hypernet/dataset/libero_90_segment}"
LIBERO_ROOT="${LIBERO_ROOT:-/share/250010208/hypernet/dataset/libero_90_no_noops}"
SAM3_CKPT="${SAM3_CKPT:-/data/lulucai/code/sam3/weights/sam3.pt}"
PYTHON="${PYTHON:-python3}"
PARALLEL="${PARALLEL:-1}"
GPU_IDS="${GPU_IDS:-0}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="${SAM3_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"

EXTRA=()
if [[ "${1:-}" == -- ]]; then
  shift
  EXTRA=("$@")
fi

CKPT_ARGS=()
if [[ -n "${SAM3_CKPT:-}" ]]; then
  if [[ -f "$SAM3_CKPT" ]]; then
    CKPT_ARGS=(--no-download-hf --checkpoint-path "$SAM3_CKPT")
  else
    echo "警告: SAM3_CKPT 已设置但文件不存在: $SAM3_CKPT（将仍尝试从 HF 下载）" >&2
  fi
fi

LIST_PY="$REPO/scripts/list_libero_hdf5.py"
RUN_PY="$REPO/scripts/run_libero_hdf5_gemini_sam3.py"

if [[ ! -f "$LIST_PY" || ! -f "$RUN_PY" ]]; then
  echo "找不到脚本: LIST_PY=$LIST_PY RUN_PY=$RUN_PY" >&2
  exit 1
fi

IFS=',' read -r -a _GPU_RAW <<< "$GPU_IDS"
GPUS=()
for g in "${_GPU_RAW[@]}"; do
  g="${g//[[:space:]]/}"
  [[ -n "$g" ]] && GPUS+=("$g")
done
((${#GPUS[@]} > 0)) || GPUS=(0)

if ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL 必须为正整数，当前: $PARALLEL" >&2
  exit 1
fi

if [[ -z "${GOOGLE_API_KEY:-}" && -n "${GOOGLE_API_KEY_FILE:-}" && -f "$GOOGLE_API_KEY_FILE" ]]; then
  GOOGLE_API_KEY=$(tr -d '\n\r' <"$GOOGLE_API_KEY_FILE")
  export GOOGLE_API_KEY
fi
if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "错误: 未设置 GOOGLE_API_KEY。请 export GOOGLE_API_KEY=... 或 export GOOGLE_API_KEY_FILE=密钥文件路径（勿把密钥写入仓库）。" >&2
  exit 1
fi

mkdir -p "$OUT"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

n=0
job_idx=0
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  if [[ ! -f "$f" ]]; then
    echo "跳过（非文件）: $f" >&2
    continue
  fi
  stem=$(basename "$f" .hdf5)
  mkdir -p "$OUT/$stem"
  ((++n))
  gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
  echo "=== [${n}] $stem  (GPU $gpu, parallel batch) ==="

  (
    set +e
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" "$RUN_PY" \
      --hdf5 "$f" \
      --output-dir "$OUT/$stem" \
      --use-gemini \
      --frame-stride 1 \
      --skip-existing \
      "${CKPT_ARGS[@]}" \
      "${EXTRA[@]}"
    ec=$?
    if (( ec != 0 )); then
      echo "FAILED (exit $ec): $f" >&2
      touch "$tmpdir/fail_${job_idx}"
    fi
    echo "=== [done] $stem ==="
    exit "$ec"
  ) &

  ((job_idx++))
  if (( PARALLEL > 1 && job_idx % PARALLEL == 0 )); then
    wait
  fi
done < <("$PYTHON" "$LIST_PY" "$LIBERO_ROOT")

wait

failed=0
shopt -s nullglob
for x in "$tmpdir"/fail_*; do
  ((failed++)) || true
done
shopt -u nullglob

echo "完成。hdf5 数: $n，失败任务数: $failed，输出根目录: $OUT，PARALLEL=$PARALLEL GPU_IDS=${GPUS[*]}"
exit $((failed > 0 ? 1 : 0))
