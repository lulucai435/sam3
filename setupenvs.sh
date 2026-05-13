export HF_MIRROR=https://hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com
# HF_TOKEN: set in your local shell or a non-committed .env file.
# Never commit a real token to git.
# export HF_TOKEN=hf_xxx
export HF_HUB_OFFLINE=1
export HF_HOME=/data/hypernet/.cache/huggingface
export UV_CACHE_DIR=/data/hypernet/.cache/uv
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

export TOKENIZERS_PARALLELISM=true
export MUJOCO_GL=egl

sudo apt install ffmpeg libavcodec-extra -y
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
if [ -d /lib/x86_64-linux-gnu ]; then
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi