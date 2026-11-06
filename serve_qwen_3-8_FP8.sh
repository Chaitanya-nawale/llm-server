#!/usr/bin/zsh

### Job name
#SBATCH --job-name=Qwen3.8

### File path where STDOUT and STDERR will be written, the %j is the job id
#SBATCH --output=logs/run_Qwen3.8_%j.log
#SBATCH --error=logs/run_Qwen3.8_%j.err

### Request time and memory
#SBATCH --time=02:50:00
#SBATCH --mem=32G

### Set Queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs

echo "=== Job Information ==="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Submit Dir:    ${SLURM_SUBMIT_DIR}"
echo "Start time:    $(date)"
echo "======================="


# Run from the directory where sbatch was submitted.
cd "${SLURM_SUBMIT_DIR}"

# Store Hugging Face model weights outside the project directory.
export HF_HOME="$HPCWORK/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$HF_HUB_CACHE"

echo "Hugging Face cache: $HF_HUB_CACHE"

# Inspect the allocated GPU.
nvidia-smi

# Create log directory before submitting with sbatch as well.
mkdir -p logs

# Remove Intel compiler environment
module unload intel/2024a

# Load GCC
module load GCC/13.3.0

# Load CUDA
module load CUDA/13.1.0

# CUDA configuration
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH

# Explicitly select GCC
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)

export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH

echo "=== Compiler configuration ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC"
echo "CXX=$CXX"

which nvcc
which gcc
which g++

gcc --version
nvcc --version

# Job-specific compilation caches
export TORCHINDUCTOR_CACHE_DIR="${HPCWORK}/torchinductor/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${HPCWORK}/triton/${SLURM_JOB_ID}"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR"

# UV local bin
export PATH="${HOME}/.local/bin:${PATH}"

# Verify the project's Python environment and dependencies.
source .venv/bin/activate

uv run python -c \
  "import torch, vllm; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('vLLM:', vllm.__version__)"

# Start the OpenAI-compatible vLLM API server.
exec uv run vllm serve Qwen/Qwen3.8-27B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
