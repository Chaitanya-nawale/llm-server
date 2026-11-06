#!/usr/bin/zsh

### Job name
#SBATCH --job-name=Qwen3.5-4B


### Log files
#SBATCH --output=logs/run_Qwen3.5-4B_%j.log
#SBATCH --error=logs/run_Qwen3.5-4B_%j.err

### Request time and memory
#SBATCH --time=00:50:00
#SBATCH --mem=16G

### Request resources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

set -euo pipefail

# Run from submission directory
cd "${SLURM_SUBMIT_DIR}"

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

echo "=== Job Information ==="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Submit Dir:    ${SLURM_SUBMIT_DIR}"
echo "Start time:    $(date)"
echo "======================="

# Hugging Face cache
: "${HPCWORK:?HPCWORK environment variable is not defined}"

export HF_HOME="${HPCWORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

mkdir -p "${HF_HUB_CACHE}"

echo "Hugging Face cache: ${HF_HUB_CACHE}"

# Inspect allocated GPU
nvidia-smi

# UV local bin
export PATH="${HOME}/.local/bin:${PATH}"

# Verify Python environment and dependencies
source .venv/bin/activate
uv run python -c \
  "import torch, vllm; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('vLLM:', vllm.__version__)"

# Start OpenAI-compatible vLLM API server
exec uv run vllm serve Qwen/Qwen3.5-4B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.80 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --gdn-prefill-backend triton
