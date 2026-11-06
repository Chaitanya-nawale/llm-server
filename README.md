# llm-server

A repository for serving Large Language Models (LLMs) with high-throughput inference using [vLLM](https://github.com/vllm-project/vllm) and SLURM workload management on HPC clusters (e.g., RWTH Aachen University HPC). It exposes an OpenAI-compatible API endpoint for easy integration with tools, SDKs, and applications.

---

## Installation & Setup

### Step 1. Install uv

Install [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and project manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> [!NOTE]
> Make sure `~/.local/bin` is in your `PATH`. If not already set, add it:
>
> ```bash
> export PATH="${HOME}/.local/bin:${PATH}"
> ```

---

### Step 2. Clone your project

Clone the repository to your remote cluster environment and navigate into the project directory:

```bash
git clone https://github.com/Chaitanya-nawale/llm-server.git
cd llm-server
```

_(Replace with `YOUR_REPOSITORY` if using a fork or custom remote URL)_

---

### Step 3. Install dependencies

Install the pinned dependencies into an isolated virtual environment using `uv`:

```bash
uv sync --frozen
```

This will automatically create `.venv` and install the exact versions specified in `uv.lock` (including `vllm` and `openai`).

---

### Step 4. Submit the SLURM job

Make sure a `logs` directory exists for job stdout/stderr:

```bash
mkdir -p logs
```

Submit the desired SLURM batch job depending on the model you want to host:

- **For Qwen 3.5 (4B):**

  ```bash
  sbatch serve_qwen_3-5.sh
  ```

- **For Qwen 3.8 (27B - FP8 quantized):**
  ```bash
  sbatch serve_qwen_3-8_FP8.sh
  ```

#### Monitor job and find allocated node

Check the status of your running job:

```bash
squeue --me
```

To find the specific compute node hostname where your job was allocated:

```bash
squeue -u $USER -o "%.10i %.9P %.20j %.8T %.10M %.15N"
```

You can also inspect the generated log file in `logs/`:

```bash
tail -f logs/run_Qwen*.log
```

Look for the line:

```text
Node: <hostnode>
```

_(e.g., `linux-gpu01` or similar compute node ID)_

---

## Port Forwarding via Terminal

Compute nodes on the cluster are behind a firewall and cannot be accessed directly from your local machine. Create an SSH tunnel via the login node to forward port `8000` to your local machine:

```bash
ssh -L 8000:<hostnode>:8000 <username>@login23-g-1.hpc.itc.rwth-aachen.de
```

### Parameter details:

- `8000` (first): Local port on your machine.
- `<hostnode>`: The compute node hostname running the job (e.g. `nxs-gpu01` or the node reported by `squeue`).
- `8000` (second): Remote port on the compute node where vLLM is listening.
- `<username>`: Your RWTH HPC username (e.g. `ab123456`).
- `login23-g-1.hpc.itc.rwth-aachen.de`: RWTH Aachen HPC login node.

Keep this terminal window open while interacting with the server.

---

## Testing with cURL

Once the SSH port forward is established, you can query the vLLM OpenAI-compatible server from your local terminal.

### 1. Check Available Models

```bash
curl http://localhost:8000/v1/models
```

Expected response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3.5-4B",
      "object": "model",
      ...
    }
  ]
}
```

---

### 2. Test Chat Completions

#### For `Qwen/Qwen3.5-4B`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello! Give a brief introduction of yourself."
      }
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

#### For `Qwen/Qwen3.8-27B-FP8`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.8-27B-FP8",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Explain what an LLM is in two sentences."
      }
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

---

## Python Client Usage (Optional)

You can also query the running server using the standard `openai` Python SDK:

```python
from openai import OpenAI

# Point client to the local forwarded port
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM does not require an API key by default
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",  # Or "Qwen/Qwen3.8-27B-FP8"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a fun science fact!"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

---

## Useful SLURM Commands

- **Check job queue:** `squeue --me`
- **Cancel a running job:** `scancel <JOB_ID>`
- **View job output in real time:** `tail -f logs/run_*.log`
- **View error logs:** `tail -f logs/run_*.err`
