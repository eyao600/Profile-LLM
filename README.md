# Profile-LLM

Standalone, minimal LLM profiling workflow extracted from `gpu-power-experiments/examples/llm`.

This duplicate keeps:
- PyTorch split JSON tracing (`prefill`, `decode_i`)
- Nsight Compute (NCU) per-range CSV profiling
- Optional post-processing from `.pt.trace.json` into `*.csv` and `*_summary.csv`
- Optional summary plotting for `aten::mm` memory accesses (default: enabled)

This duplicate intentionally removes:
- Power profiling
- Docker workflow

## Project setup (uv)

Run this once on a new machine/user account.

```bash
# 1) Clone and enter the project
git clone <your-repo-url>
cd Profile-LLM

# 2) Install project dependencies into a local .venv
uv sync

# 3) Verify required system tools
uv --version
ncu --version

# 4) Verify CUDA is visible from Python
uv run python -c "import torch; print('cuda_available=', torch.cuda.is_available())"
```

If `ncu` is not found, add CUDA tools to your `PATH` (example):

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

If model download/auth fails, log in to Hugging Face:

```bash
export HF_TOKEN="<your_token>"
huggingface-cli login --token "$HF_TOKEN"
```

## First run (sanity check)

For a minimal single-run check on a fresh setup:

```bash
uv run run_workflow.py \
  --prefill-lens 32 \
  --decode-len 4 \
  --decode-sweep-prefill 32 \
  --decode-sweep-lengths 4
```

## One-command run

From this directory:

```bash
uv run run_workflow.py
```

Default run matrix (union of both sweeps):
- Prompt sweep: `prefill in {32,64,128,256,512,1024}`, `decode=4`
- Decode sweep: `prefill=128`, `decode in {4,8,16,32,64,128}`

Each run writes to:
- `logs/<model_basename>/prompt_<prefill>_predict_<decode>/`

For example:
- `logs/Llama-2-7B-fp16/prompt_32_predict_4/`
- `logs/Llama-2-7B-fp16/prompt_128_predict_128/`

Default aggregate outputs:
- `output/<model_basename>/total_mem_access_by_prompt_len_predict_4.png`
- `output/<model_basename>/total_mem_access_by_prompt_len_predict_4.csv`
- `output/<model_basename>/total_mem_access_by_decode_len_prefill_128.png`
- `output/<model_basename>/total_mem_access_by_decode_len_prefill_128.csv`

## Metrics used in plots

`plot_mm_total_access.py` plots total memory access bytes for `aten::mm` kernels.

- DRAM bytes:
  - `dram__sectors_read.sum + dram__sectors_write.sum`
  - converted with `32 bytes/sector`
- L2 bytes:
  - `lts__t_sectors_op_read.sum + lts__t_sectors_op_write.sum`
  - converted with `32 bytes/sector`
- SHMEM bytes:
  - `sm__sass_data_bytes_mem_shared_op_ld.sum`
  - `sm__sass_data_bytes_mem_shared_op_ldsm.sum`
  - `sm__sass_data_bytes_mem_shared_op_st.sum`

Totals are aggregated over:
- `prefill` + decode phases `decode_0..decode_(N-2)` for decode length `N`

## How to add new metrics

To add metrics end-to-end:

1. Add metric names to NCU collection list in [`run_workflow.py`](/app/nanocad/projects/personal/yaoe888/Profile-LLM/run_workflow.py):
   - update `NCU_METRICS`
2. Consume them in [`plot_mm_total_access.py`](/app/nanocad/projects/personal/yaoe888/Profile-LLM/plot_mm_total_access.py):
   - add metric constants
   - if needed for long-format NCU CSVs, include them in `REQUIRED_METRICS`
   - update `_aggregate_phase_mm_bytes()` formulas
3. If you want them in outputs/plots:
   - add columns in `_write_prompt_csv()` / `_write_decode_csv()`
   - update `_write_stacked_plot()` series and labels

Note:
- NCU CSV parsing supports both wide-format and long-format files automatically.

## Post-processing toggle

Post-processing is enabled by default.

Disable it:

```bash
uv run run_workflow.py --no-postprocess
```

## Plot toggle

Plotting is enabled by default (after profiling + post-processing).

Disable plotting:

```bash
uv run run_workflow.py --no-plot
```

## Key outputs per run

- `json/*.pt.trace.json` (split PyTorch traces)
- `ncu/<model>_<device>_prefill.csv`
- `ncu/<model>_<device>_decode_<i>.csv`
- If post-processing is enabled:
  - `prefill.csv`, `decode_i.csv` (linked CPU-op/input-dims/kernel rows)
  - `prefill_summary.csv`, `decode_i_summary.csv`
- If plotting is enabled:
  - prompt-length sweep stacked plot/CSV in `output/<model_basename>/`
  - decode-length sweep stacked plot/CSV in `output/<model_basename>/`
  - if a decode-length run directory is missing, plotting reuses an available
    longer decode run for the same prefill and aggregates phases up to the
    requested decode length

## Optional overrides

Examples:

```bash
# Change model and GPU
uv run run_workflow.py --model meta-llama/Llama-3.2-1B --device-id 1

# Change decode-sweep settings
uv run run_workflow.py --decode-sweep-prefill 128 --decode-sweep-lengths 4 16 64 128

# Restrict to a subset of prompt sizes
uv run run_workflow.py --prefill-lens 64 128
```

## Change model

Set `--model` to any Hugging Face model ID compatible with `transformers`.

```bash
uv run run_workflow.py --model meta-llama/Llama-3.2-1B
```

Optional:
- use `--cache-dir <path>` to control model cache location
- use `--device-id <gpu_index>` to select GPU

For gated/private models, authenticate first:

```bash
export HF_TOKEN="<your_token>"
huggingface-cli login --token "$HF_TOKEN"
```

## Requirements

- NVIDIA GPU + CUDA driver/toolkit
- `ncu` available in `PATH`
- `uv` installed
- Hugging Face auth (for gated/private models):

```bash
export HF_TOKEN="<your_token>"
huggingface-cli login --token "$HF_TOKEN"
```
