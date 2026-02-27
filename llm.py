#!/usr/bin/env python3
"""Run and profile a HuggingFace causal LLM with synthetic prompts.

This script supports:
- Split PyTorch profiler traces (`prefill`, `decode_i`) via `-json --split_pt_traces`
- NVTX ranges suitable for Nsight Compute filtering (`prefill`, `decode_i`)

Power profiling is intentionally removed in this standalone workflow.
"""

import argparse
import os
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield
from datetime import datetime
from pathlib import Path
from time import sleep, time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _nvtx_range(message: str):
    nvtx = getattr(torch.cuda, "nvtx", None)
    if nvtx is None or not hasattr(nvtx, "range"):
        return nullcontext()
    try:
        return nvtx.range(message)
    except Exception:
        return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device_id", type=int, default=0, help="GPU device ID.")
    parser.add_argument(
        "-delay",
        "--delay",
        type=int,
        default=500,
        help="Delay in milliseconds before and after the profiled section.",
    )
    parser.add_argument("-r", "--repeat", type=int, default=1, help="Inference repeats.")
    parser.add_argument(
        "-json",
        "--do_json_profile",
        default=False,
        action="store_true",
        help="Enable PyTorch profiler JSON traces.",
    )
    parser.add_argument(
        "--split_pt_traces",
        default=False,
        action="store_true",
        help="Emit one PyTorch trace step for prefill and each decode forward.",
    )
    parser.add_argument("--model", type=str, required=True, help="HF model name/path.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Model/tokenizer cache directory.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory where trace outputs are written.",
    )
    parser.add_argument(
        "-ncu",
        "--do_ncu_profile",
        default=False,
        action="store_true",
        help="Disable preheat + time guard for NCU-driven runs.",
    )
    parser.add_argument(
        "--max_trace_time",
        type=int,
        default=None,
        help="Maximum non-NCU runtime (seconds).",
    )
    parser.add_argument(
        "--prefill_len",
        type=int,
        default=128,
        help="Synthetic prompt length.",
    )
    parser.add_argument(
        "--decode_len",
        type=int,
        default=0,
        help=(
            "Tokens to decode after prefill. Token 1 is selected from prefill logits; "
            "remaining tokens are decoded with KV cache."
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def run_prefill_and_decode(model, prompt, decode_len: int, split_trace_profiler=None):
    with torch.profiler.record_function("prefill"), _nvtx_range("prefill"):
        out = model(**prompt, use_cache=True)

    if split_trace_profiler is not None:
        split_trace_profiler.step()

    if decode_len <= 0:
        return out

    past_key_values = getattr(out, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError(
            "Model forward did not return past_key_values; cannot run cached decoding."
        )

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    attention_mask = prompt.get("attention_mask")

    if attention_mask is not None:
        prefill_len = attention_mask.shape[1]
        full_attention_mask = torch.ones(
            (attention_mask.shape[0], prefill_len + decode_len),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        full_attention_mask[:, :prefill_len] = attention_mask
    else:
        prefill_len = prompt["input_ids"].shape[1]
        full_attention_mask = None

    cur_len = prefill_len
    for _ in range(1, decode_len):
        cur_len += 1
        decode_range = f"decode_{cur_len - prefill_len - 1}"
        with torch.profiler.record_function(decode_range), _nvtx_range(decode_range):
            out = model(
                input_ids=next_token,
                attention_mask=full_attention_mask[:, :cur_len]
                if full_attention_mask is not None
                else None,
                past_key_values=past_key_values,
                use_cache=True,
            )

        if split_trace_profiler is not None:
            split_trace_profiler.step()

        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return out


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available. This workflow expects an NVIDIA GPU.")
        device = torch.device("cpu")
        cuid = "cpu"
    else:
        cuid = f"cuda:{args.device_id}"
        device = torch.device(cuid)
        torch.cuda.set_device(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_basename = os.path.basename(args.model)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    cache_dir = (
        args.cache_dir
        if args.cache_dir is not None
        else f"{Path(__file__).parent.resolve()}/.cache"
    )

    tok = AutoTokenizer.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )

    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=cuid,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )

    rand_ids = torch.randint(0, tok.vocab_size, (args.prefill_len,), dtype=torch.long)
    input_ids = rand_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    prompt = {
        "input_ids": input_ids.to(cuid),
        "attention_mask": attention_mask.to(cuid),
    }

    if not args.do_ncu_profile and torch.cuda.is_available():
        for _ in range(10):
            _ = run_prefill_and_decode(model, prompt, args.decode_len)
        torch.cuda.synchronize(cuid)

    if args.do_json_profile:
        json_path = str(Path(args.log_dir) / "json")
        profiler_kwargs = dict(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=json_path,
                worker_name=f"{model_basename}_{args.device_id}_{timestamp}_json"
                + ("_split" if args.split_pt_traces else ""),
            ),
        )

        if args.split_pt_traces:
            steps_per_iter = max(args.decode_len, 1)
            profiler_kwargs["schedule"] = torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
                repeat=steps_per_iter * args.repeat,
            )

        profiler = torch.profiler.profile(**profiler_kwargs)
        profiler.start()
    else:
        profiler = None

    if args.delay > 0:
        sleep(args.delay / 1000)

    if not args.do_ncu_profile and args.max_trace_time is not None:
        start_time = time()
    else:
        start_time = None

    for _ in range(args.repeat):
        _ = run_prefill_and_decode(
            model,
            prompt,
            args.decode_len,
            split_trace_profiler=profiler if args.split_pt_traces else None,
        )

        if start_time is not None and time() - start_time >= args.max_trace_time:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize(cuid)

    if args.delay > 0:
        sleep(args.delay / 1000)

    if profiler is not None:
        profiler.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
