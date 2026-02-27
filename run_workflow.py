#!/usr/bin/env python3
"""One-command LLM profiling workflow (JSON + NCU + postprocess + plotting)."""

import argparse
import ast
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_PREFILL_LENS = [32, 64, 128, 256, 512, 1024]
DEFAULT_PROMPT_SWEEP_DECODE_LEN = 4
DEFAULT_DECODE_SWEEP_PREFILL = 128
DEFAULT_DECODE_SWEEP_LENS = [4, 8, 16, 32, 64, 128]

NCU_METRICS = [
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_aperture_device_op_write_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "sm__sass_data_bytes_mem_shared_op_ld.sum",
    "sm__sass_data_bytes_mem_shared_op_ldsm.sum",
    "sm__sass_data_bytes_mem_shared_op_st.sum",
    "sm__sass_data_bytes_mem_shared_op_ldgsts.sum",
    "sm__sass_data_bytes_mem_shared_op_ldgsts_cache_access.sum",
    "sm__sass_data_bytes_mem_shared_op_ldgsts_cache_bypass.sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="TheBloke/Llama-2-7B-fp16")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--json-repeats", type=int, default=1)
    parser.add_argument("--ncu-repeats", type=int, default=1)
    parser.add_argument("--delay-ms", type=int, default=500)
    parser.add_argument("--max-trace-time", type=int, default=300)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--prefill-lens", type=int, nargs="*", default=DEFAULT_PREFILL_LENS)
    parser.add_argument("--decode-len", type=int, default=DEFAULT_PROMPT_SWEEP_DECODE_LEN)
    parser.add_argument("--decode-sweep-prefill", type=int, default=DEFAULT_DECODE_SWEEP_PREFILL)
    parser.add_argument(
        "--decode-sweep-lengths",
        type=int,
        nargs="*",
        default=DEFAULT_DECODE_SWEEP_LENS,
        help="Decode lengths for fixed-prefill sweep.",
    )
    parser.add_argument(
        "--postprocess",
        dest="postprocess",
        action="store_true",
        help="Enable trace post-processing into CSV summaries (default).",
    )
    parser.add_argument(
        "--no-postprocess",
        dest="postprocess",
        action="store_false",
        help="Disable trace post-processing.",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Enable summary plots from postprocessed + NCU data (default).",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable summary plot generation.",
    )
    parser.set_defaults(postprocess=True)
    parser.set_defaults(plot=True)
    return parser.parse_args()


def run_cmd(cmd: Sequence[str], *, stdout_path: Optional[Path] = None, cwd: Optional[Path] = None) -> None:
    printable = " ".join(subprocess.list2cmdline([part]) for part in cmd)
    if stdout_path:
        print(f"$ {printable} > {stdout_path}")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=cwd, stdout=f, stderr=sys.stderr, check=True)
    else:
        print(f"$ {printable}")
        subprocess.run(cmd, cwd=cwd, check=True)


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found in PATH: {name}")


def llm_cmd_base(args: argparse.Namespace, *, prefill_len: int, decode_len: int, log_dir: Path, ncu: bool) -> List[str]:
    cmd = [
        "uv",
        "run",
        "llm.py",
        "-d",
        str(args.device_id),
        "-r",
        str(args.ncu_repeats if ncu else args.json_repeats),
        "--delay",
        str(args.delay_ms),
        "--model",
        args.model,
        "--log_dir",
        str(log_dir),
        "--prefill_len",
        str(prefill_len),
        "--decode_len",
        str(decode_len),
    ]
    if args.cache_dir:
        cmd.extend(["--cache_dir", args.cache_dir])
    if ncu:
        cmd.append("-ncu")
    else:
        cmd.extend(["--max_trace_time", str(args.max_trace_time), "-json", "--split_pt_traces"])
    return cmd


def run_json_profile(args: argparse.Namespace, *, prefill_len: int, decode_len: int, log_dir: Path, cwd: Path) -> None:
    cmd = llm_cmd_base(args, prefill_len=prefill_len, decode_len=decode_len, log_dir=log_dir, ncu=False)
    run_cmd(cmd, cwd=cwd)


def run_ncu_prefill(args: argparse.Namespace, *, prefill_len: int, log_dir: Path, model_basename: str, cwd: Path) -> None:
    metrics = ",".join(NCU_METRICS)
    out_path = log_dir / "ncu" / f"{model_basename}_{args.device_id}_prefill.csv"
    cmd = [
        "ncu",
        "--nvtx",
        "--nvtx-include",
        "prefill/",
        "--target-processes",
        "all",
        "--metrics",
        metrics,
        "--replay-mode",
        "application",
        "--disable-extra-suffixes",
        "--csv",
        "--page",
        "details",
        *llm_cmd_base(
            args,
            prefill_len=prefill_len,
            decode_len=0,
            log_dir=log_dir,
            ncu=True,
        ),
    ]
    run_cmd(cmd, stdout_path=out_path, cwd=cwd)


def run_ncu_decode(
    args: argparse.Namespace,
    *,
    prefill_len: int,
    decode_index: int,
    log_dir: Path,
    model_basename: str,
    cwd: Path,
) -> None:
    metrics = ",".join(NCU_METRICS)
    out_path = log_dir / "ncu" / f"{model_basename}_{args.device_id}_decode_{decode_index}.csv"
    cmd = [
        "ncu",
        "--nvtx",
        "--nvtx-include",
        f"decode_{decode_index}/",
        "--target-processes",
        "all",
        "--metrics",
        metrics,
        "--replay-mode",
        "application",
        "--disable-extra-suffixes",
        "--csv",
        "--page",
        "details",
        *llm_cmd_base(
            args,
            prefill_len=prefill_len,
            decode_len=decode_index + 2,
            log_dir=log_dir,
            ncu=True,
        ),
    ]
    run_cmd(cmd, stdout_path=out_path, cwd=cwd)


def load_trace(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def trace_events(data):
    return data.get("traceEvents", data if isinstance(data, list) else [])


def discover_trace_files(trace_dir: Path) -> List[Path]:
    return sorted(trace_dir.rglob("*.pt.trace.json"))


def canonicalize_input_dims(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        parsed = ast.literal_eval(s)
        return repr(parsed)
    except Exception:
        return s


def build_cpu_index(events) -> Dict[object, Dict[str, str]]:
    out: Dict[object, Dict[str, str]] = {}
    for ev in events:
        if ev.get("cat") != "cpu_op":
            continue
        args = ev.get("args", {})
        ext_id = args.get("External id")
        if ext_id is None:
            continue
        out[ext_id] = {
            "name": str(ev.get("name") or "").strip(),
            "dims": canonicalize_input_dims(args.get("Input Dims")),
        }
    return out


def extract_annotation(events) -> str:
    candidates: Dict[str, float] = {}
    for ev in events:
        cat = ev.get("cat") or ""
        if cat not in {"user_annotation", "gpu_user_annotation"}:
            continue
        name = ev.get("name") or ""
        if not (name == "prefill" or name.startswith("decode_")):
            continue
        dur = float(ev.get("dur") or 0)
        candidates[name] = candidates.get(name, 0.0) + dur
    if not candidates:
        return "unknown"
    return max(candidates.items(), key=lambda kv: kv[1])[0]


def kernel_rows(events) -> List[Tuple[str, str, str]]:
    cpu_index = build_cpu_index(events)
    rows: List[Tuple[str, str, str]] = []
    for ev in events:
        if ev.get("cat") != "kernel":
            continue
        args = ev.get("args", {})
        ext_id = args.get("External id")
        cpu = cpu_index.get(ext_id, {})
        cpu_op_name = cpu.get("name", "")
        input_dims = cpu.get("dims", "")
        kernel_name = str(ev.get("name") or "").strip()
        rows.append((cpu_op_name, input_dims, kernel_name))
    return rows


def unique_output_path(output_dir: Path, base_name: str) -> Path:
    path = output_dir / f"{base_name}.csv"
    if not path.exists():
        return path
    for i in range(1, 10000):
        candidate = output_dir / f"{base_name}_{i}.csv"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate output path for {base_name}")


def write_linked_csv(path: Path, rows: Iterable[Tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cpu_op_name", "input_dims", "kernel_name"])
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path: Path, rows: Iterable[Tuple[str, str, str]]) -> None:
    grouped = defaultdict(lambda: [0, set()])
    for cpu_op_name, input_dims, kernel_name in rows:
        key = (cpu_op_name, input_dims)
        grouped[key][0] += 1
        if kernel_name:
            grouped[key][1].add(kernel_name)

    ordered = sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1][0]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cpu_op_name", "input_dims", "kernel_names", "rows", "unique_kernels"])
        for (cpu_op_name, input_dims), (count, kernels) in ordered:
            writer.writerow([cpu_op_name, input_dims, ";".join(sorted(kernels)), count, len(kernels)])


def postprocess_traces(trace_dir: Path, out_dir: Path) -> None:
    traces = discover_trace_files(trace_dir)
    if not traces:
        raise RuntimeError(f"No .pt.trace.json files found in {trace_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for trace_path in traces:
        data = load_trace(trace_path)
        events = trace_events(data)
        annotation = extract_annotation(events)

        linked_rows = kernel_rows(events)

        linked_csv = unique_output_path(out_dir, annotation)
        write_linked_csv(linked_csv, linked_rows)

        summary_csv = unique_output_path(out_dir, f"{annotation}_summary")
        write_summary_csv(summary_csv, linked_rows)

        print(f"Wrote {linked_csv.name} and {summary_csv.name} from {trace_path.name}")


def run_single_config(
    args: argparse.Namespace,
    *,
    prefill_len: int,
    decode_len: int,
    model_basename: str,
    script_dir: Path,
) -> None:
    log_dir = Path(args.log_root) / model_basename / f"prompt_{prefill_len}_predict_{decode_len}"
    (log_dir / "ncu").mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print(f"Run: prefill={prefill_len}, decode={decode_len}")
    print(f"Output: {log_dir}")

    run_json_profile(args, prefill_len=prefill_len, decode_len=decode_len, log_dir=log_dir, cwd=script_dir)

    run_ncu_prefill(
        args,
        prefill_len=prefill_len,
        log_dir=log_dir,
        model_basename=model_basename,
        cwd=script_dir,
    )

    if decode_len > 1:
        total = decode_len - 1
        for i in range(total):
            print(f"NCU decode range {i + 1}/{total} (decode_{i})")
            run_ncu_decode(
                args,
                prefill_len=prefill_len,
                decode_index=i,
                log_dir=log_dir,
                model_basename=model_basename,
                cwd=script_dir,
            )

    if args.postprocess:
        postprocess_traces(log_dir / "json", log_dir)


def build_run_plan(args: argparse.Namespace) -> List[Tuple[int, int]]:
    seen = set()
    plan: List[Tuple[int, int]] = []

    # Prompt-length sweep at fixed decode length.
    for prefill_len in args.prefill_lens:
        pair = (int(prefill_len), int(args.decode_len))
        if pair not in seen:
            plan.append(pair)
            seen.add(pair)

    # Decode-length sweep at fixed prefill length.
    for decode_len in args.decode_sweep_lengths:
        pair = (int(args.decode_sweep_prefill), int(decode_len))
        if pair not in seen:
            plan.append(pair)
            seen.add(pair)

    return plan


def run_plots(args: argparse.Namespace, *, model_basename: str, script_dir: Path) -> None:
    root_dir = Path(args.log_root) / model_basename
    output_dir = script_dir / "output" / model_basename
    prompt_lengths = ",".join(str(x) for x in args.prefill_lens)
    decode_lengths = ",".join(str(x) for x in args.decode_sweep_lengths)

    cmd = [
        "uv",
        "run",
        "plot_mm_total_access.py",
        "--root-dir",
        str(root_dir),
        "--output-dir",
        str(output_dir),
        "--prompt-lengths",
        prompt_lengths,
        "--prompt-sweep-decode-len",
        str(args.decode_len),
        "--decode-sweep-prefill",
        str(args.decode_sweep_prefill),
        "--decode-lengths",
        decode_lengths,
    ]
    run_cmd(cmd, cwd=script_dir)


def main() -> int:
    args = parse_args()

    require_command("uv")
    require_command("ncu")

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    model_basename = os.path.basename(args.model.rstrip("/"))
    run_plan = build_run_plan(args)

    for prefill_len, decode_len in run_plan:
        run_single_config(
            args,
            prefill_len=prefill_len,
            decode_len=decode_len,
            model_basename=model_basename,
            script_dir=script_dir,
        )

    if args.plot:
        if not args.postprocess:
            print(
                "Plotting with --no-postprocess: expecting existing per-phase linked CSVs in each run dir."
            )
        run_plots(args, model_basename=model_basename, script_dir=script_dir)

    print("\nAll runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
