#!/usr/bin/env python3
"""Plot NCU-based total memory access for aten::mm across run sweeps.

This script reads, per run:
- linked trace CSVs: <run_dir>/<phase>.csv (phase: prefill, decode_i)
- NCU CSVs: <run_dir>/ncu/*_<phase>.csv

It maps NCU kernel rows to CPU ops using linked trace kernel names, filters to
`aten::mm`, then aggregates memory access bytes using:
- DRAM bytes = (dram__sectors_read.sum + dram__sectors_write.sum) * 32
- L2 bytes = (lts__t_sectors_op_read.sum + lts__t_sectors_op_write.sum) * 32
- SHMEM bytes = sm__sass_data_bytes_mem_shared_op_ld.sum
                + sm__sass_data_bytes_mem_shared_op_ldsm.sum
                + sm__sass_data_bytes_mem_shared_op_st.sum

Outputs:
1) Prompt-length sweep plot and CSV (decode fixed)
2) Decode-length sweep plot and CSV (prefill fixed)
"""

import argparse
import csv
import io
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_PROMPT_LENGTHS = [32, 64, 128, 256, 512, 1024]
DEFAULT_DECODE_SWEEP_LENGTHS = [4, 8, 16, 32, 64, 128]

METRIC_DRAM_READ = "dram__sectors_read.sum"
METRIC_DRAM_WRITE = "dram__sectors_write.sum"
METRIC_L2_READ = "lts__t_sectors_op_read.sum"
METRIC_L2_WRITE = "lts__t_sectors_op_write.sum"
METRIC_SHMEM_LD = "sm__sass_data_bytes_mem_shared_op_ld.sum"
METRIC_SHMEM_LDSM = "sm__sass_data_bytes_mem_shared_op_ldsm.sum"
METRIC_SHMEM_ST = "sm__sass_data_bytes_mem_shared_op_st.sum"
REQUIRED_METRICS = {
    METRIC_DRAM_READ,
    METRIC_DRAM_WRITE,
    METRIC_L2_READ,
    METRIC_L2_WRITE,
    METRIC_SHMEM_LD,
    METRIC_SHMEM_LDSM,
    METRIC_SHMEM_ST,
}

_PHASE_TOTAL_CACHE: Dict[Tuple[str, str], Dict[str, float]] = {}


def _parse_number(value) -> float:
    if value is None:
        return 0.0
    s = str(value).strip().strip('"')
    if not s or s.lower() in ("na", "n/a", "none", "nan"):
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def _parse_int_csv(spec: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in str(spec).split(","):
        t = tok.strip()
        if not t:
            continue
        v = int(t)
        if v <= 0:
            raise ValueError("Values must be > 0, got %s" % v)
        if v not in seen:
            out.append(v)
            seen.add(v)
    if not out:
        raise ValueError("No integer values parsed from: %s" % spec)
    return out


def _strip_top_level_params(s: str) -> str:
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "(" and depth == 0:
            return s[:i]
    return s


def _normalize_kernel_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""

    s = re.sub(r"^void\s+", "", s)
    s = s.replace("at::native::", "at::")
    s = s.replace("(anonymous namespace)", "unnamed")
    s = s.replace("(anonymous_namespace)", "unnamed")
    s = s.replace("<unnamed>", "unnamed")
    s = re.sub(r"\[lambda\(\)\s*\(instance\s*(\d+)\)\]", r"{lambda()#\1}", s)
    s = re.sub(r"\s+", "", s)
    s = _strip_top_level_params(s)
    return s


def _extract_base_symbol(s_norm: str) -> str:
    return s_norm.split("<", 1)[0]


def _extract_op_marker(s_norm: str) -> Optional[str]:
    patterns = [
        r"([A-Za-z0-9_]+_cuda_out)",
        r"([A-Za-z0-9_]+_kernel_cuda)",
        r"binary_internal::([A-Za-z0-9_]+Functor)",
        r"(CUDAFunctorOnSelf_[A-Za-z0-9_]+)",
        r"(CUDAFunctor_[A-Za-z0-9_]+)",
        r"(CUDAFunctor[A-Za-z0-9_]+)",
        r"(MeanOps|SumOps|MaxOps|MinOps)",
        r"(CatArrayBatchedCopy)",
        r"(silu_kernel)",
        r"(pow_tensor_scalar_kernel_impl)",
        r"([A-Za-z0-9_]+_kernel_impl[A-Za-z0-9_]*)",
    ]
    for pat in patterns:
        m = re.search(pat, s_norm, flags=re.IGNORECASE)
        if m:
            marker = m.group(1)
            if marker.lower().startswith("gpu_kernel_impl"):
                continue
            return marker
    return None


def _kernel_keys(kernel_name: str) -> List[str]:
    n = _normalize_kernel_name(kernel_name).lower()
    if not n:
        return []
    base = _extract_base_symbol(n)
    marker = _extract_op_marker(n)

    out = [n]
    if marker:
        out.append("%s|%s" % (base, marker.lower()))
    out.append(base)
    return out


def _read_linked_cpu_map(linked_csv: Path) -> Tuple[Dict[str, Counter], set]:
    key_to_ops: Dict[str, Counter] = defaultdict(Counter)
    cpu_ops = set()

    with linked_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Empty linked CSV: %s" % linked_csv)
        if "kernel_name" not in reader.fieldnames or "cpu_op_name" not in reader.fieldnames:
            raise ValueError("Linked CSV missing required columns: %s" % linked_csv)

        for row in reader:
            cpu_op = (row.get("cpu_op_name") or "").strip()
            kernel_name = (row.get("kernel_name") or "").strip()
            if not cpu_op or not kernel_name:
                continue
            cpu_ops.add(cpu_op)
            for key in _kernel_keys(kernel_name):
                key_to_ops[key][cpu_op] += 1

    return key_to_ops, cpu_ops


def _choose_cpu_op_for_kernel(kernel_name: str, key_to_ops: Dict[str, Counter]) -> Tuple[Optional[str], Optional[str], bool]:
    best_ambiguous = None
    for key in _kernel_keys(kernel_name):
        ops = key_to_ops.get(key)
        if not ops:
            continue
        if len(ops) == 1:
            return next(iter(ops.keys())), key, False
        if best_ambiguous is None:
            best_ambiguous = (ops.most_common(1)[0][0], key)
    if best_ambiguous is not None:
        return best_ambiguous[0], best_ambiguous[1], True
    return None, None, False


def _gemm_like_cpu_op_fallback(kernel_name: str, cpu_ops_in_summary: set) -> Optional[str]:
    s = (kernel_name or "").lower()
    if not s:
        return None
    if not any(tok in s for tok in ("gemm", "sgemm", "hgemm", "xmma_gemm", "cublas", "cutlass")):
        return None
    if "stridedbatched" in s or "batched" in s:
        if "aten::bmm" in cpu_ops_in_summary:
            return "aten::bmm"
    if "aten::mm" in cpu_ops_in_summary:
        return "aten::mm"
    return None


def _find_ncu_header_start(lines: List[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        if line.startswith('"ID"') and '"Kernel Name"' in line:
            return i
    return None


def _iter_ncu_rows(ncu_csv: Path):
    text = ncu_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = _find_ncu_header_start(text)
    if start is None:
        raise ValueError("Could not find NCU CSV header in %s" % ncu_csv)

    reader = csv.DictReader(io.StringIO("\n".join(text[start:])))
    fieldnames = set(reader.fieldnames or [])

    # NCU can emit two different CSV layouts:
    # 1) Wide format: one row per kernel with metric columns.
    # 2) Long format: one row per (kernel, metric) with "Metric Name"/"Metric Value".
    if "Metric Name" in fieldnames and "Metric Value" in fieldnames:
        by_kernel: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = {}
        for row in reader:
            kernel_name = (row.get("Kernel Name") or "").strip()
            if not kernel_name:
                continue
            key = (
                str(row.get("ID") or ""),
                str(row.get("Process ID") or ""),
                kernel_name,
                str(row.get("Context") or ""),
                str(row.get("Stream") or ""),
            )
            agg = by_kernel.get(key)
            if agg is None:
                agg = {"Kernel Name": kernel_name}
                by_kernel[key] = agg

            metric_name = (row.get("Metric Name") or "").strip()
            if metric_name in REQUIRED_METRICS:
                agg[metric_name] = float(agg.get(metric_name, 0.0)) + _parse_number(row.get("Metric Value"))

        for agg in by_kernel.values():
            yield agg
        return

    for row in reader:
        kernel_name = (row.get("Kernel Name") or "").strip()
        if not kernel_name:
            continue
        yield row


def _phase_ncu_file(run_dir: Path, phase_name: str) -> Path:
    ncu_dir = run_dir / "ncu"
    matches = sorted(ncu_dir.glob("*_%s.csv" % phase_name))
    if not matches:
        raise FileNotFoundError("Missing NCU CSV for %s in %s" % (phase_name, ncu_dir))
    return matches[0]


def _phase_linked_file(run_dir: Path, phase_name: str) -> Path:
    path = run_dir / ("%s.csv" % phase_name)
    if not path.is_file():
        raise FileNotFoundError("Missing linked CSV for %s: %s" % (phase_name, path))
    return path


def _aggregate_phase_mm_bytes(run_dir: Path, phase_name: str) -> Dict[str, float]:
    cache_key = (str(run_dir.resolve()), phase_name)
    cached = _PHASE_TOTAL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    linked_csv = _phase_linked_file(run_dir, phase_name)
    ncu_csv = _phase_ncu_file(run_dir, phase_name)

    key_to_ops, cpu_ops = _read_linked_cpu_map(linked_csv)

    dram_sectors = 0.0
    l2_sectors = 0.0
    shmem_bytes = 0.0

    for row in _iter_ncu_rows(ncu_csv):
        kernel_name = (row.get("Kernel Name") or "").strip()
        cpu_op, _key, _amb = _choose_cpu_op_for_kernel(kernel_name, key_to_ops)
        if cpu_op is None:
            cpu_op = _gemm_like_cpu_op_fallback(kernel_name, cpu_ops)
        if cpu_op != "aten::mm":
            continue

        dram_sectors += _parse_number(row.get(METRIC_DRAM_READ)) + _parse_number(row.get(METRIC_DRAM_WRITE))
        l2_sectors += _parse_number(row.get(METRIC_L2_READ)) + _parse_number(row.get(METRIC_L2_WRITE))
        shmem_bytes += (
            _parse_number(row.get(METRIC_SHMEM_LD))
            + _parse_number(row.get(METRIC_SHMEM_LDSM))
            + _parse_number(row.get(METRIC_SHMEM_ST))
        )

    out = {
        "dram_bytes": dram_sectors * 32.0,
        "l2_bytes": l2_sectors * 32.0,
        "shmem_bytes": shmem_bytes,
    }
    _PHASE_TOTAL_CACHE[cache_key] = out
    return out


def _aggregate_run_mm_bytes(run_dir: Path, decode_len: int) -> Dict[str, float]:
    if decode_len <= 1:
        raise ValueError("decode_len must be > 1 for aggregate plots")

    prefill_tot = _aggregate_phase_mm_bytes(run_dir, "prefill")
    decode_tot = {"dram_bytes": 0.0, "l2_bytes": 0.0, "shmem_bytes": 0.0}

    for phase in ["decode_%d" % i for i in range(decode_len - 1)]:
        phase_tot = _aggregate_phase_mm_bytes(run_dir, phase)
        decode_tot["dram_bytes"] += phase_tot["dram_bytes"]
        decode_tot["l2_bytes"] += phase_tot["l2_bytes"]
        decode_tot["shmem_bytes"] += phase_tot["shmem_bytes"]

    total = {
        "dram_prefill_bytes": prefill_tot["dram_bytes"],
        "l2_prefill_bytes": prefill_tot["l2_bytes"],
        "shmem_prefill_bytes": prefill_tot["shmem_bytes"],
        "dram_decode_bytes": decode_tot["dram_bytes"],
        "l2_decode_bytes": decode_tot["l2_bytes"],
        "shmem_decode_bytes": decode_tot["shmem_bytes"],
    }
    total["dram_bytes"] = total["dram_prefill_bytes"] + total["dram_decode_bytes"]
    total["l2_bytes"] = total["l2_prefill_bytes"] + total["l2_decode_bytes"]
    total["shmem_bytes"] = total["shmem_prefill_bytes"] + total["shmem_decode_bytes"]
    total["total_prefill_bytes"] = (
        total["dram_prefill_bytes"] + total["l2_prefill_bytes"] + total["shmem_prefill_bytes"]
    )
    total["total_decode_bytes"] = (
        total["dram_decode_bytes"] + total["l2_decode_bytes"] + total["shmem_decode_bytes"]
    )

    total["total_bytes"] = total["dram_bytes"] + total["l2_bytes"] + total["shmem_bytes"]
    return total


def _write_prompt_csv(path: Path, rows: List[Tuple[int, int, Dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_length",
                "decode_length",
                "decode_step_range",
                "dram_prefill_bytes",
                "dram_decode_bytes",
                "dram_bytes",
                "l2_prefill_bytes",
                "l2_decode_bytes",
                "l2_bytes",
                "shmem_prefill_bytes",
                "shmem_decode_bytes",
                "shmem_bytes",
                "total_prefill_bytes",
                "total_decode_bytes",
                "total_bytes",
            ],
        )
        writer.writeheader()
        for prompt_len, decode_len, totals in rows:
            writer.writerow(
                {
                    "prompt_length": str(prompt_len),
                    "decode_length": str(decode_len),
                    "decode_step_range": "0..%d" % (decode_len - 2),
                    "dram_prefill_bytes": "%.0f" % totals["dram_prefill_bytes"],
                    "dram_decode_bytes": "%.0f" % totals["dram_decode_bytes"],
                    "dram_bytes": "%.0f" % totals["dram_bytes"],
                    "l2_prefill_bytes": "%.0f" % totals["l2_prefill_bytes"],
                    "l2_decode_bytes": "%.0f" % totals["l2_decode_bytes"],
                    "l2_bytes": "%.0f" % totals["l2_bytes"],
                    "shmem_prefill_bytes": "%.0f" % totals["shmem_prefill_bytes"],
                    "shmem_decode_bytes": "%.0f" % totals["shmem_decode_bytes"],
                    "shmem_bytes": "%.0f" % totals["shmem_bytes"],
                    "total_prefill_bytes": "%.0f" % totals["total_prefill_bytes"],
                    "total_decode_bytes": "%.0f" % totals["total_decode_bytes"],
                    "total_bytes": "%.0f" % totals["total_bytes"],
                }
            )


def _write_decode_csv(path: Path, rows: List[Tuple[int, int, Dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prefill_length",
                "decode_length",
                "decode_step_range",
                "dram_prefill_bytes",
                "dram_decode_bytes",
                "dram_bytes",
                "l2_prefill_bytes",
                "l2_decode_bytes",
                "l2_bytes",
                "shmem_prefill_bytes",
                "shmem_decode_bytes",
                "shmem_bytes",
                "total_prefill_bytes",
                "total_decode_bytes",
                "total_bytes",
            ],
        )
        writer.writeheader()
        for prefill_len, decode_len, totals in rows:
            writer.writerow(
                {
                    "prefill_length": str(prefill_len),
                    "decode_length": str(decode_len),
                    "decode_step_range": "0..%d" % (decode_len - 2),
                    "dram_prefill_bytes": "%.0f" % totals["dram_prefill_bytes"],
                    "dram_decode_bytes": "%.0f" % totals["dram_decode_bytes"],
                    "dram_bytes": "%.0f" % totals["dram_bytes"],
                    "l2_prefill_bytes": "%.0f" % totals["l2_prefill_bytes"],
                    "l2_decode_bytes": "%.0f" % totals["l2_decode_bytes"],
                    "l2_bytes": "%.0f" % totals["l2_bytes"],
                    "shmem_prefill_bytes": "%.0f" % totals["shmem_prefill_bytes"],
                    "shmem_decode_bytes": "%.0f" % totals["shmem_decode_bytes"],
                    "shmem_bytes": "%.0f" % totals["shmem_bytes"],
                    "total_prefill_bytes": "%.0f" % totals["total_prefill_bytes"],
                    "total_decode_bytes": "%.0f" % totals["total_decode_bytes"],
                    "total_bytes": "%.0f" % totals["total_bytes"],
                }
            )


def _write_stacked_plot(
    out_png: Path,
    x_labels: List[str],
    dram: List[float],
    l2: List[float],
    shmem: List[float],
    dram_prefill: List[float],
    l2_prefill: List[float],
    shmem_prefill: List[float],
    totals: List[float],
    title: str,
    xlabel: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch
    except Exception as e:
        raise RuntimeError(
            "matplotlib import failed (%s). Install it with: uv add matplotlib" % str(e)
        )

    x = np.arange(len(x_labels))
    fig_w = max(10.0, 1.6 * len(x_labels))
    fig, ax = plt.subplots(figsize=(fig_w, 6.3))

    ax.bar(x, dram, color="#E45756", edgecolor="#222222", linewidth=0.8)
    ax.bar(x, l2, bottom=dram, color="#72B7B2", edgecolor="#222222", linewidth=0.8)
    stacked_base = [a + b for a, b in zip(dram, l2)]
    ax.bar(x, shmem, bottom=stacked_base, color="#54A24B", edgecolor="#222222", linewidth=0.8)

    # Overlay hatched prefill-only portions at the bottom of each category segment.
    ax.bar(x, dram_prefill, color="none", hatch="////", edgecolor="#111111", linewidth=0.6)
    ax.bar(x, l2_prefill, bottom=dram, color="none", hatch="////", edgecolor="#111111", linewidth=0.6)
    ax.bar(
        x,
        shmem_prefill,
        bottom=stacked_base,
        color="none",
        hatch="////",
        edgecolor="#111111",
        linewidth=0.6,
    )

    max_total = max(totals) if totals else 0.0
    for xi, total in zip(x, totals):
        ax.text(xi, total + max_total * 0.01, "%.2fT" % (total / 1e12), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Total aten::mm Memory Access (bytes)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    legend_handles = [
        Patch(facecolor="#E45756", edgecolor="#222222", label="DRAM"),
        Patch(facecolor="#72B7B2", edgecolor="#222222", label="L2"),
        Patch(facecolor="#54A24B", edgecolor="#222222", label="SHMEM"),
        Patch(facecolor="white", edgecolor="#222222", hatch="////", label="Prefill portion"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    note = (
        "NCU-only. DRAM/L2 sectors converted with 32 bytes/sector. "
        "SHMEM bytes = ld + ldsm + st. Hatched area = prefill contribution. "
        "Aggregated over prefill + decode steps 0..N-2."
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_prompt_sweep(
    root_dir: Path,
    prompt_lengths: List[int],
    decode_len: int,
) -> List[Tuple[int, int, Dict[str, float]]]:
    rows: List[Tuple[int, int, Dict[str, float]]] = []
    for prompt_len in prompt_lengths:
        run_dir = root_dir / ("prompt_%d_predict_%d" % (prompt_len, decode_len))
        if not run_dir.is_dir():
            raise FileNotFoundError("Missing run directory: %s" % run_dir)
        rows.append((prompt_len, decode_len, _aggregate_run_mm_bytes(run_dir, decode_len)))
    return rows


def _available_runs_for_prefill(root_dir: Path, prefill_len: int) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    pat = re.compile(r"^prompt_(\d+)_predict_(\d+)$")
    for child in root_dir.iterdir():
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if not m:
            continue
        p = int(m.group(1))
        d = int(m.group(2))
        if p != prefill_len:
            continue
        out[d] = child
    return out


def _build_decode_sweep(
    root_dir: Path,
    prefill_len: int,
    decode_lengths: List[int],
) -> List[Tuple[int, int, Dict[str, float]]]:
    available = _available_runs_for_prefill(root_dir, prefill_len)
    if not available:
        raise FileNotFoundError(
            "No run directories found for prefill=%d under %s" % (prefill_len, root_dir)
        )

    rows: List[Tuple[int, int, Dict[str, float]]] = []
    for decode_len in decode_lengths:
        run_dir = available.get(decode_len)
        if run_dir is None:
            # If an exact run is missing, use the smallest available superset run.
            # Example: prompt_128_predict_128 can serve decode lengths 8/16/32/64.
            candidates = sorted(d for d in available if d >= decode_len)
            if not candidates:
                raise FileNotFoundError(
                    "Missing run directory for prefill=%d decode=%d and no superset decode run is available"
                    % (prefill_len, decode_len)
                )
            source_decode_len = candidates[0]
            run_dir = available[source_decode_len]
            print(
                "Decode length %d missing for prefill=%d; reusing %s and aggregating up to decode_%d."
                % (decode_len, prefill_len, run_dir.name, decode_len - 2)
            )
        rows.append((prefill_len, decode_len, _aggregate_run_mm_bytes(run_dir, decode_len)))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing prompt_<P>_predict_<N> run directories.",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PROMPT_LENGTHS),
        help="Comma-separated prompt lengths for fixed decode-length sweep.",
    )
    parser.add_argument(
        "--prompt-sweep-decode-len",
        type=int,
        default=4,
        help="Decode length used for prompt-length sweep.",
    )
    parser.add_argument(
        "--decode-sweep-prefill",
        type=int,
        default=128,
        help="Prefill length used for decode-length sweep.",
    )
    parser.add_argument(
        "--decode-lengths",
        type=str,
        default=",".join(str(x) for x in DEFAULT_DECODE_SWEEP_LENGTHS),
        help="Comma-separated decode lengths for fixed-prefill sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for aggregate plots/CSVs. "
            "If set, defaults become <output-dir>/... instead of writing under <root-dir>."
        ),
    )
    parser.add_argument(
        "--out-prompt-png",
        type=str,
        default=None,
        help=(
            "Output PNG for prompt-length sweep "
            "(default: <output-dir>/... if set, else <root>/total_mem_access_by_prompt_len_predict_<N>.png)."
        ),
    )
    parser.add_argument(
        "--out-prompt-csv",
        type=str,
        default=None,
        help=(
            "Output CSV for prompt-length sweep "
            "(default: <output-dir>/... if set, else <root>/total_mem_access_by_prompt_len_predict_<N>.csv)."
        ),
    )
    parser.add_argument(
        "--out-decode-png",
        type=str,
        default=None,
        help=(
            "Output PNG for decode-length sweep "
            "(default: <output-dir>/... if set, else <root>/prompt_<P>_predict_<maxN>/total_mem_access_by_decode_len.png)."
        ),
    )
    parser.add_argument(
        "--out-decode-csv",
        type=str,
        default=None,
        help=(
            "Output CSV for decode-length sweep "
            "(default: <output-dir>/... if set, else <root>/prompt_<P>_predict_<maxN>/total_mem_access_by_decode_len.csv)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root_dir = Path(args.root_dir)
    prompt_lengths = _parse_int_csv(args.prompt_lengths)
    decode_lengths = _parse_int_csv(args.decode_lengths)

    prompt_decode_len = int(args.prompt_sweep_decode_len)
    decode_sweep_prefill = int(args.decode_sweep_prefill)
    if prompt_decode_len <= 1:
        raise SystemExit("--prompt-sweep-decode-len must be > 1")

    prompt_rows = _build_prompt_sweep(root_dir, prompt_lengths, prompt_decode_len)
    decode_rows = _build_decode_sweep(root_dir, decode_sweep_prefill, decode_lengths)

    output_dir = Path(args.output_dir) if args.output_dir else None

    out_prompt_png = (
        Path(args.out_prompt_png)
        if args.out_prompt_png
        else (
            output_dir / ("total_mem_access_by_prompt_len_predict_%d.png" % prompt_decode_len)
            if output_dir
            else root_dir / ("total_mem_access_by_prompt_len_predict_%d.png" % prompt_decode_len)
        )
    )
    out_prompt_csv = (
        Path(args.out_prompt_csv)
        if args.out_prompt_csv
        else (
            output_dir / ("total_mem_access_by_prompt_len_predict_%d.csv" % prompt_decode_len)
            if output_dir
            else root_dir / ("total_mem_access_by_prompt_len_predict_%d.csv" % prompt_decode_len)
        )
    )

    decode_anchor = root_dir / ("prompt_%d_predict_%d" % (decode_sweep_prefill, max(decode_lengths)))
    out_decode_png = (
        Path(args.out_decode_png)
        if args.out_decode_png
        else (
            output_dir / ("total_mem_access_by_decode_len_prefill_%d.png" % decode_sweep_prefill)
            if output_dir
            else decode_anchor / "total_mem_access_by_decode_len.png"
        )
    )
    out_decode_csv = (
        Path(args.out_decode_csv)
        if args.out_decode_csv
        else (
            output_dir / ("total_mem_access_by_decode_len_prefill_%d.csv" % decode_sweep_prefill)
            if output_dir
            else decode_anchor / "total_mem_access_by_decode_len.csv"
        )
    )

    _write_prompt_csv(out_prompt_csv, prompt_rows)
    _write_decode_csv(out_decode_csv, decode_rows)

    _write_stacked_plot(
        out_png=out_prompt_png,
        x_labels=["prompt=%d" % p for p, _d, _t in prompt_rows],
        dram=[t["dram_bytes"] for _p, _d, t in prompt_rows],
        l2=[t["l2_bytes"] for _p, _d, t in prompt_rows],
        shmem=[t["shmem_bytes"] for _p, _d, t in prompt_rows],
        dram_prefill=[t["dram_prefill_bytes"] for _p, _d, t in prompt_rows],
        l2_prefill=[t["l2_prefill_bytes"] for _p, _d, t in prompt_rows],
        shmem_prefill=[t["shmem_prefill_bytes"] for _p, _d, t in prompt_rows],
        totals=[t["total_bytes"] for _p, _d, t in prompt_rows],
        title="Total aten::mm Memory Access by Prompt Length (decode fixed)",
        xlabel="Prompt Length (decode fixed at %d)" % prompt_decode_len,
    )

    _write_stacked_plot(
        out_png=out_decode_png,
        x_labels=["decode=%d" % d for _p, d, _t in decode_rows],
        dram=[t["dram_bytes"] for _p, _d, t in decode_rows],
        l2=[t["l2_bytes"] for _p, _d, t in decode_rows],
        shmem=[t["shmem_bytes"] for _p, _d, t in decode_rows],
        dram_prefill=[t["dram_prefill_bytes"] for _p, _d, t in decode_rows],
        l2_prefill=[t["l2_prefill_bytes"] for _p, _d, t in decode_rows],
        shmem_prefill=[t["shmem_prefill_bytes"] for _p, _d, t in decode_rows],
        totals=[t["total_bytes"] for _p, _d, t in decode_rows],
        title="Total aten::mm Memory Access by Decode Length (prefill fixed)",
        xlabel="Decode Length (prefill fixed at %d)" % decode_sweep_prefill,
    )

    print("Wrote prompt sweep plot: %s" % out_prompt_png)
    print("Wrote prompt sweep CSV:  %s" % out_prompt_csv)
    print("Wrote decode sweep plot: %s" % out_decode_png)
    print("Wrote decode sweep CSV:  %s" % out_decode_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
