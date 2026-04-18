#!/usr/bin/env python3
"""Quantize MiniMax dense weights into flash-moe's affine q4 layout.

Routed experts are handled separately by `repack_experts_minimax_quant.py`. This
script converts the non-expert side into the same `weight/scales/biases` format
used by the current runtime so the later Metal port can reuse the existing
matvec kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit("numpy is required for MiniMax dense quantization") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_utils import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    DEFAULT_GROUP_SIZE,
    bf16_to_f32,
    build_fp8_e4m3_table,
    dequantize_fp8_blocked,
    quantize_affine_per_group,
)
from safetensors_utils import (  # noqa: E402
    align_up,
    build_header_cache,
    detect_model_path,
    ensure_dir,
    load_model_config,
    load_weight_map,
    tensor_abs_offset,
    tensor_nbytes,
)


EXPERT_RE = re.compile(r"\.block_sparse_moe\.experts\.\d+\.")
DEFAULT_ALIGN = 64
OUTPUT_BITS = 4
GROUP_SIZE = DEFAULT_GROUP_SIZE
BLOCK_SIZE = DEFAULT_BLOCK_SIZE
FP8_TABLE = None


def categorize_tensor(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if ".self_attn." in name:
        return "attention"
    if ".block_sparse_moe.gate.weight" in name:
        return "moe_router"
    if "e_score_correction_bias" in name:
        return "moe_router_bias"
    if ".norm" in name:
        return "norm"
    return "other"


def weight_to_aux_name(name: str, suffix: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"Expected a *.weight tensor, got {name}")
    return name[: -len("weight")] + suffix


def dtype_itemsize(dtype: str) -> int:
    return {
        "BF16": 2,
        "F32": 4,
        "F8_E4M3": 1,
    }[dtype]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize MiniMax dense weights into runtime-friendly q4 model_weights.bin/json"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a MiniMax Hugging Face snapshot directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <model>/flash_moe_dense_q4)",
    )
    parser.add_argument(
        "--tensor-regex",
        type=str,
        default=None,
        help="Only process source tensor names that match this regex",
    )
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=512,
        help="How many matrix rows to process at a time for BF16/F32 tensors",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned output layout",
    )
    return parser.parse_args()


def build_tensor_entry(model_path: Path, filename: str, header_cache: dict, name: str) -> dict:
    header, data_start = header_cache[filename]
    meta = header.get(name)
    if meta is None:
        raise KeyError(f"{name} not found in {filename}")
    return {
        "name": name,
        "file": filename,
        "shape": list(meta["shape"]),
        "dtype": meta["dtype"],
        "size": tensor_nbytes(meta),
        "abs_offset": tensor_abs_offset(data_start, meta),
    }


def make_quant_plan(source: dict) -> dict:
    rows, cols = source["shape"]
    if cols % GROUP_SIZE != 0:
        raise ValueError(f"{source['name']} has input dim {cols}, not divisible by {GROUP_SIZE}")
    num_groups = cols // GROUP_SIZE
    packed_cols = cols // 8
    return {
        "kind": "quantized",
        "source": source,
        "category": categorize_tensor(source["name"]),
        "rows": rows,
        "cols": cols,
        "packed_cols": packed_cols,
        "num_groups": num_groups,
        "weight_name": source["name"],
        "scales_name": weight_to_aux_name(source["name"], "scales"),
        "biases_name": weight_to_aux_name(source["name"], "biases"),
        "weight_size": rows * packed_cols * 4,
        "scales_size": rows * num_groups * 2,
        "biases_size": rows * num_groups * 2,
    }


def make_copy_plan(source: dict) -> dict:
    return {
        "kind": "copy",
        "source": source,
        "category": categorize_tensor(source["name"]),
    }


def read_bytes(fd: int, size: int, offset: int, label: str) -> bytes:
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise IOError(f"Short read for {label}: expected {size}, got {len(data)}")
    return data


def read_matrix_rows(source: dict, fd: int, row_start: int, row_count: int) -> np.ndarray:
    rows, cols = source["shape"]
    if row_start < 0 or row_start + row_count > rows:
        raise ValueError(f"Row slice [{row_start}, {row_start + row_count}) out of range for {source['name']}")
    offset = source["abs_offset"] + row_start * cols * dtype_itemsize(source["dtype"])
    byte_len = row_count * cols * dtype_itemsize(source["dtype"])

    if source["dtype"] == "BF16":
        raw = np.frombuffer(read_bytes(fd, byte_len, offset, source["name"]), dtype=np.uint16)
        return bf16_to_f32(raw).reshape(row_count, cols)
    if source["dtype"] == "F32":
        return np.frombuffer(read_bytes(fd, byte_len, offset, source["name"]), dtype=np.float32).reshape(
            row_count, cols
        )
    raise ValueError(f"Unsupported row-chunk dtype {source['dtype']} for {source['name']}")


def read_fp8_weight_chunk(
    weight_source: dict,
    scale_source: dict,
    fd_weight: int,
    fd_scale: int,
    row_start: int,
    row_count: int,
) -> np.ndarray:
    if row_start % BLOCK_SIZE != 0 or row_count % BLOCK_SIZE != 0:
        raise ValueError("FP8 chunks must align to 128-row blocks")

    rows, cols = weight_source["shape"]
    if row_start < 0 or row_start + row_count > rows:
        raise ValueError(
            f"Row slice [{row_start}, {row_start + row_count}) out of range for {weight_source['name']}"
        )

    weight_offset = weight_source["abs_offset"] + row_start * cols
    weight_size = row_count * cols
    weight = np.frombuffer(
        read_bytes(fd_weight, weight_size, weight_offset, weight_source["name"]),
        dtype=np.uint8,
    ).reshape(row_count, cols)

    block_row_start = row_start // BLOCK_SIZE
    block_row_count = row_count // BLOCK_SIZE
    _, block_cols = scale_source["shape"]
    scale_offset = scale_source["abs_offset"] + block_row_start * block_cols * 4
    scale_size = block_row_count * block_cols * 4
    scale_inv = np.frombuffer(
        read_bytes(fd_scale, scale_size, scale_offset, scale_source["name"]),
        dtype=np.float32,
    ).reshape(block_row_count, block_cols)

    return dequantize_fp8_blocked(weight, scale_inv, FP8_TABLE, BLOCK_SIZE)


def plan_output(model_path: Path, weight_map: dict, header_cache: dict, tensor_regex: re.Pattern | None):
    plans = []
    skipped = defaultdict(int)

    for name in sorted(weight_map):
        if EXPERT_RE.search(name):
            skipped["experts"] += 1
            continue
        if tensor_regex and not tensor_regex.search(name):
            skipped["regex"] += 1
            continue
        if name.endswith(".weight_scale_inv"):
            skipped["weight_scale_inv"] += 1
            continue

        source = build_tensor_entry(model_path, weight_map[name], header_cache, name)
        is_weight_matrix = len(source["shape"]) == 2 and name.endswith(".weight")

        if is_weight_matrix and source["dtype"] == "F8_E4M3":
            scale_name = name[: -len(".weight")] + ".weight_scale_inv"
            scale_source = build_tensor_entry(model_path, weight_map[scale_name], header_cache, scale_name)
            plan = make_quant_plan(source)
            plan["scale_source"] = scale_source
            plans.append(plan)
            continue

        if is_weight_matrix and source["dtype"] in {"BF16", "F32"}:
            plans.append(make_quant_plan(source))
            continue

        plans.append(make_copy_plan(source))

    return plans, skipped


def assign_offsets(plans: list[dict]) -> tuple[dict, int, dict]:
    manifest_tensors = {}
    category_bytes = defaultdict(int)
    offset = 0

    for plan in plans:
        if plan["kind"] == "quantized":
            offset = align_up(offset, DEFAULT_ALIGN)
            plan["weight_offset"] = offset
            manifest_tensors[plan["weight_name"]] = {
                "offset": offset,
                "size": plan["weight_size"],
                "shape": [plan["rows"], plan["packed_cols"]],
                "dtype": "U32",
            }
            category_bytes[plan["category"]] += plan["weight_size"]
            offset += plan["weight_size"]

            offset = align_up(offset, DEFAULT_ALIGN)
            plan["scales_offset"] = offset
            manifest_tensors[plan["scales_name"]] = {
                "offset": offset,
                "size": plan["scales_size"],
                "shape": [plan["rows"], plan["num_groups"]],
                "dtype": "BF16",
            }
            category_bytes[plan["category"]] += plan["scales_size"]
            offset += plan["scales_size"]

            offset = align_up(offset, DEFAULT_ALIGN)
            plan["biases_offset"] = offset
            manifest_tensors[plan["biases_name"]] = {
                "offset": offset,
                "size": plan["biases_size"],
                "shape": [plan["rows"], plan["num_groups"]],
                "dtype": "BF16",
            }
            category_bytes[plan["category"]] += plan["biases_size"]
            offset += plan["biases_size"]
        else:
            source = plan["source"]
            offset = align_up(offset, DEFAULT_ALIGN)
            plan["offset"] = offset
            manifest_tensors[source["name"]] = {
                "offset": offset,
                "size": source["size"],
                "shape": source["shape"],
                "dtype": source["dtype"],
            }
            category_bytes[plan["category"]] += source["size"]
            offset += source["size"]

    return manifest_tensors, offset, category_bytes


def quantize_plan(plan: dict, source_fds: dict, out_f, rows_per_chunk: int) -> tuple[float, int]:
    source = plan["source"]
    rows = plan["rows"]
    packed_cols = plan["packed_cols"]
    num_groups = plan["num_groups"]
    total_sq_err = 0.0
    total_values = 0

    if source["dtype"] == "F8_E4M3":
        chunk_rows = max(BLOCK_SIZE, rows_per_chunk // BLOCK_SIZE * BLOCK_SIZE)
        if chunk_rows == 0:
            chunk_rows = BLOCK_SIZE
    else:
        chunk_rows = max(1, rows_per_chunk)

    for row_start in range(0, rows, chunk_rows):
        row_count = min(chunk_rows, rows - row_start)
        if source["dtype"] == "F8_E4M3" and row_count % BLOCK_SIZE != 0:
            raise ValueError(
                f"Non-block-aligned FP8 tail for {source['name']}: {row_count} rows at {row_start}"
            )

        if source["dtype"] == "F8_E4M3":
            dense = read_fp8_weight_chunk(
                source,
                plan["scale_source"],
                source_fds[source["file"]],
                source_fds[plan["scale_source"]["file"]],
                row_start,
                row_count,
            )
        else:
            dense = read_matrix_rows(source, source_fds[source["file"]], row_start, row_count)

        packed, scales_bf16, biases_bf16, sq_err, value_count = quantize_affine_per_group(
            dense, OUTPUT_BITS, GROUP_SIZE
        )
        total_sq_err += sq_err
        total_values += value_count

        out_f.seek(plan["weight_offset"] + row_start * packed_cols * 4)
        out_f.write(packed.tobytes(order="C"))
        out_f.seek(plan["scales_offset"] + row_start * num_groups * 2)
        out_f.write(scales_bf16.tobytes(order="C"))
        out_f.seek(plan["biases_offset"] + row_start * num_groups * 2)
        out_f.write(biases_bf16.tobytes(order="C"))

    return total_sq_err, total_values


def copy_plan(plan: dict, source_fds: dict, out_f) -> None:
    source = plan["source"]
    out_f.seek(plan["offset"])
    out_f.write(read_bytes(source_fds[source["file"]], source["size"], source["abs_offset"], source["name"]))


def print_plan_summary(plans: list[dict], category_bytes: dict, total_size: int, skipped: dict) -> None:
    quantized = sum(1 for plan in plans if plan["kind"] == "quantized")
    copied = len(plans) - quantized
    print(f"Planned tensors: {len(plans)} source tensors")
    print(f"  quantized matrices: {quantized}")
    print(f"  copied raw tensors: {copied}")
    if skipped:
        print("Skipped:")
        for key in sorted(skipped):
            print(f"  {key:16s}: {skipped[key]}")
    print(f"Output size: {total_size / 1e9:.2f} GB")
    print("Output categories:")
    for category in sorted(category_bytes):
        print(f"  {category:16s}: {category_bytes[category] / 1e6:9.1f} MB")


def main() -> int:
    global FP8_TABLE

    args = parse_args()
    tensor_regex = re.compile(args.tensor_regex) if args.tensor_regex else None

    candidate_paths = [
        Path(os.environ.get("MINIMAX_MODEL_PATH", "")),
        REPO_ROOT.parent / "MiniMax-M2.7",
        REPO_ROOT.parent / "minimax-m2.7-hf",
        Path.cwd(),
    ]
    model_path = detect_model_path(args.model, candidate_paths)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (model_path / "flash_moe_dense_q4").resolve()
    )
    ensure_dir(output_dir)

    config = load_model_config(model_path)
    weight_map = load_weight_map(model_path)
    header_cache = build_header_cache(model_path, weight_map.values())

    FP8_TABLE = build_fp8_e4m3_table()

    plans, skipped = plan_output(model_path, weight_map, header_cache, tensor_regex)
    manifest_tensors, total_size, category_bytes = assign_offsets(plans)

    manifest = {
        "model": str(model_path),
        "format": "flash_moe_affine_q4",
        "num_tensors": len(manifest_tensors),
        "tensors": manifest_tensors,
        "config": config,
        "quantization": {
            "bits": OUTPUT_BITS,
            "group_size": GROUP_SIZE,
            "fp8_block_size": BLOCK_SIZE,
            "experts": "excluded",
        },
        "excluded": ["*.block_sparse_moe.experts.*"],
    }

    print(f"Model: {model_path}")
    print(f"Model type: {config.get('model_type', 'unknown')}")
    print(f"Output: {output_dir}")
    print_plan_summary(plans, category_bytes, total_size, skipped)

    if args.dry_run:
        return 0

    source_files = sorted(
        {plan["source"]["file"] for plan in plans}
        | {
            plan["scale_source"]["file"]
            for plan in plans
            if plan["kind"] == "quantized" and "scale_source" in plan
        }
    )
    source_fds = {filename: os.open(model_path / filename, os.O_RDONLY) for filename in source_files}

    bin_path = output_dir / "model_weights.bin"
    json_path = output_dir / "model_weights.json"
    t0 = time.time()
    rmse_by_category = defaultdict(lambda: [0.0, 0])
    bytes_written = 0

    try:
        with open(bin_path, "wb+") as out_f:
            out_f.truncate(total_size)
            for idx, plan in enumerate(plans, start=1):
                source_name = plan["source"]["name"]
                if plan["kind"] == "quantized":
                    sq_err, value_count = quantize_plan(plan, source_fds, out_f, args.rows_per_chunk)
                    rmse_by_category[plan["category"]][0] += sq_err
                    rmse_by_category[plan["category"]][1] += value_count
                    bytes_written += plan["weight_size"] + plan["scales_size"] + plan["biases_size"]
                    rmse = math.sqrt(sq_err / value_count) if value_count else 0.0
                    print(
                        f"  [{idx}/{len(plans)}] q4 {source_name} -> "
                        f"{(plan['weight_size'] + plan['scales_size'] + plan['biases_size']) / 1e6:.1f} MB, "
                        f"rmse={rmse:.6f}"
                    )
                else:
                    copy_plan(plan, source_fds, out_f)
                    bytes_written += plan["source"]["size"]
                    print(
                        f"  [{idx}/{len(plans)}] copy {source_name} -> "
                        f"{plan['source']['size'] / 1e6:.1f} MB"
                    )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    finally:
        for fd in source_fds.values():
            os.close(fd)

    elapsed = time.time() - t0
    throughput = bytes_written / elapsed / 1e9 if elapsed > 0 else 0.0

    print(f"\nDone: {bytes_written / 1e9:.2f} GB written in {elapsed:.1f}s ({throughput:.2f} GB/s)")
    print(f"Binary:   {bin_path}")
    print(f"Manifest: {json_path}")

    print("\nRMSE by category:")
    for category in sorted(rmse_by_category):
        sq_err, value_count = rmse_by_category[category]
        rmse = math.sqrt(sq_err / value_count) if value_count else 0.0
        print(f"  {category:16s}: {rmse:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
