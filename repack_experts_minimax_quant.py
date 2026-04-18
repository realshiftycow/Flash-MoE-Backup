#!/usr/bin/env python3
"""Quantize MiniMax FP8 experts into a flash-moe-friendly packed format.

Input format:
- Native Hugging Face MiniMax checkpoint
- Expert tensors stored as FP8 E4M3 weights + F32 per-128x128 scale blocks

Output format:
- Per-layer contiguous expert files
- Each expert stores affine group-quantized projections in the order:
  w1.weight/scales/biases, w3.weight/scales/biases, w2.weight/scales/biases
- Supports 4-bit and 2-bit output packing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit("numpy is required for MiniMax expert quantization") from exc

from repack_experts_minimax import build_layout, discover_expert_tensors, parse_layers
from safetensors_utils import detect_model_path, ensure_dir, load_model_config, load_weight_map
from quant_utils import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_GROUP_SIZE,
    build_fp8_e4m3_table,
    dequantize_fp8_blocked,
    quantize_affine_per_group,
)


GROUP_SIZE = DEFAULT_GROUP_SIZE
BLOCK_SIZE = DEFAULT_BLOCK_SIZE
FP8_E4M3_TABLE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize MiniMax FP8 experts into q4/q2 packed layer files"
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
        help="Output directory (default: <model>/packed_experts_minimax_q<bits>)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help='Layer spec: "all", "0-3", or "0,5,9"',
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 4],
        help="Output quantization bits",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute layout and output sizes",
    )
    return parser.parse_args()


def component_shape(native_layout, name: str):
    for component in native_layout["components"]:
        if component["name"] == name:
            return tuple(component["shape"])
    raise KeyError(f"{name} not found in native layout")


def build_quant_layout(native_layout, bits: int):
    components = []
    offset = 0
    vals_per_u32 = 32 // bits

    for prefix in ("w1", "w3", "w2"):
        out_dim, in_dim = component_shape(native_layout, f"{prefix}.weight")
        num_groups = in_dim // GROUP_SIZE
        packed_cols = in_dim // vals_per_u32
        weight_size = out_dim * packed_cols * 4
        scale_size = out_dim * num_groups * 2
        bias_size = scale_size

        entries = [
            {
                "name": f"{prefix}.weight",
                "offset": offset,
                "size": weight_size,
                "shape": [out_dim, packed_cols],
                "dtype": "U32",
            },
            {
                "name": f"{prefix}.scales",
                "offset": offset + weight_size,
                "size": scale_size,
                "shape": [out_dim, num_groups],
                "dtype": "BF16",
            },
            {
                "name": f"{prefix}.biases",
                "offset": offset + weight_size + scale_size,
                "size": bias_size,
                "shape": [out_dim, num_groups],
                "dtype": "BF16",
            },
        ]
        components.extend(entries)
        offset += weight_size + scale_size + bias_size

    return {
        "expert_size": offset,
        "components": components,
    }


def open_source_files(model_path: Path, expert_entries, layers):
    needed_files = set()
    for layer_idx in layers:
        for expert in expert_entries[layer_idx].values():
            for info in expert.values():
                needed_files.add(info["file"])
    return {filename: os.open(model_path / filename, os.O_RDONLY) for filename in sorted(needed_files)}


def read_tensor(fd: int, info: dict, dtype, shape):
    data = os.pread(fd, info["size"], info["abs_offset"])
    if len(data) != info["size"]:
        raise IOError(
            f"Short read for {info['tensor_name']}: expected {info['size']}, got {len(data)}"
        )
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def quantize_expert(expert: dict, fds: dict, bits: int):
    stats = {}
    chunks = []
    for prefix in ("w1", "w3", "w2"):
        w_info = expert[f"{prefix}.weight"]
        s_info = expert[f"{prefix}.weight_scale_inv"]

        weight = read_tensor(
            fds[w_info["file"]],
            w_info,
            np.uint8,
            tuple(w_info["shape"]),
        )
        scale_inv = read_tensor(
            fds[s_info["file"]],
            s_info,
            np.float32,
            tuple(s_info["shape"]),
        )

        dense = dequantize_fp8_blocked(weight, scale_inv, FP8_E4M3_TABLE, BLOCK_SIZE)
        packed, scales_bf16, biases_bf16, sq_err, value_count = quantize_affine_per_group(
            dense, bits, GROUP_SIZE
        )
        chunks.extend(
            [
                packed.tobytes(order="C"),
                scales_bf16.tobytes(order="C"),
                biases_bf16.tobytes(order="C"),
            ]
        )
        stats[prefix] = (sq_err, value_count)

    return b"".join(chunks), stats


def main() -> int:
    global FP8_E4M3_TABLE
    FP8_E4M3_TABLE = build_fp8_e4m3_table()

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    candidate_paths = [
        Path(os.environ.get("MINIMAX_MODEL_PATH", "")),
        repo_root.parent / "MiniMax-M2.7",
        repo_root.parent / "minimax-m2.7-hf",
        Path.cwd(),
    ]
    model_path = detect_model_path(args.model, candidate_paths)

    try:
        config = load_model_config(model_path)
        weight_map = load_weight_map(model_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_local_experts"])
    layers = parse_layers(args.layers, num_layers)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else model_path / f"packed_experts_minimax_q{args.bits}"
    )
    ensure_dir(output_dir)

    print("Discovering expert tensors...")
    expert_entries = discover_expert_tensors(model_path, weight_map, num_layers, num_experts)
    native_layout = build_layout(expert_entries, num_layers, num_experts)
    quant_layout = build_quant_layout(native_layout, args.bits)
    layer_size = quant_layout["expert_size"] * num_experts

    layout_manifest = {
        "model": str(model_path),
        "model_type": config.get("model_type"),
        "bits": args.bits,
        "group_size": GROUP_SIZE,
        "source_format": "MiniMax native FP8 E4M3 + weight_scale_inv",
        "source_expert_size": native_layout["expert_size"],
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_size": quant_layout["expert_size"],
        "layer_size": layer_size,
        "components": quant_layout["components"],
    }
    layout_path = output_dir / "layout.json"
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout_manifest, f, indent=2)

    print(f"Model: {model_path}")
    print(f"Bits: {args.bits}, group size: {GROUP_SIZE}")
    print(f"Output directory: {output_dir}")
    print(f"Layout manifest: {layout_path}")
    print(f"Per-expert size: {quant_layout['expert_size'] / 1e6:.2f} MB")
    print(f"Per-layer size:  {layer_size / 1e9:.2f} GB")

    if args.dry_run:
        total_gib = layer_size * len(layers) / (1024 ** 3)
        print(f"Dry run complete: {len(layers)} layers, {total_gib:.2f} GiB total")
        return 0

    fds = open_source_files(model_path, expert_entries, layers)
    print(f"Opened {len(fds)} source shard files")

    sq_err_totals = {"w1": 0.0, "w3": 0.0, "w2": 0.0}
    value_totals = {"w1": 0, "w3": 0, "w2": 0}

    try:
        for layer_idx in layers:
            out_path = output_dir / f"layer_{layer_idx:02d}.bin"
            t0 = time.monotonic()
            with open(out_path, "wb") as out_f:
                for expert_idx in range(num_experts):
                    blob, stats = quantize_expert(expert_entries[layer_idx][expert_idx], fds, args.bits)
                    out_f.write(blob)
                    for key, (sq_err, value_count) in stats.items():
                        sq_err_totals[key] += sq_err
                        value_totals[key] += value_count
                    if (expert_idx + 1) % 32 == 0 or expert_idx + 1 == num_experts:
                        print(
                            f"  Layer {layer_idx:2d}: quantized {expert_idx + 1}/{num_experts} experts",
                            file=sys.stderr,
                        )

            actual_size = out_path.stat().st_size
            if actual_size != layer_size:
                raise RuntimeError(
                    f"{out_path} has size {actual_size}, expected {layer_size}"
                )
            elapsed = time.monotonic() - t0
            gib = actual_size / (1024 ** 3)
            speed = gib / elapsed if elapsed > 0 else 0.0
            print(
                f"Layer {layer_idx:2d}: wrote {gib:.2f} GiB in {elapsed:.1f}s "
                f"({speed:.2f} GiB/s)"
            )
    finally:
        for fd in fds.values():
            os.close(fd)

    if any(value_totals.values()):
        print("Average per-expert RMSE:")
        for key in ("w1", "w3", "w2"):
            rmse = (sq_err_totals[key] / value_totals[key]) ** 0.5 if value_totals[key] else 0.0
            print(f"  {key}: {rmse:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
