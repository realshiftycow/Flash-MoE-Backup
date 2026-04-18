#!/usr/bin/env python3
"""Repack MiniMax expert tensors into contiguous per-layer binary files.

Unlike the current Qwen repacker, this version preserves the native MiniMax
expert tensors byte-for-byte and writes a layout manifest beside the packed
files. That gives us a stable intermediate artifact for later quantization and
Metal runtime work, without having to hard-code FP8 tensor assumptions today.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from safetensors_utils import (
    build_header_cache,
    detect_model_path,
    ensure_dir,
    load_model_config,
    load_weight_map,
    tensor_abs_offset,
    tensor_nbytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack MiniMax expert tensors into per-layer packed binaries"
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
        help="Output directory (default: <model>/packed_experts_minimax_native)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help='Layer spec: "all", "0-3", or "0,5,9"',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover layout and print what would be written without writing files",
    )
    parser.add_argument(
        "--verify-only",
        type=int,
        default=None,
        metavar="LAYER",
        help="Verify one packed layer file against the source tensors",
    )
    return parser.parse_args()


def parse_layers(spec: str, num_layers: int) -> List[int]:
    if spec in ("all", "", None):
        return list(range(num_layers))
    layers = []
    for part in spec.split(","):
        item = part.strip()
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            layers.extend(range(int(start_s), int(end_s) + 1))
        else:
            layers.append(int(item))
    unique = sorted(set(layers))
    for layer in unique:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} is outside 0..{num_layers - 1}")
    return unique


def component_sort_key(name: str):
    module_order = {"w1": 0, "w3": 1, "w2": 2}
    suffix_order = {
        "weight": 0,
        "weight_scale_inv": 1,
        "scales": 2,
        "biases": 3,
    }
    parts = name.split(".")
    module = parts[0]
    suffix = ".".join(parts[1:]) if len(parts) > 1 else ""
    return (
        module_order.get(module, 99),
        suffix_order.get(suffix, 99),
        name,
    )


def discover_expert_tensors(
    model_path: Path, weight_map: Dict[str, str], num_layers: int, num_experts: int
):
    expert_entries = defaultdict(lambda: defaultdict(dict))

    def parse_name(name: str):
        parts = name.split(".")
        try:
            idx = parts.index("layers")
        except ValueError:
            return None
        if idx + 5 >= len(parts):
            return None
        if parts[idx + 2] != "block_sparse_moe" or parts[idx + 3] != "experts":
            return None
        layer_idx = int(parts[idx + 1])
        expert_idx = int(parts[idx + 4])
        component = ".".join(parts[idx + 5 :])
        return layer_idx, expert_idx, component

    expert_tensors = {
        name: filename for name, filename in weight_map.items() if ".block_sparse_moe.experts." in name
    }
    header_cache = build_header_cache(model_path, expert_tensors.values())

    for name, filename in expert_tensors.items():
        parsed = parse_name(name)
        if parsed is None:
            continue
        layer_idx, expert_idx, component = parsed
        if layer_idx >= num_layers or expert_idx >= num_experts:
            continue
        header, data_start = header_cache[filename]
        meta = header.get(name)
        if meta is None:
            raise KeyError(f"{name} not found in {filename}")
        expert_entries[layer_idx][expert_idx][component] = {
            "tensor_name": name,
            "file": filename,
            "abs_offset": tensor_abs_offset(data_start, meta),
            "size": tensor_nbytes(meta),
            "shape": meta["shape"],
            "dtype": meta["dtype"],
        }

    if not expert_entries:
        raise RuntimeError(
            "No expert tensors were discovered. This script expects Hugging Face "
            "tensor names containing '.block_sparse_moe.experts.<idx>.'"
        )

    return expert_entries


def build_layout(expert_entries, num_layers: int, num_experts: int):
    component_specs = {}
    component_names = None

    for layer_idx in range(num_layers):
        layer = expert_entries.get(layer_idx)
        if layer is None:
            raise RuntimeError(f"Missing expert tensors for layer {layer_idx}")
        if len(layer) != num_experts:
            raise RuntimeError(
                f"Layer {layer_idx} has {len(layer)} experts, expected {num_experts}"
            )

        for expert_idx in range(num_experts):
            expert = layer.get(expert_idx)
            if expert is None:
                raise RuntimeError(f"Missing layer {layer_idx} expert {expert_idx}")
            names = sorted(expert.keys(), key=component_sort_key)
            if component_names is None:
                component_names = names
            elif names != component_names:
                raise RuntimeError(
                    f"Component mismatch at layer {layer_idx} expert {expert_idx}: "
                    f"{names} != {component_names}"
                )

            for component in names:
                current = {
                    "size": expert[component]["size"],
                    "shape": expert[component]["shape"],
                    "dtype": expert[component]["dtype"],
                }
                baseline = component_specs.get(component)
                if baseline is None:
                    component_specs[component] = current
                elif baseline != current:
                    raise RuntimeError(
                        f"Component shape/dtype mismatch for {component}: "
                        f"{current} != {baseline}"
                    )

    offset = 0
    components = []
    for component in component_names:
        spec = component_specs[component]
        components.append(
            {
                "name": component,
                "offset": offset,
                "size": spec["size"],
                "shape": spec["shape"],
                "dtype": spec["dtype"],
            }
        )
        offset += spec["size"]

    return {
        "expert_size": offset,
        "components": components,
    }


def open_source_files(model_path: Path, expert_entries, layers: List[int]):
    needed_files = set()
    for layer_idx in layers:
        for expert in expert_entries[layer_idx].values():
            for info in expert.values():
                needed_files.add(info["file"])
    fds = {}
    for filename in sorted(needed_files):
        fds[filename] = os.open(model_path / filename, os.O_RDONLY)
    return fds


def repack_layer(layer_idx, expert_entries, layout, fds, output_dir: Path, dry_run: bool):
    out_path = output_dir / f"layer_{layer_idx:02d}.bin"
    expert_size = layout["expert_size"]
    components = layout["components"]
    num_experts = len(expert_entries[layer_idx])
    layer_size = expert_size * num_experts

    if dry_run:
        print(
            f"  Layer {layer_idx:2d}: DRY RUN OK - would write "
            f"{layer_size / 1e9:.2f} GB to {out_path}"
        )
        return layer_size, 0.0

    t0 = time.monotonic()
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)

    read_plan = []
    component_offsets = {item["name"]: item["offset"] for item in components}
    for expert_idx in range(num_experts):
        expert = expert_entries[layer_idx][expert_idx]
        for component in components:
            info = expert[component["name"]]
            read_plan.append(
                (
                    fds[info["file"]],
                    info["abs_offset"],
                    expert_idx * expert_size + component_offsets[component["name"]],
                    component["size"],
                )
            )
    read_plan.sort(key=lambda item: (item[0], item[1]))

    bytes_written = 0
    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(
                f"Short read: expected {size}, got {len(data)} at offset {src_offset}"
            )
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    return bytes_written, time.monotonic() - t0


def verify_layer(layer_idx, expert_entries, layout, fds, output_dir: Path):
    out_path = output_dir / f"layer_{layer_idx:02d}.bin"
    if not out_path.exists():
        print(f"  Layer {layer_idx}: packed file not found: {out_path}")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)
    component_offsets = {item["name"]: item["offset"] for item in layout["components"]}
    expert_size = layout["expert_size"]
    mismatches = 0

    sample_experts = [0, min(1, len(expert_entries[layer_idx]) - 1), len(expert_entries[layer_idx]) - 1]
    for expert_idx in sorted(set(sample_experts)):
        expert = expert_entries[layer_idx][expert_idx]
        for component in layout["components"]:
            info = expert[component["name"]]
            original = os.pread(fds[info["file"]], component["size"], info["abs_offset"])
            packed = os.pread(
                fd_packed,
                component["size"],
                expert_idx * expert_size + component_offsets[component["name"]],
            )
            if original != packed:
                print(
                    f"  MISMATCH: layer {layer_idx} expert {expert_idx} "
                    f"component {component['name']}"
                )
                mismatches += 1

    os.close(fd_packed)
    if mismatches == 0:
        print(f"  Layer {layer_idx}: verification PASSED")
        return True
    print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")
    return False


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    candidate_paths = [
        Path(os.environ.get("MINIMAX_MODEL_PATH", "")),
        repo_root.parent / "minimax-m2.7-hf",
        Path.cwd(),
    ]
    model_path = detect_model_path(args.model, candidate_paths)

    try:
        config = load_model_config(model_path)
        weight_map = load_weight_map(model_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if model_path.name == "minimax-m2.7-hf":
            print(
                "Hint: this local folder currently contains config/docs only. "
                "Point --model at the downloaded Hugging Face snapshot when it exists.",
                file=sys.stderr,
            )
        return 1

    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_local_experts"])
    try:
        layers = [args.verify_only] if args.verify_only is not None else parse_layers(args.layers, num_layers)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else model_path / "packed_experts_minimax_native"
    )
    ensure_dir(output_dir)

    print("Discovering expert tensors...")
    expert_entries = discover_expert_tensors(model_path, weight_map, num_layers, num_experts)
    layout = build_layout(expert_entries, num_layers, num_experts)
    layer_size = layout["expert_size"] * num_experts

    layout_manifest = {
        "model": str(model_path),
        "model_type": config.get("model_type"),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_size": layout["expert_size"],
        "layer_size": layer_size,
        "components": layout["components"],
        "notes": [
            "Packed expert data is copied byte-for-byte from the original safetensors shards.",
            "This is a native MiniMax intermediate format for later quantization and runtime work.",
        ],
    }

    layout_path = output_dir / "layout.json"
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout_manifest, f, indent=2)

    print(f"Model: {model_path}")
    print(f"Layers: {num_layers}, experts/layer: {num_experts}")
    print(f"Output directory: {output_dir}")
    print(f"Layout manifest: {layout_path}")
    print(f"Component count: {len(layout['components'])}")
    for component in layout["components"]:
        print(
            f"  {component['name']:24s} offset={component['offset']:10d} "
            f"size={component['size']:10d} dtype={component['dtype']}"
        )
    print(f"Per-expert size: {layout['expert_size'] / 1e6:.2f} MB")
    print(f"Per-layer size:  {layer_size / 1e9:.2f} GB")

    fds = open_source_files(model_path, expert_entries, layers)
    print(f"Opened {len(fds)} source shard files")

    try:
        if args.verify_only is not None:
            ok = verify_layer(args.verify_only, expert_entries, layout, fds, output_dir)
            return 0 if ok else 1

        total_written = 0
        total_elapsed = 0.0
        for layer_idx in layers:
            bytes_written, elapsed = repack_layer(
                layer_idx, expert_entries, layout, fds, output_dir, args.dry_run
            )
            total_written += bytes_written
            total_elapsed += elapsed
            if not args.dry_run:
                gib = bytes_written / (1024 ** 3)
                speed = gib / elapsed if elapsed > 0 else 0.0
                print(
                    f"  Layer {layer_idx:2d}: wrote {gib:.2f} GiB in "
                    f"{elapsed:.1f}s ({speed:.2f} GiB/s)"
                )

        if args.dry_run:
            total_gib = total_written / (1024 ** 3)
            print(f"\nDry run complete: {len(layers)} layers, {total_gib:.2f} GiB total")
        else:
            total_gib = total_written / (1024 ** 3)
            speed = total_gib / total_elapsed if total_elapsed > 0 else 0.0
            print(
                f"\nDone: wrote {total_gib:.2f} GiB across {len(layers)} layers "
                f"in {total_elapsed:.1f}s ({speed:.2f} GiB/s)"
            )
    finally:
        for fd in fds.values():
            os.close(fd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
