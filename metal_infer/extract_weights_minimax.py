#!/usr/bin/env python3
"""Extract non-expert MiniMax weights into a single mmap-friendly binary blob.

This is the MiniMax analogue of extract_weights.py, but it stays intentionally
close to the original Hugging Face safetensors layout:

- every non-expert tensor is copied byte-for-byte into model_weights.bin
- model_weights.json records offset, size, shape, and dtype for each tensor
- the full Hugging Face config is embedded in the manifest for later runtime use

The script does not need any third-party Python packages.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safetensors_utils import (  # noqa: E402
    align_up,
    build_header_cache,
    detect_model_path,
    ensure_dir,
    load_model_config,
    load_weight_map,
    tensor_nbytes,
)


EXPERT_RE = re.compile(r"\.block_sparse_moe\.experts\.\d+\.")


def categorize_tensor(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if ".self_attn." in name:
        return "attention"
    if ".block_sparse_moe.gate" in name or "e_score_correction_bias" in name:
        return "moe_router"
    if ".norm" in name:
        return "norm"
    if "mtp" in name:
        return "mtp"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MiniMax non-expert tensors to model_weights.bin/json"
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
        default=".",
        help="Directory for model_weights.bin and model_weights.json",
    )
    parser.add_argument(
        "--include-experts",
        action="store_true",
        help="Also copy routed expert tensors (usually not what you want)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    candidate_paths = [
        Path(os.environ.get("MINIMAX_MODEL_PATH", "")),
        REPO_ROOT.parent / "minimax-m2.7-hf",
        Path.cwd(),
    ]
    model_path = detect_model_path(args.model, candidate_paths)
    output_dir = Path(args.output).expanduser().resolve()
    ensure_dir(output_dir)

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

    tensors_to_extract = {}
    skipped_experts = 0
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})

    for name, filename in weight_map.items():
        if not args.include_experts and EXPERT_RE.search(name):
            skipped_experts += 1
            continue
        tensors_to_extract[name] = filename

    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    print(f"Model: {model_path}")
    print(f"Model type: {config.get('model_type', 'unknown')}")
    print(f"Total weights in index: {len(weight_map)}")
    print(f"Skipped expert tensors: {skipped_experts}")
    print(f"Extracting tensors: {len(tensors_to_extract)}")

    print("\nParsing safetensors headers...")
    header_cache = build_header_cache(model_path, by_file.keys())

    all_tensors = [(name, tensors_to_extract[name]) for name in sorted(tensors_to_extract)]

    manifest = {
        "model": str(model_path),
        "format": "raw_safetensors_bytes",
        "num_tensors": len(all_tensors),
        "tensors": {},
        "config": config,
        "excluded": [] if args.include_experts else ["*.block_sparse_moe.experts.*"],
    }

    bin_path = output_dir / "model_weights.bin"
    json_path = output_dir / "model_weights.json"
    align = 64
    total_bytes = 0
    offset = 0
    t0 = time.time()

    print(f"\nWriting {bin_path}...")
    with open(bin_path, "wb") as out_f:
        for idx, (name, filename) in enumerate(all_tensors, start=1):
            header, data_start = header_cache[filename]
            meta = header.get(name)
            if meta is None:
                print(f"WARNING: {name} not found in {filename}, skipping", file=sys.stderr)
                continue

            byte_len = tensor_nbytes(meta)
            next_offset = align_up(offset, align)
            if next_offset != offset:
                out_f.write(b"\x00" * (next_offset - offset))
                offset = next_offset

            with open(model_path / filename, "rb") as src_f:
                src_f.seek(data_start + meta["data_offsets"][0])
                data = src_f.read(byte_len)
            if len(data) != byte_len:
                print(
                    f"ERROR: short read for {name} from {filename}: "
                    f"expected {byte_len}, got {len(data)}",
                    file=sys.stderr,
                )
                return 1

            out_f.write(data)
            manifest["tensors"][name] = {
                "offset": offset,
                "size": byte_len,
                "shape": meta["shape"],
                "dtype": meta["dtype"],
            }
            category = categorize_tensor(name)
            categories[category]["count"] += 1
            categories[category]["bytes"] += byte_len

            offset += byte_len
            total_bytes += byte_len
            if idx % 100 == 0 or idx == len(all_tensors):
                print(f"  [{idx}/{len(all_tensors)}] {total_bytes / 1e9:.2f} GB written")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    throughput = total_bytes / elapsed / 1e9 if elapsed > 0 else 0.0

    print(f"\nDone: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s ({throughput:.1f} GB/s)")
    print(f"Binary:   {bin_path}")
    print(f"Manifest: {json_path}")
    print("\nTensor categories:")
    for category in sorted(categories):
        info = categories[category]
        print(
            f"  {category:12s}: {info['count']:5d} tensors, "
            f"{info['bytes'] / 1e6:9.1f} MB"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
