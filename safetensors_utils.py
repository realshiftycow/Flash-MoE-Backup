#!/usr/bin/env python3
"""Small helpers for reading Hugging Face safetensors checkpoints.

These helpers intentionally avoid third-party dependencies so the conversion
scripts can run on a fresh machine before the model weights are downloaded.
"""

from __future__ import annotations

import json
import os
import struct
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_safetensors_header(filepath: str) -> Tuple[dict, int]:
    """Return (header_dict, data_start_offset) for a safetensors shard."""
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found under {model_path}")
    return load_json(config_path)


def load_weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"model.safetensors.index.json not found under {model_path}"
        )
    index = load_json(index_path)
    if "weight_map" not in index:
        raise KeyError(f"'weight_map' missing from {index_path}")
    return index["weight_map"]


def group_tensors_by_file(weight_map: Dict[str, str]) -> Dict[str, list]:
    grouped = defaultdict(list)
    for tensor_name, filename in weight_map.items():
        grouped[filename].append(tensor_name)
    return grouped


def build_header_cache(model_path: Path, filenames: Iterable[str]) -> Dict[str, tuple]:
    cache = {}
    for filename in sorted(set(filenames)):
        cache[filename] = parse_safetensors_header(str(model_path / filename))
    return cache


def tensor_nbytes(meta: dict) -> int:
    start, end = meta["data_offsets"]
    return end - start


def tensor_abs_offset(data_start: int, meta: dict) -> int:
    return data_start + meta["data_offsets"][0]


def align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    if remainder == 0:
        return value
    return value + (alignment - remainder)


def detect_model_path(explicit_path: str | None, candidates: Iterable[Path]) -> Path:
    """Pick the first path that looks like a Hugging Face model directory."""
    tried = []
    for candidate in [explicit_path, *candidates]:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        tried.append(str(path))
        if (path / "config.json").exists():
            return path
    tried_str = "\n  ".join(tried) if tried else "(none)"
    raise FileNotFoundError(
        "Could not locate a model directory with config.json.\n"
        f"Tried:\n  {tried_str}"
    )


def ensure_dir(path: Path) -> None:
    os.makedirs(path, exist_ok=True)
