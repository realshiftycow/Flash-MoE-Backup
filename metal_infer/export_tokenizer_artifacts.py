#!/usr/bin/env python3
"""Export Hugging Face BPE tokenizer assets into flash-moe binary artifacts.

This writes:
  - tokenizer.bin: compact BPE tokenizer format used by tokenizer.h
  - vocab.bin: token-id indexed decode table used by infer.m

Usage:
  python metal_infer/export_tokenizer_artifacts.py /path/to/model_dir
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def build_char_byte_table() -> dict[int, int]:
    byte_char: dict[int, int] = {}
    n = 0
    for b in range(256):
        if (0x21 <= b <= 0x7E) or (0xA1 <= b <= 0xAC) or (0xAE <= b <= 0xFF):
            byte_char[b] = b
        else:
            byte_char[b] = 256 + n
            n += 1
    return {cp: b for b, cp in byte_char.items()}


def decode_bpe_token(token: str, char_byte: dict[int, int]) -> bytes:
    out = bytearray()
    for ch in token:
        cp = ord(ch)
        mapped = char_byte.get(cp)
        if mapped is not None:
            out.append(mapped)
        else:
            out.extend(ch.encode("utf-8"))
    return bytes(out)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_merge(entry: object) -> tuple[str, str]:
    if isinstance(entry, (list, tuple)):
        if len(entry) != 2:
            raise ValueError(f"expected merge pair of length 2, got {entry!r}")
        return str(entry[0]), str(entry[1])
    if isinstance(entry, str):
        left, sep, right = entry.partition(" ")
        if not sep:
            raise ValueError(f"expected string merge pair, got {entry!r}")
        return left, right
    raise TypeError(f"unsupported merge entry {entry!r}")


def export_tokenizer_bin(tokenizer: dict, output_path: Path) -> None:
    model = tokenizer["model"]
    if model.get("type") != "BPE":
        raise ValueError(f"only BPE tokenizers are supported, got {model.get('type')!r}")

    vocab = model["vocab"]
    merges = [parse_merge(entry) for entry in model["merges"]]
    added = sorted(tokenizer.get("added_tokens", []), key=lambda item: item["id"])
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

    with output_path.open("wb") as handle:
        handle.write(b"BPET")
        handle.write(struct.pack("<I", 1))
        handle.write(struct.pack("<I", len(sorted_vocab)))
        handle.write(struct.pack("<I", len(merges)))
        handle.write(struct.pack("<I", len(added)))

        for token_str, token_id in sorted_vocab:
            encoded = token_str.encode("utf-8")
            handle.write(struct.pack("<I", token_id))
            handle.write(struct.pack("<H", len(encoded)))
            handle.write(encoded)

        for left, right in merges:
            left_bytes = left.encode("utf-8")
            right_bytes = right.encode("utf-8")
            handle.write(struct.pack("<H", len(left_bytes)))
            handle.write(left_bytes)
            handle.write(struct.pack("<H", len(right_bytes)))
            handle.write(right_bytes)

        for token in added:
            encoded = token["content"].encode("utf-8")
            handle.write(struct.pack("<I", token["id"]))
            handle.write(struct.pack("<H", len(encoded)))
            handle.write(encoded)


def export_vocab_bin(tokenizer: dict, config: dict, output_path: Path) -> tuple[int, int]:
    char_byte = build_char_byte_table()
    model_vocab = tokenizer["model"]["vocab"]
    added = tokenizer.get("added_tokens", [])

    token_bytes: dict[int, bytes] = {}
    max_token_id = -1

    for token_str, token_id in model_vocab.items():
        token_id = int(token_id)
        token_bytes[token_id] = decode_bpe_token(token_str, char_byte)
        max_token_id = max(max_token_id, token_id)

    for token in added:
        token_id = int(token["id"])
        token_bytes[token_id] = token["content"].encode("utf-8")
        max_token_id = max(max_token_id, token_id)

    config_vocab_size = int(config.get("vocab_size", max_token_id + 1))
    num_entries = max(config_vocab_size, max_token_id + 1)
    padding_entries = num_entries - (max_token_id + 1)

    with output_path.open("wb") as handle:
        handle.write(struct.pack("<I", num_entries))
        handle.write(struct.pack("<I", num_entries - 1))
        for token_id in range(num_entries):
            payload = token_bytes.get(token_id, b"")
            handle.write(struct.pack("<H", len(payload)))
            handle.write(payload)

    return num_entries, padding_entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Directory containing tokenizer.json and config.json")
    parser.add_argument("--tokenizer-json", type=Path, help="Path to tokenizer.json")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--tokenizer-out", type=Path, help="Output path for tokenizer.bin")
    parser.add_argument("--vocab-out", type=Path, help="Output path for vocab.bin")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    tokenizer_json = args.tokenizer_json or (model_dir / "tokenizer.json")
    config_json = args.config or (model_dir / "config.json")
    tokenizer_out = args.tokenizer_out or (model_dir / "tokenizer.bin")
    vocab_out = args.vocab_out or (model_dir / "vocab.bin")

    tokenizer = load_json(tokenizer_json)
    config = load_json(config_json)

    export_tokenizer_bin(tokenizer, tokenizer_out)
    num_entries, padding_entries = export_vocab_bin(tokenizer, config, vocab_out)

    tokenizer_size = tokenizer_out.stat().st_size
    vocab_size = vocab_out.stat().st_size
    print(f"tokenizer.bin: {tokenizer_out}")
    print(f"  size: {tokenizer_size / (1024 * 1024):.2f} MiB")
    print(f"  vocab entries: {len(tokenizer['model']['vocab'])}")
    print(f"  merge rules: {len(tokenizer['model']['merges'])}")
    print(f"  added tokens: {len(tokenizer.get('added_tokens', []))}")
    print(f"vocab.bin: {vocab_out}")
    print(f"  size: {vocab_size / (1024 * 1024):.2f} MiB")
    print(f"  decode entries: {num_entries}")
    print(f"  padded empty ids: {padding_entries}")


if __name__ == "__main__":
    main()
