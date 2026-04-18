#!/usr/bin/env python3
"""Shared quantization helpers for MiniMax conversion tools."""

from __future__ import annotations

import numpy as np


DEFAULT_GROUP_SIZE = 64
DEFAULT_BLOCK_SIZE = 128


def build_fp8_e4m3_table() -> np.ndarray:
    table = np.empty(256, dtype=np.float32)
    for raw in range(256):
        sign = -1.0 if (raw & 0x80) else 1.0
        exponent = (raw >> 3) & 0x0F
        mantissa = raw & 0x07

        if exponent == 0:
            if mantissa == 0:
                table[raw] = -0.0 if sign < 0 else 0.0
            else:
                table[raw] = sign * (2.0**-6) * (mantissa / 8.0)
        elif exponent == 0x0F and mantissa == 0x07:
            table[raw] = np.nan
        else:
            table[raw] = sign * (2.0 ** (exponent - 7)) * (1.0 + mantissa / 8.0)
    return table


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    widened = np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16
    return widened.view(np.float32)


def pack_nbit(values: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    vals_per_u32 = 32 // bits
    if values.shape[-1] % vals_per_u32 != 0:
        raise ValueError(f"Last dimension {values.shape[-1]} is not divisible by {vals_per_u32}")

    flat = values.reshape(-1, values.shape[-1])
    packed = np.zeros((flat.shape[0], flat.shape[1] // vals_per_u32), dtype=np.uint32)
    for i in range(vals_per_u32):
        packed |= flat[:, i::vals_per_u32].astype(np.uint32) << (i * bits)
    return packed.reshape(values.shape[:-1] + (values.shape[-1] // vals_per_u32,))


def dequantize_fp8_blocked(
    weight_u8: np.ndarray,
    scale_inv: np.ndarray,
    fp8_table: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    if weight_u8.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError("Expected 2D weight and scale tensors")
    rows, cols = weight_u8.shape
    block_rows, block_cols = scale_inv.shape
    if block_rows * block_size != rows or block_cols * block_size != cols:
        raise ValueError(
            f"Scale shape {scale_inv.shape} does not match weight shape {weight_u8.shape}"
        )

    fp = fp8_table[weight_u8]
    fp = fp.reshape(block_rows, block_size, block_cols, block_size)
    dequant = fp * scale_inv[:, None, :, None]
    return dequant.reshape(rows, cols)


def quantize_affine_per_group(
    matrix: np.ndarray,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
):
    rows, cols = matrix.shape
    if cols % group_size != 0:
        raise ValueError(f"Input dimension {cols} is not divisible by {group_size}")
    num_groups = cols // group_size
    grouped = matrix.reshape(rows, num_groups, group_size)

    mins = grouped.min(axis=2)
    maxs = grouped.max(axis=2)
    levels = float((1 << bits) - 1)
    scales = (maxs - mins) / levels
    biases = mins

    zero_mask = scales == 0.0
    safe_scales = np.where(zero_mask, 1.0, scales)

    q = np.round((grouped - biases[:, :, None]) / safe_scales[:, :, None])
    q = np.clip(q, 0, int(levels)).astype(np.uint8)
    q = np.where(zero_mask[:, :, None], 0, q)

    recon = q.astype(np.float32) * scales[:, :, None] + biases[:, :, None]
    diff = recon.astype(np.float64) - grouped.astype(np.float64)
    sq_err = float(np.sum(diff * diff))
    value_count = int(grouped.size)

    packed = pack_nbit(q.reshape(rows, cols), bits)
    return packed, f32_to_bf16(scales), f32_to_bf16(biases), sq_err, value_count
