# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile-time configuration selection for gfx1201 FlyDSL Flash Attention."""

from __future__ import annotations

import math
from typing import NamedTuple


class FlyDSLFlashAttentionConfig(NamedTuple):
    block_m: int
    block_n: int
    waves_per_eu: int
    global_load_vector_width: int


LEGACY_CONFIG = FlyDSLFlashAttentionConfig(128, 32, 2, 16)
# Direct A/B winner across all six TI2V/A14B gfx1201 production shapes.
PRODUCTION_CONFIG = FlyDSLFlashAttentionConfig(256, 64, 3, 16)
_MAX_NONCAUSAL_PAD_RATIO = 0.005


def select_flydsl_flash_attention_config(
    *,
    arch: str,
    self_attention: bool,
    dtype_str: str,
    head_dim: int,
    causal: bool,
) -> FlyDSLFlashAttentionConfig:
    """Select tuned production config, retaining legacy behavior out of scope."""
    arch_base = arch.lower().split(":")[0]
    use_production_config = (
        arch_base.startswith("gfx1201")
        and self_attention
        and dtype_str == "bf16"
        and head_dim == 128
        and not causal
    )
    if not use_production_config:
        return LEGACY_CONFIG
    return PRODUCTION_CONFIG


def can_use_gfx1201_flydsl_dense_attention(
    *,
    arch: str,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    q_dtype: str,
    k_dtype: str,
    v_dtype: str,
    same_cuda_device: bool,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, ...],
    has_bias: bool,
    has_alibi: bool,
    return_lse: bool,
    return_attn_probs: bool,
    has_cu_seqlens: bool,
    has_sink: bool,
    num_splits: int,
    requires_grad: bool,
) -> bool:
    """Return whether the public dense MHA API can preserve FlyDSL semantics."""
    if (
        not same_cuda_device
        or q_shape != k_shape
        or q_shape != v_shape
        or len(q_shape) != 4
        or not (q_dtype == k_dtype == v_dtype == "bf16")
    ):
        return False

    batch_size, sequence_length, num_heads, head_dim = q_shape
    if batch_size <= 0 or sequence_length <= 0 or num_heads <= 0:
        return False
    config = select_flydsl_flash_attention_config(
        arch=arch,
        self_attention=True,
        dtype_str=q_dtype,
        head_dim=head_dim,
        causal=causal,
    )
    if config != PRODUCTION_CONFIG:
        return False

    standard_scale = head_dim**-0.5
    if softmax_scale is not None and not math.isclose(
        float(softmax_scale), standard_scale, rel_tol=1e-7, abs_tol=0.0
    ):
        return False
    if len(window_size) < 2 or tuple(window_size[:2]) != (-1, -1):
        return False
    if len(window_size) > 2 and window_size[2] != 0:
        return False
    if (
        dropout_p != 0.0
        or has_bias
        or has_alibi
        or return_lse
        or return_attn_probs
        or has_cu_seqlens
        or has_sink
        or num_splits != 0
        or requires_grad
    ):
        return False

    tile_multiple = math.lcm(config.block_m, config.block_n)
    sequence_length_padded = (
        (sequence_length + tile_multiple - 1) // tile_multiple
    ) * tile_multiple
    padding_tokens = sequence_length_padded - sequence_length
    return (
        padding_tokens == 0
        or padding_tokens / sequence_length_padded <= _MAX_NONCAUSAL_PAD_RATIO
    )
