# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental native SageAttention2 orchestration for gfx1201.

The attention core consumes pre-quantized tensors. This module owns the V2
preprocessing contract and deliberately keeps the first dispatch surface
narrow: dense, non-causal, head-dim-128 self-attention without GQA.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import flydsl
import torch
from packaging.version import Version

from aiter import logger
from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant
from aiter.utility import dtypes as aiter_dtypes

from .kernels.sage_attention_gfx1201 import build_sage_attention_v2_core

_MIN_NATIVE_FLYDSL_VERSION = Version("0.2.0")

__all__ = [
    "PreparedSageAttentionGfx1201",
    "SageAttentionGfx1201Config",
    "flydsl_sage_attention_v2_func",
    "launch_prepared_sage_attention_v2_gfx1201",
    "prepare_sage_attention_v2_gfx1201",
    "sage_attention_v2_gfx1201_support_reason",
]


@dataclass(frozen=True)
class SageAttentionGfx1201Config:
    block_m: int = 128
    block_n: int = 32
    waves_per_eu: int = 2
    lds_padding: int = 16
    pre_load_v: bool = False
    kv_prefetch_mode: str = "none"
    use_fp8_p_offset: bool = False

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object] | None,
    ) -> "SageAttentionGfx1201Config":
        if config is None:
            return cls()
        return cls(
            block_m=int(config.get("BLOCK_M", config.get("block_m", 128))),
            block_n=int(config.get("BLOCK_N", config.get("block_n", 32))),
            waves_per_eu=int(
                config.get("waves_per_eu", config.get("WAVES_PER_EU", 2))
            ),
            lds_padding=int(
                config.get("lds_padding", config.get("LDS_PADDING", 16))
            ),
            pre_load_v=bool(
                config.get("pre_load_v", config.get("PRE_LOAD_V", False))
            ),
            kv_prefetch_mode=str(
                config.get(
                    "kv_prefetch_mode",
                    config.get("KV_PREFETCH_MODE", "none"),
                )
            ).lower(),
            use_fp8_p_offset=bool(
                config.get(
                    "use_fp8_p_offset",
                    config.get("USE_FP8_P_OFFSET", False),
                )
            ),
        )

    def validate(self) -> None:
        if self.block_m not in (64, 128, 256):
            raise ValueError("native gfx1201 Sage BLOCK_M must be 64, 128 or 256")
        if self.block_n not in (32, 64):
            raise ValueError("native gfx1201 Sage BLOCK_N must be 32 or 64")
        if self.waves_per_eu not in (1, 2, 3, 4):
            raise ValueError("native gfx1201 Sage waves_per_eu must be in [1, 4]")
        if self.lds_padding not in (4, 8, 16):
            raise ValueError("native gfx1201 Sage LDS padding must be 4, 8, or 16")
        if self.kv_prefetch_mode not in ("none", "v", "k", "kv"):
            raise ValueError(
                "native gfx1201 Sage KV prefetch mode must be none, v, k or kv"
            )
        if self.kv_prefetch_mode != "none" and (
            self.block_m not in (64, 128) or self.block_n != 32
        ):
            raise ValueError(
                "native gfx1201 Sage KV prefetch requires BM64/128 and BN32"
            )


@dataclass
class PreparedSageAttentionGfx1201:
    q_int8: torch.Tensor
    k_int8: torch.Tensor
    v_fp8_transposed: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    out_padded: torch.Tensor
    batch_size: int
    valid_seq_len: int
    padded_seq_len: int
    num_heads: int
    output_dtype: str
    layout: str
    config: SageAttentionGfx1201Config


def sage_attention_v2_gfx1201_support_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    sm_margin: int = 0,
    return_lse: bool = False,
    layout: str = "bshd",
    block_lut=None,
) -> str | None:
    """Return ``None`` when the first native V2 path supports this call."""

    if layout not in ("bshd", "bhsd"):
        return f"layout={layout!r} is not supported"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return "Q, K and V must be rank-4 tensors"
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        return "Q, K and V must be on the same CUDA/HIP device"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"dtype={q.dtype} is not supported"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "Q, K and V must have the same dtype"
    if q.requires_grad or k.requires_grad or v.requires_grad:
        return "the native path is inference-only"
    if causal:
        return "causal attention is not implemented"
    if window_size != (-1, -1):
        return "sliding-window attention is not implemented"
    if attention_chunk not in (0, 1):
        return "attention_chunk > 1 is not implemented"
    if softcap != 0.0 or sm_margin != 0:
        return "softcap and sm_margin are not implemented"
    if return_lse:
        return "LSE output is not implemented"
    if block_lut is not None:
        return "block-sparse attention is not implemented"

    seq_dim, head_dim = ((1, 2) if layout == "bshd" else (2, 1))
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return "head dimension must be 128"
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        return "Q, K and V batch sizes must match"
    if q.shape[seq_dim] != k.shape[seq_dim] or q.shape[seq_dim] != v.shape[seq_dim]:
        return "the first native path supports self-attention only"
    if q.shape[head_dim] != k.shape[head_dim] or q.shape[head_dim] != v.shape[head_dim]:
        return "the first native path does not support GQA/MQA"
    if q.shape[seq_dim] <= 0:
        return "sequence length must be positive"
    return None


@lru_cache(maxsize=32)
def _get_kernel(
    num_heads: int,
    output_dtype: str,
    block_m: int,
    block_n: int,
    waves_per_eu: int,
    lds_padding: int,
    pre_load_v: bool,
    kv_prefetch_mode: str,
    use_fp8_p_offset: bool,
):
    logger.info(
        "[FlyDSL] dispatching native gfx1201 SageAttention2: "
        f"dtype={output_dtype}, H={num_heads}, D=128, "
        f"BM={block_m}, BN={block_n}, WPE={waves_per_eu}, "
        f"LDS_PAD={lds_padding}, PRE_LOAD_V={pre_load_v}, "
        f"KV_PREFETCH={kv_prefetch_mode}, FP8_P_OFFSET={use_fp8_p_offset}"
    )
    return build_sage_attention_v2_core(
        num_heads=num_heads,
        head_dim=128,
        output_dtype=output_dtype,
        block_m=block_m,
        block_n=block_n,
        waves_per_eu=waves_per_eu,
        lds_padding=lds_padding,
        pre_load_v=pre_load_v,
        kv_prefetch_mode=kv_prefetch_mode,
        use_fp8_p_offset=use_fp8_p_offset,
    )


def _to_bshd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "bshd":
        return tensor.contiguous()
    return tensor.transpose(1, 2).contiguous()


def _pad_bshd(tensor: torch.Tensor, padded_seq_len: int) -> torch.Tensor:
    if tensor.shape[1] == padded_seq_len:
        return tensor.contiguous()
    padded = torch.zeros(
        (tensor.shape[0], padded_seq_len, tensor.shape[2], tensor.shape[3]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, : tensor.shape[1]].copy_(tensor)
    return padded


def _pad_q_scale(q_scale: torch.Tensor, padded_seq_len: int) -> torch.Tensor:
    target_groups = padded_seq_len // 32
    if q_scale.shape[-1] == target_groups:
        return q_scale.contiguous()
    padded = torch.ones(
        (*q_scale.shape[:-1], target_groups),
        dtype=q_scale.dtype,
        device=q_scale.device,
    )
    padded[..., : q_scale.shape[-1]].copy_(q_scale)
    return padded


def _transpose_pad_v(
    v_fp8_bshd: torch.Tensor,
    padded_seq_len: int,
) -> torch.Tensor:
    """Create RDNA-native contiguous ``[B, H, D, padded_S]`` V storage."""

    batch, seq_len, heads, head_dim = v_fp8_bshd.shape
    transposed = torch.zeros(
        (batch, heads, head_dim, padded_seq_len),
        dtype=v_fp8_bshd.dtype,
        device=v_fp8_bshd.device,
    )
    transposed[..., :seq_len].copy_(v_fp8_bshd.permute(0, 2, 3, 1))
    return transposed


def _prepare_sage_attention_v2_gfx1201_on_current_stream(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    layout: str = "bshd",
    smooth_k: bool = True,
    config: Mapping[str, object] | None = None,
) -> PreparedSageAttentionGfx1201:
    """Quantize and lay out inputs for the native gfx1201 V2 core."""

    reason = sage_attention_v2_gfx1201_support_reason(q, k, v, layout=layout)
    if reason is not None:
        raise ValueError(f"native gfx1201 SageAttention2 is unavailable: {reason}")

    selected = SageAttentionGfx1201Config.from_mapping(config)
    selected.validate()
    q_bshd = _to_bshd(q, layout)
    k_bshd = _to_bshd(k, layout)
    v_bshd = _to_bshd(v, layout)
    batch, valid_seq_len, num_heads, head_dim = q_bshd.shape
    padded_seq_len = (
        (valid_seq_len + selected.block_n - 1) // selected.block_n
    ) * selected.block_n
    softmax_scale = softmax_scale or head_dim**-0.5

    fp8_dtype = aiter_dtypes.fp8
    quantized = sage_quant(
        q_bshd,
        k_bshd,
        v_bshd,
        fp8_dtype,
        torch.finfo(fp8_dtype).max,
        BLKQ=32,
        BLKK=selected.block_n,
        sm_scale=softmax_scale,
        layout="bshd",
        smooth_k=smooth_k,
        return_lse=False,
    )
    q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = quantized

    q_padded = _pad_bshd(q_int8, padded_seq_len)
    k_padded = _pad_bshd(k_int8, padded_seq_len)
    q_scale_padded = _pad_q_scale(q_scale, padded_seq_len)
    k_scale = k_scale.contiguous()
    v_rdna = _transpose_pad_v(v_fp8, padded_seq_len)
    v_scale = v_scale.contiguous()
    out_padded = torch.empty(
        (batch, padded_seq_len, num_heads, head_dim),
        dtype=q.dtype,
        device=q.device,
    )

    output_dtype = "bf16" if q.dtype == torch.bfloat16 else "f16"
    return PreparedSageAttentionGfx1201(
        q_int8=q_padded,
        k_int8=k_padded,
        v_fp8_transposed=v_rdna,
        q_scale=q_scale_padded,
        k_scale=k_scale,
        v_scale=v_scale,
        out_padded=out_padded,
        batch_size=batch,
        valid_seq_len=valid_seq_len,
        padded_seq_len=padded_seq_len,
        num_heads=num_heads,
        output_dtype=output_dtype,
        layout=layout,
        config=selected,
    )


def prepare_sage_attention_v2_gfx1201(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    layout: str = "bshd",
    smooth_k: bool = True,
    config: Mapping[str, object] | None = None,
    stream: torch.cuda.Stream | None = None,
) -> PreparedSageAttentionGfx1201:
    """Prepare V2 inputs on the same device/stream used by the core."""

    flydsl_version = Version(flydsl.__version__.split("+")[0])
    if flydsl_version < _MIN_NATIVE_FLYDSL_VERSION:
        raise RuntimeError(
            "native gfx1201 SageAttention2 requires FlyDSL "
            f">={_MIN_NATIVE_FLYDSL_VERSION}, found {flydsl.__version__}"
        )
    reason = sage_attention_v2_gfx1201_support_reason(q, k, v, layout=layout)
    if reason is not None:
        raise ValueError(f"native gfx1201 SageAttention2 is unavailable: {reason}")
    with torch.cuda.device(q.device.index):
        prepare_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if prepare_stream.device != q.device:
            raise ValueError(f"stream must be on {q.device}, got {prepare_stream.device}")
        with torch.cuda.stream(prepare_stream):
            return _prepare_sage_attention_v2_gfx1201_on_current_stream(
                q,
                k,
                v,
                softmax_scale,
                layout=layout,
                smooth_k=smooth_k,
                config=config,
            )


def launch_prepared_sage_attention_v2_gfx1201(
    prepared: PreparedSageAttentionGfx1201,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Launch only the native attention core on already prepared tensors."""

    q_device = prepared.q_int8.device
    selected = prepared.config
    with torch.cuda.device(q_device.index):
        launch_stream = (
            torch.cuda.current_stream(q_device) if stream is None else stream
        )
        if launch_stream.device != q_device:
            raise ValueError(f"stream must be on {q_device}, got {launch_stream.device}")
        kernel = _get_kernel(
            prepared.num_heads,
            prepared.output_dtype,
            selected.block_m,
            selected.block_n,
            selected.waves_per_eu,
            selected.lds_padding,
            selected.pre_load_v,
            selected.kv_prefetch_mode,
            selected.use_fp8_p_offset,
        )
        kernel(
            prepared.q_int8,
            prepared.k_int8,
            prepared.v_fp8_transposed,
            prepared.q_scale,
            prepared.k_scale,
            prepared.v_scale,
            prepared.out_padded,
            prepared.batch_size,
            prepared.padded_seq_len,
            prepared.valid_seq_len,
            stream=launch_stream,
        )

    out = prepared.out_padded[:, : prepared.valid_seq_len]
    if prepared.layout == "bhsd":
        return out.transpose(1, 2)
    return out


def flydsl_sage_attention_v2_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    layout: str = "bshd",
    smooth_k: bool = True,
    config: Mapping[str, object] | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run preprocessing and the native gfx1201 SageAttention2 core."""

    prepared = prepare_sage_attention_v2_gfx1201(
        q,
        k,
        v,
        softmax_scale,
        layout=layout,
        smooth_k=smooth_k,
        config=config,
        stream=stream,
    )
    out = launch_prepared_sage_attention_v2_gfx1201(prepared, stream=stream)
    return out.contiguous()
