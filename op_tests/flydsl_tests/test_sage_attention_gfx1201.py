# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for native gfx1201 SageAttention2."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flydsl")
from aiter.ops.flydsl import (  # noqa: E402
    flydsl_sage_attention_v2_func,
    is_flydsl_available,
)

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)


def _is_gfx1201() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return arch.lower().split(":")[0].startswith("gfx1201")


pytestmark = pytest.mark.skipif(
    not _is_gfx1201(),
    reason="native SageAttention2 is gfx1201-only",
)


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    )
    return out.transpose(1, 2).contiguous()


@pytest.mark.parametrize("seq_len", [31, 32, 33, 63, 64, 65, 129])
def test_native_sage_v2_masks_padded_keys(seq_len: int):
    torch.manual_seed(20260802)
    shape = (1, seq_len, 2, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    out = flydsl_sage_attention_v2_func(
        q,
        k,
        v,
        config={"BLOCK_M": 128, "BLOCK_N": 32, "waves_per_eu": 2},
    )
    ref = _reference(q, k, v)

    cosine = F.cosine_similarity(
        out.float().reshape(-1, 128),
        ref.float().reshape(-1, 128),
        dim=-1,
    )
    assert cosine.min().item() > 0.98
    assert cosine.mean().item() > 0.995


@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("block_n", [32, 64])
def test_native_sage_v2_layout_and_bn(layout: str, block_n: int):
    torch.manual_seed(7)
    shape = (1, 257, 5, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    ref = _reference(q, k, v)
    if layout == "bhsd":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        ref = ref.transpose(1, 2).contiguous()

    out = flydsl_sage_attention_v2_func(
        q,
        k,
        v,
        layout=layout,
        config={"BLOCK_M": 128, "BLOCK_N": block_n, "waves_per_eu": 2},
    )
    cosine = F.cosine_similarity(
        out.float().reshape(-1, 128),
        ref.float().reshape(-1, 128),
        dim=-1,
    )
    assert cosine.min().item() > 0.98
    assert cosine.mean().item() > 0.995
