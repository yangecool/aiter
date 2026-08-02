#!/usr/bin/env python3
"""CPU-only tests for native gfx1201 SageAttention2 configuration defaults."""

from __future__ import annotations

import ast
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "aiter/ops/flydsl/sage_attention.py"
KERNEL_PATH = REPO_ROOT / "aiter/ops/flydsl/kernels/sage_attention_gfx1201.py"
WRAPPER_PATH = REPO_ROOT / "aiter/ops/triton/attention/fav3_sage.py"
SOURCE = CONFIG_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
CONFIG_CLASS = next(
    node
    for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "SageAttentionGfx1201Config"
)
NAMESPACE = {
    "dataclass": dataclass,
    "Mapping": Mapping,
}
exec(
    compile(ast.Module(body=[CONFIG_CLASS], type_ignores=[]), str(CONFIG_PATH), "exec"),
    NAMESPACE,
)
CONFIG = NAMESPACE["SageAttentionGfx1201Config"]


class _FakeDevice:
    type = "cuda"


class _FakeTensor:
    def __init__(self, dtype):
        self.shape = (1, 64, 4, 128)
        self.ndim = 4
        self.device = _FakeDevice()
        self.dtype = dtype
        self.requires_grad = False


FAKE_BF16 = object()
SUPPORT_NAMESPACE = {
    "torch": types.SimpleNamespace(
        Tensor=_FakeTensor,
        bfloat16=FAKE_BF16,
        float16=object(),
    )
}
SUPPORT_FUNCTION = next(
    node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "sage_attention_v2_gfx1201_support_reason"
)
exec(
    compile(
        ast.Module(body=[SUPPORT_FUNCTION], type_ignores=[]),
        str(CONFIG_PATH),
        "exec",
    ),
    SUPPORT_NAMESPACE,
)
SUPPORT_REASON = SUPPORT_NAMESPACE["sage_attention_v2_gfx1201_support_reason"]


class SageAttentionGfx1201ConfigTests(unittest.TestCase):
    def test_native_contract_accepts_lse_for_ring_attention(self):
        q = _FakeTensor(FAKE_BF16)
        self.assertIsNone(SUPPORT_REASON(q, q, q, return_lse=True))

        kernel_source = KERNEL_PATH.read_text(encoding="utf-8")
        self.assertIn("RETURN_LSE = bool(return_lse)", kernel_source)
        self.assertIn("log2_l = rocdl.log", kernel_source)
        self.assertIn("lse_global_index(q_row)", kernel_source)

    def test_canonical_backend_names_sage_and_target_arch(self):
        wrapper_tree = ast.parse(WRAPPER_PATH.read_text(encoding="utf-8"))
        backends = next(
            ast.literal_eval(node.value)
            for node in wrapper_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_GFX1201_NATIVE_BACKENDS"
                for target in node.targets
            )
        )
        self.assertEqual(backends[0], "sage_attn_v2_gfx1201")
        self.assertIn("flydsl_v2", backends)
        self.assertIn("native_v2", backends)

    def test_production_default_uses_k_prefetch_and_fp8_offset(self):
        config = CONFIG()
        self.assertEqual((config.block_m, config.block_n), (128, 32))
        self.assertEqual(config.kv_prefetch_mode, "k")
        self.assertFalse(config.pre_load_v)
        self.assertTrue(config.use_fp8_p_offset)
        config.validate()

    def test_partial_winner_mapping_uses_production_schedule(self):
        config = CONFIG.from_mapping({"backend": "sage_attn_v2_gfx1201"})
        self.assertEqual(config.kv_prefetch_mode, "k")
        self.assertTrue(config.use_fp8_p_offset)
        config.validate()

    def test_other_tiles_do_not_inherit_unsupported_k_prefetch(self):
        for mapping in (
            {"BLOCK_M": 64, "BLOCK_N": 32},
            {"BLOCK_M": 128, "BLOCK_N": 64},
            {"BLOCK_M": 256, "BLOCK_N": 32},
        ):
            with self.subTest(mapping=mapping):
                config = CONFIG.from_mapping(mapping)
                self.assertEqual(config.kv_prefetch_mode, "none")
                self.assertTrue(config.use_fp8_p_offset)
                config.validate()

    def test_explicit_tuning_controls_override_defaults(self):
        config = CONFIG.from_mapping(
            {
                "KV_PREFETCH_MODE": "none",
                "USE_FP8_P_OFFSET": False,
            }
        )
        self.assertEqual(config.kv_prefetch_mode, "none")
        self.assertFalse(config.use_fp8_p_offset)
        config.validate()


if __name__ == "__main__":
    unittest.main()
