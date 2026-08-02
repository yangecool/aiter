#!/usr/bin/env python3
"""CPU-only tests for native gfx1201 SageAttention2 configuration defaults."""

from __future__ import annotations

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "aiter/ops/flydsl/sage_attention.py"
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


class SageAttentionGfx1201ConfigTests(unittest.TestCase):
    def test_production_default_uses_k_prefetch_and_fp8_offset(self):
        config = CONFIG()
        self.assertEqual((config.block_m, config.block_n), (128, 32))
        self.assertEqual(config.kv_prefetch_mode, "k")
        self.assertFalse(config.pre_load_v)
        self.assertTrue(config.use_fp8_p_offset)
        config.validate()

    def test_partial_winner_mapping_uses_production_schedule(self):
        config = CONFIG.from_mapping({"backend": "flydsl_v2"})
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
