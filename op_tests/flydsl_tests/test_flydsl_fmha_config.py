#!/usr/bin/env python3

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "aiter/ops/flydsl_fmha_config.py"
SPEC = importlib.util.spec_from_file_location("flydsl_fmha_config", CONFIG_PATH)
CONFIG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CONFIG
SPEC.loader.exec_module(CONFIG)


class FlyDSLFMHAConfigTests(unittest.TestCase):
    def select(self, **overrides):
        values = {
            "arch": "gfx1201:sramecc+:xnack-",
            "self_attention": True,
            "dtype_str": "bf16",
            "head_dim": 128,
            "causal": False,
        }
        values.update(overrides)
        return CONFIG.select_flydsl_flash_attention_config(**values)

    def test_production_dispatch(self):
        self.assertEqual(
            CONFIG.PRODUCTION_CONFIG,
            CONFIG.FlyDSLFlashAttentionConfig(256, 64, 3, 16),
        )
        self.assertEqual(self.select(), CONFIG.PRODUCTION_CONFIG)

    def test_out_of_scope_uses_legacy_config(self):
        overrides = (
            {"arch": "gfx1200"},
            {"self_attention": False},
            {"dtype_str": "f16"},
            {"head_dim": 64},
            {"causal": True},
        )
        for override in overrides:
            with self.subTest(override=override):
                self.assertEqual(self.select(**override), CONFIG.LEGACY_CONFIG)

    def route(self, **overrides):
        values = {
            "arch": "gfx1201",
            "q_shape": (1, 109120, 3, 128),
            "k_shape": (1, 109120, 3, 128),
            "v_shape": (1, 109120, 3, 128),
            "q_dtype": "bf16",
            "k_dtype": "bf16",
            "v_dtype": "bf16",
            "same_cuda_device": True,
            "dropout_p": 0.0,
            "softmax_scale": 128**-0.5,
            "causal": False,
            "window_size": (-1, -1, 0),
            "has_bias": False,
            "has_alibi": False,
            "return_lse": False,
            "return_attn_probs": False,
            "has_cu_seqlens": False,
            "has_sink": False,
            "num_splits": 0,
            "requires_grad": False,
        }
        values.update(overrides)
        return CONFIG.can_use_gfx1201_flydsl_dense_attention(**values)

    def test_public_route_accepts_all_six_production_shapes(self):
        shapes = (
            (1, 109120, 24, 128),
            (1, 109120, 3, 128),
            (1, 75600, 10, 128),
            (1, 75600, 5, 128),
            (1, 32760, 10, 128),
            (1, 32760, 5, 128),
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                self.assertTrue(
                    self.route(q_shape=shape, k_shape=shape, v_shape=shape)
                )

    def test_public_route_is_not_limited_to_tuned_sequence_lengths(self):
        shape = (1, 65520, 10, 128)
        self.assertTrue(self.route(q_shape=shape, k_shape=shape, v_shape=shape))

    def test_public_route_is_not_limited_to_tuned_head_counts(self):
        for heads in (2, 12, 40):
            shape = (1, 65536, heads, 128)
            with self.subTest(heads=heads):
                self.assertTrue(
                    self.route(q_shape=shape, k_shape=shape, v_shape=shape)
                )

    def test_public_route_rejects_unsupported_semantics(self):
        cases = (
            {"q_dtype": "f16"},
            {"same_cuda_device": False},
            {"dropout_p": 0.1},
            {"softmax_scale": 0.5},
            {"causal": True},
            {"window_size": (128, 128, 0)},
            {"has_bias": True},
            {"has_alibi": True},
            {"return_lse": True},
            {"return_attn_probs": True},
            {"has_cu_seqlens": True},
            {"has_sink": True},
            {"num_splits": 2},
            {"requires_grad": True},
            {
                "q_shape": (1, 0, 3, 128),
                "k_shape": (1, 0, 3, 128),
                "v_shape": (1, 0, 3, 128),
            },
            {
                "q_shape": (1, 1024, 0, 128),
                "k_shape": (1, 1024, 0, 128),
                "v_shape": (1, 1024, 0, 128),
            },
            {
                "q_shape": (1, 109120, 3, 128),
                "k_shape": (1, 1024, 3, 128),
            },
            {
                "q_shape": (1, 129, 3, 128),
                "k_shape": (1, 129, 3, 128),
                "v_shape": (1, 129, 3, 128),
            },
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(self.route(**case))

    def test_public_mha_source_calls_flydsl_gate(self):
        source = (REPO_ROOT / "aiter/ops/mha.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        public_mha = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "flash_attn_func"
        )
        called_names = {
            node.func.id
            for node in ast.walk(public_mha)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn(
            "can_use_gfx1201_flydsl_dense_attention", called_names
        )
        self.assertIn("flydsl_flash_attn_func", called_names)

    def test_kernel_cache_signature_covers_compile_time_config(self):
        source = (REPO_ROOT / "aiter/ops/flydsl/fmha_kernels.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        get_kernel = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_get_kernel"
        )
        argument_names = {argument.arg for argument in get_kernel.args.args}
        expected = {
            "waves_per_eu",
            "block_m",
            "block_n",
            "global_load_vector_width",
        }
        self.assertTrue(expected <= argument_names)
        builder_call = next(
            node
            for node in ast.walk(get_kernel)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_flash_attn_func_module"
        )
        self.assertTrue(expected <= {keyword.arg for keyword in builder_call.keywords})

    def test_kernel_cache_logs_flydsl_dispatch(self):
        source = (REPO_ROOT / "aiter/ops/flydsl/fmha_kernels.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        get_kernel = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_get_kernel"
        )
        log_call = next(
            node
            for node in ast.walk(get_kernel)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
            and node.func.attr == "info"
        )
        message = ast.get_source_segment(source, log_call.args[0])
        self.assertIn("[FlyDSL]", message)
        self.assertIn("Flash Attention", message)

    def test_builder_exposes_explicit_vector_width(self):
        source = (
            REPO_ROOT / "aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        builder = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_flash_attn_func_module_primary"
        )
        argument_names = {argument.arg for argument in builder.args.args}
        self.assertIn("global_load_vector_width", argument_names)
        self.assertNotIn("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", source)


if __name__ == "__main__":
    unittest.main()
