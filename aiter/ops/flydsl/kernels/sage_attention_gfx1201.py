# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native RDNA4 building blocks for SageAttention2 on gfx1201.

The public attention kernel is added incrementally.  This module starts with
small ABI probes for the two matrix operations that define the V2 compute
path.  They are also useful as numerical tests on real hardware because their
inputs and outputs are the packed per-lane WMMA fragments, with no attention
or quantization logic around them.

gfx1201 wave32 fragment ABI for one 16x16x16 operation:

* INT8/FP8 A and B: two packed i32 VGPRs per lane (eight elements);
* INT8 result: eight i32 VGPRs per lane;
* FP8 result: eight f32 VGPRs per lane.

The probes intentionally use the ROCDL operations instead of a generic dot so
compilation must either produce the required native ISA or fail.
"""

from __future__ import annotations

import argparse
import math as host_math
import os
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


GFX1201_WAVE_SIZE = 32
GFX1201_WMMA_M = 16
GFX1201_WMMA_N = 16
GFX1201_WMMA_K = 16
GFX1201_WMMA_INPUT_I32S_PER_LANE = 2
GFX1201_WMMA_ACCUMULATORS_PER_LANE = 8

_QK_PROBE_NAME = "sage_gfx1201_qk_i8_wmma_16x16x16"
_PV_PROBE_NAME = "sage_gfx1201_pv_fp8_wmma_16x16x16"
_QK_ISA = "v_wmma_i32_16x16x16_iu8"
_PV_ISA = "v_wmma_f32_16x16x16_fp8_fp8"
_LOG2E = host_math.log2(host_math.e)


def _raw(value):
    if isinstance(value, ir.Value):
        return value
    if hasattr(value, "ir_value"):
        return _raw(value.ir_value())
    return ir.Value._CAPICreate(value._CAPIPtr)


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _pointer_to_llvm_ptr(ptr) -> ir.Value:
    ptr_i64 = arith.index_cast(T.i64, fx.ptrtoint(ptr))
    return _llvm.IntToPtrOp(_llvm_ptr_ty(), ptr_i64).result


def _load_vector(ptr, element_index, elem_type, vector_type):
    address = buffer_ops.get_element_ptr(
        ptr,
        fx.Int64(element_index),
        elem_type=elem_type,
    )
    return _llvm.LoadOp(vector_type, _raw(address)).result


def _store_vector(ptr, element_index, elem_type, value):
    address = buffer_ops.get_element_ptr(
        ptr,
        fx.Int64(element_index),
        elem_type=elem_type,
    )
    _llvm.StoreOp(_raw(value), _raw(address))


def _set_waves_per_eu(waves_per_eu: int):
    ctx = CompilationContext.get_current()
    for op in ctx.gpu_module_body.operations:
        if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
            op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                T.i32,
                int(waves_per_eu),
            )


def build_sage_wmma_qk_probe(waves_per_eu: int = 2):
    """Build a signed INT8 16x16x16 WMMA packed-fragment probe."""

    @flyc.kernel(name=_QK_PROBE_NAME, known_block_size=[GFX1201_WAVE_SIZE, 1, 1])
    def qk_probe(a: fx.Pointer, b: fx.Pointer, d: fx.Pointer):
        v2i32_type = Vec.make_type(2, fx.Int32)
        v8i32_type = Vec.make_type(8, fx.Int32)
        lane = fx.Index(gpu.thread_idx.x)
        a_ptr = _pointer_to_llvm_ptr(a)
        b_ptr = _pointer_to_llvm_ptr(b)
        d_ptr = _pointer_to_llvm_ptr(d)

        input_offset = lane * GFX1201_WMMA_INPUT_I32S_PER_LANE
        output_offset = lane * GFX1201_WMMA_ACCUMULATORS_PER_LANE
        a_frag = _load_vector(a_ptr, input_offset, T.i32, v2i32_type)
        b_frag = _load_vector(b_ptr, input_offset, T.i32, v2i32_type)
        accum = Vec.from_elements([fx.Int32(0)] * 8, fx.Int32).ir_value()

        result = rocdl.wmma_i32_16x16x16_iu8(
            v8i32_type,
            a_frag,
            b_frag,
            accum,
            signA=True,
            signB=True,
            clamp=False,
        ).result
        _store_vector(d_ptr, output_offset, T.i32, result)

    @flyc.jit
    def launch_qk_probe(
        a: fx.Pointer,
        b: fx.Pointer,
        d: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = qk_probe(a, b, d)
        _set_waves_per_eu(waves_per_eu)
        launcher.launch(
            grid=(1, 1, 1),
            block=(GFX1201_WAVE_SIZE, 1, 1),
            stream=stream,
        )

    return launch_qk_probe


def build_sage_wmma_pv_probe(waves_per_eu: int = 2):
    """Build an E4M3 FP8 16x16x16 WMMA packed-fragment probe."""

    @flyc.kernel(name=_PV_PROBE_NAME, known_block_size=[GFX1201_WAVE_SIZE, 1, 1])
    def pv_probe(a: fx.Pointer, b: fx.Pointer, d: fx.Pointer):
        v2i32_type = Vec.make_type(2, fx.Int32)
        v8f32_type = Vec.make_type(8, fx.Float32)
        lane = fx.Index(gpu.thread_idx.x)
        a_ptr = _pointer_to_llvm_ptr(a)
        b_ptr = _pointer_to_llvm_ptr(b)
        d_ptr = _pointer_to_llvm_ptr(d)

        input_offset = lane * GFX1201_WMMA_INPUT_I32S_PER_LANE
        output_offset = lane * GFX1201_WMMA_ACCUMULATORS_PER_LANE
        a_frag = _load_vector(a_ptr, input_offset, T.i32, v2i32_type)
        b_frag = _load_vector(b_ptr, input_offset, T.i32, v2i32_type)
        accum = Vec.from_elements([fx.Float32(0.0)] * 8, fx.Float32).ir_value()

        result = rocdl.wmma_f32_16x16x16_fp8_fp8(
            v8f32_type,
            a_frag,
            b_frag,
            accum,
        ).result
        _store_vector(d_ptr, output_offset, T.f32, result)

    @flyc.jit
    def launch_pv_probe(
        a: fx.Pointer,
        b: fx.Pointer,
        d: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = pv_probe(a, b, d)
        _set_waves_per_eu(waves_per_eu)
        launcher.launch(
            grid=(1, 1, 1),
            block=(GFX1201_WAVE_SIZE, 1, 1),
            stream=stream,
        )

    return launch_pv_probe


def build_sage_attention_v2_core(
    num_heads: int,
    head_dim: int = 128,
    output_dtype: str = "bf16",
    block_m: int = 128,
    block_n: int = 32,
    waves_per_eu: int = 2,
    lds_padding: int = 16,
):
    """Build the pre-quantized, non-causal gfx1201 SageAttention2 core.

    Q and K are sequence-major signed INT8. Q scales cover 32 query rows and
    already include ``softmax_scale * log2(e)``; K scales cover ``block_n``
    rows. V is E4M3 FP8 in RDNA-native ``[B, H, D, padded_S]`` order with one
    FP32 scale for every ``[B, H, D]`` channel. Output is BSHD BF16/FP16.

    Storage sequence length must be padded to ``block_n`` by the host wrapper.
    The separate valid sequence length masks padded keys out of the online
    softmax. The first production path deliberately handles self-attention
    only; asymmetric lengths and GQA stay on the existing fallback until
    separately validated.
    """

    if head_dim != 128:
        raise ValueError("gfx1201 SageAttention2 currently requires head_dim=128")
    if output_dtype not in ("bf16", "f16"):
        raise ValueError(f"unsupported Sage output dtype: {output_dtype}")
    if block_m not in (128, 256):
        raise ValueError("Sage block_m must be 128 or 256")
    if block_n not in (32, 64):
        raise ValueError("Sage block_n must be 32 or 64")
    if lds_padding not in (4, 8, 16):
        raise ValueError("Sage lds_padding must be 4, 8, or 16")

    WARP_SIZE = GFX1201_WAVE_SIZE
    WMMA_K = GFX1201_WMMA_K
    ROWS_PER_WAVE = GFX1201_WMMA_M
    WMMA_LANE_K = 8
    K_SUB_N = 32
    VEC_WIDTH = 16
    Q_SCALE_ROWS = 32

    BLOCK_M = int(block_m)
    BLOCK_N = int(block_n)
    BLOCK_SIZE = (BLOCK_M // ROWS_PER_WAVE) * WARP_SIZE
    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE
    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8
    K_STEPS_QK = head_dim // WMMA_K
    D_CHUNKS = head_dim // GFX1201_WMMA_N
    PV_K_STEPS = K_SUB_N // WMMA_K

    K_STRIDE = head_dim + lds_padding
    V_STRIDE = BLOCK_N + lds_padding
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_BASE = LDS_K_TILE_SIZE
    LDS_V_TILE_SIZE = head_dim * V_STRIDE
    LDS_TOTAL_BYTES = LDS_K_TILE_SIZE + LDS_V_TILE_SIZE

    K_THREADS_PER_ROW = head_dim // VEC_WIDTH
    K_LOAD_ITEMS = BLOCK_N * K_THREADS_PER_ROW
    K_LOAD_BATCHES = (K_LOAD_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE
    V_CHUNKS_PER_ROW = BLOCK_N // VEC_WIDTH
    V_LOAD_ITEMS = head_dim * V_CHUNKS_PER_ROW
    V_LOAD_BATCHES = (V_LOAD_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE

    gpu_arch = os.environ.get("FLYDSL_GPU_ARCH", "gfx1201")
    path_tag = f"M{BLOCK_M}N{BLOCK_N}P{lds_padding}W{waves_per_eu}"
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"sage_attention_v2_gfx1201_smem_{path_tag}",
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_TOTAL_BYTES

    output_numeric = fx.BFloat16 if output_dtype == "bf16" else fx.Float16

    @flyc.kernel(
        name=f"sage_attention_v2_gfx1201_{path_tag}",
        known_block_size=[BLOCK_SIZE, 1, 1],
    )
    def sage_attention_core(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        QScale: fx.Pointer,
        KScale: fx.Pointer,
        VScale: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        padded_seq_len: fx.Int32,
        valid_seq_len: fx.Int32,
    ):
        fm_fast = arith.FastMathFlags.fast
        v8i8_type = Vec.make_type(8, fx.Int8)
        v16i8_type = Vec.make_type(16, fx.Int8)
        v2i32_type = Vec.make_type(2, fx.Int32)
        v8i32_type = Vec.make_type(8, fx.Int32)
        v8f32_type = Vec.make_type(8, fx.Float32)
        v8out_type = Vec.make_type(8, output_numeric)

        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        qs_ptr = _pointer_to_llvm_ptr(QScale)
        ks_ptr = _pointer_to_llvm_ptr(KScale)
        vs_ptr = _pointer_to_llvm_ptr(VScale)
        o_ptr = _pointer_to_llvm_ptr(O)

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def _global_load(ptr, index, elem_type, result_type):
            address = buffer_ops.get_element_ptr(
                ptr,
                fx.Int64(index),
                elem_type=elem_type,
            )
            return _llvm.LoadOp(result_type, _raw(address)).result

        def _global_store(ptr, index, elem_type, value):
            address = buffer_ops.get_element_ptr(
                ptr,
                fx.Int64(index),
                elem_type=elem_type,
            )
            _llvm.StoreOp(_raw(value), _raw(address))

        def _pack_i8_fragment(v8):
            return Vec(v8).bitcast(fx.Int32).ir_value()

        def _pack_fp8_probability(values):
            zero_i32 = arith.constant(0, type=T.i32)
            p0 = rocdl.cvt_pk_fp8_f32(
                T.i32, values[0], values[1], zero_i32, 0
            )
            p0 = rocdl.cvt_pk_fp8_f32(T.i32, values[2], values[3], p0, 1)
            p1 = rocdl.cvt_pk_fp8_f32(
                T.i32, values[4], values[5], zero_i32, 0
            )
            p1 = rocdl.cvt_pk_fp8_f32(T.i32, values[6], values[7], p1, 1)
            return Vec.from_elements([p0, p1], fx.Int32).ir_value()

        def _wmma_qk(a, b, accum):
            return rocdl.wmma_i32_16x16x16_iu8(
                v8i32_type,
                a,
                b,
                accum,
                signA=True,
                signB=True,
                clamp=False,
            ).result

        def _wmma_pv(a, b, accum):
            return rocdl.wmma_f32_16x16x16_fp8_fp8(
                v8f32_type,
                a,
                b,
                accum,
            ).result

        seq = fx.Index(padded_seq_len)
        valid_seq = fx.Index(valid_seq_len)
        base_ptr = allocator.get_base()
        lds = SmemPtr(
            base_ptr,
            lds_offset,
            T.i8,
            shape=(LDS_TOTAL_BYTES,),
        ).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        q_tiles = (valid_seq + BLOCK_M - 1) // BLOCK_M
        head_idx = block_id % num_heads
        batch_q_tile = block_id // num_heads
        q_tile_idx = batch_q_tile % q_tiles
        batch_idx = batch_q_tile // q_tiles
        q_start = q_tile_idx * BLOCK_M
        q_row = q_start + wave_id * ROWS_PER_WAVE + lane16

        def qk_global_index(token, col):
            return ((batch_idx * seq + token) * num_heads + head_idx) * head_dim + col

        def v_global_index(d, token):
            return ((batch_idx * num_heads + head_idx) * head_dim + d) * seq + token

        q_in_bounds = arith.cmpi(
            arith.CmpIPredicate.slt,
            _raw(q_row),
            _raw(valid_seq),
        )
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        zero_v8i8 = Vec.filled(8, 0, fx.Int8).ir_value()
        q_fragments = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
            q_raw = _global_load(
                q_ptr,
                qk_global_index(q_row_safe, q_col),
                T.i8,
                v8i8_type,
            )
            q_safe = ArithValue(q_in_bounds).select(q_raw, zero_v8i8)
            q_fragments.append(_pack_i8_fragment(q_safe))

        q_scale_blocks = seq // Q_SCALE_ROWS
        q_scale_index = (
            (batch_idx * num_heads + head_idx) * q_scale_blocks
            + q_row_safe // Q_SCALE_ROWS
        )
        q_scale = _global_load(qs_ptr, q_scale_index, T.f32, T.f32)

        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32).ir_value()
        c_zero_v8i32 = Vec.filled(8, 0, fx.Int32).ir_value()
        width_i32 = fx.Int32(WARP_SIZE)
        peer_i32 = fx.Int32(16)

        def reduction_peer(value):
            return fx.Float32(value).shuffle_xor(peer_i32, width_i32)

        init_args = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v8f32)

        loop_results = init_args
        for kv_start, inner in range(0, seq, BLOCK_N, init=init_args):
            m_running = inner[0]
            l_running = inner[1]
            o_accs = [inner[2 + dc] for dc in range_constexpr(D_CHUNKS)]

            for load_batch in range_constexpr(K_LOAD_BATCHES):
                k_item = tid + load_batch * BLOCK_SIZE
                if k_item < K_LOAD_ITEMS:
                    k_row = k_item // K_THREADS_PER_ROW
                    k_col = (k_item % K_THREADS_PER_ROW) * VEC_WIDTH
                    k_vec = _global_load(
                        k_ptr,
                        qk_global_index(kv_start + k_row, k_col),
                        T.i8,
                        v16i8_type,
                    )
                    Vec(k_vec).store(lds, [k_row * K_STRIDE + k_col])

            for load_batch in range_constexpr(V_LOAD_BATCHES):
                v_item = tid + load_batch * BLOCK_SIZE
                if v_item < V_LOAD_ITEMS:
                    d_row = v_item // V_CHUNKS_PER_ROW
                    token_offset = (v_item % V_CHUNKS_PER_ROW) * VEC_WIDTH
                    v_vec = _global_load(
                        v_ptr,
                        v_global_index(d_row, kv_start + token_offset),
                        T.i8,
                        v16i8_type,
                    )
                    Vec(v_vec).store(
                        lds,
                        [LDS_V_BASE + d_row * V_STRIDE + token_offset],
                    )

            gpu.barrier()

            s_accs = [c_zero_v8i32 for _ in range(NUM_S_ACCS)]
            for ks in range_constexpr(K_STEPS_QK):
                k_col = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base = st_idx * K_SUB_N
                    k_row_a = lane16 + st_base
                    k_row_b = lane16 + st_base + 16
                    k_a = Vec.load(
                        v8i8_type,
                        lds,
                        [k_row_a * K_STRIDE + k_col],
                    )
                    k_b = Vec.load(
                        v8i8_type,
                        lds,
                        [k_row_b * K_STRIDE + k_col],
                    )
                    acc_a = st_idx * 2
                    acc_b = acc_a + 1
                    s_accs[acc_a] = _wmma_qk(
                        _pack_i8_fragment(k_a),
                        q_fragments[ks],
                        s_accs[acc_a],
                    )
                    s_accs[acc_b] = _wmma_qk(
                        _pack_i8_fragment(k_b),
                        q_fragments[ks],
                        s_accs[acc_b],
                    )

            k_scale_blocks = seq // BLOCK_N
            k_scale_index = (
                (batch_idx * num_heads + head_idx) * k_scale_blocks
                + kv_start // BLOCK_N
            )
            k_scale = _global_load(ks_ptr, k_scale_index, T.f32, T.f32)
            score_scale = _fmul(q_scale, k_scale)

            scores = []
            for st in range_constexpr(NUM_S_ACCS):
                for item in range_constexpr(8):
                    score_f32 = arith.sitofp(T.f32, Vec(s_accs[st])[item])
                    score = _fmul(score_f32, score_scale)
                    key_offset = (
                        (st // 2) * K_SUB_N
                        + (st % 2) * ROWS_PER_WAVE
                        + klane * WMMA_LANE_K
                        + item
                    )
                    key_in_bounds = arith.cmpi(
                        arith.CmpIPredicate.slt,
                        _raw(kv_start + key_offset),
                        _raw(valid_seq),
                    )
                    scores.append(
                        ArithValue(key_in_bounds).select(score, c_neg_inf)
                    )

            local_max = scores[0]
            for item in range_constexpr(NUM_S_VALS - 1):
                local_max = _fmax(local_max, scores[item + 1])
            row_max = _fmax(local_max, reduction_peer(local_max))
            m_new = _fmax(m_running, row_max)
            corr = rocdl.exp2(
                T.f32,
                _raw(_fsub(m_running, m_new)),
            )

            p_values = []
            local_sum = _raw(c_zero_f)
            neg_m_new = _fsub(c_zero_f, m_new)
            for item in range_constexpr(NUM_S_VALS):
                shifted = fmath.fma(scores[item], _raw(c_one_f), neg_m_new)
                probability = rocdl.exp2(T.f32, _raw(shifted))
                p_values.append(probability)
                local_sum = _fadd(local_sum, probability)

            tile_sum = _fadd(local_sum, reduction_peer(local_sum))
            l_new = _fadd(_fmul(corr, l_running), tile_sum)
            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec)

            p_fragments = []
            for st_idx in range_constexpr(N_SUB_TILES):
                p_subtile = []
                for pks in range_constexpr(PV_K_STEPS):
                    p_base = (st_idx * 2 + pks) * 8
                    p_subtile.append(
                        _pack_fp8_probability(
                            [p_values[p_base + item] for item in range(8)]
                        )
                    )
                p_fragments.append(p_subtile)

            for pks in range_constexpr(PV_K_STEPS):
                for dc in range_constexpr(D_CHUNKS):
                    d_pos = fx.Index(dc * GFX1201_WMMA_N) + lane16
                    for st_idx in range_constexpr(N_SUB_TILES):
                        token_pos = (
                            st_idx * K_SUB_N
                            + pks * WMMA_K
                            + klane * WMMA_LANE_K
                        )
                        v_bytes = Vec.load(
                            v8i8_type,
                            lds,
                            [LDS_V_BASE + d_pos * V_STRIDE + token_pos],
                        )
                        o_accs[dc] = _wmma_pv(
                            _pack_i8_fragment(v_bytes),
                            p_fragments[st_idx][pks],
                            o_accs[dc],
                        )

            gpu.barrier()
            m_running = m_new
            l_running = l_new
            loop_results = yield [m_running, l_running] + o_accs

        inv_l = arith.divf(
            _raw(c_one_f),
            _raw(loop_results[1]),
            fastmath=fm_fast,
        )
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8).ir_value()
        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                d_col = fx.Index(dc * GFX1201_WMMA_N) + klane * 8
                scale_index = (batch_idx * num_heads + head_idx) * head_dim + d_col
                v_scale = _global_load(
                    vs_ptr,
                    scale_index,
                    T.f32,
                    v8f32_type,
                )
                normalized = _fmul(loop_results[2 + dc], inv_l_vec)
                scaled = _fmul(normalized, v_scale)
                out = Vec(scaled).to(output_numeric).ir_value()
                _global_store(
                    o_ptr,
                    qk_global_index(q_row, d_col),
                    output_numeric.ir_type,
                    out,
                )

    @flyc.jit
    def launch_sage_attention_core(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        QScale: fx.Pointer,
        KScale: fx.Pointer,
        VScale: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        batch_size: fx.Int32,
        padded_seq_len: fx.Int32,
        valid_seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        valid_seq = fx.Index(valid_seq_len)
        grid_x = (
            fx.Index(batch_size)
            * ((valid_seq + BLOCK_M - 1) // BLOCK_M)
            * num_heads
        )
        launcher = sage_attention_core(
            Q,
            K,
            V,
            QScale,
            KScale,
            VScale,
            O,
            padded_seq_len,
            valid_seq_len,
        )
        _set_waves_per_eu(waves_per_eu)
        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    launch_sage_attention_core.compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _ptr_arg(value):
        if hasattr(value, "data_ptr"):
            type_name = type(value).__name__
            module_name = type(value).__module__
            pointer = (
                0
                if type_name == "FakeTensor" or "fake_tensor" in module_name
                else value.data_ptr()
            )
            return flyc.from_c_void_p(fx.Uint8, pointer)
        return value

    def _launch(
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        out,
        batch_size,
        padded_seq_len,
        valid_seq_len,
        stream=None,
    ):
        args = tuple(
            _ptr_arg(value)
            for value in (q, k, v, q_scale, k_scale, v_scale, out)
        )
        compiled = getattr(launch_sage_attention_core, "_compiled", None)
        runtime_args = (
            *args,
            batch_size,
            padded_seq_len,
            valid_seq_len,
            fx.Stream(stream),
        )
        if compiled is None:
            compiled = flyc.compile(launch_sage_attention_core, *runtime_args)
            launch_sage_attention_core._compiled = compiled
        else:
            compiled(*runtime_args)

    _launch.jit_function = launch_sage_attention_core
    return _launch


def compile_sage_wmma_probes():
    """Compile both probes using null pointers; intended for COMPILE_ONLY=1."""

    null_i8 = flyc.from_c_void_p(fx.Uint8, 0)
    stream = fx.Stream(None)
    build_sage_wmma_qk_probe()(null_i8, null_i8, null_i8, stream)
    build_sage_wmma_pv_probe()(null_i8, null_i8, null_i8, stream)


def verify_dumped_wmma_isa(dump_dir: str | os.PathLike[str]) -> dict[str, Path]:
    """Find and validate the final ISA files emitted for both probes."""

    root = Path(dump_dir)
    expected = {
        _QK_PROBE_NAME: _QK_ISA,
        _PV_PROBE_NAME: _PV_ISA,
    }
    found: dict[str, Path] = {}
    for kernel_name, instruction in expected.items():
        candidates = sorted((root / kernel_name).glob("*_final_isa.s"))
        if not candidates:
            candidates = sorted((root / kernel_name).glob("*.s"))
        if not candidates:
            raise RuntimeError(f"no ISA dump found for {kernel_name} under {root}")
        isa_path = candidates[-1]
        isa = isa_path.read_text(encoding="utf-8").lower()
        if instruction not in isa:
            raise RuntimeError(
                f"{kernel_name} did not lower to {instruction}; inspect {isa_path}"
            )
        found[kernel_name] = isa_path
    return found


def _main():
    parser = argparse.ArgumentParser(description="Compile gfx1201 SageAttention2 WMMA probes")
    parser.add_argument("--dump-dir", default="/tmp/sage_gfx1201_wmma_isa")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("COMPILE_ONLY", "1")
    os.environ.setdefault("ARCH", "gfx1201")
    os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1201")
    os.environ.setdefault("FLYDSL_DUMP_IR", "1")
    os.environ.setdefault("FLYDSL_DUMP_DIR", args.dump_dir)

    compile_sage_wmma_probes()
    if args.verify:
        for name, path in verify_dumped_wmma_isa(args.dump_dir).items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    _main()
