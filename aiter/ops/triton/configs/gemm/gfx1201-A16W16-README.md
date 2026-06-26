# gfx1201 (RDNA4) GEMM A16W16 调优配置 — 起点

本目录新增三个 gfx1201 的 GEMM-A16W16 系列 Triton 调优配置，填补 gfx1201 在 BF16 矩阵乘上的配置空白。

## 背景

gfx1201 此前仅有 45 个 `GEMM-A8W8*`（FP8 量化权重）Triton 配置，`GEMM-A16W16*`（非量化 BF16 矩阵乘）系列**完全缺失**。而 gfx950 有 10 个、gfx1250 有 8 个 A16W16 配置。

A16W16 是最通用的 GEMM 变体——任何 BF16 模型推理的线性层都走它。配置缺失的后果是：gfx1201 上调用 `aiter` 的 A16W16 GEMM 时，`get_gemm_config("GEMM-A16W16", M, N, K)` 因文件不存在直接报 `AssertionError: Required config file doesn't exist`（`gemm_config_utils.py` L86 `fpath_should_exist=True`），上层（如 ATOM）被迫回退到 `torch.nn.functional.linear`。这已由 ATOM PR #811 实测印证。

## 新增文件

| 文件 | 对应 kernel | 说明 |
|---|---|---|
| `gfx1201-GEMM-A16W16.json` | `aiter.ops.triton.gemm.basic.gemm_a16w16`（Triton `_gemm_a16_w16_kernel`） | 通用 A16W16，9 个 M 档 |
| `gfx1201-GEMM-A16W16-ATOMIC.json` | `aiter.ops.triton.gemm.basic.gemm_a16w16_atomic` | atomic 归约变体 |
| `gfx1201-BATCHED_GEMM-A16W16.json` | batched A16W16 | 大 M 场景（M_GEQ_4096 档） |

## Schema

与 `gfx950-GEMM-A16W16.json` 完全一致（同一 Triton kernel 路径，gfx1250 的 gluon 路径不适用 gfx1201）：

```json
{
  "M_LEQ_<N>": {
    "BLOCK_SIZE_M": <int>, "BLOCK_SIZE_N": <int>, "BLOCK_SIZE_K": <int>,
    "GROUP_SIZE_M": <int>, "num_warps": <int>, "num_stages": <int>,
    "waves_per_eu": <int>, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg" | null, "NUM_KSPLIT": 1
  },
  "any": { ... }
}
```

`matrix_instr_nonkdim: 16` 对应 gfx1201 的 WMMA-128b（`wmma_f32_16x16x16_f16/bf16_w32_gfx12`，K 维宽度 16）。

## 取值依据

- **schema/字段**：对齐 gfx950（同 Triton kernel、同 wave32 WMMA-128b 特征）。
- **block 量级**：参照 gfx1201 自身已有 A8W8 配置（`any` 档 BM16/BN64/waves1，wave32 风格）与 gfx950 A16W16 折中。
- **`waves_per_eu`**：小 M 档用 2-4（提高占用），大 M 档用 2（减少调度开销）。

## ⚠️ 状态：starting-point，未经真机调优

这些是**起点配置，不是生产级调优**。block/waves/stages 的取值是基于架构特征的合理推断，**未在 RX 9070 XT (gfx1201) 上做过 tuning run**。它们能：

- ✅ 消除 `Required config file doesn't exist` 报错，让 A16W16 GEMM 在 gfx1201 上**可运行**（不再回退 torch）。
- ✅ 提供一个合理的调优起点（优于默认配置）。
- ❌ 不保证性能最优——需在真机上用 `aiter` 的 tuning 流程逐形状 sweep 后替换。

## 后续调优

在 gfx1201 真机上执行 `aiter` 的 GEMM tuning 流程（如 ATOM 的 `gemm_a8w8_sweep.py` 模式），对每个 (M, N, K) 形状 sweep 出最优 block/waves，生成 `gfx1201-GEMM-A16W16-N={N}-K={K}.json` 专用配置，逐步替换本文件的通用档。

## 验证

加载逻辑已用 `gemm_config_utils.py` 的文件名约定与 M 档选择逻辑模拟验证：`get_arch()=gfx1201` 时，三类配置对 M=1/8/64/512/4096/8192 均能正确命中对应档（见提交说明）。
