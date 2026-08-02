# AMD RDNA4 SageAttention2 Support Plan

Status: native self-attention hardware gates passed; LightX2V integration and
production video validation pending, 2026-08-02
Target: `gfx1201` (RDNA4, wave32)
Primary workload: Wan2.1 / Wan2.2 720P inference, head dimension 128

## 1. Objective

Implement a native RDNA4 SageAttention2/2++ path for `gfx1201` in Aiter. The
algorithmic reference is the SageAttention SM89 path used by RTX 4090:

- BF16/FP16 input and output;
- Q/K quantized to signed INT8;
- softmax probabilities P converted to FP8;
- V quantized per channel to FP8;
- QK computed by native INT8 WMMA;
- PV computed by native FP8 WMMA with FP32 accumulation;
- K smoothing preserved and LSE correction added when LSE support lands;
- dense self-attention first, then Wan cross-attention after separate gates;
- Aiter dispatches the native FlyDSL implementation on `gfx1201` and keeps the
  current Triton Sage v1 path as a fallback.

This is not a source-language rewrite of `fav3_sage`. A kernel is considered a
SageAttention2 implementation only when P and V both enter the native FP8 PV
matrix operation and the V2 quantization/layout contract is preserved.

## 2. Source Baselines

The initial implementation is based on the following local revisions:

| Component | Revision | Role |
| --- | --- | --- |
| Aiter | `efc70fb06` on `gfx1201-hvat-scratch` | integration and FlyDSL runtime |
| Aiter Triton dependency | `9c8287e49` | current Sage v1 baseline/fallback |
| SageAttention | `d1a57a5` | SM89 SageAttention2++ reference |
| RDNA4 ISA | user-provided 2025 ISA PDF | instruction and hazard contract |

The existing local changes in `fav3_sage.py` and its tests/benchmarks are part
of the baseline and must not be discarded.

## 3. Why the Current Aiter Kernel Is Not V2

`aiter/ops/triton/attention/fav3_sage.py` explicitly exposes Sage Attention v1.
It performs per-block INT8 Q/K quantization and stores V as FP8, but the
attention kernel uses a generic Triton `tl.dot` path. It does not implement the
SM89 SageAttention2++ contract:

- no V2 per-warp Q quantization;
- no V2 transposed and padded per-channel V layout;
- no explicit P-to-FP8 conversion contract;
- no guaranteed native FP8 x FP8 PV WMMA;
- no V2 instruction-buffer scheduling strategy.

The first gfx1201 tuning result (`BM256/BN64/WPE3`) therefore tunes a Sage v1
Triton implementation. Its loss to the tuned BF16 FlyDSL FMHA is expected and
does not answer whether a native RDNA4 SageAttention2 kernel can win.

## 4. RDNA4 ISA Mapping

### 4.1 Required native instructions

The RDNA4 ISA defines the two dense wave32 operations needed by V2:

| Attention stage | RDNA4 instruction | Operands | Accumulator |
| --- | --- | --- | --- |
| QK | `V_WMMA_I32_16X16X16_IU8` | signed/unsigned INT8 selected by IU8 NEG bits | I32 |
| PV | `V_WMMA_F32_16X16X16_FP8_FP8` | FP8 E4M3 x FP8 E4M3 | F32 |

For the signed Q/K path both IU8 signedness controls must select signed input.
Composable Kernel confirms the gfx12 builtin ABI:

- each lane supplies an eight-element INT8 or FP8 fragment, packed as two I32
  registers;
- each lane owns eight I32/F32 accumulator elements;
- the dense operation is `16x16x16`, not K64 or K128.

Any larger K shape must be constructed from K16 instructions. The gfx1250
K128/scaled-WMMA implementation must not be reused as an instruction-level
assumption on gfx1201.

### 4.2 Accumulation policy

RDNA4 FP8 WMMA exposes F32 accumulation. Unlike the SM89 implementation, there
is no native FP16 accumulator variant to reproduce `fp32+fp16` literally. The
gfx1201 path will use:

- I32 accumulation for QK, followed by explicit F32 dequantization;
- F32 accumulation for PV;
- an optional F32 long-lived instruction buffer flushed at a tunable K-tile
  interval only if ISA profiling shows a register-pressure or dependency win.

The initial correctness implementation uses F32 throughout PV. A mixed
precision buffer is an optimization, not part of the V2 functional contract.

### 4.3 WMMA hazards and scheduling

RDNA4 requires at least one independent VALU instruction or `V_NOP` between
WMMAs when the next A/B fragment aliases the previous D fragment. Overlapped D
to C dependencies can also stall. The kernel must satisfy this deliberately:

- interleave independent LDS reads, FP8 packing, score conversion, max/sum
  reduction, or address arithmetic between dependent WMMAs;
- use explicit `V_NOP` only when no useful independent instruction is
  available;
- inspect final ISA rather than assuming compiler scheduling is sufficient.

### 4.4 Memory pipeline

The gfx1201 implementation will reuse the proven RDNA4 FlyDSL FMHA pattern:

- vector VMEM loads;
- padded or permuted LDS tiles;
- Q register preload;
- K/V global prefetch before the current tile completes;
- software-pipelined LDS reads for GEMM2;
- wave32 reductions and native `exp2`.

The gfx1250 TDM pipeline is not available on gfx1201. Only its fine-grained
schedule representation is reusable as a design pattern.

## 5. SageAttention2 Data Contract

The first native path targets head dimension 128 and BSHD/BHSD inputs.

### 5.1 Q/K preprocessing

- Compute `K_mean` over the sequence and quantize `K - K_mean` when
  `smooth_k=True`.
- Quantize Q per 32 query rows, matching SM89 `CTA_Q=128, WARP_Q=32`.
- Quantize K per selected 32/64-row key tile. BN64 matches the SM89 contract;
  BN32 is an RDNA4 occupancy/loop-overhead candidate.
- Store signed INT8 Q/K and F32 dequantization scales.
- When LSE is requested, restore the smoothing correction
  `softmax_scale * Q dot K_mean`.

The bring-up version may launch preprocessing kernels separately. Fusion is
required later for full-call performance, but must not hide attention-kernel
correctness or ISA validation.

### 5.2 V preprocessing

Implement SageAttention2 `per_channel_fp8` semantics:

- transpose V from sequence-major to `[B, Hkv, D, padded_Sk]`;
- pad the sequence dimension to the selected BN32/64 tile;
- do not apply the SM89 16-element sequence permutation: that permutation is
  part of NVIDIA's fragment-load contract, while gfx1201 consumes each K8 FP8
  fragment contiguously from the transposed RDNA-native layout;
- quantize each `[B, Hkv, D]` channel to FP8 E4M3;
- retain an F32 V scale per channel.

This layout is a performance contract for contiguous FP8 PV fragments, not
only a wrapper detail.

### 5.3 Attention core

For every query tile and KV tile:

1. Load INT8 Q/K fragments and issue K16 INT8 WMMAs.
2. Convert I32 scores to F32 and apply Q scale, K scale and
   `softmax_scale * log2(e)`.
3. Update online row max and denominator with native `exp2`.
4. Convert normalized block probabilities to FP8 E4M3 fragments.
5. Load contiguous FP8 V fragments and issue K16 FP8 WMMAs.
6. Rescale the running output after a row-max change.
7. Normalize by the denominator, apply per-channel V scale, and store BF16 or
   FP16 output.

GQA maps each query head to `kv_head = q_head // (Hq / Hkv)`.

## 6. Kernel Shape Strategy

The first correctness tile is `BM128/BN32` with eight wave32 waves (256
threads). BN32 lowers LDS and score state; BN64 remains the closest SM89
algorithmic tile. Neither is assumed to be the final gfx1201 winner.

The first performance sweep will compare:

| Candidate family | Purpose |
| --- | --- |
| `BM128/BN64/WPE2,3` | closest SM89 KV tile with 256-thread workgroup |
| `BM128/BN32/WPE2,3` | lower LDS/register footprint and bring-up reference |
| `BM256/BN64/WPE2,3` | high Q reuse with 512-thread workgroup |
| `BM256/BN32/WPE2,3` | diagnostic 512-thread path; half of load threads idle |

Pipeline depth, V preloading, LDS padding and global vector width are tuned
after both native WMMA stages pass ISA and correctness gates. Triton
`num_stages` is not a direct FlyDSL tuning parameter and will not be carried
over mechanically.

### 6.1 First gfx1201 hardware result

The 2026-08-02 720P sweep selected `BM128/BN32/WPE2/LDS_PAD16`, which is also
the current production default. Across Wan2.1 A14B, Wan2.2 A14B and Wan2.2
TI2V self-attention under USP8 it achieved:

- `1.3359x` kernel-only geomean versus tuned FlyDSL BF16;
- `1.2874x` full-call geomean versus tuned FlyDSL BF16;
- `3.43x` to `3.55x` full-call speedup versus Triton Sage v1;
- 40/40 correctness cases passed, including sequence tails 31/33/63/65/257;
- minimum measured cosine similarity approximately `0.9988`.

The winning final ISA uses wave32 and contains native INT8 QK WMMA, FP8 PV
WMMA and packed FP8 conversion. Its recorded resources are 181 VGPR, 32 SGPR
and 10,752 bytes LDS. BN64 increased pressure substantially (up to 256 VGPR
and 19,456 bytes LDS), while WPE2 and WPE3 were effectively tied. The default
therefore remains frozen at the measured winner.

The second sweep is intentionally local rather than another Cartesian grid:

| Candidate | Reason |
| --- | --- |
| `BM64/BN32/WPE2` | test a four-wave workgroup and two-workgroup residency |
| `BM128/BN32/WPE2/LDS_PAD4,8` | isolate bank-conflict and LDS-footprint effects |
| winner plus V LDS-register preload | port the proven GEMM2 latency-hiding pattern |
| `BN16` prototype | exploratory register/LDS reduction; lower priority because KV iterations and barriers double |

WPE4 and further BM256 variants are deprioritized because the first sweep
shows that WPE is not the current ceiling and larger query workgroups lose
consistently.

## 7. Aiter Integration Boundary

New code should be isolated under the existing FlyDSL ownership boundary:

- `aiter/ops/flydsl/kernels/sage_attention_gfx1201.py`: native attention core
  and microkernels;
- `aiter/ops/flydsl/sage_attention.py`: validation, build cache, allocations
  and prepare/launch orchestration. Bring-up reuses the existing Triton
  `sage_quant` preprocessing and performs the RDNA V transpose in Torch.

`fav3_sage_wrapper_func` remains the compatibility entry point. Dispatch order:

1. On gfx1201, check the native V2 support predicate and FlyDSL availability.
2. Launch the native V2 path when every requested semantic is supported.
3. Otherwise call the existing Triton Sage v1 implementation unchanged.

The initial support predicate is intentionally narrow:

- gfx1201 only;
- inference forward only;
- BF16 or FP16 inputs with equal dtype;
- head dimension 128;
- BSHD or BHSD dense tensors;
- no dropout, softcap, sliding window, bias, ALiBi or block-sparse LUT;
- causal and non-causal handled only after each passes its own test gate;
- self-attention, GQA and cross-attention enabled independently as they pass.

The bring-up path is opt-in with `AITER_SAGE_GFX1201_NATIVE=1` or
`config={"backend": "flydsl_v2"}`. This acts as the pre-production kill switch:
unsupported calls and default calls retain Triton v1 until hardware correctness
and full-call performance pass. The Triton kernel remains callable explicitly
for A/B measurements and rollback.

## 8. Implementation Phases

### Phase 0: toolchain and ISA proof

Status: complete on gfx1201 for the required native instruction and numerical
gates.

- Add an INT8 QK `16x16x16` FlyDSL microkernel.
- Add an FP8 PV `16x16x16` FlyDSL microkernel.
- Compile for gfx1201 in the Aiter-required FlyDSL/Triton image.
- Disassemble the code object and verify the exact two RDNA4 WMMA mnemonics.
- Record VGPR, SGPR, LDS, occupancy and instruction counts.
- Verify signed INT8 controls and FP8 E4M3 bit interpretation numerically.

Exit gate: both microkernels are correct and use the required native ISA.

### Phase 1: pre-quantized V2 attention core

Status: dense non-causal self-attention core implemented and hardware-validated
for BM128/256 and BN32/64. The selected production tile beats the tuned BF16
baseline at kernel-only and full-call scope.

- Implement dense non-causal head-dim-128 core.
- Accept pre-quantized Q/K/V and scales.
- Add online softmax, P-to-FP8 conversion, F32 PV and output scaling.
- Add self-attention first, then asymmetric Q/K lengths and GQA.

Exit gate: kernel-only correctness and speed pass on representative 720P
shapes.

### Phase 2: SageAttention2 preprocessing

Status: bring-up wrapper reuses Aiter `sage_quant` with Q groups of 32 and K
groups of BN, then creates RDNA-native transposed FP8 V storage. Full-wrapper
self-attention correctness and performance gates pass. Native fusion, LSE and
cross-attention preprocessing remain pending.

- Implement per-warp Q and per-block K INT8 quantization.
- Fuse K smoothing subtraction into K quantization.
- Implement V transpose/pad and per-channel FP8 quantization without the
  NVIDIA-specific permutation.
- Add LSE and smoothing correction.

Exit gate: full wrapper is numerically equivalent to SageAttention2 semantics.

### Phase 3: Aiter dispatch

Status: strict dispatch, explicit backend selection and Triton fallback are
implemented. LightX2V routes supported gfx1201 self-attention to V2 and keeps
cross-attention on its tuned BF16 path; image-only integration validation is
still required.

- Add strict gfx1201 native support predicate.
- Route `fav3_sage_wrapper_func` to FlyDSL V2.
- Preserve explicit Triton fallback and add a kill switch.
- Add tests proving unsupported semantics still reach the old path.

Exit gate: LightX2V can select the native path without API changes.

### Phase 4: schedule and fusion optimization

- Pipeline K/V VMEM and LDS traffic across KV iterations.
- Interleave useful instructions to satisfy WMMA hazard spacing.
- Tune BM/BN/NW/WPE, LDS layout, prefetch distance and PV buffer interval.
- Fuse quantization work where full-call profiling proves it is required.
- Produce per-family configs for self, text-cross and image-cross attention.

Exit gate: full-call latency beats the tuned BF16 FlyDSL baseline on the
weighted Wan production matrix; no family may silently regress behind the
selected fallback.

## 9. Validation Matrix

All performance runs use batch 1 and head dimension 128.

| Workload | Q length | K length | Q/KV heads | Purpose |
| --- | ---: | ---: | ---: | --- |
| Wan2.1 A14B self USP4 | 75600 | 75600 | 10/10 | long self-attention |
| Wan2.1 A14B self USP8 | 75600 | 75600 | 5/5 | low-head long self-attention |
| Wan2.1 A14B text cross USP8 | 9450 | 512 | 40/40 | medium cross-attention |
| Wan2.1 A14B image cross USP8 | 9450 | 257 | 40/40 | short odd-K cross-attention |
| Wan2.2 A14B self USP8 | 75600 | 75600 | 5/5 | production A14B representative |
| Wan2.2 TI2V self single | 109120 | 109120 | 24/24 | maximum production load |
| Wan2.2 TI2V self USP8 | 109120 | 109120 | 3/3 | low parallel head count |
| Wan2.2 TI2V text cross USP8 | 13640 | 512 | 24/24 | TI2V cross-attention |

Add small synthetic cases for tails, causal masking and GQA before enabling
those branches.

The executable GPU gate lives in `aiter-tune-gfx1201`:

- `tune/tune_sage_attention_v2_gfx1201.py` checks tail lengths 31/33/63/65/257,
  exact WMMA/conversion ISA, kernel-only latency and full-call latency;
- `tune/launch_sage_attention_v2_tuning.sh` validates the Aiter revision baked
  into a prebuilt image on tune hosts without an Aiter checkout or source
  mount;
- the 720P sweep compares native V2 with FlyDSL BF16 as the primary baseline
  and Triton Sage v1 as the secondary baseline.

## 10. Acceptance Gates

### Correctness

- Microkernels match CPU/PyTorch references exactly for INT8 QK and within FP8
  rounding tolerance for PV.
- No NaN/Inf on the complete workload matrix.
- Output cosine similarity and error thresholds are at least as strict as the
  existing Aiter Sage tests; report `cos_min`, max absolute error and relative
  error rather than only a pass/fail count.
- LSE is compared after K-smoothing correction.
- Tail tokens, odd K length 257, GQA mapping and multi-GPU device/stream
  behavior have dedicated tests.

### ISA

- QK contains `V_WMMA_I32_16X16X16_IU8`.
- PV contains `V_WMMA_F32_16X16X16_FP8_FP8`.
- No scalarized integer dot-product or BF16 fallback in the measured hot loop.
- Hazard spacing is checked in the disassembly around repeated WMMAs.

### Performance

- Measure quantization-only, attention-kernel-only and full-wrapper latency.
- Compare against both tuned Triton MHA and the tuned gfx1201 BF16 FlyDSL FMHA;
  the FlyDSL result is the primary baseline.
- Require a repeatable full-call win before enabling native V2 by default.
- Cross-attention may retain Triton/BF16 dispatch if preprocessing dominates
  and no full-call win is achieved.
- Report first-call compilation separately from steady-state execution.

## 11. Main Risks and Decisions

| Risk | Decision |
| --- | --- |
| FlyDSL lacks gfx12 INT8/FP8 WMMA wrappers | add narrow ROCDL/LLVM intrinsic bindings; do not use generic `tl.dot` |
| Compiler emits a different WMMA form | fail Phase 0 ISA gate and keep Triton fallback |
| FP8 conversion dominates | use packed `V_CVT_PK_FP8_F32`, then schedule conversion between WMMAs |
| V transpose/quant dominates cross-attention | cache transformed V when lifetime permits, fuse preprocessing, or retain fallback |
| BM256 inherits too much register state | retain the 256-thread BM128 path as the reference and tune from measured VGPR occupancy |
| Tail padding changes softmax | mask real K length in the core; never rely on zero padding alone |
| Accuracy differs from SM89 mixed accumulation | keep full FP32 PV as the gfx1201 default and validate on generated frames |

## 12. Immediate Work Order

1. Run the image-only LightX2V integration gate against the already validated
   image and confirm native self-attention plus BF16 cross-attention routing.
2. Generate representative Wan2.1 and Wan2.2 720P videos and compare quality,
   peak memory and end-to-end DiT latency with the BF16 baseline.
3. Run the focused BM64, LDS padding and V-register-preload sweep described in
   Section 6.1; require at least a repeatable 2% full-call gain before changing
   the frozen default.
4. Treat BN16 as an isolated prototype and retain it only if reduced resource
   pressure outweighs the doubled KV-loop/barrier count.
5. Keep short cross-attention on the tuned BF16 path unless an asymmetric V2
   implementation independently passes correctness and full-call gates.
