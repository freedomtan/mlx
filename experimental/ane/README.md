# ANE backend exploration (v1)

Experimental files, not part of the mlx build. Branch: `ane-backend-exploration`.
Core changes live in `mlx/primitives.h` (`CompiledANE`), `mlx/compile.cpp`
(`is_fusable_ane`/`compile_fuse_ane`), and `mlx/backend/metal/compiled_ane.mm`
(real `eval_gpu`, routes an eligible `Convolution`+bias(+relu) through
`MPSGraph` with the private `preferredDevice=2` ANE hint). Build with
`-DMLX_BUILD_METAL=ON -DMLX_BUILD_ANE=ON`.

## Files

- `ane_placement_probe.mm`, `pmu_gate_check.mm`, `pmu_probe.mm` -- standalone
  MPSGraph/ANEClient probes, predate the mlx integration, no `libmlx.a`
  needed. Confirm `preferredDevice=2` actually targets the ANE.
- `trailing_op_check.mm`, `layout_isolation_check.mm`,
  `resnet50_shape_placement_check.mm` -- standalone probes that found a bare
  convolution never reaches the ANE (needs a real consumer like a bias-add),
  regardless of layout/transpose/chain length.
- `correctness_test.cpp` -- `CompiledANE` vs CPU, single conv+bias(+relu),
  a linear 2-conv chain, a branch-merge (downsample-style) pattern, and an
  ineligible (grouped) conv to check the fallback path.
- `export_resnet50.py` -- exports real ImageNet-pretrained torchvision
  ResNet50 weights (BatchNorm folded into each conv, OIHW->OHWI) plus a
  PyTorch reference output. Needs `pip install torch torchvision`.
- `resnet50_real_bench.cpp` -- real pretrained weights, cross-checked
  against both mlx-CPU and an independent PyTorch reference. Also contains
  (commented-out-by-disuse, left in place) the bisection harness
  (`forward_n_blocks`, `isolated_fun`, `shadowed_fun`, `block0_only_fun`,
  `stem_block0_fun`, `stem_split_fun`) used to localize the bug below --
  instructive if this needs revisiting.
- `resnet50_bench.cpp` -- **do not trust this file's numbers.** Its
  "5.65x speedup" was a benchmark bug: `metal_fun` regenerated random
  weights *inside* the timed loop while the compiled path only paid that
  cost once. Kept only as a cautionary example.

## Finding 1: a bare convolution never reaches the ANE

`MPSGraph` with `preferredDevice=2` only routes a convolution to the ANE if
it has a real (non-constant-foldable) consumer -- confirmed in
`trailing_op_check.mm`: `conv` alone never gets an ANE bundle; `conv + bias`
or `conv + relu` always does, regardless of NCHW/OIHW vs NHWC/OHWI+transpose
or chain length. This invalidated the original v1 scope (bare-conv-only)
and is why `CompiledANE` fuses conv+bias, optionally extended with relu
(`maximum(x, 0)`, mlx's actual relu lowering) -- see `is_fusable_ane`/
`compile_fuse_ane` in `mlx/compile.cpp`.

## Finding 2: a non-ANE op between two ANE ops silently corrupts the result

`CompiledANE::eval_gpu` runs on its own Metal command queue, synchronized
against mlx's queue only via `encoder.end_encoding(wait_in_flight=true)`,
which inserts GPU-side fences between encoders on mlx's *own* queue but
never actually commits or waits. Confirmed by direct experiment
(`resnet50_real_bench.cpp`'s bisection harness): a `maxpool` (regular Metal
op) between the stem's ANE-fused conv and the first bottleneck block's ANE
convs produced `max_abs_diff` in the tens, compounding to `inf` by the end
of a real ResNet50 forward pass -- not a crash, silently wrong numbers.
Splitting the graph into two `compile()`+`eval()` calls with a sync
checkpoint in between made it correct again, confirming the cause: maxpool's
output (computed via mlx's regular, batched/deferred queue) may not have
finished executing on the GPU by the time the next `CompiledANE`, on its
separate queue, reads it.

Tried to fix this by having `eval_gpu` itself commit+wait on mlx's queue
before running MPSGraph (`encoder.synchronize()` instead of
`end_encoding()`). Both attempts crashed instead of just being slow:

1. Calling `encoder.synchronize()` directly: use-after-free. mlx's own
   `gpu::eval()` dispatcher (`mlx/backend/metal/eval.cpp`) caches a raw
   `MTL::CommandBuffer*` *before* calling a primitive's `eval_gpu`, then uses
   that same pointer *after* it returns. `synchronize()` commits the buffer
   and replaces it with a fresh one; the old buffer's last reference drops
   when `synchronize()` returns, before control passes back to the caller,
   which then dereferences a freed object.
2. Holding an extra reference on the old buffer to survive that window (a
   small deferred-release queue) fixed the use-after-free, but exposed a
   harder problem: `gpu::eval()`'s caller unconditionally tries to attach a
   completion handler to the buffer it cached, assuming it's still
   uncommitted. Since we'd already committed it, Metal enforces its own
   contract and aborts: `-[_MTLCommandBuffer addCompletedHandler:]:
   Completed handler provided after commit call`.

Both failures trace to the same root cause: `eval.cpp`'s dispatcher
structurally assumes no primitive commits mlx's command buffer mid-stream.
Fixing that requires changing `mlx/backend/metal/eval.cpp` itself (re-fetch
the command buffer pointer *after* calling `eval_gpu` instead of before) --
shared dispatch code affecting every primitive, not an ANE-only change, so
it wasn't done here.

## The fix actually shipped: restrict fusion to provably-synchronized inputs

Since the sync gap can't be closed from inside `eval_gpu` alone,
`is_fusable_ane()` now only fuses a convolution when its input `x` is
either a raw graph input/constant (never touched by a Metal op) or the
output of another `CompiledANE` (same queue, already correctly ordered via
its own `waitUntilCompleted`). A non-ANE op's output in between blocks
fusion rather than risk silent corruption. Verified on the full real
ResNet50: `mlx-ane vs mlx-cpu max_abs_diff` back to normal fp16-rounding
levels (0.06, same order as the mlx-cpu vs PyTorch baseline of 0.16), no
more `inf`.

The cost: this makes fusion coverage very sparse on a real network. In
ResNet50, only the stem conv's input is a raw, never-touched graph input --
every other conv's input comes from a non-ANE op (maxpool, or the
non-fused residual `add`+`relu` between blocks) sitting in between, which
this restriction correctly blocks. Measured result: `ratio (metal/ane):
0.92x` -- essentially parity, no speedup, because essentially nothing
fuses.

## Where this left v1

- Mechanism works and is numerically correct, including at full
  real-model scale with real pretrained weights -- this took two rounds of
  real bugs (ANE-placement eligibility, then a genuine cross-queue race) to
  get right, not just careful scoping.
- No performance benefit on a real network as shipped, because the fix for
  the race condition (restrict fusion to ANE-to-ANE chains) also kills
  almost all fusion opportunities on networks with pooling/non-fused
  activations/residual connections between convs -- which is most real
  networks.
- Before touching `eval.cpp`, re-measured with the restriction temporarily
  bypassed (correctness aside) to see whether full per-conv fusion would
  even be faster: 53 separate `CompiledANE` dispatches for the real
  ResNet50 measured **20.59ms vs Metal's 5.36ms -- 3.8x slower**, not
  faster. So the gap wasn't about the sync bug; `eval.cpp` was left alone.

## v2: true multi-op fusion

The v1 numbers left two live hypotheses for *why* the ANE path was slower:
(a) per-dispatch overhead -- one `CompiledANE`/MPSGraph dispatch per
convolution, 53 of them, each paying queue/`waitUntilCompleted`/MPSGraph
call overhead; or (b) the ANE's own compute throughput for this workload.
v2 tests (a) directly by fusing whole connected regions -- not just one
conv at a time -- into a single `CompiledANE`/MPSGraph dispatch.

### What changed

- `CompiledANE` (`mlx/primitives.h`) is no longer conv-specific. It now
  holds a leaf-input list and an ordered tape of arbitrary fused ops,
  exactly like `Compiled` does for the elementwise fuser.
- `is_ane_op()`/`compile_fuse_ane()` (`mlx/compile.cpp`) no longer pattern-
  match "conv then optionally bias then optionally relu". They union-find
  every `Convolution`/`Add`/`Maximum`/`Pad`/`Slice` node (fp16, GPU stream)
  into maximal connected regions, then fuse a region if: it contains at
  least one convolution; it has exactly one exit value (so `CompiledANE`
  can stay single-output); and every external leaf it reads is provably
  already synchronized (a raw graph input/constant, or another
  `CompiledANE`'s output) -- the same cross-queue-race guard v1 had, just
  checked at the region's boundary instead of at every convolution's input.
  `Pad`/`Slice` are included because mlx's max-pool lowers to exactly that
  idiom (`pad` + 9 strided `slice`s + `maximum`, see
  `max_pool_3x3_s2` in `resnet50_real_bench.cpp`) -- including them means
  pooling is no longer a non-ANE op sitting between two ANE regions, so for
  a real ResNet50 the race-condition boundary essentially disappears: the
  whole backbone collapses into *one* connected region.
- `mlx/backend/metal/compiled_ane.mm` replaced the single hard-coded
  conv+bias+relu graph builder with a generic tape interpreter: it walks
  `CompiledANE`'s ops in order, building the matching MPSGraph node per
  primitive type (conv2d, add, maximum, pad, strided slice), looking up
  already-built tensors by array id. mlx represents broadcasting with an
  explicit `Broadcast` primitive rather than implicit shape promotion, so
  both the fuser and the graph builder look through it to find the real
  leaf underneath (MPSGraph's own binary ops broadcast on their own).

### A second real bug, not just a bigger feature

The MPSGraph executable cache (`get_or_build_executable`) used to key on
`(CompiledANE instance pointer, leaf shapes)`. Once a `CompiledANE` is
destroyed (its compile() cache entry evicted), the allocator can hand that
exact address to a *different*, differently-structured `CompiledANE`; if
the new one's leaves happen to have the same shapes (easy for two small
test convs with the same channel counts but different padding), the cache
silently reused the wrong compiled graph for the new region -- confirmed by
`experimental/ane/correctness_test.cpp`'s `valid_padding_conv_bias_eligible`
case blowing up to `max_abs_diff=14.65` against a same-shaped, differently
padded neighbor case. Fixed by keying on a structural fingerprint of the
whole tape (every op's type and numeric parameters: strides/padding for
conv, axes/sizes for pad, starts/ends/strides for slice) plus leaf shapes,
instead of the object's address.

### Result: fusion granularity was never the bottleneck

With both fixes in, a real ResNet50 (ImageNet-pretrained weights, BatchNorm
folded, cross-checked against both mlx-CPU and PyTorch) fuses into exactly
**one** `CompiledANE` covering the whole backbone (confirmed via
`ANE_DEBUG_CACHE=1`: a single tape of 189 ops, 107 leaf inputs, one
MPSGraph build, cache-hit on every subsequent call) -- not 53 dispatches,
one. Correctness stayed at normal fp16-rounding levels end to end
(`ANE_DEBUG_STAGES=1` shows the ane-vs-cpu diff growing gradually from
0.0 to ~0.06 over 26 stage checkpoints, no blow-up). Independently
confirmed the whole region really is placed on the ANE, not silently
falling back to GPU: `$TMPDIR/com.apple.MetalPerformanceShadersGraph/`
contains a single `main_ANE_region_0_0.bc.mlir` for the run.

Performance did not improve -- if anything it's worse than the per-conv
version: **Metal ~5.8-8.9ms/iter vs CompiledANE ~18-20ms/iter (ratio
~0.3-0.45x)**, consistently across repeated runs. Collapsing 53 dispatches
into 1 left the ANE wall-clock time roughly where it was with 53 separate
dispatches (~20ms either way). That rules out hypothesis (a): this was
never a per-dispatch-overhead problem. What's left is (b) -- for this
workload (ResNet50, fp16, batch 1, 224x224) on this hardware, the ANE's own
compute throughput through MPSGraph's `preferredDevice` hint is slower than
mlx's tuned Metal GPU kernels, and no amount of fusing dispatches together
changes that.

## Where this leaves v2

- The multi-op fusion mechanism itself is real, correct, and more general
  than v1 (arbitrary conv/add/relu/pool regions, not just one conv), and it
  resolved its own real bug independent of the original cross-queue race.
- It does not deliver a speedup, and the measurements now point at the ANE's
  raw throughput for this workload, not at anything left to fix in mlx's
  fusion or dispatch code. `mlx/backend/metal/eval.cpp` was never touched,
  consistent with the project's direction to keep ANE-specific logic on its
  own path; there is no evidence here that touching it would help.
