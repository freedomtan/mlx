// Copyright © 2026 Apple Inc.
//
// eval_gpu for CompiledANE: runs a fused region of Convolution/Add/Maximum/
// Pad/Slice ops through a single MPSGraph with the private
// preferredDevice=ANE hint, instead of mlx's regular Metal kernels. See
// is_ane_op()/compile_fuse_ane() in mlx/compile.cpp for what graphs are
// eligible to reach this path, and CompiledANE in mlx/primitives.h for how
// a region is represented (leaf inputs + an ordered tape of its ops).
//
// mlx convolutions are NHWC with OHWI weights, but MPSGraph's private ANE
// compiler only places Conv2D on the real NE matrix array when the data
// layout is NCHW (NHWC falls back to the much slower Planar/Elementwise
// engine for the tensor sizes a real conv net actually uses) -- so the
// whole fused region is converted to NCHW internally, with weights
// permuted to OIHW, right after the input placeholder, and converted back
// to NHWC right before the compiled graph's output.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/primitives.h"

// Private API, guarded at every call site by respondsToSelector:. If it
// disappears in a future OS, ane_is_available() reports false and
// compile_fuse_ane() never creates a CompiledANE node.
@interface MPSGraphCompilationDescriptor (ANEHint)
@property(nonatomic, assign) unsigned long long preferredDevice;
- (void)setEnableCompileResourcesForPackage:(BOOL)val;
@end

// Debug-only (ANE_DUMP_HWX env var): serializes the compiled executable's
// actual .hwx ANE binary to disk, so it can be inspected directly rather
// than inferred from power telemetry -- see
// experimental/ane/resnet50_hwx_check.md for how this is used.
@interface MPSGraphExecutableSerializationDescriptor (Private)
@property(readwrite, assign) BOOL serializeOriginalModule;
@end

// Offline target-architecture device, used only for the ANE_DUMP_HWX debug
// path: compiling against the live MTLDevice and then asking for a
// serializable package tries to load the program onto real ANE hardware
// (hits a com.apple.appleneuralengine entitlement error in this sandbox);
// compiling against an explicit architecture target instead never touches
// real hardware, matching measure_ane_capacity/convert_fp8_to_hwx.m's
// approach.
@interface MPSGraphDeviceDescriptor : NSObject
- (instancetype)initWithSerializedProperties:(NSArray*)props;
@end

@interface MPSGraphDevice (TargetArch)
- (instancetype)initWithDeviceDescriptor:(MPSGraphDeviceDescriptor*)desc;
@end

@interface MPSGraphExecutable (Package)
- (void)serializeToMPSGraphPackageAtURL:(NSURL*)url descriptor:(id)desc;
@end

namespace mlx::core {

namespace {

array ensure_row_contiguous(const array& x, const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  auto result = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(result);
  return result;
}

MPSShape* to_mps_shape(const Shape& shape) {
  NSMutableArray<NSNumber*>* dims =
      [NSMutableArray arrayWithCapacity:shape.size()];
  for (auto d : shape) {
    [dims addObject:@(d)];
  }
  return dims;
}

MPSShape* to_mps_shape(const std::vector<int>& shape) {
  NSMutableArray<NSNumber*>* dims =
      [NSMutableArray arrayWithCapacity:shape.size()];
  for (auto d : shape) {
    [dims addObject:@(d)];
  }
  return dims;
}

// Expands a sparse (axes, values) pair into a dense, full-rank vector with
// `fill` everywhere except the given axes -- used to turn Pad's per-axis
// state into the full-rank left/right padding MPSGraph expects.
std::vector<int> expand_axes(
    int ndim,
    const std::vector<int>& axes,
    const Shape& values,
    int fill) {
  std::vector<int> out(ndim, fill);
  for (size_t i = 0; i < axes.size(); i++) {
    out[axes[i]] = values[i];
  }
  return out;
}

struct GraphExecutable {
  MPSGraphExecutable* executable;
  // Leaf index (into CompiledANE::ane_inputs()) that each entry of
  // executable.feedTensors corresponds to -- the only leaves actually bound
  // at run time are the non-constant ones, in whatever order the MPSGraph
  // compiler chose for its placeholders, which is not guaranteed to match
  // either construction order or CompiledANE's own leaf order.
  std::vector<size_t> placeholder_leaf_index;
  // Parallel to placeholder_leaf_index: true for the 4D image-like
  // placeholders, whose declared shape is already NCHW (see
  // get_or_build_executable) -- eval_gpu must feed them an actual NCHW
  // buffer, not the leaf's native NHWC one.
  std::vector<bool> placeholder_needs_nchw;
  // True if the graph's own output is 4D image-like and so is declared NCHW
  // (see get_or_build_executable) -- eval_gpu must run it into an NCHW
  // scratch buffer and convert back to NHWC, instead of mlx's out buffer
  // directly.
  bool output_needs_nchw;
};

std::mutex& cache_mutex() {
  static std::mutex m;
  return m;
}

// unordered_map guarantees reference stability on insert (only erase
// invalidates element references), so it is safe to return a reference here
// after releasing the lock.
std::unordered_map<std::string, GraphExecutable>& executable_cache() {
  static std::unordered_map<std::string, GraphExecutable> cache;
  return cache;
}

// A fresh MTLCommandQueue per eval_gpu call is pure overhead; MPSGraph
// executables can be run repeatedly on the same queue.
id<MTLCommandQueue> ane_command_queue(id<MTLDevice> device) {
  static id<MTLCommandQueue> queue = [device newCommandQueue];
  return queue;
}

// NHWC->NCHW for the 4D image-like leaves, as a plain Metal compute kernel
// instead of an in-graph transposeTensor: (see its call site in eval_gpu).
// Deliberately not using mlx's own transpose/copy kernels or command
// encoder: this needs to be encoded onto the ANE call's own self-managed
// MPSCommandBuffer, ahead of the MPSGraph executable, so Metal's normal
// same-command-buffer ordering guarantees the copy is done before the ANE
// reads it -- without calling commit/wait on mlx's shared CommandEncoder,
// which corrupts mlx's own gpu::eval() dispatcher (see eval_gpu).
id<MTLComputePipelineState> nhwc_to_nchw_pipeline(id<MTLDevice> device) {
  static id<MTLComputePipelineState> pipeline = [device] {
    NSString* src = @"#include <metal_stdlib>\n"
                     "using namespace metal;\n"
                     "kernel void nhwc_to_nchw_f16(\n"
                     "    device const half* in_buf [[buffer(0)]],\n"
                     "    device half* out_buf [[buffer(1)]],\n"
                     "    constant uint& H [[buffer(2)]],\n"
                     "    constant uint& W [[buffer(3)]],\n"
                     "    constant uint& C [[buffer(4)]],\n"
                     "    uint3 gid [[thread_position_in_grid]]) {\n"
                     "  uint w = gid.x, h = gid.y, nc = gid.z;\n"
                     "  if (w >= W || h >= H) return;\n"
                     "  uint n = nc / C, c = nc % C;\n"
                     "  uint in_idx = ((n * H + h) * W + w) * C + c;\n"
                     "  uint out_idx = ((n * C + c) * H + h) * W + w;\n"
                     "  out_buf[out_idx] = in_buf[in_idx];\n"
                     "}\n";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src
                                               options:nil
                                                 error:&err];
    if (!lib) {
      fprintf(
          stderr, "[ane nchw kernel] library compile failed: %s\n",
          err.localizedDescription.UTF8String);
      return (id<MTLComputePipelineState>)nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"nhwc_to_nchw_f16"];
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      fprintf(
          stderr, "[ane nchw kernel] pipeline state failed: %s\n",
          err.localizedDescription.UTF8String);
    }
    return pso;
  }();
  return pipeline;
}

// NCHW->NHWC for the graph's own output, as a plain Metal compute kernel
// instead of an in-graph transposeTensor: -- same reasoning as
// nhwc_to_nchw_pipeline above, encoded onto the ANE call's own
// MPSCommandBuffer, after the MPSGraph executable, so the ANE's write is
// guaranteed visible to this read by same-command-buffer ordering.
id<MTLComputePipelineState> nchw_to_nhwc_pipeline(id<MTLDevice> device) {
  static id<MTLComputePipelineState> pipeline = [device] {
    NSString* src = @"#include <metal_stdlib>\n"
                     "using namespace metal;\n"
                     "kernel void nchw_to_nhwc_f16(\n"
                     "    device const half* in_buf [[buffer(0)]],\n"
                     "    device half* out_buf [[buffer(1)]],\n"
                     "    constant uint& H [[buffer(2)]],\n"
                     "    constant uint& W [[buffer(3)]],\n"
                     "    constant uint& C [[buffer(4)]],\n"
                     "    uint3 gid [[thread_position_in_grid]]) {\n"
                     "  uint w = gid.x, h = gid.y, nc = gid.z;\n"
                     "  if (w >= W || h >= H) return;\n"
                     "  uint n = nc / C, c = nc % C;\n"
                     "  uint in_idx = ((n * C + c) * H + h) * W + w;\n"
                     "  uint out_idx = ((n * H + h) * W + w) * C + c;\n"
                     "  out_buf[out_idx] = in_buf[in_idx];\n"
                     "}\n";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src
                                               options:nil
                                                 error:&err];
    if (!lib) {
      fprintf(
          stderr, "[ane nchw kernel] library compile failed: %s\n",
          err.localizedDescription.UTF8String);
      return (id<MTLComputePipelineState>)nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"nchw_to_nhwc_f16"];
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      fprintf(
          stderr, "[ane nchw kernel] pipeline state failed: %s\n",
          err.localizedDescription.UTF8String);
    }
    return pso;
  }();
  return pipeline;
}

// Keyed on the CompiledANE instance (its tape/structure is fixed once
// built) plus the runtime leaf shapes (defensive: mlx's compile cache keys
// on input shape too, so a given instance should only ever see one set of
// leaf shapes. Keying on the CompiledANE instance's address alone is not
// enough: once a CompiledANE is destroyed (e.g. a previous compile() cache
// entry is evicted), the allocator can hand that exact address to an
// unrelated, differently-shaped CompiledANE, silently reusing a stale
// MPSGraph executable with the wrong convolution/pad/slice parameters for
// identically-shaped leaves -- confirmed by two single-conv correctness
// cases with the same input/weight/bias shapes but different padding
// corrupting each other's result. The structural fingerprint below (every
// tape op's type and numeric parameters) makes the key depend on what the
// region actually computes, not where its C++ object happens to live.
template <typename Vec>
void append_ints(std::ostringstream& os, const Vec& v) {
  for (auto x : v) {
    os << x << ',';
  }
}

std::string cache_key(const CompiledANE* prim, const std::vector<array>& inputs) {
  std::ostringstream os;
  for (auto& node : prim->tape()) {
    auto& p = node.primitive();
    os << typeid(p).name() << ';';
    if (typeid(p) == typeid(Convolution)) {
      auto& conv = static_cast<Convolution&>(p);
      auto [strides, padding_lo, padding_hi, kernel_dilation, input_dilation,
            groups, flip] = conv.state();
      append_ints(os, strides);
      append_ints(os, padding_lo);
      append_ints(os, padding_hi);
    } else if (typeid(p) == typeid(Pad)) {
      auto& pad = static_cast<Pad&>(p);
      auto [axes, low, high] = pad.state();
      append_ints(os, axes);
      append_ints(os, low);
      append_ints(os, high);
    } else if (typeid(p) == typeid(Slice)) {
      auto& sl = static_cast<Slice&>(p);
      auto [starts, ends, strides] = sl.state();
      append_ints(os, starts);
      append_ints(os, ends);
      append_ints(os, strides);
    }
    os << '|';
  }
  os << '_';
  for (size_t i = 0; i < inputs.size(); i++) {
    append_ints(os, inputs[i].shape());
    // Constant leaves are baked into the graph by value, not bound at run
    // time, so two CompiledANE instances that happen to share a structural
    // fingerprint and leaf shapes but have different constant *values*
    // (e.g. a different trained model's weights) must not share a cached
    // executable. Array ids are cheap and distinguish that case without
    // hashing the full (possibly multi-MB) weight content on every call.
    if (prim->is_constant(i)) {
      os << '#' << inputs[i].id();
    }
    os << '|';
  }
  return os.str();
}

// True if `a` is an already-materialized scalar equal to 0 -- the shape
// Maximum's inputs take when mlx lowers relu(x) to maximum(x, 0).
bool is_scalar_zero(const array& a) {
  return a.size() == 1 && static_cast<float>(a.item<float16_t>()) == 0.0f;
}

// mlx arrays are NHWC; the whole fused region is converted to NCHW
// internally (one transposeTensor: right after the input placeholder, one
// back to NHWC before the final output -- see get_or_build_executable)
// because MPSGraph's private ANE compiler only places a convolution on the
// real NE matrix array when its data layout is NCHW: below that, Conv2D
// silently falls back to the much slower Planar/Elementwise engine
// regardless of tensor size, and for NHWC that fallback engages even for
// typical conv-net feature map sizes (confirmed empirically: NCHW reaches
// the NE array at every tested size including resnet50's 7x7..112x112
// range; NHWC only did so at >=128, which resnet50 never reaches past its
// stem). mlx weights are OHWI; MPSGraph has no OHWI layout option, only
// OIHW. Permuting the bytes ourselves once, here, and feeding the result in
// as an already-OIHW constant avoids a transposeTensor: node per
// convolution -- one fewer op (and ANE compiler procedure) per conv, for
// every conv whose weight is a compile-time constant (the common case for
// inference; see CompiledANE's comment in mlx/primitives.h). A weight
// that's still a runtime placeholder (e.g. chained from another
// CompiledANE's output) can't be pre-permuted since its data isn't known
// yet, so that case still transposes at graph-build time instead.
MPSGraphTensor* build_oihw_weight_constant(MPSGraph* graph, const array& w) {
  auto& s = w.shape(); // OHWI
  int O = s[0], H = s[1], W = s[2], I = s[3];
  std::vector<float16_t> oihw(static_cast<size_t>(O) * I * H * W);
  const float16_t* ohwi = w.data<float16_t>();
  for (int o = 0; o < O; o++) {
    for (int h = 0; h < H; h++) {
      for (int wi = 0; wi < W; wi++) {
        for (int i = 0; i < I; i++) {
          oihw[((static_cast<size_t>(o) * I + i) * H + h) * W + wi] =
              ohwi[((static_cast<size_t>(o) * H + h) * W + wi) * I + i];
        }
      }
    }
  }
  NSData* data = [NSData dataWithBytes:oihw.data()
                                 length:oihw.size() * sizeof(float16_t)];
  return [graph constantWithData:data
                           shape:@[ @(O), @(I), @(H), @(W) ]
                        dataType:MPSDataTypeFloat16];
}

MPSGraphTensor* build_convolution(
    MPSGraph* graph,
    const array& node,
    MPSGraphTensor* input,
    MPSGraphTensor* weight_oihw) {
  auto& conv = static_cast<Convolution&>(node.primitive());
  auto [strides, padding_lo, padding_hi, kernel_dilation, input_dilation,
        groups, flip] = conv.state();

  MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
      descriptorWithStrideInX:static_cast<NSUInteger>(strides[1])
                     strideInY:static_cast<NSUInteger>(strides[0])
               dilationRateInX:1
               dilationRateInY:1
                        groups:1
                   paddingLeft:static_cast<NSUInteger>(padding_lo[1])
                  paddingRight:static_cast<NSUInteger>(padding_lo[1])
                    paddingTop:static_cast<NSUInteger>(padding_lo[0])
                 paddingBottom:static_cast<NSUInteger>(padding_lo[0])
                  paddingStyle:MPSGraphPaddingStyleExplicit
                    dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                 weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
  return [graph convolution2DWithSourceTensor:input
                                 weightsTensor:weight_oihw
                                    descriptor:desc
                                          name:nil];
}

// mlx's Pad/Slice primitive state is a full-rank vector indexed in mlx's own
// (NHWC) axis order. Every intermediate tensor in this fused region is now
// NCHW (see build_oihw_weight_constant's comment), so any such vector must
// be reordered to NCHW axis order -- [N,H,W,C] -> [N,C,H,W] -- before it's
// positionally correct for the actual (now-NCHW) tensor it applies to.
template <typename Vec>
Vec nhwc_to_nchw(const Vec& v) {
  return {v[0], v[3], v[1], v[2]};
}

MPSGraphTensor* build_pad(MPSGraph* graph, const array& node, MPSGraphTensor* input) {
  auto& pad = static_cast<Pad&>(node.primitive());
  auto [axes, low, high] = pad.state();
  int ndim = node.ndim();
  auto left = nhwc_to_nchw(expand_axes(ndim, axes, low, 0));
  auto right = nhwc_to_nchw(expand_axes(ndim, axes, high, 0));
  double fill = static_cast<double>(static_cast<float>(node.inputs()[1].item<float16_t>()));
  return [graph padTensor:input
             withPaddingMode:MPSGraphPaddingModeConstant
                 leftPadding:to_mps_shape(left)
                rightPadding:to_mps_shape(right)
               constantValue:fill
                        name:nil];
}

MPSGraphTensor* build_slice(MPSGraph* graph, const array& node, MPSGraphTensor* input) {
  auto& slice = static_cast<Slice&>(node.primitive());
  auto [starts, ends, strides] = slice.state();
  return [graph sliceTensor:input
                      starts:to_mps_shape(nhwc_to_nchw(starts))
                        ends:to_mps_shape(nhwc_to_nchw(ends))
                     strides:to_mps_shape(nhwc_to_nchw(strides))
                        name:nil];
}

// mlx has no native pooling primitive (mlx.nn's max-pool, matching
// experimental/ane/resnet50_real_bench.cpp's max_pool_3x3_s2, lowers to
// pad + k*k overlapping strided slices + (k*k-1) pairwise maximums).
// is_ane_op()/compile_fuse_ane() in mlx/compile.cpp fuse all of those
// generically, but translating them 1:1 costs k*k-1 extra ops (and ANE
// compiler procedures) per pool. Recognizing the exact idiom here and
// emitting one native maxPooling2D call instead is a pure win: same NHWC
// explicit-padding semantics, same max-reduction, far fewer ops.
//
// Matching does NOT assume the slices/maximums sit at consecutive tape
// positions right after the Pad -- compile_dfs's topological order only
// guarantees producers precede consumers, not that a producer's consumers
// are contiguous -- so this scans the rest of the tape by structure
// instead. It's still strict (exact slice-count-is-a-perfect-square,
// identical slice strides, a slice/pad/intermediate-max used nowhere else)
// so anything that doesn't fit this exact shape safely falls through to
// the generic per-op path instead of risking a silent mistranslation.
//
// On a match, returns true and fills `skip_ids` with every tape node's id
// that's now folded into the single pooled result (the Pad, its k*k
// slices, and all k*k-1 maximums, including the terminal one) and
// `terminal_id` with the one among them whose value callers should
// actually use. Returns false (touching neither) otherwise.
bool find_maxpool_idiom(
    const std::vector<array>& tape,
    size_t pad_idx,
    std::unordered_set<uintptr_t>* skip_ids,
    uintptr_t* terminal_id,
    int* kernel_size,
    Shape* strides_out) {
  auto& pad_node = tape[pad_idx];
  uintptr_t pad_id = pad_node.id();

  std::vector<size_t> slice_idxs;
  for (size_t i = pad_idx + 1; i < tape.size(); i++) {
    if (typeid(tape[i].primitive()) == typeid(Slice) &&
        tape[i].inputs()[0].id() == pad_id) {
      slice_idxs.push_back(i);
    }
  }
  size_t n = slice_idxs.size();
  int k = static_cast<int>(std::lround(std::sqrt(static_cast<double>(n))));
  if (n < 4 || static_cast<size_t>(k * k) != n) {
    return false;
  }

  auto& s0 = static_cast<Slice&>(tape[slice_idxs[0]].primitive());
  auto [starts0, ends0, strides0] = s0.state();
  for (size_t si : slice_idxs) {
    auto& sl = static_cast<Slice&>(tape[si].primitive());
    auto [st, en, str] = sl.state();
    if (str != strides0) {
      return false;
    }
  }

  std::unordered_set<uintptr_t> pool_ids = {pad_id};
  for (size_t si : slice_idxs) {
    pool_ids.insert(tape[si].id());
  }

  // Fixed-point scan: a Maximum node joins the group once both its inputs
  // are already in it, which may take several passes if the reduction
  // isn't laid out as one monotonically increasing run.
  size_t max_count = 0;
  uintptr_t terminal = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t j = pad_idx + 1; j < tape.size(); j++) {
      if (typeid(tape[j].primitive()) != typeid(Maximum) ||
          pool_ids.count(tape[j].id())) {
        continue;
      }
      auto& in0 = tape[j].inputs()[0];
      auto& in1 = tape[j].inputs()[1];
      if (pool_ids.count(in0.id()) && pool_ids.count(in1.id())) {
        pool_ids.insert(tape[j].id());
        terminal = tape[j].id();
        max_count++;
        changed = true;
      }
    }
  }
  if (max_count != n - 1) {
    return false;
  }

  // Every matched node's result must stay internal to the group, except
  // the terminal's -- that's the group's actual output, meant to be
  // consumed externally (e.g. by the next conv). Skipping the others'
  // individual translation would otherwise leave some unrelated consumer
  // unable to find them in tensor_map.
  for (auto& other : tape) {
    for (auto& in : other.inputs()) {
      if (pool_ids.count(in.id()) && in.id() != terminal &&
          !pool_ids.count(other.id())) {
        return false;
      }
    }
  }

  *skip_ids = std::move(pool_ids);
  *terminal_id = terminal;
  *kernel_size = k;
  *strides_out = strides0;
  return true;
}

MPSGraphTensor* build_maxpool(
    MPSGraph* graph,
    const array& pad_node,
    int k,
    const Shape& strides0,
    MPSGraphTensor* input) {
  auto& pad_prim = static_cast<Pad&>(pad_node.primitive());
  auto [axes, low, high] = pad_prim.state();
  int ndim = pad_node.ndim();
  auto left = expand_axes(ndim, axes, low, 0);
  auto right = expand_axes(ndim, axes, high, 0);

  MPSGraphPooling2DOpDescriptor* desc = [MPSGraphPooling2DOpDescriptor
      descriptorWithKernelWidth:static_cast<NSUInteger>(k)
                    kernelHeight:static_cast<NSUInteger>(k)
                       strideInX:static_cast<NSUInteger>(strides0[2])
                       strideInY:static_cast<NSUInteger>(strides0[1])
                 dilationRateInX:1
                 dilationRateInY:1
                     paddingLeft:static_cast<NSUInteger>(left[2])
                    paddingRight:static_cast<NSUInteger>(right[2])
                      paddingTop:static_cast<NSUInteger>(left[1])
                   paddingBottom:static_cast<NSUInteger>(right[1])
                    paddingStyle:MPSGraphPaddingStyleExplicit
                      dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
  return [graph maxPooling2DWithSourceTensor:input descriptor:desc name:nil];
}

const GraphExecutable& get_or_build_executable(
    id<MTLDevice> device,
    const CompiledANE* prim,
    const std::vector<array>& inputs) {
  auto key = cache_key(prim, inputs);
  std::lock_guard<std::mutex> lock(cache_mutex());
  auto it = executable_cache().find(key);
  if (it != executable_cache().end()) {
    if (std::getenv("ANE_DEBUG_CACHE")) {
      fprintf(stderr, "[ane cache] HIT prim=%p tape_size=%zu\n", (void*)prim,
              prim->tape().size());
    }
    return it->second;
  }
  if (std::getenv("ANE_DEBUG_CACHE")) {
    fprintf(stderr, "[ane cache] MISS/BUILD prim=%p tape_size=%zu leaves=%zu\n",
            (void*)prim, prim->tape().size(), prim->ane_inputs().size());
    std::unordered_map<std::string, int> counts;
    for (auto& n : prim->tape()) {
      counts[n.primitive().name()]++;
    }
    for (auto& [name, c] : counts) {
      fprintf(stderr, "[ane cache]   %s: %d\n", name.c_str(), c);
    }
  }

  @autoreleasepool {
    MPSGraph* graph = [MPSGraph new];
    auto& leaves = prim->ane_inputs();

    // Conv weights that are constants get pre-permuted to OIHW and built
    // directly at their Convolution node (build_oihw_weight_constant), not
    // here, so they skip a separate transposeTensor: node entirely. A
    // weight is only ever consumed by the one conv it belongs to, so this
    // set is exact, not just a superset.
    std::unordered_set<uintptr_t> conv_weight_leaf_ids;
    for (auto& node : prim->tape()) {
      if (typeid(node.primitive()) == typeid(Convolution)) {
        conv_weight_leaf_ids.insert(node.inputs()[1].id());
      }
    }
    // Bias leaves: mlx's conv+bias idiom adds a raw 1D [C] array directly
    // (relying on mlx's own last-axis broadcast), identified generically as
    // any rank-1 array feeding an Add -- the only other Add in this region
    // is the residual/shortcut sum, which is 4D+4D, not 1D+4D. Built
    // directly as (1,C,1,1), matching the NCHW pipeline's channel axis,
    // instead of a (C,) constant plus a reshapeTensor: node: that reshape
    // was confirmed empirically to block live ANE placement even though it
    // doesn't change the result.
    std::unordered_set<uintptr_t> bias_leaf_ids;
    for (auto& node : prim->tape()) {
      if (typeid(node.primitive()) == typeid(Add)) {
        for (auto& in : node.inputs()) {
          if (in.ndim() == 1) {
            bias_leaf_ids.insert(in.id());
          }
        }
      }
    }
    std::unordered_map<uintptr_t, size_t> leaf_index;
    for (size_t i = 0; i < leaves.size(); i++) {
      leaf_index[leaves[i].id()] = i;
    }

    std::unordered_map<uintptr_t, MPSGraphTensor*> tensor_map;
    NSMutableDictionary* feeds = [NSMutableDictionary new];
    // MPSGraphTensor pointers are unique objects, safe to use as a map key
    // directly (ARC keeps them alive via tensor_map/feeds/the graph itself).
    std::unordered_map<MPSGraphTensor*, size_t> placeholder_leaf_of;
    for (size_t i = 0; i < leaves.size(); i++) {
      if (prim->is_constant(i) && conv_weight_leaf_ids.count(leaves[i].id())) {
        continue; // Built at its Convolution node instead; see above.
      }
      if (prim->is_constant(i) && bias_leaf_ids.count(leaves[i].id())) {
        // Built directly as (1,C,1,1), matching the NCHW pipeline's
        // channel axis; see bias_leaf_ids' comment above. Same raw data as
        // the array's own (C,) shape, just declared with a different rank.
        NSData* data = [NSData dataWithBytes:inputs[i].data<float16_t>()
                                       length:inputs[i].nbytes()];
        int C = static_cast<int>(inputs[i].shape(0));
        MPSGraphTensor* c = [graph constantWithData:data
                                              shape:@[ @1, @(C), @1, @1 ]
                                           dataType:MPSDataTypeFloat16];
        tensor_map[leaves[i].id()] = c;
        continue;
      }
      if (prim->is_constant(i)) {
        // MPSGraph's private ANE placement only engages a convolution when
        // its weight is a compile-time constant, not a runtime placeholder
        // -- confirmed empirically (see CompiledANE's comment in
        // mlx/primitives.h). Embed the leaf's current value directly;
        // correctness here depends on this leaf never changing again for
        // the life of this cached executable, which holds for inference
        // with weights loaded once (this backend's only intended use).
        NSData* data = [NSData dataWithBytes:inputs[i].data<float16_t>()
                                       length:inputs[i].nbytes()];
        MPSGraphTensor* c = [graph constantWithData:data
                                              shape:to_mps_shape(inputs[i].shape())
                                           dataType:MPSDataTypeFloat16];
        tensor_map[leaves[i].id()] = c;
        continue;
      }
      // Every leaf that isn't a conv weight or a bias is an "image-like"
      // NHWC feature map (the network input, or a non-constant chain from
      // another CompiledANE's output). The internal pipeline needs NCHW
      // (see this file's header comment), but declaring the placeholder
      // itself directly in NCHW -- instead of an NHWC placeholder plus an
      // in-graph transposeTensor: -- avoids a dedicated ANE task for that
      // transpose entirely: eval_gpu feeds this placeholder an already-NCHW
      // buffer, materialized by nhwc_to_nchw_pipeline on the ANE call's own
      // command buffer (see eval_gpu).
      bool is_image_like =
          !conv_weight_leaf_ids.count(leaves[i].id()) &&
          !bias_leaf_ids.count(leaves[i].id()) && inputs[i].ndim() == 4;
      MPSShape* ph_shape = is_image_like
          ? to_mps_shape(nhwc_to_nchw(inputs[i].shape()))
          : to_mps_shape(inputs[i].shape());
      MPSGraphTensor* ph = [graph placeholderWithShape:ph_shape
                                               dataType:MPSDataTypeFloat16
                                                   name:nil];
      tensor_map[leaves[i].id()] = ph;
      placeholder_leaf_of[ph] = i;
      feeds[ph] = [[MPSGraphShapedType alloc] initWithShape:ph.shape
                                                     dataType:MPSDataTypeFloat16];
    }
    if (std::getenv("ANE_DEBUG_CACHE")) {
      fprintf(stderr, "[ane cache] placeholder (non-constant) inputs: %lu\n",
              (unsigned long)feeds.count);
    }

    std::function<MPSGraphTensor*(const array&)> lookup = [&](const array& raw_a) -> MPSGraphTensor* {
      // mlx materializes broadcasting via an explicit Broadcast primitive;
      // compile_fuse_ane() in mlx/compile.cpp looks through it the same way
      // when building the leaf list, so tensor_map/leaf ids line up with
      // the real value underneath, not the Broadcast node.
      const array& a = (raw_a.has_primitive() &&
                         typeid(raw_a.primitive()) == typeid(Broadcast))
          ? raw_a.inputs()[0]
          : raw_a;
      auto it2 = tensor_map.find(a.id());
      if (it2 != tensor_map.end()) {
        return it2->second;
      }
      // An already-materialized scalar not wired up as a leaf placeholder
      // (is_ane_op()/compile_fuse_ane() in compile.cpp only let these
      // through as inline constants): bake its value in directly.
      return [graph constantWithScalar:static_cast<double>(static_cast<float>(a.item<float16_t>()))
                               dataType:MPSDataTypeFloat16];
    };

    auto& tape = prim->tape();

    // Structural discovery pass: find every Pad-rooted max-pool idiom up
    // front (regardless of where its slices/maximums happen to land in
    // topological order), before building any MPSGraph nodes. Each match
    // folds into one maxPooling2D built when the loop reaches its Pad;
    // every other node it covers (slices, intermediate/terminal maximums)
    // is skipped individually.
    std::unordered_set<uintptr_t> maxpool_skip_ids;
    struct MaxpoolMatch {
      uintptr_t terminal_id;
      int kernel_size;
      Shape strides;
    };
    std::unordered_map<size_t, MaxpoolMatch> maxpool_at_pad_idx;
    for (size_t i = 0; i < tape.size(); i++) {
      if (typeid(tape[i].primitive()) != typeid(Pad)) {
        continue;
      }
      std::unordered_set<uintptr_t> skip_ids;
      uintptr_t terminal_id;
      int k;
      Shape strides;
      if (find_maxpool_idiom(tape, i, &skip_ids, &terminal_id, &k, &strides)) {
        maxpool_at_pad_idx[i] = {terminal_id, k, strides};
        maxpool_skip_ids.insert(skip_ids.begin(), skip_ids.end());
        if (std::getenv("ANE_DEBUG_CACHE")) {
          fprintf(stderr,
                  "[ane maxpool fusion] matched %dx%d pool, stride (%d,%d)\n",
                  k, k, (int)strides[1], (int)strides[2]);
        }
      }
    }

    for (size_t idx = 0; idx < tape.size(); idx++) {
      auto& node = tape[idx];
      auto& node_prim = node.primitive();
      if (auto it = maxpool_at_pad_idx.find(idx); it != maxpool_at_pad_idx.end()) {
        MPSGraphTensor* pooled = build_maxpool(
            graph, node, it->second.kernel_size, it->second.strides,
            lookup(node.inputs()[0]));
        tensor_map[it->second.terminal_id] = pooled;
        continue;
      }
      if (maxpool_skip_ids.count(node.id())) {
        continue;
      }
      MPSGraphTensor* out;
      if (typeid(node_prim) == typeid(Convolution)) {
        auto& w_leaf = node.inputs()[1];
        MPSGraphTensor* weight_oihw;
        if (conv_weight_leaf_ids.count(w_leaf.id())) {
          weight_oihw =
              build_oihw_weight_constant(graph, inputs[leaf_index.at(w_leaf.id())]);
        } else {
          // Not a constant (e.g. a placeholder chained from another
          // CompiledANE's output): its data isn't known at graph-build
          // time, so permute at run time instead via a transpose node.
          // OHWI -> OIHW: O stays put, I moves from the last axis to the
          // second, H and W each shift one slot over.
          weight_oihw = [graph transposeTensor:lookup(w_leaf)
                                    permutation:@[ @0, @3, @1, @2 ]
                                           name:nil];
        }
        out = build_convolution(graph, node, lookup(node.inputs()[0]), weight_oihw);
      } else if (typeid(node_prim) == typeid(Add)) {
        MPSGraphTensor* lhs = lookup(node.inputs()[0]);
        MPSGraphTensor* rhs = lookup(node.inputs()[1]);
        // A constant bias leaf is already (1,C,1,1) (built above in the
        // leaf loop); a non-constant one (rare -- a runtime placeholder,
        // not this backend's common inference case) is still (C,) here, so
        // reshape it to broadcast against the NCHW channel axis instead of
        // NHWC's last axis.
        if (lhs.shape.count == 1 && rhs.shape.count == 4) {
          lhs = [graph reshapeTensor:lhs withShape:@[ @1, lhs.shape[0], @1, @1 ] name:nil];
        } else if (rhs.shape.count == 1 && lhs.shape.count == 4) {
          rhs = [graph reshapeTensor:rhs withShape:@[ @1, rhs.shape[0], @1, @1 ] name:nil];
        }
        out = [graph additionWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
      } else if (typeid(node_prim) == typeid(Maximum)) {
        // relu(x) lowers to maximum(x, 0) in mlx. MPSGraph's ANE compiler
        // only seems to recognize the conv+bias+activation idiom (and fold
        // it into far fewer ANE procedures) when the activation is the
        // native reLUWithTensor: op, not a generic binary maximum against a
        // zero constant -- so detect that shape here instead of always
        // emitting maximumWithPrimaryTensor:secondaryTensor:.
        auto& in0 = node.inputs()[0];
        auto& in1 = node.inputs()[1];
        if (is_scalar_zero(in1)) {
          out = [graph reLUWithTensor:lookup(in0) name:nil];
        } else if (is_scalar_zero(in0)) {
          out = [graph reLUWithTensor:lookup(in1) name:nil];
        } else {
          out = [graph maximumWithPrimaryTensor:lookup(in0)
                                  secondaryTensor:lookup(in1)
                                             name:nil];
        }
      } else if (typeid(node_prim) == typeid(Pad)) {
        out = build_pad(graph, node, lookup(node.inputs()[0]));
      } else if (typeid(node_prim) == typeid(Slice)) {
        out = build_slice(graph, node, lookup(node.inputs()[0]));
      } else {
        throw std::runtime_error(
            std::string("[CompiledANE] unsupported op in fused region: ") +
            node_prim.name());
      }
      tensor_map[node.id()] = out;
    }

    MPSGraphTensor* target = tensor_map.at(prim->tape().back().id());
    // Internal pipeline is NCHW (see build_oihw_weight_constant's comment);
    // mlx's own array/buffer layout is NHWC. Leave the graph's own output
    // NCHW -- same reasoning as the input side above -- and convert back to
    // NHWC with a Metal kernel in eval_gpu instead of an in-graph
    // transposeTensor:, which used to cost a dedicated ANE task.
    bool output_needs_nchw = target.shape.count == 4;

    const char* dump_dir = std::getenv("ANE_DUMP_HWX");
    const char* dump_arch = std::getenv("ANE_DUMP_HWX_ARCH"); // e.g. "h18g"

    MPSGraphDevice* mps_device;
    if (dump_dir && dump_arch) {
      MPSGraphDeviceDescriptor* devDesc = [[MPSGraphDeviceDescriptor alloc]
          initWithSerializedProperties:@[
            @(0), @(6), [NSString stringWithUTF8String:dump_arch]
          ]];
      mps_device = [[MPSGraphDevice alloc] initWithDeviceDescriptor:devDesc];
    } else {
      mps_device = [MPSGraphDevice deviceWithMTLDevice:device];
    }

    MPSGraphCompilationDescriptor* cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel = MPSGraphOptimizationLevel1;
    if ([cd respondsToSelector:@selector(setPreferredDevice:)]) {
      cd.preferredDevice = 2; // MPSGraphDeviceTypeANE (private).
    }
    // KVC, not a declared selector: aneBondedCompileMode isn't in any public
    // or informally-declared private header, found by extracting mps.*
    // property-name strings out of our own dumped .mpsgraph MLIR bytecode.
    // Fixes a real MPSGraph ANE-compiler correctness bug for fused graphs
    // 14+ conv blocks deep on some chip generations (confirmed on H18g/M6),
    // and collapses the compiled graph from dozens of ANE/GPU region pairs
    // down to one ANE region matching Core ML's procedure count. Confirmed
    // harmless (same correctness/performance) on chips where the bug
    // doesn't reproduce (M4 Pro/t8140). setValue:forKey: throws if the
    // property doesn't exist on this MPSGraph version, so KVC itself is the
    // availability check -- just skip it on failure.
    {
      // TEMP (testing Core AI's own combination, found via its cached
      // manifest.plist): aneBondedCompileMode=0, aneCompilerSpatialSplitting=3,
      // enableANECHWRankPromotion=true -- overridable via env for comparison
      // against our current aneBondedCompileMode=1-only default.
      int bonded = std::getenv("ANE_BONDED_MODE") ? atoi(std::getenv("ANE_BONDED_MODE")) : 1;
      @try {
        [cd setValue:@(bonded) forKey:@"aneBondedCompileMode"];
      } @catch (NSException* e) {
      }
      if (const char* v = std::getenv("ANE_SPATIAL_SPLIT_LIVE")) {
        @try {
          [cd setValue:@(atoi(v)) forKey:@"aneCompilerSpatialSplitting"];
        } @catch (NSException* e) {
        }
      }
      if (std::getenv("ANE_CHW_RANK_PROMOTION")) {
        @try {
          [cd setValue:@YES forKey:@"enableANECHWRankPromotion"];
        } @catch (NSException* e) {
        }
      }
    }
    if (dump_dir) {
      // KVC debug knobs for the ANE_DUMP_HWX investigation; not needed for
      // normal execution.
      auto try_set_int = [&](const char* key, const char* env) {
        if (const char* v = std::getenv(env)) {
          @try {
            [cd setValue:@(atoi(v)) forKey:[NSString stringWithUTF8String:key]];
            fprintf(stderr, "[ane hwx dump] set %s = %s\n", key, v);
          } @catch (NSException* e) {
            fprintf(stderr, "[ane hwx dump] %s unavailable: %s\n", key,
                    e.reason.UTF8String);
          }
        }
      };
      try_set_int("aneCompilerSpatialSplitting", "ANE_SPATIAL_SPLIT");
      try_set_int("maximumNumberOfParallelEncodingRegions", "ANE_MAX_PARALLEL_REGIONS");
      try_set_int("minimumNumberOfOpsInParallelRegion", "ANE_MIN_OPS_PARALLEL_REGION");
      if (std::getenv("ANE_PRINT_PLACEMENT")) {
        @try {
          [cd setValue:@YES forKey:@"printANEPlacementAnalysis"];
        } @catch (NSException* e) {
          fprintf(stderr, "[ane hwx dump] printANEPlacementAnalysis unavailable: %s\n",
                  e.reason.UTF8String);
        }
      }
    }
    if (dump_dir && std::getenv("ANE_ENABLE_PACKAGE_RESOURCES") &&
        [cd respondsToSelector:@selector(setEnableCompileResourcesForPackage:)]) {
      [cd setEnableCompileResourcesForPackage:YES];
    }
    if (dump_dir && std::getenv("ANE_GENERATE_ANALYTICS")) {
      // mil_to_hwx.cc (ANECompiler.framework's own tool) sets these exact
      // keys in a flagsDict passed straight to the compiler, independent of
      // MIL -- probing whether MPSGraphCompilationDescriptor forwards the
      // same named flags via KVC.
      for (NSString* key in @[
             @"GenerateAnalyticsBuffer", @"GenerateStaticPerfAnalytics",
             @"DumpStatusDictionaryToFile"
           ]) {
        @try {
          [cd setValue:@YES forKey:key];
          fprintf(stderr, "[ane analytics] set %s = YES\n", key.UTF8String);
        } @catch (NSException* e) {
          fprintf(stderr, "[ane analytics] %s unavailable: %s\n",
                  key.UTF8String, e.reason.UTF8String);
        }
      }
    }

    MPSGraphExecutable* exe = [graph compileWithDevice:mps_device
                                                  feeds:feeds
                                          targetTensors:@[ target ]
                                       targetOperations:nil
                                  compilationDescriptor:cd];
    if (!exe) {
      throw std::runtime_error("[CompiledANE] MPSGraph compilation failed.");
    }

    if (dump_dir) {
      NSString* pkgPath = [[NSString stringWithUTF8String:dump_dir]
          stringByAppendingPathComponent:@"resnet50.mpsgraphpackage"];
      [[NSFileManager defaultManager] removeItemAtPath:pkgPath error:nil];
      MPSGraphExecutableSerializationDescriptor* sDesc =
          [MPSGraphExecutableSerializationDescriptor new];
      if ([sDesc respondsToSelector:@selector(setSerializeOriginalModule:)]) {
        sDesc.serializeOriginalModule = YES;
      }
      [exe serializeToMPSGraphPackageAtURL:[NSURL fileURLWithPath:pkgPath]
                                 descriptor:sDesc];
      fprintf(stderr, "[ane hwx dump] wrote %s\n", pkgPath.UTF8String);
    }

    // exe.feedTensors is the authoritative order for runWithMTLCommandQueue's
    // inputsArray; it's the compiler's own placeholder order, not insertion
    // order, so map it back to CompiledANE's leaf indices rather than
    // assuming it matches `feeds`.
    std::vector<size_t> placeholder_leaf_index;
    std::vector<bool> placeholder_needs_nchw;
    for (MPSGraphTensor* t in exe.feedTensors) {
      size_t i = placeholder_leaf_of.at(t);
      placeholder_leaf_index.push_back(i);
      placeholder_needs_nchw.push_back(
          !conv_weight_leaf_ids.count(leaves[i].id()) &&
          !bias_leaf_ids.count(leaves[i].id()) && inputs[i].ndim() == 4);
    }

    auto result = executable_cache().emplace(
        key,
        GraphExecutable{
            exe, std::move(placeholder_leaf_index),
            std::move(placeholder_needs_nchw), output_needs_nchw});
    return result.first->second;
  }
}

} // namespace

namespace metal {

bool ane_is_available() {
  static bool available = [] {
    @autoreleasepool {
      if (![[MPSGraphCompilationDescriptor new]
              respondsToSelector:@selector(setPreferredDevice:)]) {
        return false;
      }
      return MTLCreateSystemDefaultDevice() != nil;
    }
  }();
  return available;
}

} // namespace metal

void CompiledANE::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  std::vector<array> contiguous_inputs;
  contiguous_inputs.reserve(inputs.size());
  for (auto& in : inputs) {
    contiguous_inputs.push_back(ensure_row_contiguous(in, s));
  }

  out.set_data(allocator::malloc(out.nbytes()));

  auto& d = metal::device(s.device);
  id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)d.mtl_device();
  auto& ge = get_or_build_executable(device, this, contiguous_inputs);

  // MPSGraph runs on its own command queue, outside mlx's command buffer.
  // end_encoding() only inserts GPU-side fences between encoders on mlx's
  // own queue; it does not commit or wait, so it does not actually
  // guarantee prior writes to the leaf inputs are visible to a different
  // queue. Committing/waiting from inside eval_gpu would corrupt mlx's own
  // gpu::eval() dispatcher, which caches the command buffer pointer across
  // this call and violates Metal's "no completed handler after commit"
  // contract -- see mlx/compile.cpp's is_ane_op()/compile_fuse_ane() for why
  // fusing the whole eligible region (so there is no non-ANE op left for a
  // leaf to race against) is what actually closes this gap, rather than a
  // change to mlx/backend/metal/eval.cpp.
  metal::get_command_encoder(s).end_encoding(/* wait_in_flight = */ true);

  @autoreleasepool {
    if (std::getenv("ANE_DUMP_HWX")) {
      // enableCompileResourcesForPackage (set in get_or_build_executable)
      // changes how the compiled program gets loaded for real execution,
      // onto a path that needs an entitlement this process doesn't have
      // (com.apple.appleneuralengine error 57, "operation not permitted").
      // measure_ane_capacity's convert_fp8_to_hwx.m sidesteps this by never
      // running the program at all -- it only compiles and serializes, for
      // offline inspection. Match that here: the .mpsgraphpackage is
      // already on disk by this point, so stop before the run that would
      // hit the load error.
      fprintf(stderr, "[ane hwx dump] skipping actual run (dump-only mode)\n");
      std::exit(0);
    }

    id<MTLCommandQueue> queue = ane_command_queue(device);
    // encodeToCommandBuffer: on a self-managed MPSCommandBuffer, instead of
    // runWithMTLCommandQueue:, shaves off some of MPSGraphExecutable's own
    // per-call command-buffer setup when multiple executions share one
    // command buffer; measured ~17% lower fixed per-call dispatch overhead
    // on a trivial single-op ANE graph. No multi-call batching here (mlx's
    // eval_gpu is one call per graph), so this only removes
    // runWithMTLCommandQueue's own internal overhead over the equivalent
    // explicit encode+commit -- still waits synchronously before returning,
    // same as before.
    MPSCommandBuffer* command_buffer =
        [MPSCommandBuffer commandBufferFromCommandQueue:queue];

    // The 4D image-like placeholders are declared NCHW directly (see
    // get_or_build_executable); materialize the actual NCHW data for them
    // here, as a plain Metal compute dispatch on this same command buffer,
    // ahead of the MPSGraph encode below -- same-command-buffer ordering
    // guarantees this write is visible to the ANE read without touching
    // mlx's own command encoder/dispatcher (see the comment above
    // end_encoding()).
    std::vector<std::optional<array>> nchw_inputs(
        ge.placeholder_leaf_index.size());
    for (size_t k = 0; k < ge.placeholder_leaf_index.size(); k++) {
      if (!ge.placeholder_needs_nchw[k]) {
        continue;
      }
      auto& in = contiguous_inputs[ge.placeholder_leaf_index[k]];
      array& nchw = nchw_inputs[k].emplace(
          nhwc_to_nchw(in.shape()), in.dtype(), nullptr, std::vector<array>{});
      nchw.set_data(allocator::malloc(nchw.nbytes()));

      id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)(void*)in.buffer().ptr();
      id<MTLBuffer> out_buf =
          (__bridge id<MTLBuffer>)(void*)nchw.buffer().ptr();
      uint32_t N = in.shape(0), H = in.shape(1), W = in.shape(2),
               C = in.shape(3);
      id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
      [enc setComputePipelineState:nhwc_to_nchw_pipeline(device)];
      [enc setBuffer:in_buf offset:0 atIndex:0];
      [enc setBuffer:out_buf offset:0 atIndex:1];
      [enc setBytes:&H length:sizeof(H) atIndex:2];
      [enc setBytes:&W length:sizeof(W) atIndex:3];
      [enc setBytes:&C length:sizeof(C) atIndex:4];
      [enc dispatchThreads:MTLSizeMake(W, H, (NSUInteger)N * C)
          threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [enc endEncoding];
    }

    // Only the non-constant leaves are real MPSGraph placeholders; feed
    // them in exe.feedTensors' order (captured as placeholder_leaf_index at
    // build time), not CompiledANE's own leaf order.
    NSMutableArray<MPSGraphTensorData*>* inputs_array = [NSMutableArray
        arrayWithCapacity:ge.placeholder_leaf_index.size()];
    for (size_t k = 0; k < ge.placeholder_leaf_index.size(); k++) {
      const array& in = ge.placeholder_needs_nchw[k]
          ? *nchw_inputs[k]
          : contiguous_inputs[ge.placeholder_leaf_index[k]];
      id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)in.buffer().ptr();
      [inputs_array addObject:[[MPSGraphTensorData alloc]
                                   initWithMTLBuffer:buf
                                               shape:to_mps_shape(in.shape())
                                            dataType:MPSDataTypeFloat16]];
    }

    // If the graph's own output is NCHW (see get_or_build_executable), run
    // it into an NCHW scratch buffer and convert back to NHWC below, same
    // same-command-buffer-ordering reasoning as the input side above.
    std::optional<array> nchw_out;
    MPSShape* result_shape;
    id<MTLBuffer> result_buf;
    if (ge.output_needs_nchw) {
      array& scratch = nchw_out.emplace(
          nhwc_to_nchw(out.shape()), out.dtype(), nullptr, std::vector<array>{});
      scratch.set_data(allocator::malloc(scratch.nbytes()));
      result_shape = to_mps_shape(scratch.shape());
      result_buf = (__bridge id<MTLBuffer>)(void*)scratch.buffer().ptr();
    } else {
      result_shape = to_mps_shape(out.shape());
      result_buf = (__bridge id<MTLBuffer>)(void*)out.buffer().ptr();
    }
    MPSGraphTensorData* out_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:result_buf
                    shape:result_shape
                 dataType:MPSDataTypeFloat16];

    MPSGraphExecutableExecutionDescriptor* ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = NO;

    [ge.executable encodeToCommandBuffer:command_buffer
                              inputsArray:inputs_array
                             resultsArray:@[ out_data ]
                      executionDescriptor:ed];

    if (ge.output_needs_nchw) {
      id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)(void*)out.buffer().ptr();
      uint32_t N = out.shape(0), H = out.shape(1), W = out.shape(2),
               C = out.shape(3);
      id<MTLComputeCommandEncoder> enc =
          [command_buffer computeCommandEncoder];
      [enc setComputePipelineState:nchw_to_nhwc_pipeline(device)];
      [enc setBuffer:result_buf offset:0 atIndex:0];
      [enc setBuffer:out_buf offset:0 atIndex:1];
      [enc setBytes:&H length:sizeof(H) atIndex:2];
      [enc setBytes:&W length:sizeof(W) atIndex:3];
      [enc setBytes:&C length:sizeof(C) atIndex:4];
      [enc dispatchThreads:MTLSizeMake(W, H, (NSUInteger)N * C)
          threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [enc endEncoding];
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }
}

} // namespace mlx::core
