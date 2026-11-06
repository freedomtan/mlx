// Copyright © 2026 Apple Inc.
//
// eval_gpu for CompiledANE: runs a single 2D convolution through MPSGraph
// with the private preferredDevice=ANE hint, instead of the regular Metal
// Convolution kernel. See is_fusable_ane()/compile_fuse_ane() in
// mlx/compile.cpp for what graphs are eligible to reach this path.
//
// MLX convolutions are NHWC with OHWI weights; MPSGraph's
// MPSGraphTensorNamedDataLayout has no OHWI option, so the weight tensor is
// permuted to HWIO inside the graph before the convolution node.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <mutex>
#include <sstream>
#include <unordered_map>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/primitives.h"

// Private API, guarded at every call site by respondsToSelector:. If it
// disappears in a future OS, ane_is_available() reports false and
// compile_fuse_ane() never creates a CompiledANE node.
@interface MPSGraphCompilationDescriptor (ANEHint)
@property(nonatomic, assign) unsigned long long preferredDevice;
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

std::string cache_key(
    const Shape& in_shape,
    const Shape& w_shape,
    const std::vector<int>& strides,
    const std::vector<int>& padding) {
  std::ostringstream os;
  for (auto v : in_shape) {
    os << v << ',';
  }
  os << '_';
  for (auto v : w_shape) {
    os << v << ',';
  }
  os << '_';
  for (auto v : strides) {
    os << v << ',';
  }
  os << '_';
  for (auto v : padding) {
    os << v << ',';
  }
  return os.str();
}

struct GraphExecutable {
  MPSGraphExecutable* executable;
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

const GraphExecutable& get_or_build_executable(
    id<MTLDevice> device,
    const Shape& in_shape,
    const Shape& w_shape,
    const std::vector<int>& strides,
    const std::vector<int>& padding) {
  auto key = cache_key(in_shape, w_shape, strides, padding);
  std::lock_guard<std::mutex> lock(cache_mutex());
  auto it = executable_cache().find(key);
  if (it != executable_cache().end()) {
    return it->second;
  }

  @autoreleasepool {
    MPSGraph* graph = [MPSGraph new];
    MPSGraphTensor* input =
        [graph placeholderWithShape:to_mps_shape(in_shape)
                            dataType:MPSDataTypeFloat16
                                name:@"input"];
    MPSGraphTensor* weight =
        [graph placeholderWithShape:to_mps_shape(w_shape)
                            dataType:MPSDataTypeFloat16
                                name:@"weight"];
    // OHWI -> HWIO, the one weight layout MPSGraph can interpret directly
    // that matches MLX's weight element order.
    MPSGraphTensor* weight_hwio = [graph transposeTensor:weight
                                              permutation:@[ @1, @2, @3, @0 ]
                                                     name:@"weight_hwio"];

    MPSGraphConvolution2DOpDescriptor* desc =
        [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:static_cast<NSUInteger>(strides[1])
                          strideInY:static_cast<NSUInteger>(strides[0])
                    dilationRateInX:1
                    dilationRateInY:1
                             groups:1
                        paddingLeft:static_cast<NSUInteger>(padding[1])
                       paddingRight:static_cast<NSUInteger>(padding[1])
                         paddingTop:static_cast<NSUInteger>(padding[0])
                      paddingBottom:static_cast<NSUInteger>(padding[0])
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];

    MPSGraphTensor* conv =
        [graph convolution2DWithSourceTensor:input
                                weightsTensor:weight_hwio
                                   descriptor:desc
                                         name:nil];

    MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:device];
    NSDictionary* feeds = @{
      input : [[MPSGraphShapedType alloc] initWithShape:input.shape
                                                dataType:MPSDataTypeFloat16],
      weight : [[MPSGraphShapedType alloc] initWithShape:weight.shape
                                                 dataType:MPSDataTypeFloat16]
    };

    MPSGraphCompilationDescriptor* cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel = MPSGraphOptimizationLevel1;
    if ([cd respondsToSelector:@selector(setPreferredDevice:)]) {
      cd.preferredDevice = 2; // MPSGraphDeviceTypeANE (private).
    }

    MPSGraphExecutable* exe = [graph compileWithDevice:mps_device
                                                  feeds:feeds
                                          targetTensors:@[ conv ]
                                       targetOperations:nil
                                  compilationDescriptor:cd];
    if (!exe) {
      throw std::runtime_error("[CompiledANE] MPSGraph compilation failed.");
    }

    auto result = executable_cache().emplace(key, GraphExecutable{exe});
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
  auto in = ensure_row_contiguous(inputs[0], s);
  auto w = ensure_row_contiguous(inputs[1], s);

  out.set_data(allocator::malloc(out.nbytes()));

  // MPSGraph manages its own command buffer/queue outside mlx's encoder and
  // fence bookkeeping, so fully serialize at this boundary rather than
  // trying to interleave with it.
  metal::get_command_encoder(s).end_encoding(/* wait_in_flight = */ true);

  @autoreleasepool {
    auto& d = metal::device(s.device);
    id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)d.mtl_device();

    auto& ge = get_or_build_executable(
        device, in.shape(), w.shape(), kernel_strides_, padding_);

    id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)(void*)in.buffer().ptr();
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)(void*)w.buffer().ptr();
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)(void*)out.buffer().ptr();

    MPSGraphTensorData* in_data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:in_buf
                                                 shape:to_mps_shape(in.shape())
                                              dataType:MPSDataTypeFloat16];
    MPSGraphTensorData* w_data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:w_buf
                                                 shape:to_mps_shape(w.shape())
                                              dataType:MPSDataTypeFloat16];
    MPSGraphTensorData* out_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:out_buf
                    shape:to_mps_shape(out.shape())
                 dataType:MPSDataTypeFloat16];

    id<MTLCommandQueue> queue = ane_command_queue(device);
    MPSGraphExecutableExecutionDescriptor* ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;

    [ge.executable runWithMTLCommandQueue:queue
                               inputsArray:@[ in_data, w_data ]
                              resultsArray:@[ out_data ]
                       executionDescriptor:ed];
  }
}

} // namespace mlx::core
