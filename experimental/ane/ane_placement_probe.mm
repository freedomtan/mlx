// Standalone probe, not part of the mlx build.
//
// Question: does a static-shape conv2d+relu MPSGraph, compiled with the
// private preferredDevice=2 (ANE) hint, actually run faster/differently than
// the same graph compiled for GPU? This is step 1 of scoping an MLX ANE
// execution path: confirm the MPSGraph-to-ANE hint works for a conv shape
// representative of a static vision model before building any MLX-side
// plumbing.
//
// Build:
//   clang++ -fobjc-arc -std=c++17 -framework Foundation -framework Metal \
//     -framework MetalPerformanceShadersGraph \
//     experimental/ane/ane_placement_probe.mm -o /tmp/ane_placement_probe
// Run:
//   /tmp/ane_placement_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <time.h>

// Private API, matches freedomtan/measure_ane_capacity::measure_conv.m.
@interface MPSGraphCompilationDescriptor (Private)
@property(nonatomic, assign) unsigned long long preferredDevice;
@end

static double run_once(
    id<MTLDevice> device,
    bool use_ane,
    NSUInteger B,
    NSUInteger C,
    NSUInteger H,
    NSUInteger W,
    NSUInteger Co,
    NSUInteger K,
    int layers,
    int iterations) {
  @autoreleasepool {
    MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSGraph* graph = [MPSGraph new];

    NSArray* in_shape = @[ @(B), @(C), @(H), @(W) ];
    MPSGraphTensor* cur = [graph placeholderWithShape:in_shape
                                              dataType:MPSDataTypeFloat16
                                                  name:@"input"];

    NSUInteger ci = C;
    for (int i = 0; i < layers; i++) {
      NSArray* w_shape = @[ @(Co), @(ci), @(K), @(K) ];
      NSMutableData* w_data =
          [NSMutableData dataWithLength:Co * ci * K * K * sizeof(uint16_t)];
      uint16_t* w_ptr = (uint16_t*)w_data.mutableBytes;
      for (NSUInteger j = 0; j < Co * ci * K * K; j++) {
        w_ptr[j] = (j % 2) ? 0xA800 : 0x2800; // +/- 0.03125, avoid all-zero.
      }
      MPSGraphTensor* w = [graph constantWithData:w_data
                                             shape:w_shape
                                          dataType:MPSDataTypeFloat16];
      MPSGraphConvolution2DOpDescriptor* desc =
          [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1
                                                            strideInY:1
                                                      dilationRateInX:1
                                                      dilationRateInY:1
                                                               groups:1
                                                         paddingStyle:
                                                             MPSGraphPaddingStyleTF_SAME
                                                           dataLayout:
                                                               MPSGraphTensorNamedDataLayoutNCHW
                                                        weightsLayout:
                                                            MPSGraphTensorNamedDataLayoutOIHW];
      cur = [graph convolution2DWithSourceTensor:cur
                                    weightsTensor:w
                                       descriptor:desc
                                             name:nil];
      cur = [graph reLUWithTensor:cur name:nil];
      ci = Co;
    }

    NSDictionary* feeds = @{
      [graph placeholderTensors][0] :
          [[MPSGraphShapedType alloc] initWithShape:in_shape
                                            dataType:MPSDataTypeFloat16]
    };

    MPSGraphCompilationDescriptor* cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel = MPSGraphOptimizationLevel1;
    bool requested_ane = false;
    if (use_ane) {
      if ([cd respondsToSelector:@selector(setPreferredDevice:)]) {
        cd.preferredDevice = 2; // MPSGraphDeviceTypeANE (private).
        requested_ane = true;
      }
    }

    MPSGraphExecutable* exe = [graph compileWithDevice:mps_device
                                                  feeds:feeds
                                          targetTensors:@[ cur ]
                                       targetOperations:nil
                                  compilationDescriptor:cd];
    if (!exe) {
      NSLog(@"compile failed (use_ane=%d)", use_ane);
      return -1;
    }
    if (use_ane && !requested_ane) {
      NSLog(@"preferredDevice unavailable on this OS; ANE hint not applied");
    }

    id<MTLBuffer> in_buf =
        [device newBufferWithLength:B * C * H * W * sizeof(uint16_t)
                             options:0];
    uint16_t* in_ptr = (uint16_t*)in_buf.contents;
    for (NSUInteger i = 0; i < B * C * H * W; i++) {
      in_ptr[i] = (i % 2) ? 0xA800 : 0x2800;
    }
    MPSGraphTensorData* in_data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:in_buf
                                                shape:in_shape
                                             dataType:MPSDataTypeFloat16];

    NSArray* out_shape = @[ @(B), @(Co), @(H), @(W) ];
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:B * Co * H * W * sizeof(uint16_t)
                             options:0];
    MPSGraphTensorData* out_data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf
                                                shape:out_shape
                                             dataType:MPSDataTypeFloat16];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor* ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;

    // Warmup.
    [exe runWithMTLCommandQueue:queue
                    inputsArray:@[ in_data ]
                   resultsArray:@[ out_data ]
            executionDescriptor:ed];

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
      [exe runWithMTLCommandQueue:queue
                      inputsArray:@[ in_data ]
                     resultsArray:@[ out_data ]
              executionDescriptor:ed];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double seconds =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return seconds / iterations;
  }
}

int main() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSLog(@"Device: %@", device.name);

  // A conv block shape representative of a static vision model stage.
  NSUInteger B = 1, C = 64, H = 56, W = 56, Co = 64, K = 3;
  int layers = 8, iterations = 30;

  double gpu_s = run_once(device, false, B, C, H, W, Co, K, layers, iterations);
  double ane_s = run_once(device, true, B, C, H, W, Co, K, layers, iterations);

  NSLog(@"GPU path:  %.3f ms/iter", gpu_s * 1000.0);
  NSLog(@"ANE hint:  %.3f ms/iter", ane_s * 1000.0);
  if (gpu_s > 0 && ane_s > 0) {
    NSLog(@"ratio (gpu/ane): %.2fx", gpu_s / ane_s);
  }
  NSLog(
      @"Note: latency ratio is a proxy, not proof. Cross-check with "
      @"MLComputePlan on a Core ML conversion of the same graph, or ANE PMU "
      @"counters, before relying on this.");
  return 0;
}
