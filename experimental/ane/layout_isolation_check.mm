// Isolates which change broke ANE placement relative to the earlier working
// probe (pmu_probe.mm, NCHW/OIHW, no in-graph transpose, 8 chained convs):
// NHWC vs NCHW layout, the in-graph weight transposeTensor: op, or chain
// length (1 conv vs 8 chained). Same shape throughout: H=W=56, Ci=Co=64,
// K=3, stride=1, pad=1, fp16.
//
// Build:
//   clang++ -fobjc-arc -std=c++20 -framework Foundation -framework Metal \
//     -framework MetalPerformanceShadersGraph \
//     experimental/ane/layout_isolation_check.mm \
//     -o /tmp/layout_isolation_check
// Run:
//   /tmp/layout_isolation_check

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MPSGraphCompilationDescriptor (ANEHint)
@property(nonatomic, assign) unsigned long long preferredDevice;
@end

static NSSet<NSString *> *existing_ane_tmp_dirs(void) {
  NSFileManager *fm = [NSFileManager defaultManager];
  NSString *base = [NSTemporaryDirectory()
      stringByAppendingPathComponent:@"com.apple.MetalPerformanceShadersGraph"];
  NSArray *dirs = [fm contentsOfDirectoryAtPath:base error:nil];
  return dirs ? [NSSet setWithArray:dirs] : [NSSet set];
}

static BOOL has_new_ane_bundle(NSSet<NSString *> *before) {
  NSFileManager *fm = [NSFileManager defaultManager];
  NSString *base = [NSTemporaryDirectory()
      stringByAppendingPathComponent:@"com.apple.MetalPerformanceShadersGraph"];
  NSArray *dirs = [fm contentsOfDirectoryAtPath:base error:nil];
  for (NSString *d in dirs) {
    if ([before containsObject:d]) continue;
    NSString *full = [base stringByAppendingPathComponent:d];
    for (NSString *sf in [fm contentsOfDirectoryAtPath:full error:nil]) {
      if ([sf hasSuffix:@".bc.mlir"]) return YES;
    }
  }
  return NO;
}

static void fill_fp16(void *buf, size_t bytes) {
  uint16_t *p = (uint16_t *)buf;
  for (size_t i = 0; i < bytes / 2; i++) p[i] = (i % 2) ? 0xA800 : 0x2800;
}

typedef NS_ENUM(int, Layout) { LayoutNCHW_OIHW, LayoutNHWC_HWIO_direct,
                                LayoutNHWC_OHWI_transpose };

static void run_case(id<MTLDevice> device, NSString *name, Layout layout,
                      int layers, BOOL apply_relu) {
  @autoreleasepool {
    const int B = 1, H = 56, W = 56, C = 64, K = 3;
    MPSGraphDevice *mps_device = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSGraph *graph = [MPSGraph new];

    MPSGraphTensorNamedDataLayout data_layout;
    NSArray *in_shape;
    if (layout == LayoutNCHW_OIHW) {
      data_layout = MPSGraphTensorNamedDataLayoutNCHW;
      in_shape = @[ @(B), @(C), @(H), @(W) ];
    } else {
      data_layout = MPSGraphTensorNamedDataLayoutNHWC;
      in_shape = @[ @(B), @(H), @(W), @(C) ];
    }

    MPSGraphTensor *cur = [graph placeholderWithShape:in_shape
                                              dataType:MPSDataTypeFloat16
                                                  name:@"in"];
    NSMutableArray *feed_tensors = [NSMutableArray array];
    NSMutableArray *feed_shapes = [NSMutableArray array];
    [feed_tensors addObject:cur];
    [feed_shapes addObject:in_shape];

    for (int i = 0; i < layers; i++) {
      NSArray *w_shape_as_stored; // the shape of the placeholder we feed
      MPSGraphTensorNamedDataLayout weights_layout;
      MPSGraphTensor *weight_for_conv;

      MPSGraphTensor *w_placeholder;
      if (layout == LayoutNCHW_OIHW) {
        w_shape_as_stored = @[ @(C), @(C), @(K), @(K) ]; // OIHW
        w_placeholder = [graph placeholderWithShape:w_shape_as_stored
                                            dataType:MPSDataTypeFloat16
                                                name:nil];
        weights_layout = MPSGraphTensorNamedDataLayoutOIHW;
        weight_for_conv = w_placeholder;
      } else if (layout == LayoutNHWC_HWIO_direct) {
        w_shape_as_stored = @[ @(K), @(K), @(C), @(C) ]; // already HWIO
        w_placeholder = [graph placeholderWithShape:w_shape_as_stored
                                            dataType:MPSDataTypeFloat16
                                                name:nil];
        weights_layout = MPSGraphTensorNamedDataLayoutHWIO;
        weight_for_conv = w_placeholder;
      } else { // LayoutNHWC_OHWI_transpose
        w_shape_as_stored = @[ @(C), @(K), @(K), @(C) ]; // OHWI, like mlx
        w_placeholder = [graph placeholderWithShape:w_shape_as_stored
                                            dataType:MPSDataTypeFloat16
                                                name:nil];
        weights_layout = MPSGraphTensorNamedDataLayoutHWIO;
        weight_for_conv = [graph transposeTensor:w_placeholder
                                      permutation:@[ @1, @2, @3, @0 ]
                                             name:nil];
      }
      [feed_tensors addObject:w_placeholder];
      [feed_shapes addObject:w_shape_as_stored];

      MPSGraphConvolution2DOpDescriptor *desc = [MPSGraphConvolution2DOpDescriptor
          descriptorWithStrideInX:1
                        strideInY:1
                  dilationRateInX:1
                  dilationRateInY:1
                           groups:1
                      paddingLeft:1
                     paddingRight:1
                       paddingTop:1
                    paddingBottom:1
                     paddingStyle:MPSGraphPaddingStyleExplicit
                       dataLayout:data_layout
                    weightsLayout:weights_layout];
      cur = [graph convolution2DWithSourceTensor:cur
                                    weightsTensor:weight_for_conv
                                       descriptor:desc
                                             name:nil];
      if (apply_relu) {
        cur = [graph reLUWithTensor:cur name:nil];
      }
    }

    NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
    for (NSUInteger i = 0; i < feed_tensors.count; i++) {
      feeds[feed_tensors[i]] = [[MPSGraphShapedType alloc]
          initWithShape:feed_shapes[i]
               dataType:MPSDataTypeFloat16];
    }

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel = MPSGraphOptimizationLevel1;
    cd.preferredDevice = 2;

    NSSet<NSString *> *before = existing_ane_tmp_dirs();

    MPSGraphExecutable *exe = [graph compileWithDevice:mps_device
                                                  feeds:feeds
                                          targetTensors:@[ cur ]
                                       targetOperations:nil
                                  compilationDescriptor:cd];
    if (!exe) {
      NSLog(@"[%@] compile FAILED", name);
      return;
    }

    NSMutableArray *inputs_array = [NSMutableArray array];
    for (NSUInteger i = 0; i < feed_tensors.count; i++) {
      NSArray *shp = feed_shapes[i];
      size_t n = 1;
      for (NSNumber *d in shp) n *= d.unsignedLongValue;
      id<MTLBuffer> buf = [device newBufferWithLength:n * 2 options:0];
      fill_fp16(buf.contents, buf.length);
      [inputs_array addObject:[[MPSGraphTensorData alloc]
                                   initWithMTLBuffer:buf
                                               shape:shp
                                            dataType:MPSDataTypeFloat16]];
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;
    [exe runWithMTLCommandQueue:queue
                    inputsArray:inputs_array
                   resultsArray:nil
            executionDescriptor:ed];

    BOOL on_ane = has_new_ane_bundle(before);
    NSLog(@"[%-40@] layers=%d -> %s", name, layers,
          on_ane ? "ANE bundle created" : "NO ane bundle (GPU fallback)");
  }
}

int main() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSLog(@"Device: %@", device.name);

  run_case(device, @"NCHW/OIHW, 8 layers, +relu (matches pmu_probe.mm)",
            LayoutNCHW_OIHW, 8, YES);
  run_case(device, @"NCHW/OIHW, 1 layer, +relu", LayoutNCHW_OIHW, 1, YES);
  run_case(device, @"NCHW/OIHW, 1 layer, NO relu (bare conv)",
            LayoutNCHW_OIHW, 1, NO);
  run_case(device, @"NHWC/OHWI+transpose, 1 layer, +relu",
            LayoutNHWC_OHWI_transpose, 1, YES);
  run_case(device, @"NHWC/OHWI+transpose, 1 layer, NO relu (bare conv, "
                    @"= CompiledANE's exact shape)",
            LayoutNHWC_OHWI_transpose, 1, NO);

  return 0;
}
