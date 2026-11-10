// Does MPSGraph actually emit an ANE-compiled (ANECIR) bundle for each
// distinct conv shape that appears in resnet50_real_bench.cpp, or does it
// silently fall back to GPU for some of them (e.g. 1x1 convs, which make up
// most of ResNet50's bottleneck convs)? Reuses the bundle-detection method
// from pmu_probe.mm: compile with preferredDevice=2, run once, then check
// whether MPSGraph's temp dir contains a .bc.mlir ANECIR bundle for that
// specific compilation.
//
// Build:
//   clang++ -fobjc-arc -std=c++20 -framework Foundation -framework Metal \
//     -framework MetalPerformanceShadersGraph \
//     experimental/ane/resnet50_shape_placement_check.mm \
//     -o /tmp/resnet50_shape_placement_check
// Run:
//   /tmp/resnet50_shape_placement_check

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

// Checks one conv shape: NHWC input [1,H,W,Ci], OHWI weight [Co,K,K,Ci].
static void check_shape(id<MTLDevice> device, NSString *name, int H, int W,
                        int Ci, int Co, int K, int stride, int pad) {
  @autoreleasepool {
    MPSGraphDevice *mps_device = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSGraph *graph = [MPSGraph new];
    NSArray *in_shape = @[ @1, @(H), @(W), @(Ci) ];
    NSArray *w_shape = @[ @(Co), @(K), @(K), @(Ci) ];

    MPSGraphTensor *input = [graph placeholderWithShape:in_shape
                                                dataType:MPSDataTypeFloat16
                                                    name:@"in"];
    MPSGraphTensor *weight = [graph placeholderWithShape:w_shape
                                                 dataType:MPSDataTypeFloat16
                                                     name:@"w"];
    MPSGraphTensor *weight_hwio = [graph transposeTensor:weight
                                              permutation:@[ @1, @2, @3, @0 ]
                                                     name:nil];
    MPSGraphConvolution2DOpDescriptor *desc = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:stride
                      strideInY:stride
                dilationRateInX:1
                dilationRateInY:1
                         groups:1
                    paddingLeft:pad
                   paddingRight:pad
                     paddingTop:pad
                  paddingBottom:pad
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                  weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
    MPSGraphTensor *conv = [graph convolution2DWithSourceTensor:input
                                                   weightsTensor:weight_hwio
                                                      descriptor:desc
                                                            name:nil];

    NSDictionary *feeds = @{
      input : [[MPSGraphShapedType alloc] initWithShape:in_shape
                                                dataType:MPSDataTypeFloat16],
      weight : [[MPSGraphShapedType alloc] initWithShape:w_shape
                                                 dataType:MPSDataTypeFloat16]
    };
    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel = MPSGraphOptimizationLevel1;
    cd.preferredDevice = 2;

    NSSet<NSString *> *before = existing_ane_tmp_dirs();

    MPSGraphExecutable *exe = [graph compileWithDevice:mps_device
                                                  feeds:feeds
                                          targetTensors:@[ conv ]
                                       targetOperations:nil
                                  compilationDescriptor:cd];
    if (!exe) {
      NSLog(@"[%@] compile FAILED", name);
      return;
    }

    id<MTLBuffer> in_buf = [device newBufferWithLength:(size_t)H * W * Ci * 2
                                                options:0];
    fill_fp16(in_buf.contents, in_buf.length);
    id<MTLBuffer> w_buf = [device newBufferWithLength:(size_t)Co * K * K * Ci * 2
                                               options:0];
    fill_fp16(w_buf.contents, w_buf.length);

    MPSGraphTensorData *in_data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:in_buf
                                                shape:in_shape
                                             dataType:MPSDataTypeFloat16];
    MPSGraphTensorData *w_data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:w_buf
                                                shape:w_shape
                                             dataType:MPSDataTypeFloat16];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;
    [exe runWithMTLCommandQueue:queue
                    inputsArray:@[ in_data, w_data ]
                   resultsArray:nil
            executionDescriptor:ed];

    BOOL on_ane = has_new_ane_bundle(before);
    NSLog(@"[%-28@] H=%-3d Ci=%-4d Co=%-4d K=%d stride=%d -> %s", name, H, Ci,
          Co, K, stride, on_ane ? "ANE bundle created" : "NO ane bundle (GPU fallback)");
  }
}

int main() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSLog(@"Device: %@", device.name);

  // Every distinct conv shape resnet50_real_bench.cpp's forward() actually
  // calls, across the stem and all 4 stages.
  check_shape(device, @"stem 7x7 s2", 224, 224, 3, 64, 7, 2, 3);

  check_shape(device, @"stage1 block0 1x1 reduce", 56, 56, 64, 64, 1, 1, 0);
  check_shape(device, @"stage1 block0 3x3", 56, 56, 64, 64, 3, 1, 1);
  check_shape(device, @"stage1 block0 1x1 expand", 56, 56, 64, 256, 1, 1, 0);
  check_shape(device, @"stage1 block0 downsample 1x1", 56, 56, 64, 256, 1, 1, 0);
  check_shape(device, @"stage1 block1+ 1x1 reduce", 56, 56, 256, 64, 1, 1, 0);

  check_shape(device, @"stage2 block0 1x1 reduce", 56, 56, 256, 128, 1, 1, 0);
  check_shape(device, @"stage2 block0 3x3 s2", 56, 56, 128, 128, 3, 2, 1);
  check_shape(device, @"stage2 block0 1x1 expand", 28, 28, 128, 512, 1, 1, 0);
  check_shape(device, @"stage2 block0 downsample 1x1 s2", 56, 56, 256, 512, 1, 2, 0);
  check_shape(device, @"stage2 block1+ 1x1 reduce", 28, 28, 512, 128, 1, 1, 0);
  check_shape(device, @"stage2 block1+ 3x3", 28, 28, 128, 128, 3, 1, 1);

  check_shape(device, @"stage3 block0 1x1 reduce", 28, 28, 512, 256, 1, 1, 0);
  check_shape(device, @"stage3 block0 3x3 s2", 28, 28, 256, 256, 3, 2, 1);
  check_shape(device, @"stage3 block1+ 1x1 reduce", 14, 14, 1024, 256, 1, 1, 0);
  check_shape(device, @"stage3 block1+ 3x3", 14, 14, 256, 256, 3, 1, 1);

  check_shape(device, @"stage4 block0 1x1 reduce", 14, 14, 1024, 512, 1, 1, 0);
  check_shape(device, @"stage4 block0 3x3 s2", 14, 14, 512, 512, 3, 2, 1);
  check_shape(device, @"stage4 block1+ 1x1 reduce", 7, 7, 2048, 512, 1, 1, 0);
  check_shape(device, @"stage4 block1+ 3x3", 7, 7, 512, 512, 3, 1, 1);

  return 0;
}
