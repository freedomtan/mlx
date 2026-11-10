// What's the minimal trailing op after a conv that gets MPSGraph to place
// it on ANE? Same shape throughout: NHWC/OHWI+transpose (CompiledANE's
// exact approach), H=W=56, Ci=Co=64, K=3, stride=1, pad=1, fp16.
//
// Build:
//   clang++ -fobjc-arc -std=c++20 -framework Foundation -framework Metal \
//     -framework MetalPerformanceShadersGraph \
//     experimental/ane/trailing_op_check.mm -o /tmp/trailing_op_check
// Run:
//   /tmp/trailing_op_check

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

typedef NS_ENUM(int, Trail) {
  TrailNone,
  TrailAddBias,
  TrailRelu,
  TrailAddBiasThenRelu,
  TrailSigmoid,
  TrailIdentity, // a no-op-ish op: add 0
  TrailSecondBareConv, // conv -> conv, neither has an activation
  TrailAddBiasThenMaxZero, // mlx's actual relu lowering: maximum(x, 0)
};

static void run_case(id<MTLDevice> device, NSString *name, Trail trail) {
  @autoreleasepool {
    const int B = 1, H = 56, W = 56, C = 64, K = 3;
    MPSGraphDevice *mps_device = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSGraph *graph = [MPSGraph new];

    NSArray *in_shape = @[ @(B), @(H), @(W), @(C) ];
    NSArray *w_shape = @[ @(C), @(K), @(K), @(C) ]; // OHWI, like mlx

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
                     dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                  weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];

    MPSGraphTensor *conv = [graph convolution2DWithSourceTensor:input
                                                   weightsTensor:weight_hwio
                                                      descriptor:desc
                                                            name:nil];

    NSMutableArray *feeds_keys = [NSMutableArray arrayWithObjects:input, weight, nil];
    NSMutableArray *feeds_shapes = [NSMutableArray arrayWithObjects:in_shape, w_shape, nil];
    MPSGraphTensor *target = conv;

    switch (trail) {
      case TrailNone:
        break;
      case TrailAddBias: {
        MPSGraphTensor *bias = [graph placeholderWithShape:@[ @(C) ]
                                                    dataType:MPSDataTypeFloat16
                                                        name:@"bias"];
        [feeds_keys addObject:bias];
        [feeds_shapes addObject:@[ @(C) ]];
        target = [graph additionWithPrimaryTensor:conv
                                   secondaryTensor:bias
                                              name:nil];
        break;
      }
      case TrailRelu:
        target = [graph reLUWithTensor:conv name:nil];
        break;
      case TrailAddBiasThenRelu: {
        MPSGraphTensor *bias = [graph placeholderWithShape:@[ @(C) ]
                                                    dataType:MPSDataTypeFloat16
                                                        name:@"bias"];
        [feeds_keys addObject:bias];
        [feeds_shapes addObject:@[ @(C) ]];
        MPSGraphTensor *biased = [graph additionWithPrimaryTensor:conv
                                                   secondaryTensor:bias
                                                              name:nil];
        target = [graph reLUWithTensor:biased name:nil];
        break;
      }
      case TrailSigmoid:
        target = [graph sigmoidWithTensor:conv name:nil];
        break;
      case TrailIdentity: {
        MPSGraphTensor *zero =
            [graph constantWithScalar:0.0 dataType:MPSDataTypeFloat16];
        target = [graph additionWithPrimaryTensor:conv
                                   secondaryTensor:zero
                                              name:nil];
        break;
      }
      case TrailAddBiasThenMaxZero: {
        MPSGraphTensor *bias = [graph placeholderWithShape:@[ @(C) ]
                                                    dataType:MPSDataTypeFloat16
                                                        name:@"bias"];
        [feeds_keys addObject:bias];
        [feeds_shapes addObject:@[ @(C) ]];
        MPSGraphTensor *biased = [graph additionWithPrimaryTensor:conv
                                                   secondaryTensor:bias
                                                              name:nil];
        MPSGraphTensor *zero =
            [graph constantWithScalar:0.0 dataType:MPSDataTypeFloat16];
        target = [graph maximumWithPrimaryTensor:biased
                                  secondaryTensor:zero
                                             name:nil];
        break;
      }
      case TrailSecondBareConv: {
        MPSGraphTensor *weight2 = [graph placeholderWithShape:w_shape
                                                      dataType:MPSDataTypeFloat16
                                                          name:@"w2"];
        [feeds_keys addObject:weight2];
        [feeds_shapes addObject:w_shape];
        MPSGraphTensor *weight2_hwio = [graph transposeTensor:weight2
                                                   permutation:@[ @1, @2, @3, @0 ]
                                                          name:nil];
        target = [graph convolution2DWithSourceTensor:conv
                                         weightsTensor:weight2_hwio
                                            descriptor:desc
                                                  name:nil];
        break;
      }
    }

    NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
    for (NSUInteger i = 0; i < feeds_keys.count; i++) {
      feeds[feeds_keys[i]] = [[MPSGraphShapedType alloc]
          initWithShape:feeds_shapes[i]
               dataType:MPSDataTypeFloat16];
    }

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel = MPSGraphOptimizationLevel1;
    cd.preferredDevice = 2;

    NSSet<NSString *> *before = existing_ane_tmp_dirs();

    MPSGraphExecutable *exe = [graph compileWithDevice:mps_device
                                                  feeds:feeds
                                          targetTensors:@[ target ]
                                       targetOperations:nil
                                  compilationDescriptor:cd];
    if (!exe) {
      NSLog(@"[%@] compile FAILED", name);
      return;
    }

    NSMutableArray *inputs_array = [NSMutableArray array];
    for (NSUInteger i = 0; i < feeds_keys.count; i++) {
      NSArray *shp = feeds_shapes[i];
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
    NSLog(@"[%-45@] -> %s", name,
          on_ane ? "ANE bundle created" : "NO ane bundle (GPU fallback)");
  }
}

int main() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSLog(@"Device: %@", device.name);

  run_case(device, @"bare conv (CompiledANE today)", TrailNone);
  run_case(device, @"conv + add(bias), no activation", TrailAddBias);
  run_case(device, @"conv + relu", TrailRelu);
  run_case(device, @"conv + add(bias) + relu", TrailAddBiasThenRelu);
  run_case(device, @"conv + add(bias) + maximum(x,0) [mlx's real relu lowering]",
            TrailAddBiasThenMaxZero);
  run_case(device, @"conv + sigmoid", TrailSigmoid);
  run_case(device, @"conv + add(0) [structurally a no-op]", TrailIdentity);
  run_case(device, @"conv -> conv (second conv also bare)",
            TrailSecondBareConv);

  return 0;
}
