// Ground-truth check: does MPSGraphCompilationDescriptor.preferredDevice=2
// actually execute our conv graph on the ANE, measured via physical silicon
// PMU registers (not inferred from latency)?
//
// Adapted from freedomtan/measure_ane_capacity::measure_ane_pmu.m (BSD
// 3-Clause, Copyright (c) 2025 Koan-Sin Tan), trimmed to one FP16 conv+relu
// chain matching the shape probed in ane_placement_probe.mm. That tool's
// README states this path needs no entitlement, boot-arg, or root: it reads
// perf counters via _ANEClient talking to the (already root-privileged)
// aned daemon.
//
// Flow: compile the MPSGraph with the private preferredDevice=2 hint, run it
// once via the normal MTLCommandQueue path (this makes MPSGraph emit a
// compiled ANECIR bundle to a temp dir), locate that temp dir, then load the
// *same compiled bundle* directly into _ANEClient and evaluate it there so
// the perf-counter IOSurface is attached to the request. The PMU numbers
// read back are for this exact compiled program, not a proxy.
//
// Build:
//   clang++ -fobjc-arc -std=c++17 -framework Foundation -framework Metal \
//     -framework MetalPerformanceShadersGraph \
//     -F/System/Library/PrivateFrameworks -framework AppleNeuralEngine \
//     -framework IOSurface -framework IOKit -framework Security \
//     experimental/ane/pmu_probe.mm -o /tmp/pmu_probe
// Run:
//   /tmp/pmu_probe

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <time.h>

@interface MPSGraphCompilationDescriptor (Private)
@property(nonatomic, assign) unsigned long long preferredDevice;
@end

// Private AppleNeuralEngine declarations, mirrored from measure_ane_pmu.m.
@interface _ANEIOSurfaceObject : NSObject
+ (instancetype)objectWithIOSurface:(IOSurfaceRef)surface;
@end

@interface _ANEPerformanceStatsIOSurface : NSObject
+ (instancetype)objectWithIOSurface:(_ANEIOSurfaceObject *)ioSurface
                            statType:(int)statType;
@end

@interface _ANEPerformanceStats : NSObject
@property(nonatomic, readonly) NSData *perfCounterData;
- (NSString *)stringForPerfCounter:(int)index;
@end

@interface _ANERequest : NSObject
+ (instancetype)requestWithInputs:(NSArray *)inputs
                      inputIndices:(NSArray *)inputIndices
                           outputs:(NSArray *)outputs
                     outputIndices:(NSArray *)outputIndices
                         perfStats:(NSArray *)perfStats
                    procedureIndex:(NSNumber *)procedureIndex;
@property(nonatomic, readonly) _ANEPerformanceStats *perfStats;
@end

@interface _ANEModel : NSObject
+ (instancetype)modelAtURL:(NSURL *)url
                        key:(NSString *)key
               mpsConstants:(NSString *)constants;
@end

@interface _ANEClient : NSObject
+ (instancetype)sharedConnection;
- (BOOL)loadModel:(_ANEModel *)model
          options:(NSDictionary *)options
              qos:(unsigned int)qos
            error:(NSError **)error;
- (BOOL)unloadModel:(_ANEModel *)model
            options:(NSDictionary *)options
                qos:(unsigned int)qos
              error:(NSError **)error;
- (BOOL)evaluateWithModel:(_ANEModel *)model
                   options:(NSDictionary *)options
                   request:(_ANERequest *)request
                       qos:(unsigned int)qos
                     error:(NSError **)error;
@end

#define kANEFModelTypeKey @"kANEFModelType"
#define kANEFModelANECIRValue @"kANEFModelANECIR"
#define kANEFPerformanceStatsMaskKey @"kANEFPerformanceStatsMask"
#define kANEFNetPlistFilenameKey @"kANEFNetPlistFilenameKey"
#define kANEFCompilerOptionsFilenameKey @"kANEFCompilerOptionsFilenameKey"
#define kANEFRetainModelsWithoutSourceURLKey \
  @"kANEFRetainModelsWithoutSourceURLKey"
#define kANEFTargetArchitectureKey @"kANEFTargetArchitectureKey"

static void fill_nonzero_fp16(void *buffer, size_t byte_count, uint64_t seed) {
  uint64_t state = seed;
  uint16_t *p = (uint16_t *)buffer;
  size_t count = byte_count / sizeof(uint16_t);
  for (size_t i = 0; i < count; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    p[i] = (state & 1) ? 0xA800 : 0x2800; // -0.03125, +0.03125
  }
}

static NSString *query_system_ane_architecture(void) {
  CFMutableDictionaryRef matching = IOServiceMatching("H11ANEIn");
  io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, matching);
  if (!service) {
    matching = IOServiceMatching("AppleH16ANEInterface");
    service = IOServiceGetMatchingService(kIOMainPortDefault, matching);
  }
  NSString *arch = @"h16g";
  if (service) {
    CFMutableDictionaryRef props = NULL;
    if (IORegistryEntryCreateCFProperties(
            service, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS &&
        props) {
      NSDictionary *dict = (__bridge NSDictionary *)props;
      NSDictionary *dev_props = dict[@"DeviceProperties"];
      if (dev_props && dev_props[@"ANEDevicePropertyTypeANEArchitectureTypeStr"]) {
        arch = [dev_props[@"ANEDevicePropertyTypeANEArchitectureTypeStr"] copy];
      }
      CFRelease(props);
    }
    IOObjectRelease(service);
  }
  return arch;
}

static IOSurfaceRef create_io_surface(size_t alloc_size) {
  NSDictionary *props = @{
    (id)kIOSurfaceWidth : @(alloc_size),
    (id)kIOSurfaceHeight : @1,
    (id)kIOSurfaceBytesPerElement : @1,
    (id)kIOSurfaceBytesPerRow : @(alloc_size),
    (id)kIOSurfaceAllocSize : @(alloc_size)
  };
  return IOSurfaceCreate((CFDictionaryRef)props);
}

static NSSet<NSString *> *existing_ane_tmp_dirs(void) {
  NSFileManager *fm = [NSFileManager defaultManager];
  NSString *base = [NSTemporaryDirectory()
      stringByAppendingPathComponent:@"com.apple.MetalPerformanceShadersGraph"];
  NSArray *dirs = [fm contentsOfDirectoryAtPath:base error:nil];
  return dirs ? [NSSet setWithArray:dirs] : [NSSet set];
}

static NSString *find_new_ane_tmp_dir(NSSet<NSString *> *before) {
  NSFileManager *fm = [NSFileManager defaultManager];
  NSString *base = [NSTemporaryDirectory()
      stringByAppendingPathComponent:@"com.apple.MetalPerformanceShadersGraph"];
  NSArray *dirs = [fm contentsOfDirectoryAtPath:base error:nil];
  for (NSString *d in dirs) {
    if ([before containsObject:d]) continue;
    NSString *full = [base stringByAppendingPathComponent:d];
    for (NSString *sf in [fm contentsOfDirectoryAtPath:full error:nil]) {
      if ([sf hasSuffix:@".bc.mlir"]) return full;
    }
  }
  NSString *best = nil;
  NSDate *best_date = [NSDate distantPast];
  for (NSString *d in dirs) {
    NSString *full = [base stringByAppendingPathComponent:d];
    NSDate *mod = [fm attributesOfItemAtPath:full error:nil][NSFileModificationDate];
    if (!mod || [mod compare:best_date] != NSOrderedDescending) continue;
    for (NSString *sf in [fm contentsOfDirectoryAtPath:full error:nil]) {
      if ([sf hasSuffix:@".bc.mlir"]) {
        best_date = mod;
        best = full;
        break;
      }
    }
  }
  return best;
}

int main() {
  @autoreleasepool {
    NSBundle *ane_bundle = [NSBundle
        bundleWithPath:@"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"];
    [ane_bundle load];

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSLog(@"Device: %@, ANE arch: %@", device.name, query_system_ane_architecture());

    // Same shape as ane_placement_probe.mm's vertical slice.
    NSUInteger B = 1, C = 64, H = 56, W = 56, Co = 64, K = 3;
    int layers = 8, iterations = 30;
    double total_macs = (double)B * H * W * C * Co * K * K * layers;

    MPSGraphDevice *mps_device = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSGraph *graph = [MPSGraph new];
    NSArray *in_shape = @[ @(B), @(C), @(H), @(W) ];
    MPSGraphTensor *input = [graph placeholderWithShape:in_shape
                                                dataType:MPSDataTypeFloat16
                                                    name:@"input"];
    MPSGraphTensor *cur = input;
    NSUInteger ci = C;
    for (int i = 0; i < layers; i++) {
      NSArray *w_shape = @[ @(Co), @(ci), @(K), @(K) ];
      NSMutableData *w_data =
          [NSMutableData dataWithLength:Co * ci * K * K * sizeof(uint16_t)];
      fill_nonzero_fp16(w_data.mutableBytes, w_data.length, 0x5EED5EED5EED5EEDULL + i);
      MPSGraphTensor *w = [graph constantWithData:w_data
                                             shape:w_shape
                                          dataType:MPSDataTypeFloat16];
      MPSGraphConvolution2DOpDescriptor *desc = [MPSGraphConvolution2DOpDescriptor
          descriptorWithStrideInX:1
                        strideInY:1
                  dilationRateInX:1
                  dilationRateInY:1
                           groups:1
                     paddingStyle:MPSGraphPaddingStyleTF_SAME
                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                    weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
      cur = [graph convolution2DWithSourceTensor:cur
                                    weightsTensor:w
                                       descriptor:desc
                                             name:nil];
      cur = [graph reLUWithTensor:cur name:nil];
      ci = Co;
    }

    NSDictionary *feeds = @{
      input : [[MPSGraphShapedType alloc] initWithShape:in_shape
                                                dataType:MPSDataTypeFloat16]
    };

    MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
    cd.optimizationLevel = MPSGraphOptimizationLevel1;
    cd.preferredDevice = 2; // MPSGraphDeviceTypeANE (private).

    NSSet<NSString *> *before_dirs = existing_ane_tmp_dirs();

    MPSGraphExecutable *exe = [graph compileWithDevice:mps_device
                                                  feeds:feeds
                                          targetTensors:@[ cur ]
                                       targetOperations:nil
                                  compilationDescriptor:cd];
    if (!exe) {
      NSLog(@"compile failed");
      return 1;
    }

    id<MTLBuffer> mtl_in_buf =
        [device newBufferWithLength:B * C * H * W * sizeof(uint16_t) options:0];
    fill_nonzero_fp16(mtl_in_buf.contents, mtl_in_buf.length, 0x9E3779B97F4A7C15ULL);
    MPSGraphTensorData *mtl_in_data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:mtl_in_buf
                                                shape:in_shape
                                             dataType:MPSDataTypeFloat16];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MPSGraphExecutableExecutionDescriptor *ed =
        [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;

    // Triggers ANE region compilation to a temp dir, which is what we go
    // read directly below.
    [exe runWithMTLCommandQueue:queue
                    inputsArray:@[ mtl_in_data ]
                   resultsArray:nil
            executionDescriptor:ed];

    NSString *bundle_dir = find_new_ane_tmp_dir(before_dirs);
    if (!bundle_dir) {
      NSLog(@"Could not locate compiled ANECIR temp dir. preferredDevice "
            @"may not have taken effect, or MPSGraph placed this graph on "
            @"GPU instead of ANE.");
      return 1;
    }
    NSLog(@"ANECIR bundle: %@", bundle_dir);

    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *bc_mlir = nil, *region_key = nil;
    for (NSString *f in [fm contentsOfDirectoryAtPath:bundle_dir error:nil]) {
      if ([f hasSuffix:@".bc.mlir"]) {
        bc_mlir = f;
        region_key = [f substringToIndex:(f.length - @".bc.mlir".length)];
        break;
      }
    }
    if (!region_key) {
      NSLog(@"No .bc.mlir in bundle dir, cannot load into _ANEClient.");
      return 1;
    }

    NSString *target_arch = query_system_ane_architecture();
    _ANEClient *client = [_ANEClient sharedConnection];
    _ANEModel *model = [_ANEModel modelAtURL:[NSURL fileURLWithPath:bundle_dir]
                                          key:region_key
                                 mpsConstants:@"constants"];

    NSDictionary *load_opts = @{
      kANEFModelTypeKey : kANEFModelANECIRValue,
      kANEFCompilerOptionsFilenameKey :
          [NSString stringWithFormat:@"compiler_options_%@.plist", region_key],
      kANEFNetPlistFilenameKey : bc_mlir,
      kANEFTargetArchitectureKey : target_arch,
      kANEFPerformanceStatsMaskKey : @(15),
      kANEFRetainModelsWithoutSourceURLKey : @1
    };

    NSError *load_err = nil;
    if (![client loadModel:model options:load_opts qos:25 error:&load_err]) {
      NSLog(@"Failed to load model into _ANEClient: %@",
            load_err.localizedDescription ?: @"unknown");
      return 1;
    }

    size_t in_bytes = B * H * W * C * sizeof(uint16_t);
    size_t out_bytes = B * H * W * Co * sizeof(uint16_t);

    IOSurfaceRef in_surf = create_io_surface(in_bytes);
    IOSurfaceLock(in_surf, 0, NULL);
    fill_nonzero_fp16(IOSurfaceGetBaseAddress(in_surf), in_bytes, 0x9E3779B97F4A7C15ULL);
    IOSurfaceUnlock(in_surf, 0, NULL);
    IOSurfaceRef out_surf = create_io_surface(out_bytes);
    IOSurfaceRef pmu_surf = create_io_surface(4096);

    _ANEIOSurfaceObject *in_obj = [_ANEIOSurfaceObject objectWithIOSurface:in_surf];
    _ANEIOSurfaceObject *out_obj = [_ANEIOSurfaceObject objectWithIOSurface:out_surf];
    _ANEIOSurfaceObject *pmu_obj = [_ANEIOSurfaceObject objectWithIOSurface:pmu_surf];
    _ANEPerformanceStatsIOSurface *pmu_stats_surf =
        [_ANEPerformanceStatsIOSurface objectWithIOSurface:pmu_obj statType:2];

    _ANERequest *req = [_ANERequest requestWithInputs:@[ in_obj ]
                                          inputIndices:@[ @0 ]
                                               outputs:@[ out_obj ]
                                         outputIndices:@[ @0 ]
                                             perfStats:@[ pmu_stats_surf ]
                                        procedureIndex:@0];

    NSDictionary *eval_opts =
        @{kANEFPerformanceStatsMaskKey : @(15), @"enableProfiling" : @YES};

    NSError *eval_err = nil;
    [client evaluateWithModel:model options:eval_opts request:req qos:25 error:&eval_err];
    uint64_t init_regs[29] = {0};
    NSData *d0 = req.perfStats.perfCounterData;
    if (d0 && d0.length >= sizeof(init_regs)) memcpy(init_regs, d0.bytes, sizeof(init_regs));

    uint64_t total_ns = 0;
    for (int i = 0; i < iterations; i++) {
      uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
      [client evaluateWithModel:model options:eval_opts request:req qos:25 error:&eval_err];
      total_ns += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
    }

    uint64_t final_regs[29] = {0};
    NSData *d1 = req.perfStats.perfCounterData;
    if (d1 && d1.length >= sizeof(final_regs)) memcpy(final_regs, d1.bytes, sizeof(final_regs));

    double ms_per_iter = (double)total_ns / (iterations * 1e6);
    uint64_t compute_cycles = (final_regs[13] - init_regs[13]) / iterations;
    uint64_t l2pe_cycles = (final_regs[21] - init_regs[21]) / iterations;
    uint64_t input_stall = (final_regs[14] - init_regs[14]) / iterations;
    uint64_t output_stall = (final_regs[15] - init_regs[15]) / iterations;
    uint64_t dma_bytes = (final_regs[17] - init_regs[17]) / iterations;
    uint64_t nominal_cycles = (final_regs[10] - init_regs[10]) / iterations;

    NSLog(@"_ANEClient driver latency: %.3f ms/iter", ms_per_iter);
    for (int i = 0; i < 29; i++) {
      uint64_t d = (final_regs[i] >= init_regs[i])
          ? (final_regs[i] - init_regs[i]) / iterations
          : 0;
      if (d != 0) {
        NSLog(@"  raw[%02d] delta = %llu/iter (init=%llu final=%llu)", i, d,
              init_regs[i], final_regs[i]);
      }
    }
    NSLog(@"kANE_NE_COMPUTE_CYCLES   : %llu cycles/iter", compute_cycles);
    NSLog(@"kANE_L2PE_COMPUTE_CYCLES : %llu cycles/iter", l2pe_cycles);
    NSLog(@"kANE_NE_INPUT_STALL      : %llu cycles/iter", input_stall);
    NSLog(@"kANE_NE_OUTPUT_STALL     : %llu cycles/iter", output_stall);
    NSLog(@"kANE_DMA_READWRITE_BYTES : %llu bytes/iter", dma_bytes);
    NSLog(@"kANE_NE_NOMINAL_CYCLES   : %llu cycles/iter", nominal_cycles);
    if (nominal_cycles > 0) {
      NSLog(@"MACs / nominal cycle (chip, 16 cores): %.1f",
            (total_macs / (double)nominal_cycles));
    }
    if (compute_cycles == 0 && nominal_cycles == 0) {
      NSLog(@"All-zero PMU deltas: either the registers aren't incrementing "
            @"on this architecture/OS, or evaluateWithModel: isn't reaching "
            @"the engine. Treat placement as UNCONFIRMED, not confirmed.");
    } else {
      NSLog(@"Non-zero ANE hardware cycles observed: this graph is CONFIRMED "
            @"to execute on physical ANE silicon, not just hinted at.");
    }

    [client unloadModel:model options:@{} qos:25 error:nil];
    CFRelease(in_surf);
    CFRelease(out_surf);
    CFRelease(pmu_surf);
  }
  return 0;
}
