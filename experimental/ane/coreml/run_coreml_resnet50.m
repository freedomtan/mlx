// Load and run the converted ResNet50 Core ML model (CPU_AND_NE), sustained,
// so procedure count / ANE engagement can be measured the same way as the
// raw-MPSGraph CompiledANE path: ANECompilerService log capture + rootless
// powermetrics sampling during the run.
//
// Build:
//   clang -fobjc-arc -O2 -framework Foundation -framework CoreML \
//     run_coreml_resnet50.m -o /tmp/run_coreml_resnet50
// Run:
//   /tmp/run_coreml_resnet50 <path-to-resnet50.mlpackage-or-mlmodelc> <seconds>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <time.h>

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) {
      fprintf(stderr, "usage: %s <model path> [seconds]\n", argv[0]);
      return 1;
    }
    NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
    double seconds = argc > 2 ? atof(argv[2]) : 5.0;
    NSURL *modelURL = [NSURL fileURLWithPath:modelPath];

    NSURL *compiledURL = modelURL;
    if ([[modelPath pathExtension] isEqualToString:@"mlpackage"]) {
      NSError *compileErr = nil;
      fprintf(stderr, "[run_coreml] compiling %s...\n", argv[1]);
      compiledURL = [MLModel compileModelAtURL:modelURL error:&compileErr];
      if (compileErr) {
        fprintf(stderr, "compile error: %s\n",
                compileErr.localizedDescription.UTF8String);
        return 1;
      }
    }

    MLModelConfiguration *cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

    NSError *loadErr = nil;
    fprintf(stderr, "[run_coreml] loading %s...\n", compiledURL.path.UTF8String);
    double tLoad0 = now_s();
    MLModel *model = [MLModel modelWithContentsOfURL:compiledURL
                                        configuration:cfg
                                                error:&loadErr];
    double tLoad1 = now_s();
    if (loadErr || !model) {
      fprintf(stderr, "load error: %s\n", loadErr.localizedDescription.UTF8String);
      return 1;
    }
    fprintf(stderr, "[run_coreml] loaded in %.1fms\n", (tLoad1 - tLoad0) * 1000.0);

    MLFeatureDescription *inDesc =
        model.modelDescription.inputDescriptionsByName[@"image"];
    NSArray<NSNumber *> *shape = inDesc.multiArrayConstraint.shape;
    fprintf(stderr, "[run_coreml] input shape: %s\n",
            shape.description.UTF8String);

    NSError *maErr = nil;
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                     dataType:MLMultiArrayDataTypeFloat32
                                                        error:&maErr];
    float *p = (float *)arr.dataPointer;
    NSUInteger count = arr.count;
    for (NSUInteger i = 0; i < count; i++) {
      p[i] = (float)((i % 997) - 498) / 498.0f;
    }

    MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"image" : arr}
                      error:&maErr];

    MLPredictionOptions *opts = [MLPredictionOptions new];

    NSError *predErr = nil;
    id<MLFeatureProvider> out = [model predictionFromFeatures:input
                                                        options:opts
                                                          error:&predErr];
    if (predErr || !out) {
      fprintf(stderr, "predict error: %s\n",
              predErr.localizedDescription.UTF8String);
      return 1;
    }
    fprintf(stderr, "[run_coreml] warm-up prediction ok, output names: %s\n",
            out.featureNames.description.UTF8String);

    fprintf(stderr, "[run_coreml] sustained starting now, running for %.1fs\n",
            seconds);
    double t0 = now_s();
    long iters = 0;
    while (now_s() - t0 < seconds) {
      id<MLFeatureProvider> r = [model predictionFromFeatures:input
                                                        options:opts
                                                          error:&predErr];
      if (!r) {
        fprintf(stderr, "predict error at iter %ld: %s\n", iters,
                predErr.localizedDescription.UTF8String);
        return 1;
      }
      iters++;
    }
    double dt = now_s() - t0;
    fprintf(stderr, "[run_coreml] sustained done: %ld iters in %.1fms (%.3fms/iter)\n",
            iters, dt * 1000.0, dt * 1000.0 / iters);
  }
  return 0;
}
