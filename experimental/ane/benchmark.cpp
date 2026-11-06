// Standalone benchmark, not part of the mlx build: is routing a conv2d
// through CompiledANE (MPSGraph + preferredDevice=ANE) actually faster than
// mlx's regular Metal Convolution kernel, once the full per-call overhead
// (end_encoding + wait, building/looking up the MPSGraphExecutable, a fresh
// MTLCommandQueue per call) is included? The earlier standalone probes only
// measured MPSGraph in isolation; this measures it through the real
// CompiledANE::eval_gpu path exercised by correctness_test.cpp.
//
// Build:
//   clang++ -std=c++20 -O2 -I. -Ibuild_ane/_deps/metal_cpp-src \
//     -Ibuild_ane/_deps/fmt-src/include \
//     experimental/ane/benchmark.cpp build_ane/libmlx.a \
//     -framework Foundation -framework Metal -framework QuartzCore \
//     -framework IOKit -framework MetalPerformanceShadersGraph \
//     -framework Accelerate -framework Security -o /tmp/ane_benchmark
// Run:
//   MLX_METAL_PATH=build_ane/mlx/backend/metal/kernels /tmp/ane_benchmark

#include <chrono>
#include <cstdio>
#include <functional>
#include <random>

#include "mlx/mlx.h"

using namespace mlx::core;

double time_ms(const std::function<array()>& step, int warmup, int iters) {
  array out = step();
  eval(out);
  for (int i = 0; i < warmup; i++) {
    out = step();
    eval(out);
  }
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) {
    out = step();
    eval(out);
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
  printf("ane_is_available: %d\n", metal::ane_is_available());

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // A single conv stage representative of a static vision model block,
  // same order of magnitude as the earlier MPSGraph-only probes.
  Shape in_shape{1, 56, 56, 64};
  Shape w_shape{64, 3, 3, 64};
  size_t in_n = 1 * 56 * 56 * 64;
  size_t w_n = 64 * 3 * 3 * 64;

  std::vector<float> in_data(in_n), w_data(w_n);
  for (auto& v : in_data) v = dist(rng);
  for (auto& v : w_data) v = dist(rng);

  array x = astype(array(in_data.begin(), in_shape), float16);
  array w = astype(array(w_data.begin(), w_shape), float16);

  // Regular Metal path: plain conv2d, no compile(), eager eval_gpu.
  double metal_ms = time_ms(
      [&]() {
        return conv2d(x, w, {1, 1}, {1, 1}, {1, 1}, 1, Device::gpu);
      },
      5,
      30);

  // ANE path: same conv2d, but traced through compile() so
  // compile_fuse_ane() swaps it for CompiledANE.
  std::function<std::vector<array>(const std::vector<array>&)> fun =
      [](const std::vector<array>& inputs) {
        auto y = conv2d(
            inputs[0], inputs[1], {1, 1}, {1, 1}, {1, 1}, 1, Device::gpu);
        return std::vector<array>{y};
      };
  auto compiled = compile(fun);
  double ane_ms = time_ms(
      [&]() { return compiled({x, w})[0]; }, 5, 30);

  printf("Metal conv2d:     %.3f ms/iter\n", metal_ms);
  printf("CompiledANE conv2d: %.3f ms/iter\n", ane_ms);
  printf("ratio (metal/ane): %.2fx\n", metal_ms / ane_ms);
  return 0;
}
