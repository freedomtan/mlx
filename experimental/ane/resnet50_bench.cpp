// Standalone benchmark/correctness check against a ResNet50-shaped conv
// population, not part of the mlx build. This is not a bit-exact ResNet50
// port (no real weights, no maxpool, stem simplified to feeding stage 1
// directly at 56x56x64) -- the point is to exercise the same conv2d shape
// and depth population a real ResNet50 forward pass has (1x1/3x3 convs
// across 4 stages, 3/4/6/3 bottleneck blocks, channel progression
// 64->256->512->1024->2048), not to reproduce ImageNet-accuracy weights.
//
// Build (after `cmake --build build_ane --target mlx` with MLX_BUILD_ANE=ON):
//   clang++ -std=c++20 -O2 -I. -Ibuild_ane/_deps/metal_cpp-src \
//     -Ibuild_ane/_deps/fmt-src/include \
//     experimental/ane/resnet50_bench.cpp build_ane/libmlx.a \
//     -framework Foundation -framework Metal -framework QuartzCore \
//     -framework IOKit -framework MetalPerformanceShadersGraph \
//     -framework Accelerate -framework Security -o /tmp/resnet50_bench
// Run:
//   MLX_METAL_PATH=build_ane/mlx/backend/metal/kernels /tmp/resnet50_bench

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>

#include "mlx/mlx.h"

using namespace mlx::core;

std::mt19937 g_rng(0);

array random_weight(int co, int kh, int kw, int ci) {
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  std::vector<float> data((size_t)co * kh * kw * ci);
  for (auto& v : data) {
    v = dist(g_rng);
  }
  return astype(array(data.begin(), Shape{co, kh, kw, ci}), float16);
}

array relu(const array& x) {
  return maximum(x, array(0.0f, float16));
}

// 1x1 -> 3x3 -> 1x1 bottleneck, with an optional 1x1 downsample on the
// shortcut when channel count or stride changes (standard ResNet bottleneck).
array bottleneck(
    const array& x,
    int in_ch,
    int mid_ch,
    int out_ch,
    int stride,
    bool downsample,
    Device::DeviceType device) {
  auto w1 = random_weight(mid_ch, 1, 1, in_ch);
  auto y = conv2d(x, w1, {1, 1}, {0, 0}, {1, 1}, 1, device);
  y = relu(y);

  auto w2 = random_weight(mid_ch, 3, 3, mid_ch);
  y = conv2d(y, w2, {stride, stride}, {1, 1}, {1, 1}, 1, device);
  y = relu(y);

  auto w3 = random_weight(out_ch, 1, 1, mid_ch);
  y = conv2d(y, w3, {1, 1}, {0, 0}, {1, 1}, 1, device);

  array shortcut = x;
  if (downsample) {
    auto wd = random_weight(out_ch, 1, 1, in_ch);
    shortcut = conv2d(x, wd, {stride, stride}, {0, 0}, {1, 1}, 1, device);
  }
  return relu(add(y, shortcut));
}

// Stage counts and channel progression match torchvision's resnet50.
array resnet50_stages(const array& stem_out, Device::DeviceType device) {
  struct Stage {
    int blocks;
    int in_ch, mid_ch, out_ch, stride;
  };
  std::vector<Stage> stages = {
      {3, 64, 64, 256, 1},
      {4, 256, 128, 512, 2},
      {6, 512, 256, 1024, 2},
      {3, 1024, 512, 2048, 2},
  };

  array x = stem_out;
  for (auto& s : stages) {
    for (int b = 0; b < s.blocks; b++) {
      bool first = b == 0;
      int in_ch = first ? s.in_ch : s.out_ch;
      int stride = first ? s.stride : 1;
      bool downsample = first; // channel or spatial change
      x = bottleneck(x, in_ch, s.mid_ch, s.out_ch, stride, downsample, device);
    }
  }
  return x;
}

int main() {
  printf("ane_is_available: %d\n", metal::ane_is_available());

  // Approximates ResNet50's stem output (conv1 + maxpool): 56x56x64.
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> stem_data(1 * 56 * 56 * 64);
  for (auto& v : stem_data) {
    v = dist(g_rng);
  }
  array stem_f32(stem_data.begin(), Shape{1, 56, 56, 64});
  array stem = astype(stem_f32, float16);

  // Correctness: same random weights (fixed seed) on CPU vs compiled GPU.
  g_rng.seed(1);
  auto ref = resnet50_stages(stem, Device::cpu);
  eval(ref);

  g_rng.seed(1);
  std::function<std::vector<array>(const std::vector<array>&)> fun =
      [](const std::vector<array>& inputs) {
        return std::vector<array>{resnet50_stages(inputs[0], Device::gpu)};
      };
  auto compiled = compile(fun);
  auto outs = compiled({stem});
  eval(outs);

  auto ref32 = astype(ref, float32);
  auto cand32 = astype(outs[0], float32);
  eval(ref32);
  eval(cand32);

  bool shape_ok = ref.shape() == outs[0].shape();
  double max_abs_diff = 0.0, max_rel_diff = 0.0;
  if (shape_ok) {
    const float* r = ref32.data<float>();
    const float* c = cand32.data<float>();
    for (size_t i = 0; i < ref32.size(); i++) {
      double d = std::abs((double)r[i] - (double)c[i]);
      max_abs_diff = std::max(max_abs_diff, d);
      max_rel_diff =
          std::max(max_rel_diff, d / std::max(1e-3, std::abs((double)r[i])));
    }
  }
  bool correctness_pass =
      shape_ok && max_abs_diff < 0.5 && max_rel_diff < 0.2;
  {
    const float* r = ref32.data<float>();
    double sum_abs = 0.0, max_val = 0.0;
    for (size_t i = 0; i < ref32.size(); i++) {
      sum_abs += std::abs((double)r[i]);
      max_val = std::max(max_val, (double)std::abs(r[i]));
    }
    printf(
        "ref signal: mean_abs=%.8f max_abs=%.8f n=%zu\n",
        sum_abs / ref32.size(),
        max_val,
        ref32.size());
  }
  printf(
      "correctness: shape_ok=%d max_abs_diff=%.8f max_rel_diff=%.8f -> %s\n",
      shape_ok,
      max_abs_diff,
      max_rel_diff,
      correctness_pass ? "PASS" : "FAIL");

  // Benchmark: all-Metal eager forward vs compile()-with-ANE-fuse forward.
  // Fixed weights (seeded once) reused across iterations so both paths see
  // identical graphs shape-for-shape across repeats.
  g_rng.seed(2);
  auto metal_fun = [&]() { return resnet50_stages(stem, Device::gpu); };

  auto time_ms = [&](const std::function<array()>& step, int warmup,
                      int iters) {
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
    return std::chrono::duration<double, std::milli>(t1 - t0).count() /
        iters;
  };

  double metal_ms = time_ms(metal_fun, 2, 10);

  auto compiled2 = compile(fun);
  double ane_ms = time_ms([&]() { return compiled2({stem})[0]; }, 2, 10);

  printf("Metal forward:       %.2f ms/iter\n", metal_ms);
  printf("CompiledANE forward:  %.2f ms/iter\n", ane_ms);
  printf("ratio (metal/ane): %.2fx\n", metal_ms / ane_ms);

  return correctness_pass ? 0 : 1;
}
