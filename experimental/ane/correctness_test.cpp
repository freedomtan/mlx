// Standalone correctness check, not part of the mlx build: does
// CompiledANE's MPSGraph/ANE conv2d path produce the same result as the
// reference CPU Convolution, for shapes within the v1 eligibility rules in
// mlx/compile.cpp (is_fusable_ane), and does an ineligible conv (groups=2)
// still fall through correctly to the regular Metal path?
//
// Build (after `cmake --build build_ane --target mlx`):
//   clang++ -std=c++20 -I. -Ibuild_ane/_deps/metal_cpp-src \
//     -Ibuild_ane/_deps/fmt-src/include \
//     experimental/ane/correctness_test.cpp build_ane/libmlx.a \
//     -framework Foundation -framework Metal -framework QuartzCore \
//     -framework IOKit -framework MetalPerformanceShadersGraph \
//     -framework Accelerate -framework Security -o /tmp/ane_correctness
// Run:
//   MLX_METAL_PATH=build_ane/mlx/backend/metal/kernels /tmp/ane_correctness

#include <cmath>
#include <cstdio>
#include <random>

#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

struct Case {
  const char* name;
  Shape in_shape; // NHWC
  Shape w_shape; // OHWI
  std::pair<int, int> stride;
  std::pair<int, int> padding;
  int groups;
  const char* expect_primitive;
};

bool run_case(const Case& c) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  size_t in_n = 1;
  for (auto d : c.in_shape) {
    in_n *= d;
  }
  size_t w_n = 1;
  for (auto d : c.w_shape) {
    w_n *= d;
  }

  std::vector<float> in_data(in_n), w_data(w_n);
  for (auto& v : in_data) {
    v = dist(rng);
  }
  for (auto& v : w_data) {
    v = dist(rng);
  }

  array x = astype(array(in_data.begin(), c.in_shape), float16);
  array w = astype(array(w_data.begin(), c.w_shape), float16);

  auto ref = conv2d(
      x, w, c.stride, c.padding, {1, 1}, c.groups, Device::cpu);
  eval(ref);

  std::function<std::vector<array>(const std::vector<array>&)> fun =
      [stride = c.stride, padding = c.padding, groups = c.groups](
          const std::vector<array>& inputs) {
        auto y = conv2d(
            inputs[0],
            inputs[1],
            stride,
            padding,
            {1, 1},
            groups,
            Device::gpu);
        return std::vector<array>{y};
      };
  auto compiled = compile(fun);
  auto outs = compiled({x, w});
  auto& candidate = outs[0];
  std::string primitive_name = candidate.primitive().name();
  eval(outs);

  bool primitive_ok = primitive_name == c.expect_primitive;

  auto ref32 = astype(ref, float32);
  auto cand32 = astype(candidate, float32);
  eval(ref32);
  eval(cand32);

  bool shape_ok = ref.shape() == candidate.shape();
  double max_abs_diff = 0.0, max_rel_diff = 0.0;
  if (shape_ok) {
    const float* r = ref32.data<float>();
    const float* k = cand32.data<float>();
    for (size_t i = 0; i < ref32.size(); i++) {
      double d = std::abs((double)r[i] - (double)k[i]);
      max_abs_diff = std::max(max_abs_diff, d);
      max_rel_diff = std::max(max_rel_diff, d / std::max(1e-3, std::abs((double)r[i])));
    }
  }

  bool numeric_ok = shape_ok && max_abs_diff < 0.05 && max_rel_diff < 0.1;
  bool pass = primitive_ok && numeric_ok;

  printf(
      "[%s] primitive=%s (expected %s) shape_ok=%d max_abs_diff=%.6f max_rel_diff=%.6f -> %s\n",
      c.name,
      primitive_name.c_str(),
      c.expect_primitive,
      shape_ok,
      max_abs_diff,
      max_rel_diff,
      pass ? "PASS" : "FAIL");
  return pass;
}

int main() {
  printf("ane_is_available: %d\n", metal::ane_is_available());

  std::vector<Case> cases = {
      {"stride1_pad1_eligible",
       {1, 16, 16, 8},
       {8, 3, 3, 8},
       {1, 1},
       {1, 1},
       1,
       "CompiledANE"},
      {"stride2_pad1_eligible",
       {1, 17, 17, 8},
       {8, 3, 3, 8},
       {2, 2},
       {1, 1},
       1,
       "CompiledANE"},
      {"valid_padding_eligible",
       {1, 16, 16, 8},
       {8, 3, 3, 8},
       {1, 1},
       {0, 0},
       1,
       "CompiledANE"},
      {"grouped_falls_back",
       {1, 16, 16, 8},
       {8, 3, 3, 4},
       {1, 1},
       {1, 1},
       2,
       "Convolution"},
  };

  bool all_pass = true;
  for (auto& c : cases) {
    all_pass &= run_case(c);
  }
  printf("%s\n", all_pass ? "ALL PASS" : "SOME FAILED");
  return all_pass ? 0 : 1;
}
