// Standalone correctness check, not part of the mlx build: does
// CompiledANE's MPSGraph/ANE conv2d+bias(+relu) path produce the same
// result as the reference CPU ops, for shapes within the eligibility rules
// in mlx/compile.cpp (is_ane_op/compile_fuse_ane)?
//
// A bare conv (no bias) is NOT expected to fuse to CompiledANE: MPSGraph
// never places a bare convolution on the ANE regardless of layout, only one
// followed by a genuine consumer (confirmed in
// experimental/ane/trailing_op_check.mm), so compile_fuse_ane requires at
// least one other fusable op (a bias-add, at minimum) before creating a
// CompiledANE node.
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
  bool with_bias;
  bool with_relu;
  bool expect_ane; // whether the conv should fuse into CompiledANE
};

array make_fp16(std::mt19937& rng, const Shape& shape) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  size_t n = 1;
  for (auto d : shape) {
    n *= d;
  }
  std::vector<float> data(n);
  for (auto& v : data) {
    v = dist(rng);
  }
  return astype(array(data.begin(), shape), float16);
}

// Two conv+bias+relu in a row, CompiledANE feeding directly into another
// CompiledANE's input -- never exercised by `forward()`/run_case(), which
// only ever builds a single conv.
bool run_chained_case() {
  std::mt19937 rng(0);
  Shape in_shape{1, 56, 56, 64};
  array x = make_fp16(rng, in_shape);
  array w1 = make_fp16(rng, {64, 1, 1, 64});
  array b1 = make_fp16(rng, {64});
  array w2 = make_fp16(rng, {64, 3, 3, 64}); // matches resnet50's conv1->conv2
  array b2 = make_fp16(rng, {64});

  auto chain = [](const array& x, const array& w1, const array& b1,
                   const array& w2, const array& b2, Device::DeviceType d) {
    auto y = conv2d(x, w1, {1, 1}, {0, 0}, {1, 1}, 1, d);
    y = add(y, b1);
    y = maximum(y, array(0.0f, float16));
    y = conv2d(y, w2, {1, 1}, {1, 1}, {1, 1}, 1, d);
    y = add(y, b2);
    y = maximum(y, array(0.0f, float16));
    return y;
  };

  auto ref = chain(x, w1, b1, w2, b2, Device::cpu);
  eval(ref);

  std::function<std::vector<array>(const std::vector<array>&)> fun =
      [&chain](const std::vector<array>& inputs) {
        return std::vector<array>{chain(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            Device::gpu)};
      };
  auto compiled = compile(fun);
  auto outs = compiled({x, w1, b1, w2, b2});
  eval(outs);

  auto ref32 = astype(ref, float32);
  auto cand32 = astype(outs[0], float32);
  eval(ref32);
  eval(cand32);
  const float* r = ref32.data<float>();
  const float* c = cand32.data<float>();
  double max_abs_diff = 0.0, max_rel_diff = 0.0, ref_max_abs = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < ref32.size(); i++) {
    double d = std::abs((double)r[i] - (double)c[i]);
    if (d > max_abs_diff) worst_i = i;
    max_abs_diff = std::max(max_abs_diff, d);
    max_rel_diff = std::max(max_rel_diff, d / std::max(1e-3, std::abs((double)r[i])));
    ref_max_abs = std::max(ref_max_abs, std::abs((double)r[i]));
  }
  bool pass = max_abs_diff < 0.5 && max_rel_diff < 0.2;
  printf(
      "[chained_conv_bias_relu] max_abs_diff=%.6f max_rel_diff=%.6f ref_max_abs=%.6f worst_i=%zu ref=%.6f cand=%.6f -> %s\n",
      max_abs_diff, max_rel_diff, ref_max_abs, worst_i, r[worst_i], c[worst_i],
      pass ? "PASS" : "FAIL");
  return pass;
}

// Matches the resnet50 downsample/residual pattern: x feeds two SEPARATE
// conv branches (one with relu, one without), both CompiledANE, whose
// outputs are later combined by an unrelated Add+relu -- never exercised by
// run_chained_case() (a linear chain) or run_case() (a single conv).
bool run_branch_merge_case() {
  std::mt19937 rng(0);
  Shape in_shape{1, 56, 56, 64};
  array x = make_fp16(rng, in_shape);
  array w_main = make_fp16(rng, {256, 1, 1, 64});
  array b_main = make_fp16(rng, {256});
  array w_ds = make_fp16(rng, {256, 1, 1, 64});
  array b_ds = make_fp16(rng, {256});

  auto branch_merge = [](const array& x, const array& w_main,
                          const array& b_main, const array& w_ds,
                          const array& b_ds, Device::DeviceType d) {
    auto main = conv2d(x, w_main, {1, 1}, {0, 0}, {1, 1}, 1, d);
    main = add(main, b_main); // no relu, like conv3 in a bottleneck
    auto shortcut = conv2d(x, w_ds, {1, 1}, {0, 0}, {1, 1}, 1, d);
    shortcut = add(shortcut, b_ds); // no relu, like the downsample conv
    return maximum(add(main, shortcut), array(0.0f, float16));
  };

  auto ref = branch_merge(x, w_main, b_main, w_ds, b_ds, Device::cpu);
  eval(ref);

  std::function<std::vector<array>(const std::vector<array>&)> fun =
      [&branch_merge](const std::vector<array>& inputs) {
        return std::vector<array>{branch_merge(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            Device::gpu)};
      };
  auto compiled = compile(fun);
  auto outs = compiled({x, w_main, b_main, w_ds, b_ds});
  eval(outs);

  auto ref32 = astype(ref, float32);
  auto cand32 = astype(outs[0], float32);
  eval(ref32);
  eval(cand32);
  const float* r = ref32.data<float>();
  const float* c = cand32.data<float>();
  double max_abs_diff = 0.0, max_rel_diff = 0.0, ref_max_abs = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < ref32.size(); i++) {
    double d = std::abs((double)r[i] - (double)c[i]);
    if (d > max_abs_diff) worst_i = i;
    max_abs_diff = std::max(max_abs_diff, d);
    max_rel_diff = std::max(max_rel_diff, d / std::max(1e-3, std::abs((double)r[i])));
    ref_max_abs = std::max(ref_max_abs, std::abs((double)r[i]));
  }
  bool pass = max_abs_diff < 0.5 * std::max(1.0, ref_max_abs / 50.0);
  printf(
      "[branch_merge] max_abs_diff=%.6f max_rel_diff=%.6f ref_max_abs=%.6f worst_i=%zu ref=%.6f cand=%.6f -> %s\n",
      max_abs_diff, max_rel_diff, ref_max_abs, worst_i, r[worst_i], c[worst_i],
      pass ? "PASS" : "FAIL");
  return pass;
}

array forward(
    const array& x,
    const array& w,
    const array* bias,
    bool with_relu,
    std::pair<int, int> stride,
    std::pair<int, int> padding,
    int groups,
    Device::DeviceType device) {
  auto y = conv2d(x, w, stride, padding, {1, 1}, groups, device);
  if (bias) {
    y = add(y, *bias);
  }
  if (with_relu) {
    y = maximum(y, array(0.0f, float16));
  }
  return y;
}

bool run_case(const Case& c) {
  std::mt19937 rng(0);
  array x = make_fp16(rng, c.in_shape);
  array w = make_fp16(rng, c.w_shape);
  int out_channels = c.w_shape[0];
  array bias = make_fp16(rng, {out_channels});
  const array* bias_ptr = c.with_bias ? &bias : nullptr;

  auto ref = forward(
      x, w, bias_ptr, c.with_relu, c.stride, c.padding, c.groups, Device::cpu);
  eval(ref);

  std::function<std::vector<array>(const std::vector<array>&)> fun =
      [&c](const std::vector<array>& inputs) {
        const array* bp = c.with_bias ? &inputs[2] : nullptr;
        auto y = forward(
            inputs[0], inputs[1], bp, c.with_relu, c.stride, c.padding,
            c.groups, Device::gpu);
        return std::vector<array>{y};
      };
  auto compiled = compile(fun);
  std::vector<array> call_inputs = {x, w};
  if (c.with_bias) {
    call_inputs.push_back(bias);
  }
  auto outs = compiled(call_inputs);
  auto& candidate = outs[0];
  std::string primitive_name = candidate.primitive().name();
  eval(outs);

  bool is_ane = primitive_name == "CompiledANE";
  bool primitive_ok = is_ane == c.expect_ane;

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
      "[%s] primitive=%s is_ane=%d (expected %d) shape_ok=%d max_abs_diff=%.6f max_rel_diff=%.6f -> %s\n",
      c.name,
      primitive_name.c_str(),
      is_ane,
      c.expect_ane,
      shape_ok,
      max_abs_diff,
      max_rel_diff,
      pass ? "PASS" : "FAIL");
  return pass;
}

int main() {
  printf("ane_is_available: %d\n", metal::ane_is_available());

  std::vector<Case> cases = {
      {"bare_conv_no_bias_does_not_fuse",
       {1, 16, 16, 8},
       {8, 3, 3, 8},
       {1, 1},
       {1, 1},
       1,
       /* with_bias */ false,
       /* with_relu */ false,
       /* expect_ane */ false},
      {"conv_bias_eligible",
       {1, 16, 16, 8},
       {8, 3, 3, 8},
       {1, 1},
       {1, 1},
       1,
       /* with_bias */ true,
       /* with_relu */ false,
       /* expect_ane */ true},
      {"conv_bias_relu_eligible",
       {1, 16, 16, 8},
       {8, 3, 3, 8},
       {1, 1},
       {1, 1},
       1,
       /* with_bias */ true,
       /* with_relu */ true,
       /* expect_ane */ true},
      {"stride2_conv_bias_relu_eligible",
       {1, 17, 17, 8},
       {8, 3, 3, 8},
       {2, 2},
       {1, 1},
       1,
       /* with_bias */ true,
       /* with_relu */ true,
       /* expect_ane */ true},
      {"valid_padding_conv_bias_eligible",
       {1, 16, 16, 8},
       {8, 3, 3, 8},
       {1, 1},
       {0, 0},
       1,
       /* with_bias */ true,
       /* with_relu */ false,
       /* expect_ane */ true},
      {"kernel1x1_conv_bias_relu_eligible",
       {1, 56, 56, 64},
       {64, 1, 1, 64},
       {1, 1},
       {0, 0},
       1,
       /* with_bias */ true,
       /* with_relu */ true,
       /* expect_ane */ true},
      {"kernel1x1_wide_conv_bias_eligible",
       {1, 56, 56, 64},
       {256, 1, 1, 64},
       {1, 1},
       {0, 0},
       1,
       /* with_bias */ true,
       /* with_relu */ false,
       /* expect_ane */ true},
      {"grouped_conv_bias_falls_back",
       {1, 16, 16, 8},
       {8, 3, 3, 4},
       {1, 1},
       {1, 1},
       2,
       /* with_bias */ true,
       /* with_relu */ false,
       /* expect_ane */ false},
  };

  bool all_pass = true;
  for (auto& c : cases) {
    all_pass &= run_case(c);
  }
  all_pass &= run_chained_case();
  all_pass &= run_branch_merge_case();
  printf("%s\n", all_pass ? "ALL PASS" : "SOME FAILED");
  return all_pass ? 0 : 1;
}
