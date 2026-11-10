// Real ResNet50 (ImageNet-pretrained torchvision weights, BatchNorm folded
// into each conv at export time) through the ANE path, cross-checked against
// both the CPU mlx path and an independent PyTorch reference. Weights/input
// are produced by experimental/ane/export_resnet50.py, which requires
// torch+torchvision (not part of mlx).
//
// Build (after `cmake --build build_ane --target mlx` with MLX_BUILD_ANE=ON):
//   clang++ -std=c++20 -O2 -I. -Ibuild_ane/_deps/metal_cpp-src \
//     -Ibuild_ane/_deps/fmt-src/include \
//     experimental/ane/resnet50_real_bench.cpp build_ane/libmlx.a \
//     -framework Foundation -framework Metal -framework QuartzCore \
//     -framework IOKit -framework MetalPerformanceShadersGraph \
//     -framework Accelerate -framework Security -o /tmp/resnet50_real_bench
// Run (after `python3 experimental/ane/export_resnet50.py <dir>`):
//   MLX_METAL_PATH=build_ane/mlx/backend/metal/kernels \
//     /tmp/resnet50_real_bench <dir>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <stdexcept>

#include "mlx/mlx.h"

using namespace mlx::core;

// Matches export_resnet50.py's write_blob: uint32 ndim, ndim x uint32 dims,
// then raw float32 data, read sequentially in export order.
class BlobReader {
 public:
  explicit BlobReader(const std::string& path) : f_(path, std::ios::binary) {
    if (!f_) {
      throw std::runtime_error("cannot open " + path);
    }
  }

  array next() {
    uint32_t ndim;
    f_.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    Shape shape(ndim);
    size_t n = 1;
    for (uint32_t i = 0; i < ndim; i++) {
      uint32_t d;
      f_.read(reinterpret_cast<char*>(&d), sizeof(d));
      shape[i] = (int)d;
      n *= d;
    }
    std::vector<float> data(n);
    f_.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
    return array(data.begin(), shape);
  }

 private:
  std::ifstream f_;
};

array relu(const array& x) {
  return maximum(x, array(0.0f, float16));
}

array conv_bias_relu(
    const array& x,
    const array& w,
    const array& bias,
    std::pair<int, int> stride,
    std::pair<int, int> padding,
    Device::DeviceType device,
    bool apply_relu) {
  auto y = conv2d(x, w, stride, padding, {1, 1}, 1, device);
  // Feed the raw 1D bias directly; mlx inserts its own Broadcast primitive
  // for the last-dim broadcast, which is what compile_fuse_ane looks for
  // (mlx/compile.cpp). Reshaping to (1,1,1,C) first defeats that match.
  y = add(y, bias);
  return apply_relu ? relu(y) : y;
}

// 3x3, stride 2, pad 1 max pool over NHWC, via 9 shifted-and-strided slices.
// Safe because this always runs after a ReLU, so all real values are >= 0
// and the pad value just needs to be negative.
array max_pool_3x3_s2(const array& x) {
  auto padded = pad(x, {1, 2}, {1, 1}, {1, 1}, array(-1.0f, float16));
  int out_h = (padded.shape(1) - 3) / 2 + 1;
  int out_w = (padded.shape(2) - 3) / 2 + 1;
  array acc = array(-1.0f, float16);
  bool first = true;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      auto s = slice(
          padded,
          {0, i, j, 0},
          {padded.shape(0), i + 2 * out_h - 1, j + 2 * out_w - 1,
           padded.shape(3)},
          {1, 2, 2, 1});
      acc = first ? s : maximum(acc, s);
      first = false;
    }
  }
  return acc;
}

struct Resnet50Weights {
  array conv1_w, conv1_b;
  struct Block {
    array w1, b1, w2, b2, w3, b3;
    bool has_downsample;
    array wd, bd;
  };
  std::vector<Block> blocks; // 3 + 4 + 6 + 3 = 16, in order
};

Resnet50Weights load_weights(const std::string& path) {
  BlobReader r(path);
  array conv1_w = astype(r.next(), float16);
  array conv1_b = astype(r.next(), float16);
  Resnet50Weights w{conv1_w, conv1_b, {}};

  int stage_blocks[4] = {3, 4, 6, 3};
  for (int s = 0; s < 4; s++) {
    for (int b = 0; b < stage_blocks[s]; b++) {
      array w1 = astype(r.next(), float16);
      array b1 = astype(r.next(), float16);
      array w2 = astype(r.next(), float16);
      array b2 = astype(r.next(), float16);
      array w3 = astype(r.next(), float16);
      array b3 = astype(r.next(), float16);
      bool has_downsample = (b == 0);
      array wd = has_downsample ? astype(r.next(), float16) : w1;
      array bd = has_downsample ? astype(r.next(), float16) : b1;
      w.blocks.push_back(
          {w1, b1, w2, b2, w3, b3, has_downsample, wd, bd});
    }
  }
  return w;
}

// Total multiply-accumulates for forward_n_blocks(), computed directly from
// the actual weight shapes and the same stride/padding/spatial-tracking
// logic forward_n_blocks() uses -- not a remembered "ResNet50 is ~4 GFLOPs"
// citation, so it can't silently disagree with what this benchmark actually
// runs (e.g. if max_blocks < 16, or input size changes).
double compute_total_macs(const Resnet50Weights& w, int H0, int W0, int max_blocks) {
  auto out_dim = [](int in, int k, int pad, int stride) {
    return (in + 2 * pad - k) / stride + 1;
  };
  // w shape is OHWI: [out_ch, kh, kw, in_ch].
  auto conv_macs_and_advance = [&](const array& wt, int& H, int& W, int pad,
                                    int stride) {
    int out_ch = wt.shape(0), kh = wt.shape(1), kw = wt.shape(2),
        in_ch = wt.shape(3);
    int oh = out_dim(H, kh, pad, stride);
    int ow = out_dim(W, kw, pad, stride);
    double macs = (double)oh * ow * out_ch * kh * kw * in_ch;
    H = oh;
    W = ow;
    return macs;
  };

  int H = H0, W = W0;
  double total = 0.0;
  total += conv_macs_and_advance(w.conv1_w, H, W, 3, 2);
  H = out_dim(H, 3, 1, 2); // maxpool, no MACs
  W = out_dim(W, 3, 1, 2);

  int stage_blocks[4] = {3, 4, 6, 3};
  int stage_stride[4] = {1, 2, 2, 2};
  int idx = 0, done = 0;
  for (int s = 0; s < 4 && done < max_blocks; s++) {
    for (int b = 0; b < stage_blocks[s] && done < max_blocks; b++) {
      auto& blk = w.blocks[idx++];
      int stride = (b == 0) ? stage_stride[s] : 1;
      int Hb = H, Wb = W; // block input spatial size, for the shortcut branch
      total += conv_macs_and_advance(blk.w1, H, W, 0, 1);
      total += conv_macs_and_advance(blk.w2, H, W, 1, stride);
      total += conv_macs_and_advance(blk.w3, H, W, 0, 1);
      if (blk.has_downsample) {
        int Hd = Hb, Wd = Wb;
        total += conv_macs_and_advance(blk.wd, Hd, Wd, 0, stride);
      }
      done++;
    }
  }
  return total;
}

array forward_n_blocks(
    const array& input,
    const Resnet50Weights& w,
    Device::DeviceType device,
    int max_blocks) {
  auto x = conv_bias_relu(
      input, w.conv1_w, w.conv1_b, {2, 2}, {3, 3}, device, true);
  x = max_pool_3x3_s2(x);

  int stage_blocks[4] = {3, 4, 6, 3};
  int stage_stride[4] = {1, 2, 2, 2};
  int idx = 0;
  int done = 0;
  for (int s = 0; s < 4 && done < max_blocks; s++) {
    for (int b = 0; b < stage_blocks[s] && done < max_blocks; b++) {
      auto& blk = w.blocks[idx++];
      int stride = (b == 0) ? stage_stride[s] : 1;
      auto y = conv_bias_relu(
          x, blk.w1, blk.b1, {1, 1}, {0, 0}, device, true);
      y = conv_bias_relu(
          y, blk.w2, blk.b2, {stride, stride}, {1, 1}, device, true);
      y = conv_bias_relu(
          y, blk.w3, blk.b3, {1, 1}, {0, 0}, device, false);
      array shortcut = x;
      if (blk.has_downsample) {
        shortcut = conv_bias_relu(
            x, blk.wd, blk.bd, {stride, stride}, {0, 0}, device, false);
      }
      x = relu(add(y, shortcut));
      done++;
    }
  }
  return x;
}

array forward(
    const array& input,
    const Resnet50Weights& w,
    Device::DeviceType device) {
  return forward_n_blocks(input, w, device, 16);
}

// Same as forward(), but also returns every stage boundary, so the
// compiled/ANE path's intermediates can be inspected too (replaying a
// cached compile() tape never re-invokes forward() as C++ code, so
// mid-function eval() calls only ever see the CPU reference call).
std::vector<array> forward_with_stages(
    const array& input,
    const Resnet50Weights& w,
    Device::DeviceType device) {
  auto x = conv_bias_relu(
      input, w.conv1_w, w.conv1_b, {2, 2}, {3, 3}, device, true);
  std::vector<array> stages;
  stages.push_back(x); // stem conv+bias+relu, before maxpool
  x = max_pool_3x3_s2(x);
  stages.push_back(x); // after maxpool

  int stage_blocks[4] = {3, 4, 6, 3};
  int stage_stride[4] = {1, 2, 2, 2};
  int idx = 0;
  for (int s = 0; s < 4; s++) {
    for (int b = 0; b < stage_blocks[s]; b++) {
      auto& blk = w.blocks[idx++];
      int stride = (b == 0) ? stage_stride[s] : 1;
      auto y = conv_bias_relu(
          x, blk.w1, blk.b1, {1, 1}, {0, 0}, device, true);
      if (s == 0 && b == 0) stages.push_back(y); // after conv1+bias+relu
      y = conv_bias_relu(
          y, blk.w2, blk.b2, {stride, stride}, {1, 1}, device, true);
      if (s == 0 && b == 0) stages.push_back(y); // after conv2+bias+relu
      y = conv_bias_relu(
          y, blk.w3, blk.b3, {1, 1}, {0, 0}, device, false);
      if (s == 0 && b == 0) stages.push_back(y); // after conv3+bias
      array shortcut = x;
      if (blk.has_downsample) {
        shortcut = conv_bias_relu(
            x, blk.wd, blk.bd, {stride, stride}, {0, 0}, device, false);
        if (s == 0 && b == 0) stages.push_back(shortcut); // after downsample+bias
      }
      x = relu(add(y, shortcut));
      stages.push_back(x); // per-block, for fine-grained localization
    }
    stages.push_back(x);
  }
  return stages;
}

double max_abs(const array& a) {
  auto a32 = astype(a, float32);
  eval(a32);
  const float* p = a32.data<float>();
  double m = 0.0;
  for (size_t i = 0; i < a32.size(); i++) {
    m = std::max(m, (double)std::abs(p[i]));
  }
  return m;
}

double max_abs_diff(const array& a, const array& b) {
  auto a32 = astype(a, float32);
  auto b32 = astype(b, float32);
  eval(a32);
  eval(b32);
  const float* pa = a32.data<float>();
  const float* pb = b32.data<float>();
  double m = 0.0;
  for (size_t i = 0; i < a32.size(); i++) {
    m = std::max(m, (double)std::abs((double)pa[i] - (double)pb[i]));
  }
  return m;
}

// Runs one path (metal or ane) in a tight loop for `seconds`, printing wall
// clock markers around the loop. Meant to be run while sampling
// `sudo powermetrics --samplers gpu_power,ane_power,cpu_power` in another
// terminal, so GPU/ANE hardware activity can be correlated to which path is
// actually running -- eval_gpu() running doesn't by itself prove which
// hardware block executed the work.
void run_sustained(
    const std::string& mode,
    const array& input,
    const Resnet50Weights& weights,
    double seconds) {
  std::function<array()> step;
  if (mode == "metal") {
    step = [&]() { return forward(input, weights, Device::gpu); };
  } else if (mode == "ane") {
    std::function<std::vector<array>(const std::vector<array>&)> fun =
        [&weights](const std::vector<array>& inputs) {
          return std::vector<array>{forward(inputs[0], weights, Device::gpu)};
        };
    auto compiled = compile(fun);
    step = [&, compiled]() mutable { return compiled({input})[0]; };
  } else {
    fprintf(stderr, "unknown mode '%s', expected 'metal' or 'ane'\n", mode.c_str());
    return;
  }

  array warm = step();
  eval(warm);
  fprintf(stderr, "[sustained] mode=%s starting now, running for %.1fs\n",
          mode.c_str(), seconds);
  fflush(stderr);
  auto t0 = std::chrono::steady_clock::now();
  int iters = 0;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
             .count() < seconds) {
    array out = step();
    eval(out);
    iters++;
  }
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  fprintf(
      stderr,
      "[sustained] mode=%s done: %d iters in %.1fms (%.2fms/iter)\n",
      mode.c_str(), iters, ms, ms / iters);
}

// Builds and runs forward_n_blocks(0..16) (0 = stem+maxpool only) one prefix
// length at a time, each as its own fresh compile() (so each gets its own
// CompiledANE tape/cache entry, forcing a real recompile every step), and
// brackets each step with a wall-clock timestamp in the exact format
// `log show --start/--end` expects, so a shell wrapper can pull the real
// ANE-driver procedure count (AppleH16ANEInterface
// traverseAllProcedureAndOperationGraphs: ProcId:N) for that exact prefix.
void run_layer_by_layer(const array& input, const Resnet50Weights& weights) {
  auto print_ts = [](const char* label) {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch())
                  .count() %
        1000;
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    fprintf(stderr, "%s %s.%03lld\n", label, buf, (long long)ms);
  };

  for (int n = 0; n <= 16; n++) {
    std::function<std::vector<array>(const std::vector<array>&)> fun_n =
        [&weights, n](const std::vector<array>& inputs) {
          return std::vector<array>{
              forward_n_blocks(inputs[0], weights, Device::gpu, n)};
        };
    auto compiled_n = compile(fun_n);

    print_ts("[layer-ts-start]");
    fprintf(stderr, "[layer] n=%d compiling+warming up\n", n);
    auto out = compiled_n({input})[0];
    eval(out); // forces the fresh compile for this n

    int iters = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() < 1.0) {
      auto o = compiled_n({input})[0];
      eval(o);
      iters++;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    print_ts("[layer-ts-end]");
    fprintf(
        stderr, "[layer] n=%d iters=%d ms_per_iter=%.3f\n", n, iters,
        iters > 0 ? ms / iters : -1.0);
    fflush(stderr);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <export_dir> [metal|ane|layers [seconds]]\n", argv[0]);
    return 2;
  }
  std::string dir = argv[1];
  std::string sustained_mode = argc > 2 ? argv[2] : "";
  double sustained_seconds = argc > 3 ? std::atof(argv[3]) : 6.0;

  printf("ane_is_available: %d\n", metal::ane_is_available());

  auto weights = load_weights(dir + "/resnet50_weights.bin");

  BlobReader input_reader(dir + "/resnet50_input.bin");
  array input_nhwc_f32 = input_reader.next();
  array input = astype(input_nhwc_f32, float16);

  if (!sustained_mode.empty()) {
    // Materialize input/weights (currently lazy AsType nodes) before
    // tracing: compile_fuse_ane()'s cross-queue-race guard in
    // mlx/compile.cpp only treats a leaf as safe to fuse if it has no
    // primitive (or is another CompiledANE's output); a lazy AsType would
    // otherwise block the whole region from fusing and silently make the
    // "ane" mode measure the plain Metal path instead.
    auto warm_cpu = forward(input, weights, Device::cpu);
    eval(warm_cpu);
    if (sustained_mode == "layers") {
      run_layer_by_layer(input, weights);
    } else {
      run_sustained(sustained_mode, input, weights, sustained_seconds);
    }
    return 0;
  }

  BlobReader ref_reader(dir + "/resnet50_torch_ref.bin");
  array torch_ref = ref_reader.next();

  auto cpu_out = forward(input, weights, Device::cpu);
  eval(cpu_out);

  if (std::getenv("ANE_DEBUG_STAGES")) {
    auto cpu_stages = forward_with_stages(input, weights, Device::cpu);
    eval(cpu_stages);
    std::function<std::vector<array>(const std::vector<array>&)> stage_fun =
        [&weights](const std::vector<array>& inputs) {
          return forward_with_stages(inputs[0], weights, Device::gpu);
        };
    auto compiled_stages = compile(stage_fun);
    auto ane_stages = compiled_stages({input});
    eval(ane_stages);
    for (size_t i = 0; i < cpu_stages.size(); i++) {
      fprintf(
          stderr, "[debug] stage %zu: ane_max_abs_diff=%.6f\n", i,
          max_abs_diff(cpu_stages[i], ane_stages[i]));
    }

    // Same real weights/input for block0's conv1, but as a fresh, isolated
    // single op -- not part of the big fused tape. If this passes, the bug
    // is about something in the full-graph context, not the data values.
    array stage1_input = cpu_stages[1]; // post-maxpool, verified correct
    auto& blk0 = weights.blocks[0];
    std::function<std::vector<array>(const std::vector<array>&)>
        isolated_fun = [&blk0](const std::vector<array>& inputs) {
          auto y = conv2d(
              inputs[0], blk0.w1, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          y = add(y, blk0.b1);
          y = maximum(y, array(0.0f, float16));
          return std::vector<array>{y};
        };
    auto isolated_ref = [&]() {
      auto y = conv2d(
          stage1_input, blk0.w1, {1, 1}, {0, 0}, {1, 1}, 1, Device::cpu);
      y = add(y, blk0.b1);
      return maximum(y, array(0.0f, float16));
    }();
    eval(isolated_ref);
    auto isolated_compiled = compile(isolated_fun);
    auto isolated_ane = isolated_compiled({stage1_input})[0];
    eval(isolated_ane);
    fprintf(
        stderr, "[debug] isolated block0-conv1 (real data): max_abs_diff=%.6f\n",
        max_abs_diff(isolated_ref, isolated_ane));

    // Same as isolated_fun, but with the REST of block0 also present and
    // traced/fused (conv2, conv3, downsample, residual) -- only reading
    // out conv1's own intermediate, same as above. If conv1's output is
    // wrong here but right in isolated_fun, downstream fusions are somehow
    // corrupting this one even though nothing should cause that.
    std::function<std::vector<array>(const std::vector<array>&)>
        shadowed_fun = [&blk0](const std::vector<array>& inputs) {
          auto x = inputs[0];
          auto y = conv2d(x, blk0.w1, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          y = add(y, blk0.b1);
          y = maximum(y, array(0.0f, float16));
          array stage2_output = y; // what we'll actually return

          auto y2 = conv2d(y, blk0.w2, {1, 1}, {1, 1}, {1, 1}, 1, Device::gpu);
          y2 = add(y2, blk0.b2);
          y2 = maximum(y2, array(0.0f, float16));
          auto y3 = conv2d(y2, blk0.w3, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          y3 = add(y3, blk0.b3);
          auto shortcut =
              conv2d(x, blk0.wd, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          shortcut = add(shortcut, blk0.bd);
          auto block_out = maximum(add(y3, shortcut), array(0.0f, float16));
          // block_out is computed and traced but NOT returned -- only
          // stage2_output is.
          (void)block_out;
          return std::vector<array>{stage2_output};
        };
    auto shadowed_compiled = compile(shadowed_fun);
    auto shadowed_ane = shadowed_compiled({stage1_input})[0];
    eval(shadowed_ane);
    fprintf(
        stderr,
        "[debug] shadowed block0-conv1 (rest of block0 also traced): max_abs_diff=%.6f\n",
        max_abs_diff(isolated_ref, shadowed_ane));

    // Full block0, real data, but ONLY block0 -- returning the block's
    // final (post-residual) output, compared against ITS OWN fresh CPU
    // reference (not the full-network cpu_stages).
    std::function<std::vector<array>(const std::vector<array>&)>
        block0_only_fun = [&blk0](const std::vector<array>& inputs) {
          auto x = inputs[0];
          auto y = conv2d(x, blk0.w1, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          y = add(y, blk0.b1);
          y = maximum(y, array(0.0f, float16));
          y = conv2d(y, blk0.w2, {1, 1}, {1, 1}, {1, 1}, 1, Device::gpu);
          y = add(y, blk0.b2);
          y = maximum(y, array(0.0f, float16));
          y = conv2d(y, blk0.w3, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          y = add(y, blk0.b3);
          auto shortcut =
              conv2d(x, blk0.wd, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          shortcut = add(shortcut, blk0.bd);
          return std::vector<array>{
              maximum(add(y, shortcut), array(0.0f, float16))};
        };
    auto block0_only_ref = [&]() {
      auto y = conv2d(
          stage1_input, blk0.w1, {1, 1}, {0, 0}, {1, 1}, 1, Device::cpu);
      y = add(y, blk0.b1);
      y = maximum(y, array(0.0f, float16));
      y = conv2d(y, blk0.w2, {1, 1}, {1, 1}, {1, 1}, 1, Device::cpu);
      y = add(y, blk0.b2);
      y = maximum(y, array(0.0f, float16));
      y = conv2d(y, blk0.w3, {1, 1}, {0, 0}, {1, 1}, 1, Device::cpu);
      y = add(y, blk0.b3);
      auto shortcut = conv2d(
          stage1_input, blk0.wd, {1, 1}, {0, 0}, {1, 1}, 1, Device::cpu);
      shortcut = add(shortcut, blk0.bd);
      return maximum(add(y, shortcut), array(0.0f, float16));
    }();
    eval(block0_only_ref);
    auto block0_only_compiled = compile(block0_only_fun);
    auto block0_only_ane = block0_only_compiled({stage1_input})[0];
    eval(block0_only_ane);
    fprintf(
        stderr,
        "[debug] block0-only, real data, final output: max_abs_diff=%.6f\n",
        max_abs_diff(block0_only_ref, block0_only_ane));

    // Same block0-only computation, but starting from the raw image (stem +
    // maxpool also traced/fused in the same compile() call), instead of a
    // precomputed stage1_input constant.
    std::function<std::vector<array>(const std::vector<array>&)>
        stem_block0_fun = [&weights, &blk0](const std::vector<array>& inputs) {
          auto x = conv_bias_relu(
              inputs[0], weights.conv1_w, weights.conv1_b, {2, 2}, {3, 3},
              Device::gpu, true);
          x = max_pool_3x3_s2(x);
          auto y = conv2d(x, blk0.w1, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          y = add(y, blk0.b1);
          y = maximum(y, array(0.0f, float16));
          y = conv2d(y, blk0.w2, {1, 1}, {1, 1}, {1, 1}, 1, Device::gpu);
          y = add(y, blk0.b2);
          y = maximum(y, array(0.0f, float16));
          y = conv2d(y, blk0.w3, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          y = add(y, blk0.b3);
          auto shortcut =
              conv2d(x, blk0.wd, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
          shortcut = add(shortcut, blk0.bd);
          return std::vector<array>{
              maximum(add(y, shortcut), array(0.0f, float16))};
        };
    auto stem_block0_compiled = compile(stem_block0_fun);
    auto stem_block0_ane = stem_block0_compiled({input})[0];
    eval(stem_block0_ane);
    fprintf(
        stderr, "[debug] stem+block0 (from raw image): max_abs_diff=%.6f\n",
        max_abs_diff(block0_only_ref, stem_block0_ane));

    // Stem + maxpool alone, no block0 at all.
    std::function<std::vector<array>(const std::vector<array>&)>
        stem_only_fun = [&weights](const std::vector<array>& inputs) {
          auto x = conv_bias_relu(
              inputs[0], weights.conv1_w, weights.conv1_b, {2, 2}, {3, 3},
              Device::gpu, true);
          x = max_pool_3x3_s2(x);
          return std::vector<array>{x};
        };
    auto stem_only_compiled = compile(stem_only_fun);
    auto stem_only_ane = stem_only_compiled({input})[0];
    eval(stem_only_ane);
    fprintf(
        stderr, "[debug] stem+maxpool only: max_abs_diff=%.6f\n",
        max_abs_diff(stage1_input, stem_only_ane));

    // Same as stem_block0_fun, but as TWO separate compile()+eval() calls
    // with a full sync in between, instead of one compiled tape. If this
    // fixes it, the bug is the known cross-queue sync gap (maxpool's output,
    // on mlx's regular queue, not actually committed/finished before
    // block0's ANE ops read it via the separate ane_command_queue).
    std::function<std::vector<array>(const std::vector<array>&)>
        stem_split_fun = [&weights](const std::vector<array>& inputs) {
          auto x = conv_bias_relu(
              inputs[0], weights.conv1_w, weights.conv1_b, {2, 2}, {3, 3},
              Device::gpu, true);
          x = max_pool_3x3_s2(x);
          return std::vector<array>{x};
        };
    auto stem_split_compiled = compile(stem_split_fun);
    auto stem_split_out = stem_split_compiled({input})[0];
    eval(stem_split_out); // full sync checkpoint
    auto block0_split_compiled = compile(block0_only_fun);
    auto block0_split_ane = block0_split_compiled({stem_split_out})[0];
    eval(block0_split_ane);
    fprintf(
        stderr,
        "[debug] stem+block0, split with eval() checkpoint: max_abs_diff=%.6f\n",
        max_abs_diff(block0_only_ref, block0_split_ane));

    for (int n = 1; n <= 16; n++) {
      auto cpu_n = forward_n_blocks(input, weights, Device::cpu, n);
      eval(cpu_n);
      std::function<std::vector<array>(const std::vector<array>&)> fun_n =
          [&weights, n](const std::vector<array>& inputs) {
            return std::vector<array>{
                forward_n_blocks(inputs[0], weights, Device::gpu, n)};
          };
      auto compiled_n = compile(fun_n);
      auto ane_n = compiled_n({input})[0];
      eval(ane_n);
      fprintf(
          stderr, "[debug] forward_n_blocks(%d): max_abs_diff=%.6f\n", n,
          max_abs_diff(cpu_n, ane_n));
    }

    // Isolate block index 13 (global, 0-based -- the 14th block, first of
    // stage3) specifically, since forward_n_blocks(13) is fine but (14)
    // isn't: narrow which sub-step (conv1/conv2/conv3/downsample/residual)
    // introduces the jump.
    {
      auto& blk = weights.blocks[13];
      auto stage_in_cpu = forward_n_blocks(input, weights, Device::cpu, 13);
      eval(stage_in_cpu);

      auto conv1_cpu = conv2d(stage_in_cpu, blk.w1, {1, 1}, {0, 0}, {1, 1}, 1, Device::cpu);
      conv1_cpu = add(conv1_cpu, blk.b1);
      conv1_cpu = maximum(conv1_cpu, array(0.0f, float16));
      eval(conv1_cpu);

      auto conv2_cpu = conv2d(conv1_cpu, blk.w2, {2, 2}, {1, 1}, {1, 1}, 1, Device::cpu);
      conv2_cpu = add(conv2_cpu, blk.b2);
      conv2_cpu = maximum(conv2_cpu, array(0.0f, float16));
      eval(conv2_cpu);

      auto conv3_cpu = conv2d(conv2_cpu, blk.w3, {1, 1}, {0, 0}, {1, 1}, 1, Device::cpu);
      conv3_cpu = add(conv3_cpu, blk.b3);
      eval(conv3_cpu);

      auto ds_cpu = conv2d(stage_in_cpu, blk.wd, {2, 2}, {0, 0}, {1, 1}, 1, Device::cpu);
      ds_cpu = add(ds_cpu, blk.bd);
      eval(ds_cpu);

      std::function<std::vector<array>(const std::vector<array>&)> conv1_fun =
          [&blk](const std::vector<array>& inputs) {
            auto y = conv2d(inputs[0], blk.w1, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
            y = add(y, blk.b1);
            y = maximum(y, array(0.0f, float16));
            return std::vector<array>{y};
          };
      auto conv1_ane = compile(conv1_fun)({stage_in_cpu})[0];
      eval(conv1_ane);
      fprintf(stderr, "[debug block13] conv1: max_abs_diff=%.6f\n",
              max_abs_diff(conv1_cpu, conv1_ane));

      std::function<std::vector<array>(const std::vector<array>&)> conv2_fun =
          [&blk](const std::vector<array>& inputs) {
            auto y = conv2d(inputs[0], blk.w2, {2, 2}, {1, 1}, {1, 1}, 1, Device::gpu);
            y = add(y, blk.b2);
            y = maximum(y, array(0.0f, float16));
            return std::vector<array>{y};
          };
      auto conv2_ane = compile(conv2_fun)({conv1_cpu})[0];
      eval(conv2_ane);
      fprintf(stderr, "[debug block13] conv2 (3x3 stride2): max_abs_diff=%.6f\n",
              max_abs_diff(conv2_cpu, conv2_ane));

      std::function<std::vector<array>(const std::vector<array>&)> conv3_fun =
          [&blk](const std::vector<array>& inputs) {
            auto y = conv2d(inputs[0], blk.w3, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
            y = add(y, blk.b3);
            return std::vector<array>{y};
          };
      auto conv3_ane = compile(conv3_fun)({conv2_cpu})[0];
      eval(conv3_ane);
      fprintf(stderr, "[debug block13] conv3: max_abs_diff=%.6f\n",
              max_abs_diff(conv3_cpu, conv3_ane));

      std::function<std::vector<array>(const std::vector<array>&)> ds_fun =
          [&blk](const std::vector<array>& inputs) {
            auto y = conv2d(inputs[0], blk.wd, {2, 2}, {0, 0}, {1, 1}, 1, Device::gpu);
            y = add(y, blk.bd);
            return std::vector<array>{y};
          };
      auto ds_ane = compile(ds_fun)({stage_in_cpu})[0];
      eval(ds_ane);
      fprintf(stderr, "[debug block13] downsample (1x1 stride2): max_abs_diff=%.6f\n",
              max_abs_diff(ds_cpu, ds_ane));

      auto block13_cpu = maximum(add(conv3_cpu, ds_cpu), array(0.0f, float16));
      eval(block13_cpu);
      std::function<std::vector<array>(const std::vector<array>&)> block13_fun =
          [&blk](const std::vector<array>& inputs) {
            auto x = inputs[0];
            auto y = conv2d(x, blk.w1, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
            y = add(y, blk.b1);
            y = maximum(y, array(0.0f, float16));
            y = conv2d(y, blk.w2, {2, 2}, {1, 1}, {1, 1}, 1, Device::gpu);
            y = add(y, blk.b2);
            y = maximum(y, array(0.0f, float16));
            y = conv2d(y, blk.w3, {1, 1}, {0, 0}, {1, 1}, 1, Device::gpu);
            y = add(y, blk.b3);
            auto shortcut = conv2d(x, blk.wd, {2, 2}, {0, 0}, {1, 1}, 1, Device::gpu);
            shortcut = add(shortcut, blk.bd);
            return std::vector<array>{maximum(add(y, shortcut), array(0.0f, float16))};
          };
      auto block13_ane = compile(block13_fun)({stage_in_cpu})[0];
      eval(block13_ane);
      fprintf(stderr, "[debug block13] whole block fused: max_abs_diff=%.6f\n",
              max_abs_diff(block13_cpu, block13_ane));
    }
  }

  std::function<std::vector<array>(const std::vector<array>&)> fun =
      [&weights](const std::vector<array>& inputs) {
        return std::vector<array>{forward(inputs[0], weights, Device::gpu)};
      };
  auto compiled = compile(fun);
  auto ane_out = compiled({input})[0];
  eval(ane_out);

  printf("torch ref shape: %s\n", torch_ref.shape().empty() ? "?" : "ok");
  printf(
      "mlx cpu shape matches torch ref: %d\n",
      cpu_out.shape() == torch_ref.shape());
  printf("torch ref max_abs: %.6f\n", max_abs(torch_ref));
  printf(
      "mlx-cpu vs torch   max_abs_diff: %.6f\n",
      max_abs_diff(cpu_out, torch_ref));
  printf(
      "mlx-ane vs torch   max_abs_diff: %.6f\n",
      max_abs_diff(ane_out, torch_ref));
  printf(
      "mlx-ane vs mlx-cpu max_abs_diff: %.6f\n",
      max_abs_diff(ane_out, cpu_out));

  // Timing: all-Metal eager vs compiled-with-ANE-fuse, real weights.
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

  double metal_ms =
      time_ms([&]() { return forward(input, weights, Device::gpu); }, 2, 10);
  auto compiled2 = compile(fun);
  double ane_ms =
      time_ms([&]() { return compiled2({input})[0]; }, 2, 10);

  printf("Metal forward:      %.2f ms/iter\n", metal_ms);
  printf("CompiledANE forward: %.2f ms/iter\n", ane_ms);
  printf("ratio (metal/ane): %.2fx\n", metal_ms / ane_ms);

  double macs = compute_total_macs(
      weights, input.shape(1), input.shape(2), 16);
  double gflops = 2.0 * macs / 1e9; // multiply + add, counted separately
  printf("forward pass: %.3f GMACs (%.3f GFLOPs)\n", macs / 1e9, gflops);
  printf("Metal achieved:      %.2f GFLOPS\n", gflops / (metal_ms / 1000.0));
  printf("CompiledANE achieved: %.2f GFLOPS\n", gflops / (ane_ms / 1000.0));

  return 0;
}
