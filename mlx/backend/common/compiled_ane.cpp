// Copyright © 2026 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"
#include "mlx/primitives.h"

namespace mlx::core {

bool CompiledANE::is_equivalent(const Primitive& other) const {
  const CompiledANE& a_other = static_cast<const CompiledANE&>(other);
  return kernel_strides_ == a_other.kernel_strides_ &&
      padding_ == a_other.padding_;
}

std::vector<Shape> CompiledANE::output_shapes(
    const std::vector<array>& inputs) {
  // CompiledANE wraps exactly one Convolution with groups=1, unit kernel and
  // input dilation, and flip=false; see is_fusable_ane() in compile.cpp.
  return {Convolution::conv_out_shape(
      inputs[0].shape(),
      inputs[1].shape(),
      kernel_strides_,
      padding_,
      padding_,
      /* kernel_dilation */ std::vector<int>(kernel_strides_.size(), 1),
      /* input_dilation */ std::vector<int>(kernel_strides_.size(), 1))};
}

std::pair<std::vector<array>, std::vector<int>> CompiledANE::vmap(
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[CompiledANE] Cannot vmap primitive.");
}

std::vector<array> CompiledANE::jvp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[CompiledANE] Cannot jvp primitive.");
}

std::vector<array> CompiledANE::vjp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("[CompiledANE] Cannot vjp primitive.");
}

void CompiledANE::eval_cpu(const std::vector<array>&, array&) {
  throw std::runtime_error(
      "[CompiledANE] eval_cpu not supported; CompiledANE only runs on the "
      "ANE execution path.");
}

#ifndef MLX_BUILD_ANE
bool metal::ane_is_available() {
  return false;
}

void CompiledANE::eval_gpu(const std::vector<array>&, array&) {
  throw std::runtime_error(
      "[CompiledANE] ANE backend not built (set MLX_BUILD_ANE=ON and "
      "MLX_BUILD_METAL=ON).");
}
#endif

} // namespace mlx::core
