// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <stdexcept>

#include "mlx/backend/metal/metal.h"
#include "mlx/primitives.h"

namespace mlx::core {

bool CompiledANE::is_equivalent(const Primitive& other) const {
  const CompiledANE& a_other = static_cast<const CompiledANE&>(other);
  return std::equal(
      tape_.begin(),
      tape_.end(),
      a_other.tape_.begin(),
      a_other.tape_.end(),
      [](const array& a1, const array& a2) {
        auto& p1 = a1.primitive();
        auto& p2 = a2.primitive();
        return typeid(p1) == typeid(p2) && p1.is_equivalent(p2);
      });
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
