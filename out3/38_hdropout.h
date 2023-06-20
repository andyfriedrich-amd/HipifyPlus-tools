// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/common.h"
#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace rocm {

template <bool UseBitmask>
class Dropout final : public RocmKernel {
 public:
  Dropout(const OpKernelInfo& info) : RocmKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace rocm
}  // namespace onnxruntime
