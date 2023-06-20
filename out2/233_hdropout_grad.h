// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <bool UseBitmask>
class DropoutGrad final : public RocmKernel {
 public:
  DropoutGrad(const OpKernelInfo& info) : RocmKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace rocm
}  // namespace onnxruntime
