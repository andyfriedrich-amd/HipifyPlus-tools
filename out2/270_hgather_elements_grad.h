// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

class GatherElementsGrad final : public RocmKernel {
 public:
  GatherElementsGrad(const OpKernelInfo& info) : RocmKernel(info) {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(0));
  }
  ~GatherElementsGrad() = default;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;

  int64_t axis_;
};

}  // namespace rocm
}  // namespace onnxruntime
