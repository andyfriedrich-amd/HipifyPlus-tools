// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
class All final : public RocmKernel {
 public:
  All(const OpKernelInfo& info) : RocmKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
