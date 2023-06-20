// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/miopen_common.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
class InstanceNorm final : public RocmKernel {
 public:
  InstanceNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  double epsilon_;
};

}  // namespace rocm
}  // namespace onnxruntime
