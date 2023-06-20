// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

class GroupNorm final : public RocmKernel {
 public:
  GroupNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool use_swish_activation_;
  float epsilon_;
  int num_groups_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
