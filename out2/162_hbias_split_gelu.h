// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class BiasSplitGelu final : public RocmKernel {
 public:
  BiasSplitGelu(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
