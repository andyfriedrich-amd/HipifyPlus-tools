// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/miopen_common.h"

namespace onnxruntime {
namespace rocm {

class WaitEvent final : public RocmKernel {
 public:
  WaitEvent(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime