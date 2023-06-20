// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/scatter_nd.h"

namespace onnxruntime {
namespace rocm {

class ScatterND final : public RocmKernel {
 public:
  explicit ScatterND(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
