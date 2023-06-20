// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/upsamplebase.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
class Upsample : public UpsampleBase, public RocmKernel {
 public:
  Upsample(const OpKernelInfo& info) : UpsampleBase(info), RocmKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
  Status BaseCompute(OpKernelContext* context, const std::vector<float>& roi, const std::vector<float>& scales,
                     const gsl::span<const int64_t>& output_dims) const;
};

}  // namespace rocm
}  // namespace onnxruntime
