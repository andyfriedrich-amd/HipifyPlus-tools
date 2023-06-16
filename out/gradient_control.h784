// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
class ZeroGradient final : public RocmKernel {
 public:
  ZeroGradient(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename T_GRAD>
class InPlaceAccumulator final : public RocmKernel {
 public:
  InPlaceAccumulator(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename T_GRAD>
class InPlaceAccumulatorV2 final : public RocmKernel {
 public:
  InPlaceAccumulatorV2(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime