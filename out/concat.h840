// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/concatbase.h"

namespace onnxruntime {
namespace rocm {

class ConcatTraining final : public RocmKernel, public ConcatBase {
 public:
  ConcatTraining(const OpKernelInfo& info) : RocmKernel(info), ConcatBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
