// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "orttraining/training_ops/cpu/optimizer/sgd/sgdbase.h"

namespace onnxruntime {
namespace rocm {

class SGDOptimizerV2 final : public RocmKernel, public contrib::SGDOptimizerV2Base {
 public:
  SGDOptimizerV2(const OpKernelInfo& info) : RocmKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
