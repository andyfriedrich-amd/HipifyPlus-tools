// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "orttraining/training_ops/cpu/optimizer/adamw/adamwbase.h"

namespace onnxruntime {
namespace rocm {

class AdamWOptimizer final : public RocmKernel, public contrib::AdamWOptimizerBase {
 public:
  AdamWOptimizer(const OpKernelInfo& info) : RocmKernel(info), contrib::AdamWOptimizerBase(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
