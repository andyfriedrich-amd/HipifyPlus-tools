// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/object_detection/non_max_suppression.h"

namespace onnxruntime {
namespace rocm {

struct NonMaxSuppression final : public RocmKernel, public NonMaxSuppressionBase {
  explicit NonMaxSuppression(const OpKernelInfo& info) : RocmKernel(info), NonMaxSuppressionBase(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NonMaxSuppression);
};
}  // namespace rocm
}  // namespace onnxruntime
