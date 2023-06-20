// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/object_detection/roialign.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
struct RoiAlign final : RocmKernel, RoiAlignBase {
  RoiAlign(const OpKernelInfo& info) : RocmKernel(info), RoiAlignBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoiAlign);
};
}  // namespace rocm
}  // namespace onnxruntime
