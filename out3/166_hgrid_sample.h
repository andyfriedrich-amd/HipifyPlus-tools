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
class GridSample final : public RocmKernel {
 public:
  explicit GridSample(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t mode_i_;          // 0: bilinear (default), 1: nearest 2: bicubic
  int64_t padding_mode_i_;  // 0:'zeros', 1: 'border', 2:'reflection'
  int64_t align_corners_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
