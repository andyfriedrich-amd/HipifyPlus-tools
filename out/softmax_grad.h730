// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

class SoftmaxGrad final : public RocmKernel {
 public:
  SoftmaxGrad(const OpKernelInfo& info) : RocmKernel{info} {
    const auto& op_type = info.node().OpType();
    is_since_opset_13_ = (op_type == "SoftmaxGrad_13" || op_type == "LogSoftmaxGrad_13");
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(is_since_opset_13_ ? -1 : 1));
    is_log_softmax_ = (op_type == "LogSoftmaxGrad" || op_type == "LogSoftmaxGrad_13");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  bool is_log_softmax_;
  bool is_since_opset_13_;
};

}  // namespace rocm
}  // namespace onnxruntime
