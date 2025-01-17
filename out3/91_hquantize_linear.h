// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <class T, class U>
class QuantizeLinear final : public RocmKernel {
 public:
  QuantizeLinear(const OpKernelInfo& info) : RocmKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
    if (!info.GetAttr<int64_t>("saturate", &saturate_).IsOK()) {
      saturate_ = 1;
    }
  }

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t axis_;
  int64_t saturate_;
};

template <class T, class U>
class DequantizeLinear final : public RocmKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : RocmKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
  }

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t axis_;
};

}  // namespace rocm
}  // namespace onnxruntime
