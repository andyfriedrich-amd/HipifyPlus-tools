// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_TORCH_INTEROP

#pragma once
#include "core/providers/rocm/rocm_kernel.h"
#include "orttraining/training_ops/cpu/torch/torch_custom_function_kernel_base.h"

namespace onnxruntime {
namespace rocm {

// Pytorch's torch.autograd.Function.apply(...) wrapper.
class PythonOp final : public contrib::PythonOpBase, public RocmKernel {
 public:
  PythonOp(const OpKernelInfo& info) : contrib::PythonOpBase(info), RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// Pytorch's torch.autograd.Function.backward(...) wrapper.
class PythonOpGrad final : public contrib::PythonOpGradBase, public RocmKernel {
 public:
  PythonOpGrad(const OpKernelInfo& info) : contrib::PythonOpGradBase(info), RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime

#endif