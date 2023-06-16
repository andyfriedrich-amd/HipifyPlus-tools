// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/miopen_common.h"

namespace onnxruntime {
namespace rocm {

class MiopenLRNDescriptor final {
 public:
  MiopenLRNDescriptor();
  ~MiopenLRNDescriptor();
  Status Set(uint32_t N, double alpha, double beta, double K);
  operator miopenLRNDescriptor_t() const { return desc_; }

 private:
  miopenLRNDescriptor_t desc_;
};

template <typename T>
class LRN : public RocmKernel {
 public:
  LRN(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  MiopenLRNDescriptor norm_desc_;
};

}  // namespace rocm
}  // namespace onnxruntime
