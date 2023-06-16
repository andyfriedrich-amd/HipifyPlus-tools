// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <typename SrcT>
class MixedPrecisionScale final : public RocmKernel {
 public:
  MixedPrecisionScale(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
  size_t bytes_per_output_elem_;
  bool fuse_outputs_;
};

}  // namespace rocm
}  // namespace onnxruntime
