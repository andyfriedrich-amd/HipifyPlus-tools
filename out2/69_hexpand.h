// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

class Expand final : public RocmKernel {
 public:
  Expand(const OpKernelInfo& info) : RocmKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

Status ComputeOutputShape(
    const std::string& node_name,
    const TensorShape& lhs_shape,
    const TensorShape& rhs_shape,
    TensorShape& out_shape);

}  // namespace rocm
}  // namespace onnxruntime
