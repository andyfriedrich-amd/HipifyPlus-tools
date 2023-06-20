// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/squeeze.h"

namespace onnxruntime {
namespace rocm {

class Squeeze final : public SqueezeBase, public RocmKernel {
 public:
  Squeeze(const OpKernelInfo& info) : SqueezeBase(info), RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
