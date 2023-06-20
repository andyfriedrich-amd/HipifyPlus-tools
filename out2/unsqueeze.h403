// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/unsqueeze.h"

namespace onnxruntime {
namespace rocm {

class Unsqueeze final : public UnsqueezeBase, public RocmKernel {
 public:
  Unsqueeze(const OpKernelInfo& info) : UnsqueezeBase(info), RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
