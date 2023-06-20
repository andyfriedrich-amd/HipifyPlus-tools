// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/concatbase.h"

namespace onnxruntime {
namespace rocm {

class Concat final : public RocmKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : RocmKernel(info), ConcatBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
