// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/padbase.h"

using onnxruntime::PadBase;

namespace onnxruntime {
namespace rocm {

template <typename T>
class Pad final : public PadBase, public RocmKernel {
 public:
  Pad(const OpKernelInfo& info) : PadBase(info), RocmKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
