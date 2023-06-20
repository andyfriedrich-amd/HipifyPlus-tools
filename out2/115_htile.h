// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/tile.h"

namespace onnxruntime {
namespace rocm {

struct Tile final : RocmKernel {
  explicit Tile(const OpKernelInfo& info) : RocmKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};
}  // namespace rocm
}  // namespace onnxruntime
