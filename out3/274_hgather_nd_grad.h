// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/tensor/gather_nd.h"

namespace onnxruntime {
namespace rocm {

template <typename Tind>
class GatherNDGrad final : public GatherNDBase {
 public:
  GatherNDGrad(const OpKernelInfo& info) : GatherNDBase(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime