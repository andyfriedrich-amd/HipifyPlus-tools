// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

class Compress final : public RocmKernel {
 public:
  Compress(const OpKernelInfo& info) : RocmKernel(info) {
    has_axis_ = info.GetAttr("axis", &axis_).IsOK();
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  bool has_axis_;
};

}  // namespace rocm
}  // namespace onnxruntime
