// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

class EyeLike final : public RocmKernel {
 public:
  EyeLike(const OpKernelInfo& info) : RocmKernel(info) {
    if (!info.GetAttr("k", &k_).IsOK()) {
      k_ = 0;
    }

    has_dtype_ = info.GetAttr("dtype", &dtype_).IsOK();
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool has_dtype_;
  int64_t dtype_;
  int64_t k_;
};

}  // namespace rocm
}  // namespace onnxruntime
