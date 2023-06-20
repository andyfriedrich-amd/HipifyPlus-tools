// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

class Trilu final : public RocmKernel {
 public:
  Trilu(const OpKernelInfo& info) : RocmKernel(info), upper_(info.GetAttrOrDefault<int64_t>("upper", 1) >= 1) {
  }
  ~Trilu() = default;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool upper_;
};

}  // namespace rocm
}  // namespace onnxruntime