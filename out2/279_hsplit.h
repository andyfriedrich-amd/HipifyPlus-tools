// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/split.h"
#include "orttraining/training_ops/cpu/tensor/split.h"

namespace onnxruntime {
namespace rocm {

class SplitTraining final : public RocmKernel, public SplitBase {
 public:
  // ONNX Split from opset 13. no support for uneven splits that was added in opset 18.
  SplitTraining(const OpKernelInfo& info) : RocmKernel(info), SplitBase(info, 13) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
