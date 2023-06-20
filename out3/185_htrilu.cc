// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/tensor/trilu.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

ONNX_OPERATOR_KERNEL_EX(
    Trilu,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    onnxruntime::rocm::Trilu);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
