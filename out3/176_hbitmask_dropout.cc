// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/nn/dropout.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

ONNX_OPERATOR_KERNEL_EX(BitmaskDropout, kMSDomain, 1, kRocmExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("T3", DataTypeImpl::GetTensorType<onnxruntime::rocm::BitmaskElementType>())
                            .InputMemoryType(OrtMemTypeCPUInput, 1)
                            .InputMemoryType(OrtMemTypeCPUInput, 2),
                        onnxruntime::rocm::Dropout<true>);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
