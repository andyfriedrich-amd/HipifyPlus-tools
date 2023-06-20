// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/math/matmul.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(op_name, T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      op_name,                                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      onnxruntime::rocm::MatMul<T>);

// TransposeMatMul is kept here for backward compatibility
REGISTER_KERNEL_TYPED(TransposeMatMul, float)
REGISTER_KERNEL_TYPED(TransposeMatMul, double)
REGISTER_KERNEL_TYPED(TransposeMatMul, MLFloat16)
REGISTER_KERNEL_TYPED(TransposeMatMul, BFloat16)

REGISTER_KERNEL_TYPED(FusedMatMul, float)
REGISTER_KERNEL_TYPED(FusedMatMul, double)
REGISTER_KERNEL_TYPED(FusedMatMul, MLFloat16)
REGISTER_KERNEL_TYPED(FusedMatMul, BFloat16)

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
