// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/activation/activations_impl.h"

namespace onnxruntime {
namespace rocm {

typedef onnxruntime::rocm::CtxNull CtxGeluGrad;
typedef onnxruntime::rocm::CtxNull CtxFastGeluGrad;
typedef onnxruntime::rocm::CtxNull CtxReluGrad;
typedef onnxruntime::rocm::CtxNull CtxSigmoidGrad;
typedef onnxruntime::rocm::CtxAlpha CtxQuickGeluGrad;
typedef onnxruntime::rocm::CtxNull CtxTanhGrad;

#define ACTIVATION_GRAD_OPS()            \
  ACTIVATION_GRAD_OP_NAME(GeluGrad)      \
  ACTIVATION_GRAD_OP_NAME(FastGeluGrad)  \
  ACTIVATION_GRAD_OP_NAME(ReluGrad)      \
  ACTIVATION_GRAD_OP_NAME(SigmoidGrad)   \
  ACTIVATION_GRAD_OP_NAME(QuickGeluGrad) \
  ACTIVATION_GRAD_OP_NAME(TanhGrad)

#define BINARY_ELEMENTWISE_IMPL_DECLARATION(name) \
  template <typename T>                           \
  void Impl_##name(hipStream_t stream,           \
                   const T* lhs_data,             \
                   const T* rhs_data,             \
                   T* output_data,                \
                   const Ctx##name* func_ctx,     \
                   size_t count)

#define ACTIVATION_GRAD_OP_NAME(name) BINARY_ELEMENTWISE_IMPL_DECLARATION(name);
ACTIVATION_GRAD_OPS()
#undef ACTIVATION_GRAD_OP_NAME

}  // namespace rocm
}  // namespace onnxruntime
