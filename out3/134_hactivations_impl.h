// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/activation/activations_impl.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

typedef onnxruntime::rocm::CtxAlphaBeta CtxAffine;
typedef onnxruntime::rocm::CtxAlphaBeta CtxParametricSoftplus;
typedef onnxruntime::rocm::CtxAlphaBeta CtxScaledTanh;
typedef onnxruntime::rocm::CtxNull CtxGelu;
typedef onnxruntime::rocm::CtxAlpha CtxQuickGelu;

#define UNARY_CONTRIB_ACTIVATION_OPS()         \
  UNARY_ACTIVATION_OP_NAME(ScaledTanh)         \
  UNARY_ACTIVATION_OP_NAME(Affine)             \
  UNARY_ACTIVATION_OP_NAME(ParametricSoftplus) \
  UNARY_ACTIVATION_OP_NAME(Gelu)               \
  UNARY_ACTIVATION_OP_NAME(QuickGelu)

#define UNARY_ACTIVATION_OP_NAME(name) UNARY_ACTIVATION_IMPL_DECLARATION(name);
UNARY_CONTRIB_ACTIVATION_OPS()
#undef UNARY_ACTIVATION_OP_NAME

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
