// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sg.h"
#include "sg_impl.h"

#include "core/providers/rocm/reduction/reduction_functions.h"
#include "core/providers/rocm/math/binary_elementwise_ops.h"

namespace onnxruntime {
namespace rocm {

ONNX_OPERATOR_KERNEL_EX(
    SGDOptimizer,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(1, 0)  // Update weights in-place
        .Alias(2, 1)  // Update gradients in-place
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SGDOptimizer);

Status SGDOptimizer::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& W = *ctx->Input<Tensor>(1);
  const Tensor& G = *ctx->Input<Tensor>(2);
  Tensor* NW = ctx->Output(0, W.Shape());
  Tensor* NG = ctx->Output(1, G.Shape());

  ORT_ENFORCE(W.Shape() == G.Shape());

  SGDOptimizerImpl(
      Stream(ctx),
      ETA.template Data<float>(),
      W.template Data<float>(),
      G.template Data<float>(),
      NW != nullptr ? NW->template MutableData<float>() : nullptr,
      NG != nullptr ? NG->template MutableData<float>() : nullptr,
      W.Shape().Size());

  return Status::OK();
}

}  // namespace rocm
}  // namespace onnxruntime
