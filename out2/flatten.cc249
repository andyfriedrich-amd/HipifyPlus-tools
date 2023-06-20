// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flatten.h"

namespace onnxruntime {
namespace rocm {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    1, 8,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Flatten);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    9, 10,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Flatten);

// explicitly support negative axis
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    11, 12,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Flatten);

ONNX_OPERATOR_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    13,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Flatten);

Status Flatten::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();

  auto axis = axis_;
  // Valid axis range is [-rank, rank] instead of [-rank, rank-1], add additional check to only handle neg axis case.
  if (axis < 0) {
    axis = HandleNegativeAxis(axis, X_shape.NumDimensions());  // handle negative and enforce axis is valid
  }

  ORT_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.NumDimensions()) >= axis, "The rank of input tensor must be >= axis");

  Tensor* Y = ctx->Output(0, {X_shape.SizeToDimension(axis), X_shape.SizeFromDimension(axis)});
  // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
  const void* source = X->DataRaw();
  void* target = Y->MutableDataRaw();
  if (target != source) {
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(target, source, X_shape.Size() * X->DataType()->Size(),
                                         hipMemcpyDeviceToDevice, Stream(ctx)));
  }

  return Status::OK();
}

}  // namespace rocm
}  // namespace onnxruntime
