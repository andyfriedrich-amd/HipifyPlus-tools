// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/rocm/math/isfinite.h"
#include "orttraining/training_ops/rocm/math/isfinite_impl.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace rocm {

#define REGISTER_ISFINITE_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      IsFinite,                                                       \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kRocmExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()), \
      IsFiniteOp<T>);

template <typename TSrc>
Status IsFiniteOp<TSrc>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToHipType<TSrc>::MappedType HipTSrc;
  const Tensor& input = *context->Input<Tensor>(0);
  Tensor& output = *context->Output(0, input.Shape());
  IsFinite(
      Stream(context),
      reinterpret_cast<const HipTSrc*>(input.Data<TSrc>()),
      output.MutableData<bool>(), input.Shape().Size());

  return Status::OK();
}

REGISTER_ISFINITE_KERNEL_TYPED(MLFloat16)
REGISTER_ISFINITE_KERNEL_TYPED(float)
REGISTER_ISFINITE_KERNEL_TYPED(double)

}  // namespace rocm
}  // namespace onnxruntime
