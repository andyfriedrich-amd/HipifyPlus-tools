// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shrink.h"
#include "shrink_impl.h"
#include "core/providers/common.h"

using namespace std;
namespace onnxruntime {
namespace rocm {

#define SHRINK_REGISTER_KERNEL(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Shrink,                                                     \
      kOnnxDomain,                                                \
      9,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .MayInplace(0, 0)                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Shrink<T>);

template <typename T>
Status Shrink<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToHipType<T>::MappedType HipT;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const auto* x_data = reinterpret_cast<const HipT*>(X->Data<T>());
  const TensorShape& x_shape = X->Shape();
  const size_t x_size = static_cast<size_t>(x_shape.Size());

  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  auto* y_data = reinterpret_cast<HipT*>(Y->MutableData<T>());

  ShrinkImpl<HipT>(Stream(p_op_kernel_context), x_data, bias_, lambd_, y_data, x_size);

  return Status::OK();
}

SHRINK_REGISTER_KERNEL(float)
SHRINK_REGISTER_KERNEL(double)
SHRINK_REGISTER_KERNEL(MLFloat16)
SHRINK_REGISTER_KERNEL(uint8_t)
SHRINK_REGISTER_KERNEL(int8_t)
SHRINK_REGISTER_KERNEL(uint16_t)
SHRINK_REGISTER_KERNEL(int16_t)
SHRINK_REGISTER_KERNEL(uint32_t)
SHRINK_REGISTER_KERNEL(int32_t)
SHRINK_REGISTER_KERNEL(uint64_t)
SHRINK_REGISTER_KERNEL(int64_t)

}  // namespace rocm
}  // namespace onnxruntime
