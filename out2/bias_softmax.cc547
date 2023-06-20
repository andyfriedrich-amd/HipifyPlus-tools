// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/math/bias_softmax.h"

#include "core/providers/rocm/rocm_common.h"
#include "contrib_ops/rocm/math/bias_softmax_impl.h"

using namespace onnxruntime;
using namespace onnxruntime::rocm;
using namespace onnxruntime::contrib::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

namespace {

template <typename T>
struct DispatchBiasSoftmaxImpl {
  Status operator()(hipStream_t stream, miopenHandle_t miopen_handle, Tensor* Y, const Tensor* X, const Tensor* B,
                    int element_count, int batch_count, bool is_inner_broadcast, int bias_broadcast_size) {
    typedef typename ToHipType<T>::MappedType HipT;
    HipT* output_data = reinterpret_cast<HipT*>(Y->template MutableData<T>());
    const HipT* input_data = reinterpret_cast<const HipT*>(X->template Data<T>());
    const HipT* bias_data = reinterpret_cast<const HipT*>(B->template Data<T>());
    return BiasSoftmaxImpl<HipT>(stream, miopen_handle, output_data, input_data, bias_data, element_count, batch_count,
                                  is_inner_broadcast, bias_broadcast_size);
  }
};

}  // namespace

// MIOpen doesn't support double so ROCm kernel doesn't have double support for now.
#ifdef USE_ROCM
#define BIAS_SOFTMAX_TYPES float, MLFloat16
#else
#define BIAS_SOFTMAX_TYPES float, MLFloat16, double
#endif

ONNX_OPERATOR_KERNEL_EX(
    BiasSoftmax, kMSDomain, 1, kRocmExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<BIAS_SOFTMAX_TYPES>()), BiasSoftmax);

Status BiasSoftmax::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const TensorShape& X_shape = X->Shape();
  const TensorShape& B_shape = B->Shape();
  Tensor* Y = ctx->Output(0, X_shape);

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, X_shape.NumDimensions()));
  const int batch_count = static_cast<int>(X_shape.SizeToDimension(axis));
  const int element_count = static_cast<int>(X_shape.SizeFromDimension(axis));
  int bias_broadcast_size = static_cast<int>(B_shape.Size() / element_count);
  if (is_inner_broadcast_) bias_broadcast_size = batch_count / bias_broadcast_size;
  utils::MLTypeCallDispatcher<BIAS_SOFTMAX_TYPES> t_disp(X->GetElementType());
  return t_disp.InvokeRet<Status, DispatchBiasSoftmaxImpl>(Stream(ctx), GetMiopenHandle(ctx), Y, X, B, element_count, batch_count,
                                                           is_inner_broadcast_, bias_broadcast_size);
}

#undef BIAS_SOFTMAX_TYPES

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
