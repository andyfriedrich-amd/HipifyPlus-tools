

#include "core/providers/rocm/rocm_common.h"
#include "contrib_ops/rocm/diffusion/bias_add.h"
#include "contrib_ops/rocm/diffusion/bias_add_impl.h"
namespace onnxruntime {
namespace contrib {
namespace rocm {
#define REGISTER_KERNEL_TYPED(T)                   ONNX_OPERATOR_TYPED_KERNEL_EX(                     BiasAdd, kMSDomain, 1, T, kRocmExecutionProvider, (*KernelDefBuilder::Create())                     .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), BiasAdd<T>);
REGISTER_KERNEL_TYPED(MLFloat16);
REGISTER_KERNEL_TYPED(float);
using namespace ONNX_NAMESPACE;
template <typename T>
BiasAdd<T>::BiasAdd(const OpKernelInfo& op_info) : RocmKernel(op_info) {
}
template <typename T>
Status BiasAdd<T>::ComputeInternal(OpKernelContext* context) const {
 
 
 
 
 const Tensor* input = context->Input<Tensor>(0);
 const auto& input_dims = input->Shape().GetDims();
 if (input_dims.size() != 3) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The input is expected to have 3 dimensions, got ", input_dims.size());
 }
 if (input_dims[2] != 320 && input_dims[2] != 640 && input_dims[2] != 1280) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Number of channels should be 320, 640 or 1280, got ", input_dims[2]);
 }
 const Tensor* bias = context->Input<Tensor>(1);
 const auto& bias_dims = bias->Shape().GetDims();
 if (bias_dims.size() != 1) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The bias is expected to have 1 dimensions, got ", bias_dims.size());
 }
 if (bias_dims[0] != input_dims[2]) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Number of channels in the last dimension of input and bias are not the same");
 }
 const Tensor* skip = context->Input<Tensor>(2);
 if (skip->Shape() != input->Shape()) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Shape of input and skip (residual) shall be the same");
 }
 Tensor* output = context->Output(0, input->Shape());
 typedef typename ToHipType<T>::MappedType HipT;
 const int32_t grid_size = static_cast<int32_t>(input_dims[0] * input_dims[1]);
 LaunchBiasAddKernel<HipT>(Stream(context), grid_size, static_cast<int32_t>(input_dims[2]), reinterpret_cast<const HipT*>(input->Data<T>()), reinterpret_cast<const HipT*>(bias->Data<T>()), reinterpret_cast<const HipT*>(skip->Data<T>()), reinterpret_cast<HipT*>(output->MutableData<T>()));
 HIP_RETURN_IF_ERROR(hipPeekAtLastError());
 return Status::OK();
}
} 
} 
} 