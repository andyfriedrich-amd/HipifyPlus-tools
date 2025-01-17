

#include "core/providers/rocm/rocm_common.h"
#include "contrib_ops/rocm/diffusion/bias_split_gelu.h"
#include "contrib_ops/rocm/diffusion/bias_split_gelu_impl.h"
namespace onnxruntime {
namespace contrib {
namespace rocm {
#define REGISTER_KERNEL_TYPED(T)                   ONNX_OPERATOR_TYPED_KERNEL_EX(                     BiasSplitGelu, kMSDomain, 1, T, kRocmExecutionProvider, (*KernelDefBuilder::Create())                     .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), BiasSplitGelu<T>);
REGISTER_KERNEL_TYPED(MLFloat16);
REGISTER_KERNEL_TYPED(float);
using namespace ONNX_NAMESPACE;
template <typename T>
BiasSplitGelu<T>::BiasSplitGelu(const OpKernelInfo& op_info) : RocmKernel(op_info) {
}
template <typename T>
Status BiasSplitGelu<T>::ComputeInternal(OpKernelContext* context) const {
 const Tensor* input = context->Input<Tensor>(0);
 const auto& input_dims = input->Shape().GetDims();
 if (input_dims.size() != 3) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input is expected to have 3 dimensions, got ", input_dims.size());
 }
 if (input_dims[2] != 2560 && input_dims[2] != 5120 && input_dims[2] != 10240) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "hidden size should be 2560, 5120 or 10240, got ", input_dims[2]);
 }
 const Tensor* bias = context->Input<Tensor>(1);
 const auto& bias_dims = bias->Shape().GetDims();
 if (bias_dims.size() != 1) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "bias is expected to have 1 dimensions, got ", bias_dims.size());
 }
 if (bias_dims[0] != input_dims[2]) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "last dimension of input and bias are not the same");
 }
 TensorShapeVector output_shape = input->Shape().AsShapeVector();
 output_shape[2] = input_dims[2] / 2;
 Tensor* output = context->Output(0, output_shape);
 typedef typename ToHipType<T>::MappedType HipT;
 const int32_t grid_size = static_cast<int32_t>(input_dims[0] * input_dims[1]);
 const int32_t half_hidden_size = static_cast<int32_t>(input_dims[2] / 2);
 LaunchBiasSplitGeluKernel<HipT>(Stream(context), grid_size, half_hidden_size, reinterpret_cast<const HipT*>(input->Data<T>()), reinterpret_cast<const HipT*>(bias->Data<T>()), reinterpret_cast<HipT*>(output->MutableData<T>()));
 HIP_RETURN_IF_ERROR(hipPeekAtLastError());
 return Status::OK();
}
} 
} 
} 