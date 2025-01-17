

#include "core/providers/rocm/tensor/scatter_nd.h"
#include "core/providers/rocm/tensor/scatter_nd_impl.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/cpu/tensor/utils.h"
namespace onnxruntime {
namespace rocm {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterND, kOnnxDomain, 11, 12, kRocmExecutionProvider, (*KernelDefBuilder::Create())
                   .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                   .MayInplace(0, 0), ScatterND);
ONNX_OPERATOR_KERNEL_EX(ScatterND, kOnnxDomain, 13, kRocmExecutionProvider, (*KernelDefBuilder::Create())
              .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
              .MayInplace(0, 0), ScatterND);
Status ScatterND::ComputeInternal(OpKernelContext* context) const {
 const auto* input_tensor = context->Input<Tensor>(0);
 const auto* indices_tensor = context->Input<Tensor>(1);
 const auto* updates_tensor = context->Input<Tensor>(2);
 const auto& input_shape = input_tensor->Shape();
 const auto& indices_shape = indices_tensor->Shape();
 const auto& updates_shape = updates_tensor->Shape();
 
 ORT_RETURN_IF_ERROR(onnxruntime::ScatterND::ValidateShapes(input_shape, indices_shape, updates_shape));
 auto* output_tensor = context->Output(0, input_shape);
 const void* input_data = input_tensor->DataRaw();
 void* output_data = output_tensor->MutableDataRaw();
 size_t element_size = input_tensor->DataType()->Size();
 if (input_data != output_data) {
  
  HIP_RETURN_IF_ERROR(
    hipMemcpyAsync(output_data, input_data, input_tensor->SizeInBytes(), hipMemcpyDeviceToDevice, Stream(context)));
 }
 
 if (indices_shape.Size() == 0) {
  return Status::OK();
 }
 auto last_index_dimension = indices_shape[indices_shape.NumDimensions() - 1];
 
 
 
 TensorPitches input_strides(input_shape);
 std::vector<int64_t> element_counts_and_input_dims(last_index_dimension * 2, 0LL);
 for (int64_t i = 0; i < last_index_dimension; ++i) {
  element_counts_and_input_dims[i] = input_strides[i];
  element_counts_and_input_dims[i + last_index_dimension] = input_shape[i];
 }
 RocmAsyncBuffer<int64_t> element_counts_and_input_dims_gpu(this, element_counts_and_input_dims);
 ORT_RETURN_IF_ERROR(element_counts_and_input_dims_gpu.CopyToGpu(context->GetComputeStream()));
 ORT_RETURN_IF_ERROR(ScatterNDImpl(
   Stream(context), output_data, element_size, indices_shape.Size() / static_cast<size_t>(last_index_dimension), indices_tensor->Data<int64_t>(), last_index_dimension, element_counts_and_input_dims_gpu.GpuPtr(), updates_tensor->DataRaw(), input_shape.SizeFromDimension(last_index_dimension)));
 return Status::OK();
}
} 
} 