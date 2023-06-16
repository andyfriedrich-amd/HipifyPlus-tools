// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/rocm/optimizer/common.h"

namespace onnxruntime {
namespace rocm {

Status CopyIfNotSameROCMBuffer(OpKernelContext* ctx, size_t number_of_values,
                               const TensorSeq* values, TensorSeq* updated_values) {
  if (values != updated_values) {
    AllocatorPtr alloc;
    ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc).IsOK(),
                "ROCM CopyIfNotSameBuffer for tensor sequence: Unable to get an allocator.");
    hipStream_t hip_stream = ctx->GetComputeStream()
                                   ? static_cast<hipStream_t>(ctx->GetComputeStream()->GetHandle())
                                   : nullptr;

    updated_values->SetType(values->DataType());
    updated_values->Reserve(number_of_values);
    for (size_t input_idx = 0; input_idx < number_of_values; ++input_idx) {
      const Tensor& source_tensor = values->Get(input_idx);
      std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(), source_tensor.Shape(), alloc);

      HIP_RETURN_IF_ERROR(hipMemcpyAsync(target_tensor->MutableDataRaw(),
                                           source_tensor.DataRaw(),
                                           source_tensor.SizeInBytes(),
                                           hipMemcpyDeviceToDevice, hip_stream));

      updated_values->Add(std::move(*target_tensor));  // Add will check for type consistency
    }
  }
  return Status::OK();
}

}  // namespace rocm
}  // namespace onnxruntime
