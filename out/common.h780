// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
Status CopyIfNotSameBuffer(hipStream_t stream, const Tensor& source_tensor, Tensor& target_tensor) {
  const T* source = source_tensor.template Data<T>();
  T* target = target_tensor.template MutableData<T>();
  if (target != source) {
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(target, source, source_tensor.SizeInBytes(), hipMemcpyDeviceToDevice,
                                         stream));
  }
  return Status::OK();
}

Status CopyIfNotSameROCMBuffer(OpKernelContext* ctx, size_t number_of_values, const TensorSeq* values,
                               TensorSeq* updated_values);

}  // namespace rocm
}  // namespace onnxruntime
