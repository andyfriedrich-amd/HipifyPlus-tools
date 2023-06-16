// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
Status SoftmaxDropoutGradImpl(hipStream_t stream, miopenHandle_t miopen_handle, T* input_grad_data,
                              const T* output_grad_data, const bool* mask_data, const T* softmax_output_data,
                              int element_count, int batch_count, const float ratio);

}  // namespace rocm
}  // namespace onnxruntime
