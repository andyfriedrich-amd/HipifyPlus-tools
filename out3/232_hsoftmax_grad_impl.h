// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {
template <typename T>
Status SoftmaxGradImpl(hipStream_t stream, miopenHandle_t miopen_handle, T* input_grad, const T* output_grad,
                       const T* softmax_output, int element_count, int batch_count, bool is_log_softmax);
}
}  // namespace onnxruntime
