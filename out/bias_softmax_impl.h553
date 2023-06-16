// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
Status BiasSoftmaxImpl(hipStream_t stream, miopenHandle_t miopen_handle, T* output_data, const T* input_data,
                       const T* bias_data, int element_count, int batch_count, bool is_inner_broadcast,
                       int bias_broadcast_size);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
