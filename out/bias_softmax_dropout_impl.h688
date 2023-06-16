// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/random_generator.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
Status BiasSoftmaxDropoutImpl(hipStream_t stream, const hipDeviceProp_t& prop, miopenHandle_t miopen_handle,
                              T* dropout_output_data, bool* mask_data, T* softmax_output_data, const T* input_data,
                              const T* bias_data, int element_count, int batch_count, bool is_inner_broadcast,
                              int bias_broadcast_size, const float ratio, PhiloxGenerator& generator);

}  // namespace rocm
}  // namespace onnxruntime
