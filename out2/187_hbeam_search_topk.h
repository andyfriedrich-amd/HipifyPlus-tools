// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
void BeamSearchTopK(
    const T* input,
    int32_t batch_size,
    int32_t num_beams,
    int32_t vocab_size,
    int32_t k,
    T* tmp_values_1st_stage,
    int32_t* tmp_indices_1st_stage,
    T* tmp_values_2st_stage,
    int32_t* tmp_indices_2st_stage,
    T* output_values,
    int32_t* output_tokens,
    int32_t* output_indices,
    hipStream_t stream);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
