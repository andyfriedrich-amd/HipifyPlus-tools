// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
void GreedySearchTopOne(
    const T* input,
    int32_t batch_size,
    int32_t vocab_size,
    T* tmp_values,
    int32_t* tmp_tokens,
    T* output_values,
    int32_t* output_tokens,
    hipStream_t stream);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
