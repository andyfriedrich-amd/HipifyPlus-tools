// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace rocm {

template <typename T>
void Impl_Scale(
    hipStream_t stream,
    const T* input_data,
    const float scale_value,
    T* output_data,
    size_t count);
}
}  // namespace onnxruntime
