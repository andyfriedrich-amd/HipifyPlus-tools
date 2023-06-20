// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace rocm {
// Implementation can be found in rocm file
template <typename T, typename T_GRAD>
void InPlaceAccumulatorImpl(
    hipStream_t stream,
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count);
}  // namespace rocm
}  // namespace onnxruntime
