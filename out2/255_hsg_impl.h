// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace rocm {
template <typename T>
void SGDOptimizerImpl(
    hipStream_t stream,
    const T* eta,
    const T* weights,
    const T* gradients,
    T* weight_out,
    T* gradients_out,
    size_t count);
}
}  // namespace onnxruntime
