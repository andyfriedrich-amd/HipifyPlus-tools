// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace rocm {
template <typename T>
void LaunchAllKernel(hipStream_t stream, const T* data, const int size, bool* output);
}
}  // namespace onnxruntime
