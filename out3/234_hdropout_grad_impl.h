// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {
namespace rocm {

template <typename T>
void DropoutGradientKernelImpl(hipStream_t stream, const int64_t N, const T* dY_data, const void* mask_data,
                               const float ratio, T* dX_data, bool use_bitmask);

}  // namespace rocm
}  // namespace onnxruntime
