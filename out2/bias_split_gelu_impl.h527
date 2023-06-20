// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/common/status.h"
#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
void LaunchBiasSplitGeluKernel(hipStream_t stream, int32_t grid_size, int32_t half_hidden_size,
                               T const* input, T const* bias, T* output);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
