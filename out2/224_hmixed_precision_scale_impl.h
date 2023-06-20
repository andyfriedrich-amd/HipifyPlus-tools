// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace rocm {
template <typename SrcT, typename DstT>
void Impl_MixedPrecisionScale(
    hipStream_t stream,
    const SrcT* input_data,
    const float* scale_data,
    DstT* output_data,
    size_t count);
}
}  // namespace onnxruntime
