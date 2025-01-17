// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
void CumSumImpl(
    hipStream_t stream,
    const T* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    T* output_data,
    int64_t output_size,
    bool exclusive,
    bool reverse);

}  // namespace rocm
}  // namespace onnxruntime
