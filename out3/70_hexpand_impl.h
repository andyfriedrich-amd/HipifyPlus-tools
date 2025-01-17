// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

Status ExpandImpl(
    hipStream_t stream,
    const size_t element_size,
    const int N_output,
    const int N_input,
    const void* input_data,
    void* output_data,
    const TArray<fast_divmod>& output_strides,
    const TArray<int64_t>& input_strides);

}  // namespace rocm
}  // namespace onnxruntime
