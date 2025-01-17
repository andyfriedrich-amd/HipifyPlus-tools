// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
void GridSampleImpl(
    hipStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t dims_input[4],
    const int64_t H_out,
    const int64_t W_out,
    T* output_data);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
