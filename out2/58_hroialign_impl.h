// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
void RoiAlignImpl(
    hipStream_t stream,
    const int64_t nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio,
    const T* bottom_rois,
    int64_t roi_cols,
    T* top_data,
    const bool is_mode_avg,
    const bool half_pixel,
    const int64_t* batch_indices_ptr);

}  // namespace rocm
}  // namespace onnxruntime
