// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
void WhereImpl(
    hipStream_t stream,
    size_t output_rank_or_simple_broadcast,
    BroadcastIndexType cond_index_type,
    const TArray<int64_t>& cond_padded_strides,
    const bool* cond_data,
    BroadcastIndexType x_index_type,
    const TArray<int64_t>& x_padded_strides,
    const T* x_data,
    BroadcastIndexType y_index_type,
    const TArray<int64_t>& y_padded_strides,
    const T* y_data,
    const TArray<fast_divmod>& fdm_output_strides,
    T* output_data,
    size_t count);

}  // namespace rocm
}  // namespace onnxruntime
