// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
void TileImpl(hipStream_t stream, const size_t shape_rank, const TArray<fast_divmod>& fdm_input_shape,
              const TArray<int64_t>& input_stride, const T* input_data, const TArray<fast_divmod>& fdm_output_strides,
              T* output_data, const size_t N);

template <typename T>
void TileMemcpyImpl(hipStream_t stream, const T* input_data, T* output_data, const size_t num_input_elements,
                    const size_t repeats);

template <typename T>
void TileBatchedMemcpyImpl(hipStream_t stream, const T* input_data, T* output_data, const size_t size_input_row,
                           const size_t num_input_elements, const size_t batch_repeats, const size_t repeats_per_batch);

}  // namespace rocm
}  // namespace onnxruntime
