// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace rocm {

template <typename InputDataArray>
Status ConcatSameConcatDimImpl(hipStream_t stream, const size_t element_bytes, const int block_size_including_axis_dim,
                               const int block_size_inside_axis_dim, const int64_t concat_size, void* output_data,
                               const InputDataArray input_data, const size_t output_size);

Status ConcatImpl(hipStream_t stream, const size_t element_bytes, const int block_size_including_axis_dim,
                  const int block_size_inside_axis_dim, const int64_t* concat_sizes, const int64_t* concat_sizes_range,
                  const int64_t* axis_dimension_input_output_mapping, void* output_data, const void** input_data,
                  const size_t output_size);

}  // namespace rocm
}  // namespace onnxruntime
