// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace rocm {

template <typename OutputDataArray>
Status SplitSameSplitDimImpl(hipStream_t stream, const size_t element_size, const int block_size_including_axis_dim,
                             const int block_size_inside_axis_dim, const int64_t split_size, const int num_outputs,
                             const void* input_data, OutputDataArray output_data, const size_t input_size);

Status SplitImpl(hipStream_t stream, const size_t element_size, const int block_size_including_axis_dim,
                 const int block_size_inside_axis_dim, const int64_t* split_sizes, const int64_t* split_sizes_range,
                 const int64_t* axis_dimension_input_output_mapping, const int num_outputs, const void* input_data,
                 void** output_data, const size_t input_size);

}  // namespace rocm
}  // namespace onnxruntime
