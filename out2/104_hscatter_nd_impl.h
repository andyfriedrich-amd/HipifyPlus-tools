// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

Status ScatterNDImpl(
    hipStream_t stream,
    void* output_data,
    const size_t element_size,
    const size_t num_indices,
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const int64_t* element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements);

}  // namespace rocm
}  // namespace onnxruntime
