// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

template <typename TIndex>
void ComputeSliceOffsetsImpl(
    hipStream_t stream,
    const int64_t batch_dims,
    const TArray<int64_t> input_dims,
    const size_t num_slices,
    const size_t num_slices_per_batch,
    const size_t input_batch_stride,
    const size_t num_slice_dims,
    const int64_t* const sizes_from_slice_dims_data,  // num_slice_dims elements
    const TIndex* const indices_data,                 // num_slices * num_slice_dims elements
    int64_t* const input_slice_offsets_data);         // num_slices elements

template <typename T>
void GatherNDImpl(
    hipStream_t stream,
    const size_t num_slices,
    const void* input_data,
    void* output_data,
    const size_t slice_size,
    const int64_t* input_slice_offsets_data);

#ifdef ENABLE_TRAINING_OPS
template <typename T>
void GatherNDGradImpl(
    hipStream_t stream,
    const size_t num_slices,
    const void* update_data,
    void* output_data,
    const size_t slice_size,
    const int64_t* input_slice_offsets_data);
#endif

}  // namespace rocm
}  // namespace onnxruntime
