// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

bool CanDoTranspose3D(const hipDeviceProp_t& prop,
                      size_t rank, const gsl::span<const int64_t>& input_dims, const gsl::span<const size_t>& permutations,
                      dim3& grid_size, dim3& block_size);
Status Transpose3DImpl(hipStream_t stream, size_t element_size, const TArray<int64_t>& input_shape, const TArray<int64_t>& input_strides, const void* input_data,
                       void* output_data, int64_t N,
                       const dim3& grid_size, const dim3& block_size);

bool CanDoTranspose4DParallelizeMultipleElementsPerThreadInInnermostDim(const hipDeviceProp_t& prop,
                                                                        size_t element_size,
                                                                        int32_t rank,
                                                                        const gsl::span<const int64_t>& input_dims,
                                                                        const gsl::span<const size_t>& permutations,
                                                                        dim3& grid_size, dim3& block_size);
Status Transpose4DParallelizeMultipleElementsPerThreadInInnermostDim(hipStream_t stream,
                                                                     size_t element_size, const TArray<int64_t>& input_shape,
                                                                     const TArray<int64_t>& input_strides, const void* input_data,
                                                                     const TArray<int64_t>& output_strides, void* output_data, int N,
                                                                     const dim3& grid_size, const dim3& block_size);

bool CanDoTranspose4DParallelizeOneElementPerThread(const hipDeviceProp_t& prop,
                                                    size_t element_size,
                                                    int32_t rank,
                                                    const gsl::span<const int64_t>& input_dims,
                                                    const gsl::span<const size_t>& permutations,
                                                    dim3& grid_size, dim3& block_size);
Status Transpose4DParallelizeOneElementPerThread(hipStream_t stream,
                                                 size_t element_size, const TArray<int64_t>& input_shape,
                                                 const TArray<int64_t>& input_strides, const void* input_data,
                                                 const TArray<int64_t>& output_strides, void* output_data, int N,
                                                 const dim3& grid_size, const dim3& block_size);

Status TransposeImpl(hipStream_t stream, size_t element_size, int32_t shape_rank, const TArray<int64_t>& input_strides,
                     const void* input_data, const TArray<fast_divmod>& fdm_output_strides, void* output_data, int N);
}  // namespace rocm
}  // namespace onnxruntime
