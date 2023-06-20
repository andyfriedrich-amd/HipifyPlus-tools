// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

int NonZeroCalcBlockCount(int64_t x_size);

hipError_t NonZeroCalcPrefixSumTempStorageBytes(hipStream_t stream, int* prefix_counts, int number_of_blocks, size_t&);

hipError_t NonZeroInclusivePrefixSum(hipStream_t stream, void* d_temp_storage, size_t temp_storage_bytes, int* prefix_counts, int number_of_blocks);

// count nonzero elements in each block into counts_in_blocks,
// the counts_in_blocks buffer is pre-allocated on gpu first.
template <typename InputT>
hipError_t NonZeroCountEachBlock(hipStream_t stream, const InputT* x, int64_t x_size, int* counts_in_blocks);

// output nonzero positions using input x and prefix_counts for each blocks
template <typename InputT>
hipError_t NonZeroOutputPositions(
    hipStream_t stream, const InputT* x, int64_t x_size, int x_rank, const TArray<fast_divmod>& x_strides,
    const int* prefix_counts, int nonzero_elements, int64_t* results);

}  // namespace rocm
}  // namespace onnxruntime
