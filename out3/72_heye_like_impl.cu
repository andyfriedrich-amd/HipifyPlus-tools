#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "eye_like_impl.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
__global__ void _EyeLikeKernel(
    size_t offset,
    size_t stripe,
    T* output_data,
    HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // offset is the first elements, stripe is width + 1.
  output_data[offset + id * stripe] = static_cast<T>(1);
}

template <typename T>
void EyeLikeImpl(
    hipStream_t stream,
    size_t offset,
    size_t stripe,
    T* output_data,
    size_t diag_count) {
  constexpr int block_size = 256;
  int blocksPerGrid = (int)(ceil(static_cast<float>(diag_count) / block_size));
  HIP_LONG N = static_cast<HIP_LONG>(diag_count);

  _EyeLikeKernel<<<blocksPerGrid, block_size, 0, stream>>>(offset, stripe, output_data, N);
}

#define SPECIALIZED_IMPL(T)                                          \
  template void EyeLikeImpl<T>(                                      \
    hipStream_t stream,                                       \
    size_t offset,                                                   \
    size_t stripe,                                                   \
    T* output_data,                                                  \
    size_t diag_count);

SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)

}  // namespace rocm
}  // namespace onnxruntime