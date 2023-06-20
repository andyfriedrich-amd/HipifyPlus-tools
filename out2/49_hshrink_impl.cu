#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/cu_inc/common.cuh"
#include "shrink_impl.h"

namespace onnxruntime {
namespace rocm {

// Generic implementation of Shrink
template <typename T>
__global__ void _ShrinkKernel(
    const T* input_data,
    const float bias,
    const float lambda,
    T* output_data,
    const HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  T x = input_data[id];
  if (x < -lambda) {
    output_data[id] = (T)(x + bias);
  } else if (x > lambda) {
    output_data[id] = (T)(x - bias);
  } else {
    output_data[id] = (T)0;
  }
}

// Specialized implementation for 'half' type
// the idea is to convert 'half' data to 'float' first,
// do the operation and convert result back to 'half'
template <>
__global__ void _ShrinkKernel(
    const half* input_data,
    const float bias,
    const float lambda,
    half* output_data,
    const HIP_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  half x = input_data[id];
  if ((float)x < -lambda) {
    output_data[id] = half((float)x + bias);
  } else if ((float)x > lambda) {
    output_data[id] = half((float)x - bias);
  } else {
    output_data[id] = (half)0;
  }
}

template <typename T>
void ShrinkImpl(
    hipStream_t stream,
    const T* input_data,
    const float bias,
    const float lambda,
    T* output_data,
    size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _ShrinkKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, bias, lambda, output_data, (HIP_LONG)N);
}

#define SPECIALIZED_IMPL(T) \
  template void ShrinkImpl<T>(hipStream_t stream, const T* input_data, const float bias, const float lambda, T* output_data, size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(uint8_t)
SPECIALIZED_IMPL(int8_t)
SPECIALIZED_IMPL(uint16_t)
SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(uint32_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(int64_t)

}  // namespace rocm
}  // namespace onnxruntime
