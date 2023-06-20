#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/rocm/math/scale_impl.h"
#include "core/providers/rocm/cu_inc/common.cuh"

namespace onnxruntime {
namespace rocm {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _Scale(
    const T* input_data,
    const T scale_value,
    T* output_data,
    HIP_LONG N) {
  HIP_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T input_value[NumElementsPerThread];
  HIP_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
        input_value[i] = input_data[id];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = input_value[i] * scale_value;
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void Impl_Scale(
    hipStream_t stream,
    const T* input_data,
    const float scale_value,
    T* output_data,
    size_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  HIP_LONG N = static_cast<HIP_LONG>(count);
  _Scale<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data,
      static_cast<T>(scale_value),
      output_data,
      N);
}

#define SPECIALIZE_SCALE_IMPL(T)        \
template void Impl_Scale<T>(            \
    hipStream_t stream,                \
    const T* input_data,                \
    const float scale_value,            \
    T* output_data,                     \
    size_t count);

SPECIALIZE_SCALE_IMPL(half)
SPECIALIZE_SCALE_IMPL(float)
SPECIALIZE_SCALE_IMPL(double)

}  // namespace rocm
}  // namespace onnxruntime
