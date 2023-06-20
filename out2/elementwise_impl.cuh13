#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/cu_inc/common.cuh"

namespace onnxruntime {
namespace rocm {

#ifdef USE_ROCM
constexpr int kElementsPerThread = 2;
constexpr int kThreadsPerBlock = 512;
#else
constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
#endif

template <typename T, typename FuncT>
__global__ void ElementwiseKernel(T* output_data, const FuncT functor, HIP_LONG N) {
  HIP_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];

  HIP_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      value[i] = functor(id);
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += kThreadsPerBlock;
    }
  }
}

template <typename T, typename FuncT>
void LaunchElementwiseKernel(hipStream_t stream, T* output_data, const FuncT& functor, size_t output_size) {
  if (output_size == 0) return;
  HIP_LONG N = static_cast<HIP_LONG>(output_size);
  int blocksPerGrid = CeilDiv(N, kThreadsPerBlock * kElementsPerThread);
  ElementwiseKernel<T, FuncT><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(output_data, functor, N);
}

}  // namespace rocm
}  // namespace onnxruntime
