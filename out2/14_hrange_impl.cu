#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hipcub/hipcub.hpp>
#include <rocblas/rocblas.h>
#include <hip/hip_fp16.h>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "range_impl.h"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace rocm {

template <typename T>
__global__ void RangeKernel(const T start, const T delta, const int count, T* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = start + delta * index;
  }
}

template <typename T>
Status RangeImpl(hipStream_t stream, const T start, const T delta, const int count, T* output) {
  constexpr int block_size = 256;
  int grid_size = (count + block_size - 1) / block_size;
  RangeKernel<T><<<grid_size, block_size, 0, stream>>>(start, delta, count, output);
  return HIP_CALL(hipGetLastError());
}

#define SPECIALIZED_IMPL(T) \
  template Status RangeImpl<T>(hipStream_t stream, const T start, const T delta, const int count, T* output);

SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)

}  // namespace rocm
}  // namespace onnxruntime
