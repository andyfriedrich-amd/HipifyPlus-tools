// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include "core/providers/rocm/multi_tensor/common.cuh"

namespace onnxruntime {
namespace rocm {
template <typename TIn, typename TOut>
struct MultiTensorReduceL2 {
  void operator()(hipStream_t stream, ChunkGroup<1> chunk_group, TOut* output);
};

template <typename Tin, typename Tout>
void ScalarSqrt(hipStream_t stream, Tin* input, Tout* output);
}  // namespace rocm
}  // namespace onnxruntime
