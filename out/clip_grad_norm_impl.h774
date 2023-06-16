// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/multi_tensor/common.cuh"

namespace onnxruntime {
namespace rocm {

constexpr int ClipGradNormGroupSize = 1;

template <typename T>
struct ClipGradNormFunctor {
  void operator()(hipStream_t stream,
                  ChunkGroup<ClipGradNormGroupSize> chunks,
                  const float* l2_norm,
                  const float epsilon,
                  const float max_norm);
};

}  // namespace rocm
}  // namespace onnxruntime
