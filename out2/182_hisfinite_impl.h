// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include "core/providers/rocm/multi_tensor/common.cuh"

namespace onnxruntime {
namespace rocm {

template <typename T>
struct IsAllFiniteFunctor {
  void operator()(hipStream_t stream, ChunkGroup<1> chunks, bool* output, const bool isinf_only, const bool isnan_only);
};

}  // namespace rocm
}  // namespace onnxruntime