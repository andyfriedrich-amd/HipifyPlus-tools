// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/multi_tensor/common.cuh"

namespace onnxruntime {
namespace rocm {

#define MTA_SGD_GROUP_SIZE 2
#define MTA_SGD_CHUNK_SIZE 2048 * 32

template <typename T_WEIGHT, typename T_GRAD>
struct SGDMTAFunctor {
  void operator()(hipStream_t stream, ChunkGroup<MTA_SGD_GROUP_SIZE> chunks, const float lr);
};

}  // namespace rocm
}  // namespace onnxruntime
