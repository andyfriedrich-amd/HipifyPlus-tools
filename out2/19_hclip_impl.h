// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/math/clip.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {
template <typename T>
void ClipImpl(hipStream_t stream, const T* input_data, T* output_data, const T* min, const T* max, T min_default, T max_default, size_t count);

}  // namespace rocm
}  // namespace onnxruntime
