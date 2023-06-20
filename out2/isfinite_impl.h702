// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace rocm {

template <typename TSrc>
void IsFinite(hipStream_t stream, const TSrc* input, bool* output, size_t N);

}
}  // namespace onnxruntime
