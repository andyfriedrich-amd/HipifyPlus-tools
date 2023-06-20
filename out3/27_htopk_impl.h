// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
Status TopKImpl(const RocmKernel* kernel, Stream* ort_stream, const T* input_x, T* output_v, int64_t* output_i, const TArray<int64_t>& elem_nums, size_t size, int32_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension);

}  // namespace rocm
}  // namespace onnxruntime
