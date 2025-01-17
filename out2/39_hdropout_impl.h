// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
void DropoutKernelImpl(const hipDeviceProp_t& prop, hipStream_t stream, const int64_t N,
                       const int64_t mask_element_count, const float ratio, PhiloxGenerator& generator, const T* X_data,
                       T* Y_data, void* mask_data, bool use_bitmask);

}  // namespace rocm
}  // namespace onnxruntime
