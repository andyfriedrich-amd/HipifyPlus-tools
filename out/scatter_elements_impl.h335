// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

struct GatherScatterElementsArgs;

template <typename T, typename TIndex>
Status ScatterElementsImpl(hipStream_t stream, const T* input_data, const TIndex* indices_data, const T* updates_data,
                           T* output_data, const GatherScatterElementsArgs& args);

}  // namespace rocm
}  // namespace onnxruntime
