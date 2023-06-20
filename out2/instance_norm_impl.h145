// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/shared_inc/fast_divmod.h"
namespace onnxruntime {
namespace rocm {

template <typename T1, typename T2>
void InstanceNormImpl(
    hipStream_t stream,
    const T1* input_data,
    const T1* scale,
    const T1* bias,
    const T2* mean,
    const T2* variance,
    const double variance_correction,
    const double epsilon,
    const fast_divmod& fdm_HW,
    const fast_divmod& fdm_C,
    T1* output_data,
    size_t count);

}  // namespace rocm
}  // namespace onnxruntime
