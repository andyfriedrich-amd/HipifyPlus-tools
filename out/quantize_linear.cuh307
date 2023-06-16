// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "quantize_linear.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

template <class T, class U>
Status CudaQuantizeLinearStd(hipStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element);

template <class T, class U>
Status CudaQuantizeLinearSat(hipStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element,
                             bool saturate);

template <class T, class U>
Status CudaQuantizeLinearAxisStd(hipStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element,
                                 size_t batch_size, size_t n_scales);

template <class T, class U>
Status CudaQuantizeLinearAxisSat(hipStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element,
                                 size_t batch_size, size_t n_scales, bool saturate);

template <class T, class U>
Status CudaDequantizeLinearStd(hipStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element);

template <class T, class U>
Status CudaDequantizeLinearSat(hipStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element);

template <class T, class U>
Status CudaDequantizeLinearAxisStd(hipStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element,
                                   size_t batch_size, size_t n_scales);

template <class T, class U>
Status CudaDequantizeLinearAxisSat(hipStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element,
                                   size_t batch_size, size_t n_scales);

}  // namespace rocm
}  // namespace onnxruntime
