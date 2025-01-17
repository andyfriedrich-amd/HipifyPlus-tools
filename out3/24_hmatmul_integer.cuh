// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

Status ReduceRowSumOnMatrixA(hipStream_t stream, const int8_t* matrix, int32_t* row_sum, const int8_t offset, const MatMulComputeHelper& helper);
Status ReduceColSumOnMatrixB(hipStream_t stream, const int8_t* matrix, int32_t* col_sum, const int8_t offset, const MatMulComputeHelper& helper);
Status OffsetOutput(hipStream_t stream,
                    const int32_t* row_sum,
                    const int32_t* col_sum,
                    int32_t* output,
                    const int8_t a_offset,
                    const int8_t b_offset,
                    const MatMulComputeHelper& helper);

}  // namespace rocm
}  // namespace onnxruntime
