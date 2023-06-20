// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

void NGramRepeatBlockImpl(
    hipStream_t stream,
    const int64_t* tokens_ptr,
    float* scores_ptr,
    int bsz,
    int step,
    int max_predict_len,
    int vocab_size,
    int beam_size,
    int no_repeat_ngram_size);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
