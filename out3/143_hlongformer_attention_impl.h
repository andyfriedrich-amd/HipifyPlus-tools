// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

size_t GetPinnedBufferSize(
    size_t batch_size);

size_t GetLongformerAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t head_size,
    size_t sequence_length,

    size_t max_num_global,
    size_t window,
    bool disable_compact_memory);

Status LaunchLongformerAttentionKernel(
    const hipDeviceProp_t& device_prop,  // Device Properties
    rocblas_handle rocblas,              // Rocblas handle
    hipStream_t stream,                // ROCM stream
    const void* input,                  // Input tensor
    const void* bias,                   // Bias tensor
    const void* attention_mask,         // Attention mask with shape (B, S)
    const void* global_input,           // Global attention input, or nullptr when max_num_global == 0.
    const void* global_bias,            // Global bias tensor
    const int* global_attention,        // Global attention flags with shape (B, S)
    const int* global_index,            // Global index
    const int* batch_global_num,        // Number of global tokens per batch. It is in device memory.
    void* pinned_buffer,                // Pinned memory: copy of batch_global_num, and a buffer to copy to scratch2.
    void* workspace,                    // Temporary buffer
    void* output,                       // Output tensor
    int batch_size,                     // Batch size (B)
    int sequence_length,                // Sequence length (S)
    int num_heads,                      // Number of attention heads (N)
    int head_size,                      // Hidden layer size per head (H)
    int window,                         // One sided attention window (W)
    int max_num_global,                 // Maximum number of global tokens (G)
    const size_t element_size,          // Element size of input tensor,
    bool disable_compact_memory,        // Disable compact memory kernel
    bool use_merged_qkv_weights,
    bool use_half4);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
