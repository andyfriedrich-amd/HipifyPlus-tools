

#pragma once
#include "core/providers/rocm/shared_inc/rocm_utils.h"
namespace onnxruntime {
namespace contrib {
namespace rocm {



















template <typename T>
void LaunchAddBiasTranspose(
  hipStream_t stream, const int num_matrices, const int format, const int max_threads_per_block, const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size, const T* input, const T* biases, T* output, bool enable_half4, const int v_head_size, T* qkv_add_bias = nullptr, int total_matrix_count = -1, bool do_rotary = false, int original_past_sequence_length = 0);







template <typename T>
void LaunchAddBiasTransposeTrt(
  hipStream_t stream, const int max_threads_per_block, const int batch_size, const int sequence_length, const int num_heads, const int head_size, const T* biases, const T* query, const T* key, const T* value, T* output, bool is_cross_attention, int kv_sequence_length = -1);




template <typename T>
void LaunchAddBias(
  hipStream_t stream, const int max_threads_per_block, const int batch_size, const int sequence_length, const int kv_sequence_length, const int num_heads, const int head_size, const int v_head_size, const T* biases, const T* query, const T* key, const T* value, T* q, T* k, T* v);
} 
} 
} 