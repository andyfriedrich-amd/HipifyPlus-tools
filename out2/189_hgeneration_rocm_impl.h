

#pragma once
#include <stdint.h>
#include <hip/hip_fp16.h>
#include <hiprand/hiprand_kernel.h>
namespace onnxruntime {
namespace contrib {
namespace rocm {
void LaunchInitKernel(
  float* beam_scores, int batch_size, int num_beams, hipStream_t stream);
template <typename T>
void LaunchAddProbsKernel(T* log_probs, T* cum_log_probs, const int batch_size, const int num_beams, const int vocab_size, hipStream_t stream);
template <typename T>
void LaunchLogitsProcessKernel(
  T* next_token_scores, const int* vocab_mask, const int* prefix_vocab_mask, int* presence_mask, float presence_penalty, float temperature, int batch_size, int num_beams, int vocab_size, int padded_vocab_size, int demote_token_id, int32_t* sequences, int max_sequence_length, int current_sequence_length, float repetition_penalty, int no_repeat_ngram_size, hipStream_t stream);
void LaunchNextTokenKernel(const int64_t* next_token_indices, int32_t* next_indices, int32_t* next_tokens, int batch_size, int top_k, int vocab_size, hipStream_t stream);
void LaunchUpdateGptKernel(const int32_t* old_mask_data, int32_t* mask_data, int32_t* next_positions, int batch_beam_size, int current_length, hipStream_t stream);
template <typename T>
void GetTempStorageSize(const T* d_keys_in, const int* d_values_in, int* d_offsets, int num_items, int num_segments, hipStream_t stream, bool is_descending, size_t& temp_storage_bytes);
void LaunchSetupParamsKernel(int* d_values_in, int* d_offsets, int batch_size, int vocab_size, hipStream_t stream);
template <typename T>
void LaunchSortPairs(void* d_temp_storage, size_t temp_storage_bytes, const T* d_keys_in, T* d_keys_out, const int* d_values_in, int* d_values_out, int num_items, int num_segments, int* d_offsets, hipStream_t stream, bool is_descending);
template <typename T>
void LaunchFilterLogitsKernel(float* d_sorted_logits_in, const int* d_sorted_indices, T* d_logits_in_out, float top_p, float filter_value, int min_tokens_to_keep, int batch_size, int vocab_size, hipStream_t stream, bool is_descending);
void TorchMultinomialKernelLauncher(float* d_input, float* d_sampled, int32_t* d_output, int batch_size, int vocab_size, int* d_presence_mask, hipStream_t stream);
void UpdateDecoderMaskedMultiHeadAttentionCacheIndirection(int32_t* tgt_indir_cache, const int32_t* src_indir_cache, const int32_t* beam_ids, int batch_size, int beam_width, int input_seq_length, int max_seq_length, int current_length, hipStream_t stream);
template <typename T>
void KeyCacheExpansionKernelLauncher(const T* key_cache, T* key_cache_expanded, int batch_size, int beam_width, int num_heads, int sequence_length, int max_seq_length, int head_size, hipStream_t stream);
template <typename T>
void BufferExpansionKernelLauncher(const T* input, T* output, int batch_size, int beam_width, int chunk_size, hipStream_t stream);
} 
} 
} 