

#pragma once
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
namespace onnxruntime {
namespace cuda {
namespace detail {
size_t compute_reduce_matrix_columns_intermediate_buffer_size(
  int element_size, int num_rows, int num_cols);
} 

template <typename TIn>
size_t compute_reduce_matrix_columns_buffer_size(int m, int n) {
 using TBuf = AccumulationType_t<TIn>;
 return detail::compute_reduce_matrix_columns_intermediate_buffer_size(
   sizeof(TBuf), m, n);
}

template <typename TIn>
size_t compute_reduction_buffer_size(int size) {
 using TBuf = AccumulationType_t<TIn>;
 return detail::compute_reduce_matrix_columns_intermediate_buffer_size(
   sizeof(TBuf), 1, size);
}

template <typename TIn, typename TOut>
Status reduce_sum(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

template <typename TIn, typename TOut>
Status reduce_square_sum(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

template <typename TIn, typename TOut>
Status reduce_l2_norm(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

template <typename TIn, typename TOut>
Status reduce_mean(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);
enum class ApplicableMatrixReduction {
 
 Rows, Columns, None, };

ApplicableMatrixReduction get_applicable_matrix_reduction(
  const cudnnReduceTensorOp_t cudnn_reduce_op, gsl::span<const int64_t> dims, gsl::span<const int64_t> axes, int& m, int& n);

template <typename TIn, typename TOut>
Status reduce_matrix_rows(cudaStream_t stream, const TIn* input, TOut* output, int m, int n, bool reset_initial_output = true);

template <typename TIn, typename TOut>
Status reduce_matrix_columns(cudaStream_t stream, const TIn* input, TOut* output, int m, int n, void* buffer, size_t buffer_size);

template <typename T>
void UnaryDiv(cudaStream_t stream, const T* input, T* output, T denominator, size_t count);
} 
} 