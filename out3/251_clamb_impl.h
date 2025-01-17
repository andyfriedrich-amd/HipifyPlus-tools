

#pragma once
#include <cuda_runtime.h>
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/multi_tensor/common.cuh"
#include "core/framework/stream_handles.h"
namespace onnxruntime {
namespace cuda {




template <typename T1, typename T2, typename T3, typename T_GRAD_NORM>
void LambComputeDirection(
  cudaStream_t stream, const T1* weights, const T2* grads, const T3* moment_1, const T3* moment_2, const T1* loss_scale, const T_GRAD_NORM* grad_norm, float alpha, float beta, float lambda, float epsilon, float max_norm, float alpha_correction, float beta_correction, T2* update_direction, T3* moment_1_out, T3* moment_2_out, size_t count);




template <typename T1, typename T2, typename T3, typename T_MIXED_PRECISION_FP>
void LambUpdate(
  cudaStream_t stream, const T1* eta, const float ratio_min, const float ratio_max, const T2* r_norm, const T2* w_norm, const T2* weights, const T3* update_direction, T2* weights_out, T3* gradients_out, T_MIXED_PRECISION_FP* mixed_precision_weights_out, size_t count);


















template <typename T1, typename T2, typename T3, typename T_GRAD_NORM>
struct LambMultiTensorComputeDirectionFunctor {
 void operator()(
   cudaStream_t stream, ChunkGroup<6> chunk_group, const T1* loss_scale, const T_GRAD_NORM* grad_norm, const float lambda, const float alpha, const float beta, const float epsilon, const float max_norm, const float alpha_correction, const float beta_correction);
};













template <typename TIn1, typename TIn2, typename TOut1, typename TOut2, typename TBuf>
struct LambMultiTensorReductionFunctor {
 void operator()(
   cudaStream_t stream, ChunkGroup<4> chunk_group, const CudaKernel& kernel, void* reduction_buffer, size_t reduction_buffer_size, onnxruntime::Stream* ort_stream);
};















struct LambMultiTensorSyncRangeAndLock {
 int leading_block;
 int number_blocks;
 int completed_blocks;
};



















template <typename T1, typename T2, typename T3, typename T_MIXED_PRECISION_FP>
struct LambMultiTensorUpdateFunctor {
 void operator()(
   cudaStream_t stream, ChunkGroup<7> chunk_group, const T1* eta, const float ratio_min, const float ratio_max);
};
} 
} 