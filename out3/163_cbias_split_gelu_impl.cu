



#include <cub/cub.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/diffusion/bias_split_gelu_impl.h"
using namespace onnxruntime::cuda;
namespace onnxruntime {
namespace contrib {
namespace cuda {
template <typename T, int32_t HHS, int32_t TPB>
__global__ void biasSplitGeluKernel(T const* input, T const* bias, T* output) {
 int32_t index_input = blockIdx.x * HHS * 2 + threadIdx.x;
 int32_t index_output = blockIdx.x * HHS + threadIdx.x;
 int32_t index_bias = threadIdx.x;
#pragma unroll
 for (int32_t i = 0; i < HHS / TPB; ++i) {
  auto value_left = (float)(input[index_input] + bias[index_bias]);
  auto value_right = (float)(input[index_input + HHS] + bias[index_bias + HHS]);
  
  float gelu_right = value_right * 0.5f * (erff(value_right / 1.41421356237f) + 1.0f);
  float result = value_left * gelu_right;
  output[index_output] = static_cast<T>(result);
  index_input += TPB;
  index_output += TPB;
  index_bias += TPB;
 }
 return;
}
template <typename T>
void LaunchBiasSplitGeluKernel(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size, T const* input, T const* bias, T* output) {
 constexpr int32_t TPB = 256; 
 switch (half_hidden_size) {
  case 1280:
   (biasSplitGeluKernel<T, 1280, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
   break;
  case 2560:
   (biasSplitGeluKernel<T, 2560, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
   break;
  case 5120:
   (biasSplitGeluKernel<T, 5120, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, output);
   break;
  default:
   ORT_NOT_IMPLEMENTED("Not implemented");
 }
}
template __global__ void biasSplitGeluKernel<float, 1280, 256>(float const*, float const*, float*);
template __global__ void biasSplitGeluKernel<float, 2560, 256>(float const*, float const*, float*);
template __global__ void biasSplitGeluKernel<float, 5120, 256>(float const*, float const*, float*);
template __global__ void biasSplitGeluKernel<half, 1280, 256>(half const*, half const*, half*);
template __global__ void biasSplitGeluKernel<half, 2560, 256>(half const*, half const*, half*);
template __global__ void biasSplitGeluKernel<half, 5120, 256>(half const*, half const*, half*);
template void LaunchBiasSplitGeluKernel<float>(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size, float const* input, float const* bias, float* output);
template void LaunchBiasSplitGeluKernel<half>(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size, half const* input, half const* bias, half* output);
} 
} 
} 