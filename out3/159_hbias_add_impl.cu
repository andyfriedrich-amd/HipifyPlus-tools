#include "hip/hip_runtime.h"




#include <hipcub/hipcub.hpp>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "contrib_ops/rocm/diffusion/bias_add_impl.h"
using namespace onnxruntime::rocm;
namespace onnxruntime {
namespace contrib {
namespace rocm {
template <typename T, int32_t C, int32_t TPB>
__global__ void BiasAddKernel(T const* input, T const* bias, T const* residual, T* output) {
 int32_t base_offset = blockIdx.x * C + threadIdx.x;
 int32_t bias_offset = threadIdx.x;
#pragma unroll
 for (int32_t i = 0; i < C / TPB; ++i) {
  output[base_offset] = input[base_offset] + bias[bias_offset] + residual[base_offset];
  base_offset += TPB;
  bias_offset += TPB;
 }
}
template __global__ void BiasAddKernel<float, 320, 320>(float const*, float const*, float const*, float*);
template __global__ void BiasAddKernel<float, 640, 320>(float const*, float const*, float const*, float*);
template __global__ void BiasAddKernel<float, 1280, 320>(float const*, float const*, float const*, float*);
template __global__ void BiasAddKernel<half, 320, 320>(half const*, half const*, half const*, half*);
template __global__ void BiasAddKernel<half, 640, 320>(half const*, half const*, half const*, half*);
template __global__ void BiasAddKernel<half, 1280, 320>(half const*, half const*, half const*, half*);
template <typename T>
void LaunchBiasAddKernel(hipStream_t stream, int32_t grid_size, int32_t num_channels, T const* input, T const* bias, T const* residual, T* output) {
 constexpr int32_t TPB = 320; 
 switch (num_channels) {
  case 320:
   (BiasAddKernel<T, 320, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, residual, output);
   break;
  case 640:
   (BiasAddKernel<T, 640, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, residual, output);
   break;
  case 1280:
   (BiasAddKernel<T, 1280, TPB>)<<<grid_size, TPB, 0, stream>>>(input, bias, residual, output);
   break;
  default:
   ORT_NOT_IMPLEMENTED("Not implemented");
 }
}
template void LaunchBiasAddKernel<float>(hipStream_t stream, int32_t grid_size, int32_t num_channels, float const* input, float const* bias, float const* residual, float* output);
template void LaunchBiasAddKernel<half>(hipStream_t stream, int32_t grid_size, int32_t num_channels, half const* input, half const* bias, half const* residual, half* output);
} 
} 
} 