#include "hip/hip_runtime.h"


#include "core/providers/rocm/math/clip_impl.h"
#include "core/providers/rocm/cu_inc/common.cuh"
namespace onnxruntime {
namespace rocm {
template <typename T>
__global__ void _Clip(const T* input, T* output, const T* min, const T* max, T min_default, T max_default, size_t N) {
 auto min_val = (min) ? *min : min_default; 
 auto max_val = (max) ? *max : max_default; 
 CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
 output[id] = (input[id] < min_val) ? min_val : ((input[id] > max_val) ? max_val : input[id]);
}
template <typename T>
void ClipImpl(hipStream_t stream, const T* input_data, T* output_data, const T* min, const T* max, T min_default, T max_default, size_t count) {
 typedef typename ToHipType<T>::MappedType HipT;
 int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
 union ConstAliasUnion {
  const T *t;
  const HipT *rocmT;
  ConstAliasUnion(const T* _t) { t = _t;}
 };
 union AliasUnion {
  T *t;
  HipT *rocmT;
  AliasUnion(T* _t) { t = _t;}
 };
 _Clip<HipT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(((union ConstAliasUnion)input_data).rocmT, ((union AliasUnion)output_data).rocmT, ((union ConstAliasUnion)min).rocmT, ((union ConstAliasUnion)max).rocmT, *((union AliasUnion)&min_default).rocmT, *((union AliasUnion)&max_default).rocmT, count);
}
template void ClipImpl<float>(hipStream_t stream, const float* input_data, float* output_data, const float* min, const float* max, float min_default, float max_default, size_t count);
template void ClipImpl<double>(hipStream_t stream, const double* input_data, double* output_data, const double* min, const double* max, double min_default, double max_default, size_t count);
template void ClipImpl<MLFloat16>(hipStream_t stream, const MLFloat16* input_data, MLFloat16* output_data, const MLFloat16* min, const MLFloat16* max, MLFloat16 min_default, MLFloat16 max_default, size_t count);
template void ClipImpl<int8_t>(hipStream_t stream, const int8_t* input_data, int8_t* output_data, const int8_t* min, const int8_t* max, int8_t min_default, int8_t max_default, size_t count);
template void ClipImpl<uint8_t>(hipStream_t stream, const uint8_t* input_data, uint8_t* output_data, const uint8_t* min, const uint8_t* max, uint8_t min_default, uint8_t max_default, size_t count);
template void ClipImpl<int64_t>(hipStream_t stream, const int64_t* input_data, int64_t* output_data, const int64_t* min, const int64_t* max, int64_t min_default, int64_t max_default, size_t count);
template void ClipImpl<uint64_t>(hipStream_t stream, const uint64_t* input_data, uint64_t* output_data, const uint64_t* min, const uint64_t* max, uint64_t min_default, uint64_t max_default, size_t count);
} 
} 