

#pragma once
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
template <int NumUnroll>
__device__ __forceinline__ void SetBitmask(const CUDA_LONG id, const CUDA_LONG mask_element_count, const fast_divmod fdm_bits_per_element, BitmaskElementType thread_bitmask, BitmaskElementType* mask_data) {
 int bitmask_idx, bitmask_shift;
 fdm_bits_per_element.divmod(id, bitmask_idx, bitmask_shift);
 BitmaskElementType bitmask = (thread_bitmask << bitmask_shift);
#if defined(USE_CUDA) && __CUDA_ARCH__ >= 800
 
 BitmaskElementType thread_mask = __match_any_sync(0xFFFFFFFF, bitmask_idx);
 
 
 bitmask = __reduce_or_sync(thread_mask, bitmask);
#else
#pragma unroll
 for (int stride = kNumBitsPerBitmaskElement / (NumUnroll * 2); stride > 0; stride /= 2) {
  bitmask |= WARP_SHFL_DOWN(bitmask, stride);
 }
#endif
 
 if (bitmask_shift == 0 && bitmask_idx < mask_element_count) {
  mask_data[bitmask_idx] = bitmask;
 }
}
template <int NumUnroll>
__device__ __forceinline__ void GetMasks(CUDA_LONG id, const fast_divmod fdm_bits_per_element, const BitmaskElementType* mask_data, bool* mask_result) {
 int bitmask_idx, bitmask_shift;
 fdm_bits_per_element.divmod(id, bitmask_idx, bitmask_shift);
 BitmaskElementType shifted_mask = mask_data[bitmask_idx] >> bitmask_shift;
#pragma unroll
 for (int i = 0; i < NumUnroll; i++) {
  mask_result[i] = (shifted_mask & (1 << i)) != 0;
 }
}
} 
} 