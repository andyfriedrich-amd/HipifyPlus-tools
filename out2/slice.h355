// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace rocm {

namespace SliceRocm {

Status Impl(hipStream_t stream,
            const void* input_data,
            const TensorShape& input_shape,
            void* output_data,
            SliceOp::PrepareForComputeMetadata& prepare_metadata,
            size_t element_size);

}  // namespace SliceRocm

template <bool dynamic>
class Slice : public RocmKernel, public SliceBase {
 public:
  Slice(const OpKernelInfo& info) : RocmKernel(info), SliceBase(info, dynamic) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  virtual const Tensor* GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const;
  virtual Status FillInputVectors(OpKernelContext* ctx, TensorShapeVector& input_starts,
                                  TensorShapeVector& input_ends, TensorShapeVector& input_axes,
                                  TensorShapeVector& input_steps) const;

  virtual Status CallSliceImp(size_t element_size, size_t dimension_count, const TArray<int64_t>& starts_buffer,
                              const TArray<int64_t>& steps_buffer, const TArray<int64_t>& input_strides,
                              const TArray<fast_divmod>& output_strides, OpKernelContext* ctx,
                              const TensorShape& output_shape) const;
};
}  // namespace rocm
}  // namespace onnxruntime
