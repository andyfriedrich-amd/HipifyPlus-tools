// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/common/gsl.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {
namespace rocm {

class Transpose final : public RocmKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : RocmKernel(info), TransposeBase(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

  static Status DoTranspose(const Transpose& transpose_kernel,
                            onnxruntime::Stream* ort_stream,
                            const gsl::span<const size_t>& permutations, const Tensor& input, Tensor& output);

  //  `input_shape_override` (if provided) overrides the shape of `input` for compute purposes
  //  `output_shape_override` (if provided) overrides the shape of `output` for compute purposes
  static Status DoTranspose(const hipDeviceProp_t& prop,
                            hipStream_t stream,
                            const rocblas_handle rocblas_handle,
                            const gsl::span<const size_t>& permutations,
                            const Tensor& input, Tensor& output,
                            const TensorShape* input_shape_override = nullptr,
                            const TensorShape* output_shape_override = nullptr);
};

}  // namespace rocm
}  // namespace onnxruntime
