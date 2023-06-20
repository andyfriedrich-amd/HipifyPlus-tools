// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/framework/random_generator.h"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
void BiasDropoutKernelImpl(const hipDeviceProp_t& prop, hipStream_t stream, const int64_t N,
                           const int64_t mask_element_count, const fast_divmod fdm_dim, const float ratio,
                           PhiloxGenerator& generator, const T* X_data, const T* bias_data, const T* residual_data,
                           T* Y_data, void* mask_data, bool has_same_shape_bias, bool use_bitmask);

template <bool UseBitmask>
class BiasDropout final : public RocmKernel {
 public:
  BiasDropout(const OpKernelInfo& info) : RocmKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
