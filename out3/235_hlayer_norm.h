#pragma once
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {
template <typename T, typename U, typename V, bool simplified>
class LayerNormGrad final : public RocmKernel {
 public:
  LayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

template <typename T, typename U, typename V>
class InvertibleLayerNormGrad final : public RocmKernel {
 public:
  InvertibleLayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

}  // namespace rocm
}  // namespace onnxruntime