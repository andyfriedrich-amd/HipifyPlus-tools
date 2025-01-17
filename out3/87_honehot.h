// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <typename in_type, typename out_type>
void OneHotImpl(
    hipStream_t stream,
    const in_type* indices,
    const fast_divmod fdm_depth_suffix,
    const fast_divmod fdm_suffix,
    const int64_t depth_val,
    const out_type on_value,
    const out_type off_value,
    out_type* output,
    size_t count);

template <typename in_type, typename out_type>
void OneHotWithZeroOffValueImpl(
    hipStream_t stream,
    const in_type* indices,
    const fast_divmod fdm_suffix,
    const int64_t depth_val,
    const out_type on_value,
    out_type* output,
    size_t count);

template <typename in_type, typename out_type, typename depth_type>
class OneHotOp final : public RocmKernel {
 public:
  explicit OneHotOp(const OpKernelInfo& info) : RocmKernel(info) {
    int64_t tmp_axis;
    if (info.GetAttr<int64_t>("axis", &tmp_axis).IsOK()) {
      axis_ = tmp_axis;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OneHotOp);

  int64_t axis_ = -1;
};

}  // namespace rocm
}  // namespace onnxruntime
