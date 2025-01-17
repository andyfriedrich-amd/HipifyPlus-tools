// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/tensor/upsample.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
class Resize : public Upsample<T> {
 public:
  Resize(const OpKernelInfo& info) : Upsample<T>(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    return Upsample<T>::ComputeInternal(context);
  }
};

}  // namespace rocm
}  // namespace onnxruntime
