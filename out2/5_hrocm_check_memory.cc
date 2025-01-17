// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_check_memory.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
void CheckIfMemoryOnCurrentGpuDevice(const void* ptr) {
  hipPointerAttribute_t attrs;
  HIP_CALL_THROW(hipPointerGetAttributes(&attrs, ptr));
  int current_device;
  HIP_CALL_THROW(hipGetDevice(&current_device));
  ORT_ENFORCE(attrs.device == current_device,
              "Current ROCM device is ", current_device,
              " but the memory of pointer ", ptr,
              " is allocated on device ", attrs.device);
}
}  // namespace onnxruntime
