// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
// Throw if "ptr" is not allocated on the ROCM device obtained by hipGetDevice.
void CheckIfMemoryOnCurrentGpuDevice(const void* ptr);
}  // namespace onnxruntime