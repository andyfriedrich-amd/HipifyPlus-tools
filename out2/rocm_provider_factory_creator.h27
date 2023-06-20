// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

struct OrtROCMProviderOptions;
struct OrtROCMProviderOptionsV2;

namespace onnxruntime {
// defined in provider_bridge_ort.cc
struct CudaProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtROCMProviderOptions* provider_options);
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtROCMProviderOptionsV2* provider_options);
};
}  // namespace onnxruntime
